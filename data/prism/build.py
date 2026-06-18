"""Build PRISM GRPO data."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.model_ids import DEFAULT_MODEL_ID as MODEL_ID, load_tokenizer
from shared.prompt_utils import (
    CONDITIONING_MODE_CHOICES,
    CONDITIONING_MODE_HISTORY,
    build_grpo_prompt_payload,
    conditioning_mode_uses_persona,
)
from shared.load_personas import get_persona_for_user, load_persona_map
from data.utils import PUBLIC_DATA_MODES, shuffle_parquet_rows
from data.prism.build_history import (
    DEFAULT_PRISM_CONFIG,
    DEFAULT_PRISM_DATASET,
    DEFAULT_PRISM_SPLIT,
    extract_selected_turns,
    load_prism_dataset,
)


DEFAULT_OUTPUT = "data/prism/history/train.parquet"
DEFAULT_MIN_CONVERSATIONS = 6
DEFAULT_HELDOUT_USER_FRAC = 0.4
DEFAULT_HISTORY_COUNT_MIN = 2
DEFAULT_HISTORY_COUNT_MAX = 4
DEFAULT_EVAL_THREAD_FRAC = 0.3
DEFAULT_PREDICTION_START_TURN = 1
DEFAULT_SPLIT_SEED = 42
DEFAULT_DATA_SOURCE = "prism_alignment_user_sim"
DEFAULT_SOURCE_NAME = "prism_alignment"

_HISTORY_HEADER = (
    "[USER HISTORY]\n"
    "Below are previous conversations involving [HUMAN]. Use them to infer [HUMAN]'s style, "
    "values, preferences, and likely responses."
)


@dataclass(frozen=True)
class PrismConversation:
    conversation_id: str
    user_id: str
    conversation_type: str
    selected_turns: list[dict[str, Any]]

    @property
    def turn_count(self) -> int:
        return len(self.selected_turns)


@dataclass(frozen=True)
class PrismSplitEntries:
    train_entries: list[dict[str, Any]]
    val_entries: list[dict[str, Any]]
    test_entries: list[dict[str, Any]]
    qualified_user_ids: list[str]
    heldout_user_ids: list[str]
    train_user_ids: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "qualified_users": len(self.qualified_user_ids),
            "heldout_users": len(self.heldout_user_ids),
            "train_users_selected": len(self.train_user_ids),
        }


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _default_val_output(output: str) -> str:
    path = Path(output)
    stem = path.stem
    suffix = path.suffix or ".parquet"
    if stem.endswith("_train"):
        stem = f"{stem[:-6]}_val"
    elif stem.endswith("train"):
        stem = f"{stem[:-5]}val"
    else:
        stem = f"{stem}_val"
    return str(path.with_name(f"{stem}{suffix}"))


def _default_test_output(output: str) -> str:
    path = Path(output)
    stem = path.stem
    suffix = path.suffix or ".parquet"
    if stem.endswith("_train"):
        stem = f"{stem[:-6]}_test"
    elif stem.endswith("train"):
        stem = f"{stem[:-5]}test"
    else:
        stem = f"{stem}_test"
    return str(path.with_name(f"{stem}{suffix}"))


def _write_rows_to_parquet(rows: list[dict[str, Any]], output_path: str) -> None:
    import pandas as pd

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_metadata(summary: dict[str, Any], output: str) -> str:
    metadata_path = f"{output}.metadata.json"
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_path


def _conversation_sort_key(conversation: PrismConversation) -> str:
    return conversation.conversation_id


def load_prism_conversations(
    *,
    dataset_name: str = DEFAULT_PRISM_DATASET,
    config_name: str = DEFAULT_PRISM_CONFIG,
    split: str = DEFAULT_PRISM_SPLIT,
) -> list[PrismConversation]:
    dataset = load_prism_dataset(dataset_name=dataset_name, config_name=config_name, split=split)
    conversations: list[PrismConversation] = []
    for row in dataset:
        conversations.append(
            PrismConversation(
                conversation_id=str(row["conversation_id"]),
                user_id=str(row["user_id"]),
                conversation_type=_normalize_text(row.get("conversation_type", "")),
                selected_turns=extract_selected_turns(row.get("conversation_history", [])),
            )
        )
    return conversations


def _group_conversations_by_user(
    conversations: Iterable[PrismConversation],
) -> dict[str, list[PrismConversation]]:
    grouped: dict[str, list[PrismConversation]] = defaultdict(list)
    for conversation in conversations:
        grouped[conversation.user_id].append(conversation)
    for user_id in grouped:
        grouped[user_id] = sorted(grouped[user_id], key=_conversation_sort_key)
    return dict(grouped)


def _per_user_rng(seed: int, user_id: str) -> random.Random:
    return random.Random(f"{seed}:prism:{user_id}")


def _shuffled_conversations_for_user(
    conversations: Sequence[PrismConversation],
    *,
    seed: int,
    user_id: str,
) -> list[PrismConversation]:
    shuffled = list(conversations)
    _per_user_rng(seed, user_id).shuffle(shuffled)
    return shuffled


def _history_count_for_user(
    conversation_count: int,
    *,
    min_count: int,
    max_count: int,
    seed: int,
    user_id: str,
) -> int:
    history_count = _per_user_rng(seed, user_id).randint(min_count, max_count)
    return min(history_count, max(1, conversation_count - 1))


def _format_messages(turns: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        user_text = _normalize_text(turn.get("user", ""))
        assistant_text = _normalize_text(turn.get("assistant", ""))
        if user_text:
            lines.append(f"[HUMAN]: {user_text}")
        if assistant_text:
            lines.append(f"[OTHER]: {assistant_text}")
    return "\n".join(lines).strip()


def format_prism_history(history_conversations: Sequence[PrismConversation]) -> str:
    sections = [_HISTORY_HEADER]
    for conversation_idx, conversation in enumerate(history_conversations, start=1):
        sections.extend(
            [
                "",
                f"<Conversation {conversation_idx}>",
                "[MESSAGES]",
                _format_messages(conversation.selected_turns),
                f"</Conversation {conversation_idx}>",
            ]
        )
    return "\n".join(section for section in sections if section is not None).strip()


def format_prism_current_context(conversation: PrismConversation, *, target_turn_idx: int) -> str:
    message_text = _format_messages(conversation.selected_turns[:target_turn_idx])
    if not message_text:
        raise ValueError(
            f"PRISM current context is empty for conversation_id={conversation.conversation_id} "
            f"target_turn_idx={target_turn_idx}"
        )
    return f"[MESSAGES SO FAR]\n{message_text}"


def _iter_entries_for_conversations(
    *,
    conversations: Sequence[PrismConversation],
    history_conversations: Sequence[PrismConversation],
    user_id: str,
    split_name: str,
    prompt_idx_start: int,
    prediction_start_turn: int,
    dataset_name: str,
    config_name: str,
    source_split: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    prompt_idx = prompt_idx_start
    history_text = format_prism_history(history_conversations)
    history_conversation_ids = [conversation.conversation_id for conversation in history_conversations]
    history_conversation_types = [conversation.conversation_type for conversation in history_conversations]

    for conversation in conversations:
        for target_turn_idx in range(prediction_start_turn, conversation.turn_count):
            ground_truth = _normalize_text(conversation.selected_turns[target_turn_idx].get("user", ""))
            if not ground_truth:
                continue
            current_context = format_prism_current_context(
                conversation,
                target_turn_idx=target_turn_idx,
            )
            entries.append(
                {
                    "prompt_idx": prompt_idx,
                    "user_id": user_id,
                    "raw_user_id": user_id,
                    "source_name": DEFAULT_SOURCE_NAME,
                    "post_id": conversation.conversation_id,
                    "target_idx": target_turn_idx,
                    "ground_truth": ground_truth,
                    "user_history": history_text,
                    "persona": "",
                    "current_context": current_context,
                    "split": split_name,
                    "history_thread_ids": history_conversation_ids,
                    "history_count": len(history_conversations),
                    "history_conversation_types": history_conversation_types,
                    "target_conversation_type": conversation.conversation_type,
                    "dataset_name": dataset_name,
                    "dataset_config": config_name,
                    "dataset_split": source_split,
                }
            )
            prompt_idx += 1
    return entries


def build_prism_split_entries(
    conversations: Iterable[PrismConversation],
    *,
    min_conversations: int = DEFAULT_MIN_CONVERSATIONS,
    heldout_user_frac: float = DEFAULT_HELDOUT_USER_FRAC,
    max_train_users: int | None = None,
    train_user_offset: int = 0,
    history_count_min: int = DEFAULT_HISTORY_COUNT_MIN,
    history_count_max: int = DEFAULT_HISTORY_COUNT_MAX,
    eval_thread_frac: float = DEFAULT_EVAL_THREAD_FRAC,
    prediction_start_turn: int = DEFAULT_PREDICTION_START_TURN,
    split_seed: int = DEFAULT_SPLIT_SEED,
    dataset_name: str = DEFAULT_PRISM_DATASET,
    config_name: str = DEFAULT_PRISM_CONFIG,
    source_split: str = DEFAULT_PRISM_SPLIT,
) -> PrismSplitEntries:
    if min_conversations < 1:
        raise ValueError(f"min_conversations must be >= 1, got {min_conversations}")
    if not 0.0 <= heldout_user_frac < 1.0:
        raise ValueError(f"heldout_user_frac must be >= 0 and < 1, got {heldout_user_frac}")
    if history_count_min < 1:
        raise ValueError(f"history_count_min must be >= 1, got {history_count_min}")
    if history_count_max < history_count_min:
        raise ValueError(
            f"history_count_max must be >= history_count_min, got min={history_count_min} max={history_count_max}"
        )
    if not 0.0 <= eval_thread_frac < 1.0:
        raise ValueError(f"eval_thread_frac must be >= 0 and < 1, got {eval_thread_frac}")
    if prediction_start_turn < 1:
        raise ValueError(f"prediction_start_turn must be >= 1, got {prediction_start_turn}")
    if train_user_offset < 0:
        raise ValueError(f"train_user_offset must be >= 0, got {train_user_offset}")
    if max_train_users is not None and max_train_users < 1:
        raise ValueError(f"max_train_users must be >= 1 when set, got {max_train_users}")

    conversations_by_user = _group_conversations_by_user(conversations)
    conversation_counts = {
        user_id: len(user_conversations)
        for user_id, user_conversations in conversations_by_user.items()
    }
    qualified_user_ids = sorted(
        user_id
        for user_id, count in conversation_counts.items()
        if count >= min_conversations
    )
    shuffled_qualified_user_ids = list(qualified_user_ids)
    random.Random(split_seed).shuffle(shuffled_qualified_user_ids)

    heldout_user_count = int(heldout_user_frac * len(shuffled_qualified_user_ids))
    heldout_user_ids = shuffled_qualified_user_ids[:heldout_user_count]
    train_user_ids = shuffled_qualified_user_ids[heldout_user_count:]
    if train_user_offset:
        train_user_ids = train_user_ids[train_user_offset:]
    if max_train_users is not None:
        train_user_ids = train_user_ids[:max_train_users]

    train_entries: list[dict[str, Any]] = []
    val_entries: list[dict[str, Any]] = []
    test_entries: list[dict[str, Any]] = []

    for user_id in train_user_ids:
        user_conversations = _shuffled_conversations_for_user(
            conversations_by_user[user_id],
            seed=split_seed,
            user_id=user_id,
        )
        history_count = _history_count_for_user(
            len(user_conversations),
            min_count=history_count_min,
            max_count=history_count_max,
            seed=split_seed,
            user_id=user_id,
        )
        history_conversations = user_conversations[:history_count]
        target_conversations = user_conversations[history_count:]
        train_entries.extend(
            _iter_entries_for_conversations(
                conversations=target_conversations,
                history_conversations=history_conversations,
                user_id=user_id,
                split_name="train",
                prompt_idx_start=len(train_entries),
                prediction_start_turn=prediction_start_turn,
                dataset_name=dataset_name,
                config_name=config_name,
                source_split=source_split,
            )
        )

    for user_id in heldout_user_ids:
        user_conversations = _shuffled_conversations_for_user(
            conversations_by_user[user_id],
            seed=split_seed,
            user_id=user_id,
        )
        history_count = _history_count_for_user(
            len(user_conversations),
            min_count=history_count_min,
            max_count=history_count_max,
            seed=split_seed,
            user_id=user_id,
        )
        history_conversations = user_conversations[:history_count]
        remaining_conversations = user_conversations[history_count:]
        eval_count = max(1, int(eval_thread_frac * len(remaining_conversations))) if remaining_conversations else 0
        if eval_count >= len(remaining_conversations) and len(remaining_conversations) > 1:
            eval_count = len(remaining_conversations) - 1
        eval_conversations = remaining_conversations[:eval_count]
        test_conversations = remaining_conversations[eval_count:]
        val_entries.extend(
            _iter_entries_for_conversations(
                conversations=eval_conversations,
                history_conversations=history_conversations,
                user_id=user_id,
                split_name="val",
                prompt_idx_start=len(val_entries),
                prediction_start_turn=prediction_start_turn,
                dataset_name=dataset_name,
                config_name=config_name,
                source_split=source_split,
            )
        )
        test_entries.extend(
            _iter_entries_for_conversations(
                conversations=test_conversations,
                history_conversations=history_conversations,
                user_id=user_id,
                split_name="test",
                prompt_idx_start=len(test_entries),
                prediction_start_turn=prediction_start_turn,
                dataset_name=dataset_name,
                config_name=config_name,
                source_split=source_split,
            )
        )

    test_entries = sorted(
        test_entries,
        key=lambda entry: (
            str(entry["user_id"]),
            str(entry["post_id"]),
            int(entry["target_idx"]),
        ),
    )
    for prompt_idx, entry in enumerate(test_entries):
        entry["prompt_idx"] = prompt_idx

    return PrismSplitEntries(
        train_entries=train_entries,
        val_entries=val_entries,
        test_entries=test_entries,
        qualified_user_ids=qualified_user_ids,
        heldout_user_ids=heldout_user_ids,
        train_user_ids=train_user_ids,
    )


def build_prism_grpo_row(
    entry: dict[str, Any],
    tokenizer: Any,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> dict[str, Any]:
    persona = str(entry.get("persona", "") or "").strip()
    if conditioning_mode_uses_persona(conditioning_mode) and not persona:
        raise ValueError(
            f"Missing persona for user_id={entry.get('user_id', '')} while "
            f"conditioning_mode={conditioning_mode!r}"
        )
    prompt_payload = build_grpo_prompt_payload(
        tokenizer,
        user_history=entry["user_history"],
        thread_context=entry["current_context"],
        prompt_mode="reasoning",
        persona=persona,
        conditioning_mode=conditioning_mode,
    )
    return {
        "data_source": data_source,
        "prompt": prompt_payload["prompt"],
        "reward_model": {
            "style": "rule",
            "ground_truth": entry["ground_truth"],
        },
        "extra_info": {
            "user_id": entry["user_id"],
            "raw_user_id": entry.get("raw_user_id", entry["user_id"]),
            "source_name": entry.get("source_name", DEFAULT_SOURCE_NAME),
            "user_history": entry["user_history"],
            "persona": persona,
            "post_id": entry["post_id"],
            "target_idx": entry["target_idx"],
            "context": entry["current_context"],
            "thread_context": entry["current_context"],
            "prompt_text": prompt_payload["prompt_text"],
            "prompt_mode": prompt_payload["prompt_mode"],
            "conditioning_mode": prompt_payload["conditioning_mode"],
            "split": entry.get("split", ""),
            "prompt_idx": entry["prompt_idx"],
            "index": entry["prompt_idx"],
            "raw_prompt": prompt_payload["raw_prompt"],
            "history_thread_ids": entry.get("history_thread_ids", []),
            "history_count": entry.get("history_count", len(entry.get("history_thread_ids", []))),
            "history_conversation_types": entry.get("history_conversation_types", []),
            "target_conversation_type": entry.get("target_conversation_type", ""),
            "dataset_name": entry.get("dataset_name", DEFAULT_PRISM_DATASET),
            "dataset_config": entry.get("dataset_config", DEFAULT_PRISM_CONFIG),
            "dataset_split": entry.get("dataset_split", DEFAULT_PRISM_SPLIT),
        },
    }


def _build_rows(
    entries: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    mode: str,
    data_source: str,
    conditioning_mode: str,
    persona_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if mode not in PUBLIC_DATA_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    persona_map = persona_map or {}
    prepared_entries: list[dict[str, Any]] = []
    for entry in entries:
        prepared_entry = dict(entry)
        if conditioning_mode_uses_persona(conditioning_mode):
            prepared_entry["persona"] = get_persona_for_user(
                persona_map,
                prepared_entry["user_id"],
                prepared_entry.get("raw_user_id", prepared_entry["user_id"]),
            )
        prepared_entries.append(prepared_entry)
    return [
        build_prism_grpo_row(
            entry,
            tokenizer,
            data_source=data_source,
            conditioning_mode=conditioning_mode,
        )
        for entry in prepared_entries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PRISM Alignment GRPO data")
    parser.add_argument("--dataset_name", default=DEFAULT_PRISM_DATASET)
    parser.add_argument("--config_name", default=DEFAULT_PRISM_CONFIG)
    parser.add_argument("--split", default=DEFAULT_PRISM_SPLIT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output train parquet path.")
    parser.add_argument("--val_output", default=None, help="Optional output val parquet path.")
    parser.add_argument("--test_output", default=None, help="Optional output test parquet path.")
    parser.add_argument("--min_conversations", type=int, default=DEFAULT_MIN_CONVERSATIONS)
    parser.add_argument("--heldout_user_frac", type=float, default=DEFAULT_HELDOUT_USER_FRAC)
    parser.add_argument("--max_train_users", type=int, default=None)
    parser.add_argument("--train_user_offset", type=int, default=0)
    parser.add_argument("--history_count_min", type=int, default=DEFAULT_HISTORY_COUNT_MIN)
    parser.add_argument("--history_count_max", type=int, default=DEFAULT_HISTORY_COUNT_MAX)
    parser.add_argument("--eval_thread_frac", type=float, default=DEFAULT_EVAL_THREAD_FRAC)
    parser.add_argument("--prediction_start_turn", type=int, default=DEFAULT_PREDICTION_START_TURN)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--mode", choices=sorted(PUBLIC_DATA_MODES), default="reasoning")
    parser.add_argument(
        "--conditioning_mode",
        choices=CONDITIONING_MODE_CHOICES,
        default=CONDITIONING_MODE_HISTORY,
    )
    parser.add_argument(
        "--persona_path",
        default=None,
        help="Persona map JSON/JSONL/pickle path required for persona-backed conditioning modes.",
    )
    parser.add_argument("--tokenizer", default=MODEL_ID)
    parser.add_argument("--data_source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument(
        "--shuffle_rows",
        action="store_true",
        help="Deterministically shuffle train/val parquet rows after writing. Test rows stay ordered.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    if conditioning_mode_uses_persona(args.conditioning_mode) and not args.persona_path:
        raise ValueError(f"--persona_path is required when --conditioning_mode={args.conditioning_mode}")
    persona_map = load_persona_map(args.persona_path)
    tokenizer = load_tokenizer(args.tokenizer)
    conversations = load_prism_conversations(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
    )
    split_entries = build_prism_split_entries(
        conversations,
        min_conversations=args.min_conversations,
        heldout_user_frac=args.heldout_user_frac,
        max_train_users=args.max_train_users,
        train_user_offset=args.train_user_offset,
        history_count_min=args.history_count_min,
        history_count_max=args.history_count_max,
        eval_thread_frac=args.eval_thread_frac,
        prediction_start_turn=args.prediction_start_turn,
        split_seed=args.split_seed,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        source_split=args.split,
    )
    if not split_entries.train_entries:
        raise SystemExit("No PRISM train entries generated after filtering and splitting.")

    train_rows = _build_rows(
        split_entries.train_entries,
        tokenizer=tokenizer,
        mode=args.mode,
        data_source=args.data_source,
        conditioning_mode=args.conditioning_mode,
        persona_map=persona_map,
    )
    val_rows = _build_rows(
        split_entries.val_entries,
        tokenizer=tokenizer,
        mode=args.mode,
        data_source=args.data_source,
        conditioning_mode=args.conditioning_mode,
        persona_map=persona_map,
    )
    test_rows = _build_rows(
        split_entries.test_entries,
        tokenizer=tokenizer,
        mode=args.mode,
        data_source=args.data_source,
        conditioning_mode=args.conditioning_mode,
        persona_map=persona_map,
    )

    val_output = args.val_output or _default_val_output(args.output)
    test_output = args.test_output or _default_test_output(args.output)
    _write_rows_to_parquet(train_rows, args.output)
    _write_rows_to_parquet(val_rows, val_output)
    _write_rows_to_parquet(test_rows, test_output)
    if args.shuffle_rows:
        shuffle_parquet_rows(args.output, seed=args.shuffle_seed)
        shuffle_parquet_rows(val_output, seed=args.shuffle_seed + 1)

    summary = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "hf_split": args.split,
        "seed": args.split_seed,
        "counts": {
            "train": {"rows": len(train_rows), "users": len(split_entries.train_user_ids)},
            "val": {"rows": len(val_rows), "users": len(split_entries.heldout_user_ids)},
            "test": {"rows": len(test_rows), "users": len(split_entries.heldout_user_ids)},
        },
        "min_conversations": args.min_conversations,
        "heldout_user_frac": args.heldout_user_frac,
        "max_train_users": args.max_train_users,
        "train_user_offset": args.train_user_offset,
        "history_count_min": args.history_count_min,
        "history_count_max": args.history_count_max,
        "eval_thread_frac": args.eval_thread_frac,
        "prediction_start_turn": args.prediction_start_turn,
        "shuffle_seed": args.shuffle_seed if args.shuffle_rows else None,
        "conditioning_mode": args.conditioning_mode,
        "persona_path": args.persona_path if conditioning_mode_uses_persona(args.conditioning_mode) else None,
        "data_source": args.data_source,
        "user_split": split_entries.to_metadata(),
    }
    metadata_path = _write_metadata(summary, args.output)

    print(f"Dataset: {args.dataset_name} / {args.config_name} / {args.split}")
    print(
        f"Qualified users: {len(split_entries.qualified_user_ids)} "
        f"(heldout={len(split_entries.heldout_user_ids)} train={len(split_entries.train_user_ids)})"
    )
    print(f"Train: {len(train_rows)} rows -> {args.output}")
    print(f"Val:   {len(val_rows)} rows -> {val_output}")
    print(f"Test:  {len(test_rows)} rows -> {test_output}")
    print(f"Metadata: {metadata_path}")
    sample = train_rows[0]["prompt"][1]["content"] if train_rows else ""
    print("Sample prompt structure:")
    for marker in ("[USER HISTORY]", "[CURRENT CONTEXT]", "[MESSAGES SO FAR]"):
        print(f"  {marker}: {marker in sample}")
    print(sample[:500])


if __name__ == "__main__":
    main()
