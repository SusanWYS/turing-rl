"""Render history/persona conditioning variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import normalize_reddit_reply_artifacts
from shared.prompt_utils import (
    CONDITIONING_MODE_CHOICES,
    CONDITIONING_MODE_HISTORY,
    CONDITIONING_MODE_HISTORY_PERSONA,
    CONDITIONING_MODE_PERSONA,
    build_grpo_prompt_payload,
    conditioning_mode_uses_history,
    conditioning_mode_uses_persona,
)
from shared.load_personas import get_persona_for_user, load_persona_map


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def render_row_for_mode(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    conditioning_mode: str,
) -> dict[str, Any]:
    """Render one row for one conditioning mode."""
    extra_info = dict(row.get("extra_info") or {})
    reward_model = dict(row.get("reward_model") or {})

    prompt_mode = str(extra_info.get("prompt_mode") or "reasoning")
    context = normalize_reddit_reply_artifacts(
        _as_text(extra_info.get("thread_context") or extra_info.get("context"))
    )
    source_history = _as_text(extra_info.get("user_history"))
    source_persona = _as_text(extra_info.get("persona"))

    if conditioning_mode_uses_history(conditioning_mode) and not source_history.strip():
        raise ValueError(
            f"Row user_id={extra_info.get('user_id')} post_id={extra_info.get('post_id')} "
            f"has empty user_history; required for conditioning_mode={conditioning_mode!r}."
        )
    if conditioning_mode_uses_persona(conditioning_mode) and not source_persona.strip():
        raise ValueError(
            f"Row user_id={extra_info.get('user_id')} post_id={extra_info.get('post_id')} "
            f"has empty persona; required for conditioning_mode={conditioning_mode!r}."
        )

    prompt_history = source_history if conditioning_mode_uses_history(conditioning_mode) else ""
    persona = source_persona if conditioning_mode_uses_persona(conditioning_mode) else ""

    payload = build_grpo_prompt_payload(
        tokenizer,
        user_history=prompt_history,
        thread_context=context,
        prompt_mode=prompt_mode,
        persona=persona,
        conditioning_mode=conditioning_mode,
    )

    new_extra = dict(extra_info)
    new_extra["conditioning_mode"] = conditioning_mode
    new_extra["context"] = context
    new_extra["thread_context"] = context
    new_extra["prompt_text"] = payload["prompt_text"]
    new_extra["raw_prompt"] = payload["raw_prompt"]
    new_extra["prompt_mode"] = payload["prompt_mode"]
    new_extra["user_history"] = source_history
    if not conditioning_mode_uses_persona(conditioning_mode):
        new_extra["persona"] = ""

    reward_model["ground_truth"] = normalize_reddit_reply_artifacts(
        _as_text(reward_model.get("ground_truth"))
    )

    new_row = dict(row)
    new_row["prompt"] = payload["prompt"]
    new_row["reward_model"] = reward_model
    new_row["extra_info"] = new_extra
    return new_row


DEFAULT_DERIVE_SPLIT_FILES = (
    "grpo/train.parquet",
    "grpo/val.parquet",
    "grpo/test.parquet",
    "sft/train.parquet",
    "sft/val.parquet",
    "sft/test.parquet",
    "test.parquet",
)


def attach_persona_from_map(
    row: dict[str, Any], persona_map: dict[str, str]
) -> dict[str, Any]:
    """Attach persona text from the map."""
    extra_info = dict(row.get("extra_info") or {})
    user_id = str(extra_info.get("user_id", "") or "")
    raw_user_id = str(extra_info.get("raw_user_id", "") or "")
    extra_info["persona"] = get_persona_for_user(persona_map, user_id, raw_user_id)
    new_row = dict(row)
    new_row["extra_info"] = extra_info
    return new_row


def derive_split_file(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer: Any,
    conditioning_mode: str,
    persona_map: dict[str, str],
) -> int:
    """Render one split parquet."""
    import pandas as pd

    attach = conditioning_mode_uses_persona(conditioning_mode)
    df = pd.read_parquet(input_path)
    rendered = []
    for record in df.to_dict(orient="records"):
        if attach:
            record = attach_persona_from_map(record, persona_map)
        rendered.append(
            render_row_for_mode(record, tokenizer=tokenizer, conditioning_mode=conditioning_mode)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rendered).to_parquet(output_path, index=False)
    return len(rendered)


def _write_derive_metadata(
    output_dir: Path,
    *,
    conditioning_mode: str,
    tokenizer_name: str,
    derived_files: list[dict[str, Any]],
) -> None:
    metadata = {
        "conditioning_mode": conditioning_mode,
        "tokenizer": tokenizer_name,
        "counts": {
            item["split"]: {"rows": item["rows"]}
            for item in derived_files
        },
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def derive_conditioning_dir(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    conditioning_mode: str,
    tokenizer: Any,
    persona_map: dict[str, str] | None = None,
    tokenizer_name: str = "",
    split_files: Iterable[str] = DEFAULT_DERIVE_SPLIT_FILES,
) -> list[dict[str, Any]]:
    """Derive one conditioning-mode build."""
    persona_map = persona_map or {}
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_files: list[dict[str, Any]] = []
    for relative in split_files:
        input_path = input_dir / relative
        if not input_path.is_file():
            continue
        output_path = output_dir / relative
        rows = derive_split_file(
            input_path,
            output_path,
            tokenizer=tokenizer,
            conditioning_mode=conditioning_mode,
            persona_map=persona_map,
        )
        derived_files.append({"split": relative, "rows": rows})
    _write_derive_metadata(
        output_dir,
        conditioning_mode=conditioning_mode,
        tokenizer_name=tokenizer_name,
        derived_files=derived_files,
    )
    return derived_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a conditioning-mode build (history_persona / persona / history) "
            "from a source build directory and an induced persona map."
        )
    )
    parser.add_argument(
        "--input-dir", required=True, help="Source build dir (history-only, split layout)."
    )
    parser.add_argument("--output-dir", required=True, help="Destination build dir.")
    parser.add_argument(
        "--conditioning-mode",
        required=True,
        choices=CONDITIONING_MODE_CHOICES,
    )
    parser.add_argument(
        "--persona-path",
        default=None,
        help="Persona map (jsonl/json/pickle); required for persona-backed modes.",
    )
    parser.add_argument(
        "--tokenizer", default=None, help="Tokenizer/model id (defaults to the shared default)."
    )
    parser.add_argument("--split-files", nargs="+", default=list(DEFAULT_DERIVE_SPLIT_FILES))
    args = parser.parse_args()

    from shared.model_ids import DEFAULT_MODEL_ID, load_tokenizer

    if conditioning_mode_uses_persona(args.conditioning_mode) and not args.persona_path:
        raise SystemExit(f"--persona-path is required for --conditioning-mode={args.conditioning_mode}")
    persona_map = load_persona_map(args.persona_path) if args.persona_path else {}
    tokenizer_name = args.tokenizer or DEFAULT_MODEL_ID
    tokenizer = load_tokenizer(tokenizer_name)

    derived = derive_conditioning_dir(
        args.input_dir,
        args.output_dir,
        conditioning_mode=args.conditioning_mode,
        tokenizer=tokenizer,
        persona_map=persona_map,
        tokenizer_name=tokenizer_name,
        split_files=args.split_files,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "conditioning_mode": args.conditioning_mode,
                "files": len(derived),
                "rows": sum(item["rows"] for item in derived),
            }
        )
    )


if __name__ == "__main__":
    main()
