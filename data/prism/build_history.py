"""Load PRISM conversations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from datasets import Dataset, load_dataset


DEFAULT_PRISM_DATASET = "HannahRoseKirk/prism-alignment"
DEFAULT_PRISM_CONFIG = "conversations"
DEFAULT_PRISM_SPLIT = "train"


def _normalize_text(text: Any) -> str:
    compact = " ".join(str(text or "").split())
    return compact.strip()


def load_prism_dataset(
    *,
    dataset_name: str = DEFAULT_PRISM_DATASET,
    config_name: str = DEFAULT_PRISM_CONFIG,
    split: str = DEFAULT_PRISM_SPLIT,
) -> Dataset:
    """Load the PRISM split."""
    return load_dataset(dataset_name, config_name, split=split)


def extract_selected_turns(conversation_history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the chosen user/assistant path."""
    turn_records: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "turn": 0,
            "user": "",
            "assistant": "",
            "assistant_model": "",
            "assistant_provider": "",
            "assistant_score": None,
            "_candidates": [],
        }
    )

    for item in conversation_history:
        turn_idx = int(item.get("turn", 0) or 0)
        record = turn_records[turn_idx]
        record["turn"] = turn_idx
        role = str(item.get("role", "") or "").strip().lower()
        content = _normalize_text(item.get("content", ""))
        if role == "user":
            if content:
                record["user"] = content
            continue
        if role != "model":
            continue
        candidate = {
            "content": content,
            "if_chosen": bool(item.get("if_chosen")),
            "score": item.get("score"),
            "within_turn_id": item.get("within_turn_id"),
            "model_name": _normalize_text(item.get("model_name", "")),
            "model_provider": _normalize_text(item.get("model_provider", "")),
        }
        record["_candidates"].append(candidate)
        if candidate["if_chosen"] and content and not record["assistant"]:
            record["assistant"] = content
            record["assistant_model"] = candidate["model_name"]
            record["assistant_provider"] = candidate["model_provider"]
            record["assistant_score"] = candidate["score"]

    selected_turns: list[dict[str, Any]] = []
    for turn_idx in sorted(turn_records):
        record = turn_records[turn_idx]
        if not record["user"]:
            continue
        if not record["assistant"] and record["_candidates"]:
            fallback = max(
                record["_candidates"],
                key=lambda candidate: (
                    float(candidate["score"]) if candidate["score"] is not None else float("-inf"),
                    -(int(candidate["within_turn_id"]) if candidate["within_turn_id"] is not None else 1_000_000),
                ),
            )
            record["assistant"] = fallback["content"]
            record["assistant_model"] = fallback["model_name"]
            record["assistant_provider"] = fallback["model_provider"]
            record["assistant_score"] = fallback["score"]
        selected_turns.append(
            {
                "turn": turn_idx,
                "user": record["user"],
                "assistant": record["assistant"],
                "assistant_model": record["assistant_model"],
                "assistant_provider": record["assistant_provider"],
                "assistant_score": record["assistant_score"],
            }
        )
    return selected_turns
