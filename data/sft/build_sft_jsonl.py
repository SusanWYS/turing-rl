"""Build SFT JSONL from prompt parquets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.prompt_utils import normalize_prompt_messages
from shared.sft_prompt_utils import format_sft_assistant_content


def extract_ground_truth(reward_model: Any) -> str:
    """Extract the target response."""
    if not isinstance(reward_model, dict):
        raise TypeError(f"Expected reward_model dict, got {type(reward_model)!r}")
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        raise ValueError("reward_model is missing ground_truth")
    return str(ground_truth)


def extract_ground_truth_reasoning(row: Any) -> str | None:
    """Extract optional reasoning."""
    for cell_name in ("extra_info", "reward_model"):
        cell = row.get(cell_name)
        if not isinstance(cell, dict):
            continue
        for key in ("ground_truth_reasoning", "thinking_trace", "cot"):
            value = cell.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def build_sft_record(row: Any) -> dict[str, list[dict[str, str]]]:
    """Build one SFT record."""
    messages = normalize_prompt_messages(row["prompt"])
    if not messages:
        raise ValueError("row has no prompt messages")
    ground_truth = extract_ground_truth(row["reward_model"])
    ground_truth_reasoning = extract_ground_truth_reasoning(row)
    if not ground_truth_reasoning:
        raise ValueError(
            "row has no ground_truth_reasoning; SFT is CoT-only. "
            "Run data.sft.generate_cot to annotate reasoning before building the JSONL."
        )
    messages = [
        *messages,
        {
            "role": "assistant",
            "content": format_sft_assistant_content(ground_truth, ground_truth_reasoning),
        },
    ]
    return {"messages": messages}


def convert_parquet_to_jsonl(
    input_parquet: str,
    output_jsonl: str,
    *,
    max_rows: int | None = None,
) -> int:
    """Convert one parquet to JSONL."""
    df = pd.read_parquet(input_parquet)
    if max_rows is not None:
        df = df.head(max_rows)
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    rows_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = build_sft_record(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_parquet", required=True, help="GRPO-format train parquet")
    parser.add_argument("--output_jsonl", required=True, help="Output SFT JSONL path")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row limit for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = convert_parquet_to_jsonl(
        args.input_parquet,
        args.output_jsonl,
        max_rows=args.max_rows,
    )
    print(f"Wrote {rows_written} SFT examples to {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
