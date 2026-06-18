"""Generate SFT reasoning traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.api_client import build_chat_payload, get_openai_max_retries, post_chat_sync
from shared.judge_utils import build_source_copy_warning

DEFAULT_COT_MODEL = "qwen/qwen3-8b"
DEFAULT_MAX_REGEN_ATTEMPTS = 10
DEFAULT_LEAKAGE_NGRAM_SIZE = 5
DEFAULT_LEAKAGE_MAX_MATCH_TOKENS = 5
DEFAULT_MAX_COMPLETION_TOKENS = 4096
THINKING_TRACE_SOURCE = "data.sft.generate_cot"

RATIONALIZE_SYSTEM_PROMPT = (
    "You reconstruct a Reddit user's private reasoning. You are given the "
    "conversation context and the reply the user actually wrote. Write the user's "
    "first-person, step-by-step reasoning that leads naturally to that exact "
    "reply: what they noticed in the context, their intent, stance, and tone. Do "
    "not quote or restate the reply verbatim; reason about it. Output only the "
    "reasoning, with no preamble and no copy of the reply."
)
RATIONALIZE_USER_TEMPLATE = (
    "[CONTEXT]\n{context}\n\n"
    "[THE USER'S ACTUAL REPLY]\n{ground_truth}\n\n"
    "Write the user's reasoning that leads to this reply."
)
REGEN_NUDGE = (
    "Your previous reasoning copied wording from the reply. Rewrite the reasoning "
    "entirely in your own words, without quoting or restating the reply."
)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _row_context(extra_info: dict[str, Any]) -> str:
    sections = []
    persona = _as_text(extra_info.get("persona")).strip()
    history = _as_text(extra_info.get("user_history")).strip()
    context = _as_text(extra_info.get("context") or extra_info.get("thread_context")).strip()
    if persona:
        sections.append(f"[PERSONA]\n{persona}")
    if history:
        sections.append(f"[USER HISTORY]\n{history}")
    if context:
        sections.append(f"[CURRENT CONTEXT]\n{context}")
    return "\n\n".join(sections)


def reasoning_leaks_reply(
    reasoning: str,
    ground_truth: str,
    *,
    ngram_size: int = DEFAULT_LEAKAGE_NGRAM_SIZE,
    max_match_tokens: int = DEFAULT_LEAKAGE_MAX_MATCH_TOKENS,
) -> bool:
    """Return whether reasoning copies the reply."""
    if not reasoning.strip() or not ground_truth.strip():
        return False
    warning = build_source_copy_warning(reasoning, thread_context=ground_truth, ngram_size=ngram_size)
    return bool(warning.get("triggered")) and int(warning.get("longest_match_tokens", 0)) >= max_match_tokens


def generate_reasoning_for_row(
    extra_info: dict[str, Any],
    ground_truth: str,
    *,
    model: str = DEFAULT_COT_MODEL,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    max_regen_attempts: int = DEFAULT_MAX_REGEN_ATTEMPTS,
    ngram_size: int = DEFAULT_LEAKAGE_NGRAM_SIZE,
    max_match_tokens: int = DEFAULT_LEAKAGE_MAX_MATCH_TOKENS,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """Generate one reasoning trace."""
    if max_retries is None:
        max_retries = get_openai_max_retries()
    base_messages = [
        {"role": "system", "content": RATIONALIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RATIONALIZE_USER_TEMPLATE.format(
                context=_row_context(extra_info), ground_truth=ground_truth
            ),
        },
    ]
    reasoning = ""
    attempts = 0
    leaked = True
    for attempt in range(1, max(1, max_regen_attempts) + 1):
        attempts = attempt
        messages = base_messages if attempt == 1 else base_messages + [{"role": "user", "content": REGEN_NUDGE}]
        payload = build_chat_payload(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            reasoning=True,
        )
        content = post_chat_sync(payload, max_retries=max_retries)
        reasoning = (content or "").strip()
        leaked = reasoning_leaks_reply(
            reasoning, ground_truth, ngram_size=ngram_size, max_match_tokens=max_match_tokens
        )
        if reasoning and not leaked:
            break
    return {
        "ground_truth_reasoning": reasoning,
        "thinking_trace_source": THINKING_TRACE_SOURCE,
        "thinking_trace_model": model,
        "thinking_trace_num_regen_attempts": attempts,
        "thinking_trace_failed_leakage_guard": bool(leaked),
    }


def annotate_rows(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, int]:
    """Annotate rows with reasoning traces."""
    written = 0
    failed_guard = 0
    skipped = 0
    for row in rows:
        extra_info = dict(row.get("extra_info") or {})
        reward_model = dict(row.get("reward_model") or {})
        ground_truth = _as_text(reward_model.get("ground_truth"))
        if not ground_truth.strip():
            skipped += 1
            continue
        trace = generate_reasoning_for_row(extra_info, ground_truth, **kwargs)
        extra_info.update(trace)
        row["extra_info"] = extra_info
        written += 1
        if trace["thinking_trace_failed_leakage_guard"]:
            failed_guard += 1
    return {"rows_written": written, "rows_failed_leakage_guard": failed_guard, "rows_skipped": skipped}


def generate_cot_for_parquet(input_path: str | Path, output_path: str | Path, **kwargs: Any) -> dict[str, int]:
    """Generate reasoning for one parquet."""
    import pandas as pd  # lazy: keeps the rationalize/leakage logic importable without pandas

    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = pd.read_parquet(input_path).to_dict(orient="records")
    rows = [dict(row) for row in rows]
    stats = annotate_rows(rows, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)

    metadata = {
        "thinking_trace_model": kwargs.get("model", DEFAULT_COT_MODEL),
        "max_regen_attempts": kwargs.get("max_regen_attempts", DEFAULT_MAX_REGEN_ATTEMPTS),
        "leakage_ngram_size": kwargs.get("ngram_size", DEFAULT_LEAKAGE_NGRAM_SIZE),
        "leakage_max_match_tokens": kwargs.get("max_match_tokens", DEFAULT_LEAKAGE_MAX_MATCH_TOKENS),
        **stats,
    }
    Path(f"{output_path}.cot_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT-think reasoning traces for an SFT parquet via OpenRouter."
    )
    parser.add_argument("--input", required=True, help="Source SFT parquet (prompt + reward_model.ground_truth).")
    parser.add_argument("--output", required=True, help="Destination parquet with ground_truth_reasoning added.")
    parser.add_argument("--model", default=DEFAULT_COT_MODEL, help="OpenRouter model slug (thinking).")
    parser.add_argument("--max_completion_tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--max_regen_attempts", type=int, default=DEFAULT_MAX_REGEN_ATTEMPTS)
    parser.add_argument("--leakage_ngram_size", type=int, default=DEFAULT_LEAKAGE_NGRAM_SIZE)
    parser.add_argument("--leakage_max_match_tokens", type=int, default=DEFAULT_LEAKAGE_MAX_MATCH_TOKENS)
    args = parser.parse_args()

    stats = generate_cot_for_parquet(
        args.input,
        args.output,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        max_regen_attempts=args.max_regen_attempts,
        ngram_size=args.leakage_ngram_size,
        max_match_tokens=args.leakage_max_match_tokens,
    )
    print(json.dumps({"output": args.output, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
