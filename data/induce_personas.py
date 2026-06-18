"""Induce personas from fixed user histories."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.api_client import build_chat_payload, get_openai_max_retries, post_chat_sync
from shared.judge_utils import build_source_copy_warning


DEFAULT_PERSONA_MODEL: Final[str] = "gpt-5.4-nano"
DEFAULT_PERSONA_TEMPERATURE: Final[float] = 0.0
DEFAULT_PERSONA_MAX_COMPLETION_TOKENS: Final[int] = 1024
DEFAULT_PERSONA_REASONING: Final[bool] = False
DEFAULT_PERSONA_MAX_HISTORY_WORDS: Final[int] = 8192
DEFAULT_PERSONA_ATTEMPTS: Final[int] = 3
DEFAULT_PERSONA_WORKERS: Final[int] = 24
DEFAULT_PERSONA_MAX_ROUNDS: Final[int] = 5
DEFAULT_PERSONA_LEAKAGE_NGRAM: Final[int] = 8
DEFAULT_PERSONA_LEAKAGE_MAX_MATCH_TOKENS: Final[int] = 8
PERSONA_MODEL_ALIASES: Final[dict[str, str]] = {
    "gpt-5.4-nano": "gpt-5.4-nano",
    "opus4.8": "anthropic/claude-opus-4.8",
    "qwen3-8b": "qwen/qwen3-8b",
}

PERSONA_INDUCTION_SYSTEM_PROMPT: Final[str] = (
    "You write compressed first-person persona notes as if the target user wrote them about their own habits.\n"
    "Sound like the user, not like an analyst, therapist, teacher, or policy brief.\n"
    "Output only the requested strict JSON object and nothing else."
)

PERSONA_JSON_KEYS: Final[tuple[str, ...]] = (
    "values",
    "verbal_quirks",
    "expression_style",
    "length_prior",
    "background",
)


def normalize_persona_model(model: str) -> str:
    """Resolve persona-inductor aliases."""
    if model not in PERSONA_MODEL_ALIASES:
        allowed = ", ".join(PERSONA_MODEL_ALIASES)
        raise ValueError(f"Unsupported persona inductor {model!r}. Expected one of: {allowed}")
    return PERSONA_MODEL_ALIASES[model]


def truncate_history_words(text: str, max_words: int) -> str:
    """Keep the last max_words words."""
    words = text.split()
    if max_words < 1 or len(words) <= max_words:
        return text
    return " ".join(words[-max_words:])


def build_persona_induction_user_prompt(selected_history: str) -> str:
    """Build the persona induction prompt."""
    return (
        "<target_speaker>[HUMAN]</target_speaker>\n"
        "<selected_history>\n"
        f"{selected_history.strip()}\n"
        "</selected_history>\n\n"
        "<task>\n"
        "Write a compact persona for [HUMAN] in [HUMAN]'s own voice.\n\n"
        "Rules:\n"
        "- Write each field in first person, as if [HUMAN] is describing their own habits.\n"
        "- Keep it casual and compressed, not polished or formal.\n"
        "- Use only [HUMAN]'s own messages as evidence.\n"
        "- Capture only stable traits that are useful for predicting future messages across contexts.\n"
        "- Prefer concrete wording habits over abstract summaries.\n"
        "- Do not sound like an analyst describing a person from the outside.\n"
        "- Do not use therapist/assistant/policy-brief language.\n"
        "- Avoid abstract labels such as fairness, accountability, autonomy, respect, boundaries, "
        "power dynamics, empathy, nuance, skepticism, advocacy, procedural correctness, or similar "
        "summary words unless those exact words are frequent in the history.\n"
        "- Do not infer demographics or sensitive attributes.\n"
        "- Do not infer motives, worldview, morals, personality traits, or values unless they are "
        "explicitly and repeatedly stated.\n"
        "- Do not include one-off topical stances, transient emotions, or local interaction goals.\n"
        "- If evidence is weak, write `unknown`.\n"
        "- Keep each field concise.\n"
        "- Write each field as short fragments or very short sentences, not polished mini-paragraphs.\n"
        "- Do not quote or reproduce complete sentences from the selected history.\n"
        "- Do not quote or reuse exact words or phrases from the selected history, including verdict "
        "labels, slang, catchphrases, or short text fragments.\n"
        "- Do not include literal examples from [HUMAN]'s messages, even single-word examples.\n"
        "- Describe lexical habits generically rather than by repeating exact tokens.\n"
        "- Non-text segments such as punctuation styles may be described, but do not quote surrounding text.\n"
        "- Do not include parenthetical example lists, `e.g.` scaffolding, or full sample sentences "
        "from [HUMAN]'s messages.\n\n"
        "Field descriptions:\n"
        "- values: only concrete recurring preferences, irritations, or things I keep siding with; "
        "avoid abstract moral language unless it is explicit in the history\n"
        "- verbal_quirks: my observable surface-form habits only, such as how I use interjections, "
        "punctuation, grammar looseness, discourse markers, formatting quirks, or short repeated "
        "lexical patterns. Do not include exact verdict labels or quoted wording from the history.\n"
        "- expression_style: how I usually come across at the sentence level; keep this concrete and "
        "plain, not psychological, evaluative, or therapist-like\n"
        "- length_prior: my usual reply length in plain language, mainly sentence count and rough brevity\n"
        "- background: concrete background cues not already captured above, especially repeated "
        "personal experiences, self-disclosed roles, responsibilities, constraints, routines, domain "
        "familiarity, sources of information, or recurring points of reference that likely shape "
        "future responses; do not restate general values, tone, or reasoning style; use `unknown` "
        "if nothing strong remains\n\n"
        "Output only this JSON object shape:\n"
        "{\n"
        '  "values": "...",\n'
        '  "verbal_quirks": "...",\n'
        '  "expression_style": "...",\n'
        '  "length_prior": "...",\n'
        '  "background": "..."\n'
        "}\n"
        "</task>"
    )


def parse_persona_json(text: str) -> dict[str, Any]:
    """Parse persona JSON."""
    stripped = (text or "").strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    persona = json.loads(stripped)
    if not isinstance(persona, dict):
        raise ValueError("Persona JSON must be an object")
    persona.pop("analysis", None)
    missing = [key for key in PERSONA_JSON_KEYS if key not in persona]
    if missing:
        raise ValueError(f"Persona JSON missing required keys: {', '.join(missing)}")
    for key in PERSONA_JSON_KEYS:
        if not isinstance(persona[key], str):
            raise ValueError(f"Persona JSON field {key!r} must be a string")
    return persona


def _persona_text_for_leakage(persona: dict[str, Any]) -> str:
    """Return persona text for leakage checks."""
    return " ".join(str(persona.get(key, "") or "") for key in PERSONA_JSON_KEYS).strip()


def persona_leaks_history(
    persona: dict[str, Any],
    selected_history: str,
    *,
    ngram_size: int = DEFAULT_PERSONA_LEAKAGE_NGRAM,
    max_match_tokens: int = DEFAULT_PERSONA_LEAKAGE_MAX_MATCH_TOKENS,
) -> bool:
    """Return whether persona text copies history."""
    persona_text = _persona_text_for_leakage(persona)
    if not persona_text or not (selected_history or "").strip():
        return False
    warning = build_source_copy_warning(
        persona_text, thread_context=selected_history, ngram_size=ngram_size
    )
    if not warning.get("triggered"):
        return False
    return int(warning.get("longest_match_tokens", 0) or 0) >= max_match_tokens


def induce_persona_for_history(
    selected_history: str,
    *,
    model: str = DEFAULT_PERSONA_MODEL,
    temperature: float = DEFAULT_PERSONA_TEMPERATURE,
    max_completion_tokens: int = DEFAULT_PERSONA_MAX_COMPLETION_TOKENS,
    reasoning: bool = DEFAULT_PERSONA_REASONING,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    attempts: int = DEFAULT_PERSONA_ATTEMPTS,
    max_history_words: int = DEFAULT_PERSONA_MAX_HISTORY_WORDS,
    ngram_size: int = DEFAULT_PERSONA_LEAKAGE_NGRAM,
    max_match_tokens: int = DEFAULT_PERSONA_LEAKAGE_MAX_MATCH_TOKENS,
    max_retries: int | None = None,
) -> dict[str, str]:
    """Induce one user persona."""
    selected_history = truncate_history_words(selected_history, max_history_words)
    model = normalize_persona_model(model)
    if max_retries is None:
        max_retries = get_openai_max_retries()
    messages = [
        {"role": "system", "content": PERSONA_INDUCTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_persona_induction_user_prompt(selected_history)},
    ]
    payload = build_chat_payload(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
        reasoning=reasoning,
    )
    payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if min_p is not None:
        payload["min_p"] = float(min_p)

    last_persona: dict[str, str] | None = None
    for _ in range(max(1, attempts)):
        content = post_chat_sync(payload, max_retries=max_retries)
        try:
            persona = parse_persona_json(content)
        except (ValueError, json.JSONDecodeError):
            continue
        last_persona = persona
        if not persona_leaks_history(
            persona, selected_history, ngram_size=ngram_size, max_match_tokens=max_match_tokens
        ):
            return persona
    if last_persona is None:
        raise ValueError("Persona induction produced no parseable JSON object")
    return last_persona


def _as_parquet_list(input_parquet: str | os.PathLike | Sequence[Any]) -> list[Any]:
    if isinstance(input_parquet, (str, os.PathLike)):
        return [input_parquet]
    return list(input_parquet)


def histories_by_user(input_parquet: str | os.PathLike | Sequence[Any]) -> dict[str, str]:
    """Return the first history seen for each user."""
    import pandas as pd

    histories: dict[str, str] = {}
    for parquet in _as_parquet_list(input_parquet):
        rows = pd.read_parquet(parquet).to_dict(orient="records")
        for row in rows:
            extra_info = row.get("extra_info") or {}
            if not isinstance(extra_info, dict):
                continue
            user_id = str(extra_info.get("user_id", "") or "").strip()
            history = str(extra_info.get("user_history", "") or "").strip()
            if user_id and history and user_id not in histories:
                histories[user_id] = history
    return histories


def induce_personas_for_parquet(
    input_parquet: str | os.PathLike | Sequence[Any],
    output_jsonl: str,
    *,
    model: str = DEFAULT_PERSONA_MODEL,
    temperature: float = DEFAULT_PERSONA_TEMPERATURE,
    max_completion_tokens: int = DEFAULT_PERSONA_MAX_COMPLETION_TOKENS,
    reasoning: bool = DEFAULT_PERSONA_REASONING,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    attempts: int = DEFAULT_PERSONA_ATTEMPTS,
    workers: int = DEFAULT_PERSONA_WORKERS,
    max_history_words: int = DEFAULT_PERSONA_MAX_HISTORY_WORDS,
    max_users: int | None = None,
    ngram_size: int = DEFAULT_PERSONA_LEAKAGE_NGRAM,
    max_match_tokens: int = DEFAULT_PERSONA_LEAKAGE_MAX_MATCH_TOKENS,
    max_rounds: int = DEFAULT_PERSONA_MAX_ROUNDS,
    require_complete: bool = True,
) -> dict[str, int]:
    """Induce and write a per-user persona map."""
    model = normalize_persona_model(model)
    histories = histories_by_user(input_parquet)
    user_ids = list(histories)
    if max_users is not None:
        user_ids = user_ids[:max_users]

    def _induce(user_id: str) -> tuple[str, dict[str, str] | None]:
        try:
            persona = induce_persona_for_history(
                histories[user_id],
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                reasoning=reasoning,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                attempts=attempts,
                max_history_words=max_history_words,
                ngram_size=ngram_size,
                max_match_tokens=max_match_tokens,
            )
            return user_id, persona
        except Exception as exc:  # noqa: BLE001 - retried in the next round
            print(
                f"[induce_personas] user {user_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return user_id, None

    persona_map: dict[str, dict[str, str]] = {}
    pending = list(user_ids)
    rounds = 0
    while pending and rounds < max(1, max_rounds):
        rounds += 1
        still_pending: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for user_id, persona in executor.map(_induce, pending):
                if persona is None:
                    still_pending.append(user_id)
                else:
                    persona_map[user_id] = persona
        if still_pending:
            print(
                f"[induce_personas] round {rounds}/{max_rounds}: "
                f"{len(still_pending)}/{len(pending)} users still missing a persona",
                flush=True,
            )
        pending = still_pending

    failed = list(pending)

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for user_id, persona in persona_map.items():
            handle.write(json.dumps({"user_id": user_id, **persona}, ensure_ascii=False) + "\n")

    metadata = {
        "model": model,
        "temperature": temperature,
        "reasoning": reasoning,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "max_completion_tokens": max_completion_tokens,
        "max_history_words": max_history_words,
        "attempts": attempts,
        "rounds": rounds,
        "max_rounds": max_rounds,
        "total_users": len(user_ids),
        "persona_map_users": len(persona_map),
        "failed_users": failed,
    }
    Path(f"{output_jsonl}.metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if failed and require_complete:
        raise RuntimeError(
            f"Persona induction incomplete: {len(failed)} of {len(user_ids)} users "
            f"still have no persona after {rounds} round(s) (e.g. {failed[:5]}). "
            f"Partial map written to {output_jsonl}; rerun to resume, raise "
            f"--max_rounds, or pass --allow_incomplete."
        )
    return {"users": len(persona_map), "failed": len(failed), "rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Induce a per-user persona map (jsonl) from a built history parquet, "
            "for use with shared.load_personas.load_persona_map."
        )
    )
    parser.add_argument(
        "--input_parquet",
        required=True,
        nargs="+",
        help=(
            "One or more built parquets carrying extra_info.user_history and "
            "extra_info.user_id (pass train/val/test together to cover every user)."
        ),
    )
    parser.add_argument("--output_jsonl", required=True, help="Destination persona-map jsonl path.")
    parser.add_argument(
        "--model",
        default=DEFAULT_PERSONA_MODEL,
        choices=tuple(PERSONA_MODEL_ALIASES),
        help="Persona inductor.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_PERSONA_TEMPERATURE)
    parser.add_argument(
        "--max_completion_tokens", type=int, default=DEFAULT_PERSONA_MAX_COMPLETION_TOKENS
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable thinking for the inductor (use for thinking models like Qwen3-8B).",
    )
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--max_history_words", type=int, default=DEFAULT_PERSONA_MAX_HISTORY_WORDS)
    parser.add_argument("--attempts", type=int, default=DEFAULT_PERSONA_ATTEMPTS)
    parser.add_argument("--workers", type=int, default=DEFAULT_PERSONA_WORKERS)
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=DEFAULT_PERSONA_MAX_ROUNDS,
        help="Re-attempt failed users this many rounds until every user is covered.",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Write a partial map instead of erroring when some users still fail.",
    )
    parser.add_argument(
        "--max_users", type=int, default=None, help="Cap the number of users (smoke tests)."
    )
    parser.add_argument(
        "--leakage_ngram_size", type=int, default=DEFAULT_PERSONA_LEAKAGE_NGRAM
    )
    parser.add_argument(
        "--leakage_max_match_tokens", type=int, default=DEFAULT_PERSONA_LEAKAGE_MAX_MATCH_TOKENS
    )
    args = parser.parse_args()

    stats = induce_personas_for_parquet(
        args.input_parquet,
        args.output_jsonl,
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        reasoning=args.reasoning,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        attempts=args.attempts,
        workers=args.workers,
        max_history_words=args.max_history_words,
        max_users=args.max_users,
        ngram_size=args.leakage_ngram_size,
        max_match_tokens=args.leakage_max_match_tokens,
        max_rounds=args.max_rounds,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps({"output": args.output_jsonl, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
