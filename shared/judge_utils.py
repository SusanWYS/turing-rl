"""Shared judge parsing and scoring helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any

try:  # pragma: no cover - exercised in runtime envs
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

from shared.api_client import (
    _OPENAI_MAX_RETRIES_CAP,
    build_chat_payload,
    get_openai_max_retries,
    post_chat_async,
)
from shared.judge_prompts import (
    RESPONSE_BREAKDOWN_PROMPT_BATCHED,
    RESPONSE_ONLY_NO_HARD_FLAGS_PROMPT_BATCHED,
    RESPONSE_ONLY_PROMPT_BATCHED,
)


_COPY_TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_COPY_EXCERPT_MAX_CHARS = 220
_ROLE_PREFIX_RE = re.compile(r"^\s*\[(?:HUMAN|OTHER)\]\s*:\s*", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^\s*(?:what|which|who|whom|whose|when|where|why|how|is|are|am|was|were|"
    r"do|does|did|can|could|would|should|will|have|has|had)\b",
    re.IGNORECASE,
)
_ASSISTANT_STYLE_RE = re.compile(
    r"\b(?:here(?:'s| is)|overall|in summary|recommendation|breakdown|"
    r"for casual players|for more dedicated players|worthwhile experience)\b",
    re.IGNORECASE,
)
_ASSISTANT_ANSWER_START_RE = re.compile(
    r"^\s*[A-Z0-9][^.?!:\n]{0,120}\b(?:is|are|was|were|means|refers to|"
    r"features|offers|has|includes|can be)\b",
    re.IGNORECASE,
)
_STRUCTURED_LIST_RE = re.compile(r"(?:^|\n)\s*\d+\.\s+\*{0,2}[A-Za-z]", re.MULTILINE)
_shared_session = None
_JUDGE_REQUEST_SEMAPHORES: dict[int, Any] = {}
_JUDGE_REQUEST_LIMITS: dict[int, int] = {}
_JUDGE_REQUEST_LOGGED_LOOPS: set[int] = set()


def _copy_tokens(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), match.start(), match.end())
        for match in _COPY_TOKEN_RE.finditer(text or "")
    ]


def _compact_excerpt(text: str, start: int, end: int) -> str:
    excerpt = re.sub(r"\s+", " ", (text or "")[start:end]).strip()
    if len(excerpt) <= _COPY_EXCERPT_MAX_CHARS:
        return excerpt
    return f"{excerpt[: _COPY_EXCERPT_MAX_CHARS - 3].rstrip()}..."


def _find_ngram_copy_matches(
    *,
    response: str,
    source_text: str,
    source_label: str,
    ngram_size: int,
) -> list[dict[str, Any]]:
    response_tokens = _copy_tokens(response)
    source_tokens = _copy_tokens(source_text)
    if len(response_tokens) < ngram_size or len(source_tokens) < ngram_size:
        return []

    source_index: dict[tuple[str, ...], list[int]] = {}
    source_words = [token for token, _, _ in source_tokens]
    response_words = [token for token, _, _ in response_tokens]
    for source_idx in range(0, len(source_words) - ngram_size + 1):
        key = tuple(source_words[source_idx: source_idx + ngram_size])
        source_index.setdefault(key, []).append(source_idx)

    matches: list[dict[str, Any]] = []
    seen_response_spans: set[tuple[int, int, str]] = set()
    for response_idx in range(0, len(response_words) - ngram_size + 1):
        key = tuple(response_words[response_idx: response_idx + ngram_size])
        for source_idx in source_index.get(key, []):
            span_len = ngram_size
            while (
                response_idx + span_len < len(response_words)
                and source_idx + span_len < len(source_words)
                and response_words[response_idx + span_len] == source_words[source_idx + span_len]
            ):
                span_len += 1

            response_span = (response_idx, response_idx + span_len, source_label)
            if response_span in seen_response_spans:
                continue
            seen_response_spans.add(response_span)

            response_start = response_tokens[response_idx][1]
            response_end = response_tokens[response_idx + span_len - 1][2]
            source_start = source_tokens[source_idx][1]
            source_end = source_tokens[source_idx + span_len - 1][2]
            matches.append(
                {
                    "source": source_label,
                    "match_tokens": span_len,
                    "response_token_start": response_idx,
                    "response_token_end": response_idx + span_len,
                    "response_excerpt": _compact_excerpt(response, response_start, response_end),
                    "source_excerpt": _compact_excerpt(source_text, source_start, source_end),
                }
            )

    matches.sort(key=lambda item: (-int(item["match_tokens"]), int(item["response_token_start"])))
    return matches


def _select_nonoverlapping_matches(
    matches: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for match in matches:
        start = int(match["response_token_start"])
        end = int(match["response_token_end"])
        if any(start < prev_end and end > prev_start for prev_start, prev_end in occupied):
            continue
        selected.append(
            {
                "source": match["source"],
                "match_tokens": int(match["match_tokens"]),
                "response_excerpt": match["response_excerpt"],
                "source_excerpt": match["source_excerpt"],
            }
        )
        occupied.append((start, end))
        if len(selected) >= max_examples:
            break
    return selected


def build_source_copy_warning(
    response: str,
    *,
    user_history: str = "",
    thread_context: str = "",
    ngram_size: int = 5,
    max_examples: int = 3,
) -> dict[str, Any]:
    """Build an advisory source-copy warning."""
    ngram_size = max(1, int(ngram_size))
    all_matches: list[dict[str, Any]] = []
    for source_label, source_text in (("history", user_history), ("context", thread_context)):
        if not source_text:
            continue
        all_matches.extend(
            _find_ngram_copy_matches(
                response=response,
                source_text=source_text,
                source_label=source_label,
                ngram_size=ngram_size,
            )
        )

    if not all_matches:
        return {
            "triggered": False,
            "ngram_size": ngram_size,
            "longest_match_tokens": 0,
            "matches": [],
        }

    all_matches.sort(
        key=lambda item: (-int(item["match_tokens"]), int(item["response_token_start"]))
    )
    selected = _select_nonoverlapping_matches(all_matches, max_examples=max_examples)
    return {
        "triggered": True,
        "ngram_size": ngram_size,
        "longest_match_tokens": int(all_matches[0]["match_tokens"]),
        "matches": selected,
    }


def format_source_copy_watchlist(
    warnings: list[dict[str, Any]],
    *,
    item_label: str = "Response",
    labels: list[str] | None = None,
) -> str:
    """Format source-copy warnings for a judge prompt."""
    if not any(bool(warning and warning.get("triggered")) for warning in warnings):
        return "5-gram source-copy scan: no response triggered a warning."

    lines = [
        "5-gram source-copy scan. This watchlist is advisory: inspect whether "
        "the overlap is a natural quote or confusing copied text before setting "
        "source_copy=true.",
    ]
    for idx, warning in enumerate(warnings):
        if not warning or not warning.get("triggered"):
            continue
        label = labels[idx] if labels and idx < len(labels) else f"{item_label} {idx + 1}"
        longest_match = int(warning.get("longest_match_tokens", 0))
        lines.append(f"{label}: triggered; longest_match_tokens={longest_match}")
        for match in warning.get("matches", []):
            response_excerpt = json.dumps(
                str(match.get("response_excerpt", "")),
                ensure_ascii=False,
            )
            source_excerpt = json.dumps(
                str(match.get("source_excerpt", "")),
                ensure_ascii=False,
            )
            lines.append(
                "  match: "
                f"source={match.get('source', '')}; "
                f"match_tokens={int(match.get('match_tokens', 0))}; "
                f"response_excerpt={response_excerpt}; "
                f"source_excerpt={source_excerpt}"
            )
    return "\n".join(lines)


def _get_session() -> "aiohttp.ClientSession":
    """Create the shared aiohttp session."""
    if aiohttp is None:
        raise ImportError("OpenRouter response-similarity judging requires aiohttp to be installed")
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        request_limit = _judge_request_limit()
        connector_limit = max(
            request_limit,
            int(os.environ.get("PERSONA_OPENAI_CONNECTION_LIMIT", str(request_limit))),
        )
        timeout_seconds = float(os.environ.get("PERSONA_OPENAI_TIMEOUT_SECONDS", "400"))
        connector = aiohttp.TCPConnector(limit=connector_limit)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        print(
            "[response_judge] aiohttp connection "
            f"limit={connector_limit} timeout_s={timeout_seconds:g}",
            flush=True,
        )
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


def _judge_request_limit() -> int:
    for env_name in ("SIM_JUDGE_MAX_CONCURRENCY", "PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(1, int(raw_value))
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return 400


def _get_judge_request_semaphore() -> Any:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    limit = _judge_request_limit()
    semaphore = _JUDGE_REQUEST_SEMAPHORES.get(loop_id)
    if semaphore is None or _JUDGE_REQUEST_LIMITS.get(loop_id) != limit:
        semaphore = asyncio.Semaphore(limit)
        _JUDGE_REQUEST_SEMAPHORES[loop_id] = semaphore
        _JUDGE_REQUEST_LIMITS[loop_id] = limit
    if loop_id not in _JUDGE_REQUEST_LOGGED_LOOPS:
        print(f"[response_judge] max concurrent requests per process: {limit}", flush=True)
        _JUDGE_REQUEST_LOGGED_LOOPS.add(loop_id)
    return semaphore


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else None


def _clip_unit(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"Could not parse boolean value from {value!r}")


def _strip_role_prefix(text: str) -> str:
    return _ROLE_PREFIX_RE.sub("", text or "", count=1).strip()


def _is_question_like(text: str) -> bool:
    normalized = _strip_role_prefix(text)
    if not normalized:
        return False
    if "?" in normalized:
        return True
    return bool(_QUESTION_START_RE.match(normalized))


def _looks_assistant_like_response(*, ground_truth: str, candidate: str) -> bool:
    normalized_ground_truth = _strip_role_prefix(ground_truth)
    normalized_candidate = _strip_role_prefix(candidate)
    if not normalized_ground_truth or not normalized_candidate:
        return False
    if not _is_question_like(normalized_ground_truth):
        return False
    if _is_question_like(normalized_candidate):
        return False

    ground_truth_tokens = len(_copy_tokens(normalized_ground_truth))
    candidate_tokens = len(_copy_tokens(normalized_candidate))
    if candidate_tokens < max(24, ground_truth_tokens * 3):
        return False

    signals = 0
    if _ASSISTANT_STYLE_RE.search(normalized_candidate):
        signals += 1
    if _STRUCTURED_LIST_RE.search(normalized_candidate):
        signals += 1
    if _ASSISTANT_ANSWER_START_RE.search(normalized_candidate):
        signals += 1
    return signals >= 1


def _build_generations_json_text(item_name: str, generations: list[str]) -> str:
    payload = {str(i + 1): generation.strip() for i, generation in enumerate(generations)}
    return (
        f"<|The Start of Generated {item_name}s|>\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
        f"<|The End of Generated {item_name}s|>"
    )


def _build_plain_generations_text(generations: list[str], item_name: str = "response") -> str:
    parts = []
    for idx, generation in enumerate(generations, 1):
        parts.append(f"<|The Start of Generated {item_name} {idx}|>")
        parts.append(generation.strip())
        parts.append(f"<|The End of Generated {item_name} {idx}|>")
        parts.append("")
    return "\n".join(parts)


def _build_response_judge_context(*, user_history: str, thread_context: str) -> str:
    return (
        "<|The Start of Past Messages|>\n"
        f"{user_history.strip()}\n"
        "<|The End of Past Messages|>\n\n"
        "<|The Start of Context|>\n"
        f"{thread_context.strip()}\n"
        "<|The End of Context|>"
    )


async def _request_judge_json(
    *,
    prompt: str,
    model: str,
    max_retry: int,
    label: str,
) -> tuple[dict[str, Any] | None, str]:
    last_content = ""
    semaphore = _get_judge_request_semaphore()
    session = _get_session()
    payload = build_chat_payload(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=int(os.environ.get("PERSONA_JUDGE_MAX_COMPLETION_TOKENS", "8192")),
        response_format={"type": "json_object"},
        reasoning=True,
    )
    for attempt in range(max_retry):
        try:
            last_content = await post_chat_async(session, payload, semaphore=semaphore)
            result = _extract_json(last_content)
            if not isinstance(result, dict):
                raise ValueError("Judge response did not parse as JSON object")
            return result, last_content
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            if attempt == max_retry - 1:
                print(
                    f"[response_judge] {label} failed after {max_retry} attempts: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                break
            print(
                f"[response_judge] {label} retry {attempt + 1}/{max_retry}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    return None, last_content


def _failure_outputs(
    candidates: list[Any],
    last_content: str,
    *,
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "score": 0.0,
            "node_score": 0.0,
            "semantic_similarity": 0.0,
            "information_completeness": 0.0,
            "source_copy": False,
            "judge_well_formed_json": False,
            "judge_parse_failed": error == "judge_parse_failed",
            "judge_schema_failed": error == "judge_schema_failed",
            "metrics_info": json.dumps({"error": error, "raw": last_content}, ensure_ascii=False),
        }
        for _ in candidates
    ]


def _parse_metric_candidate(
    item: dict[str, Any],
    *,
    include_breakdown: bool = False,
) -> dict[str, Any]:
    if "score" not in item or "thought" not in item:
        raise ValueError("Response judge item missing score or thought")
    raw_score = _clip_unit(item.get("score"))
    wrong_perspective = (
        _coerce_bool(item.get("wrong_perspective"))
        if "wrong_perspective" in item else False
    )
    source_copy = _coerce_bool(item.get("source_copy")) if "source_copy" in item else False
    parsed = {
        "thought": str(item.get("thought", "") or ""),
        "wrong_perspective": wrong_perspective,
        "source_copy": source_copy,
        "raw_score": raw_score,
        "score": 0.0 if wrong_perspective or source_copy else raw_score,
    }
    if include_breakdown:
        if "semantic_similarity" not in item or "information_completeness" not in item:
            raise ValueError(
                "Response judge breakdown item missing semantic_similarity "
                "or information_completeness"
            )
        semantic_similarity = _clip_unit(item.get("semantic_similarity"))
        information_completeness = _clip_unit(item.get("information_completeness"))
        if wrong_perspective or source_copy:
            semantic_similarity = 0.0
            information_completeness = 0.0
        parsed["semantic_similarity"] = semantic_similarity
        parsed["information_completeness"] = information_completeness
    return parsed


def _strip_hard_flags(parsed: dict[str, Any]) -> dict[str, Any]:
    score = _clip_unit(parsed.get("raw_score", parsed.get("score", 0.0)))
    stripped = dict(parsed)
    stripped["wrong_perspective"] = False
    stripped["source_copy"] = False
    stripped["assistant_like_response"] = False
    stripped["raw_score"] = score
    stripped["score"] = score
    return stripped


def _build_output(*, parsed: dict[str, Any], key_points: str) -> dict[str, Any]:
    wrong_perspective = bool(parsed["wrong_perspective"])
    source_copy = bool(parsed.get("source_copy", False))
    assistant_like_response = bool(parsed.get("assistant_like_response", False))
    raw_score = _clip_unit(parsed["raw_score"])
    score = 0.0 if wrong_perspective or source_copy else raw_score
    metrics_info = {
        "key_points": key_points,
        "wrong_perspective": wrong_perspective,
        "source_copy": source_copy,
        "assistant_like_response": assistant_like_response,
        "response": {
            "thought": parsed["thought"],
            "wrong_perspective": wrong_perspective,
            "source_copy": source_copy,
            "assistant_like_response": assistant_like_response,
            "raw_score": raw_score,
            "score": score,
        },
        "score": float(score),
    }
    output = {
        "score": score,
        "node_score": score,
        "source_copy": source_copy,
        "wrong_perspective": wrong_perspective,
        "assistant_like_response": assistant_like_response,
        "judge_well_formed_json": True,
        "judge_parse_failed": False,
        "judge_schema_failed": False,
    }
    if "semantic_similarity" in parsed:
        semantic_similarity = _clip_unit(parsed["semantic_similarity"])
        information_completeness = _clip_unit(parsed["information_completeness"])
        if wrong_perspective or source_copy:
            semantic_similarity = 0.0
            information_completeness = 0.0
        metrics_info["semantic_similarity"] = semantic_similarity
        metrics_info["information_completeness"] = information_completeness
        output["semantic_similarity"] = semantic_similarity
        output["information_completeness"] = information_completeness
    output["metrics_info"] = json.dumps(metrics_info, ensure_ascii=False)
    return output


def _parse_judge_outputs(
    result: dict[str, Any],
    *,
    num_candidates: int,
    include_breakdown: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    key_points = str(result.get("key_points", "") or "")
    outputs = []
    for idx in range(1, num_candidates + 1):
        item = result.get(str(idx))
        if not isinstance(item, dict):
            raise ValueError(f"Missing response score fields for candidate {idx}")
        outputs.append(_parse_metric_candidate(item, include_breakdown=include_breakdown))
    return key_points, outputs


def _apply_assistant_like_wrong_perspective(
    *,
    parsed: dict[str, Any],
    ground_truth: str,
    candidate: str,
) -> dict[str, Any]:
    if parsed.get("wrong_perspective") or parsed.get("source_copy"):
        return parsed
    if not _looks_assistant_like_response(ground_truth=ground_truth, candidate=candidate):
        return parsed

    updated = dict(parsed)
    updated["wrong_perspective"] = True
    updated["assistant_like_response"] = True
    thought = str(updated.get("thought", "") or "").strip()
    suffix = "Auto-flagged assistant-like answer to a user follow-up."
    updated["thought"] = f"{thought} {suffix}".strip() if thought else suffix
    return updated


async def judge_response_batch(
    *,
    user_history: str,
    thread_context: str,
    ground_truth: str,
    candidates: list[str],
    model: str,
    copy_warnings: list[dict[str, Any] | None] | None = None,
    include_breakdown: bool = False,
    enable_hard_flags: bool = True,
    max_retry: int | None = None,
    label: str = "response judge",
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if max_retry is None:
        max_retry = get_openai_max_retries()
    else:
        try:
            max_retry = max(1, min(_OPENAI_MAX_RETRIES_CAP, int(max_retry)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_retry must be an integer, got {max_retry!r}") from exc

    shared_kwargs = {
        "context": _build_response_judge_context(
            user_history=user_history,
            thread_context=thread_context,
        ),
        "ground_truth": ground_truth,
        "generations_text": (
            _build_generations_json_text("response", candidates)
            if include_breakdown else _build_plain_generations_text(candidates, "response")
        ),
        "num_generations": len(candidates),
    }
    if include_breakdown:
        prompt_template = RESPONSE_BREAKDOWN_PROMPT_BATCHED
    elif enable_hard_flags:
        prompt_template = RESPONSE_ONLY_PROMPT_BATCHED
    else:
        prompt_template = RESPONSE_ONLY_NO_HARD_FLAGS_PROMPT_BATCHED
    response_prompt = prompt_template.format(**shared_kwargs)
    response_result, response_last_content = await _request_judge_json(
        prompt=response_prompt,
        model=model,
        max_retry=max_retry,
        label=f"{label} response",
    )
    if response_result is None:
        return _failure_outputs(candidates, response_last_content, error="judge_parse_failed")

    try:
        key_points, parsed_outputs = _parse_judge_outputs(
            response_result,
            num_candidates=len(candidates),
            include_breakdown=include_breakdown,
        )
        outputs = []
        for idx, parsed in enumerate(parsed_outputs):
            if enable_hard_flags:
                parsed = _apply_assistant_like_wrong_perspective(
                    parsed=parsed,
                    ground_truth=ground_truth,
                    candidate=candidates[idx],
                )
            else:
                parsed = _strip_hard_flags(parsed)
            outputs.append(
                _build_output(
                    parsed=parsed,
                    key_points=key_points,
                )
            )
        return outputs
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[response_judge] {label} parse failure: {type(exc).__name__}: {exc}", flush=True)
        return _failure_outputs(candidates, response_last_content, error="judge_schema_failed")


def _coerce_turing_rating(value: Any) -> int | None:
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 7 else None


def _extract_turing_rating(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    patterns = (
        r'(?i)(?:^|[{\s,\n])"?rating"?\s*[:=]\s*"?([1-7])"?\b',
        r'(?i)\bfinal\s+rating\s*(?:is|=|:)?\s*"?([1-7])"?\b',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _rating_from_turing_score_gap(score_gap: float) -> int:
    if score_gap <= -2.0:
        return 1
    if score_gap <= -1.0:
        return 2
    if score_gap <= -0.25:
        return 3
    if score_gap < 0.25:
        return 4
    if score_gap < 1.0:
        return 5
    if score_gap < 2.0:
        return 6
    return 7


def _turing_parse_failure_result(
    *,
    rating: int | None = None,
    raw_text: str | None = None,
) -> dict[str, Any]:
    recovered = rating is not None
    return {
        "response_a_source_copy": "",
        "response_b_source_copy": "",
        "source_copy_penalty_a": 0.0,
        "source_copy_penalty_b": 0.0,
        "response_a_assistant_like": "",
        "response_b_assistant_like": "",
        "assistant_like_penalty_a": 0.0,
        "assistant_like_penalty_b": 0.0,
        "response_a_wrong_target_or_role": "",
        "response_b_wrong_target_or_role": "",
        "wrong_target_or_role_penalty_a": 0.0,
        "wrong_target_or_role_penalty_b": 0.0,
        "response_a_unsupported_adversarial_reframing": "",
        "response_b_unsupported_adversarial_reframing": "",
        "unsupported_adversarial_reframing_penalty_a": 0.0,
        "unsupported_adversarial_reframing_penalty_b": 0.0,
        "immediate_target_score_a": 0.0,
        "immediate_target_score_b": 0.0,
        "human_goal_score_a": 0.0,
        "human_goal_score_b": 0.0,
        "communication_style_score_a": 0.0,
        "communication_style_score_b": 0.0,
        "base_score_a": 0.0,
        "base_score_b": 0.0,
        "penalty_a": 0.0,
        "penalty_b": 0.0,
        "response_a_score": 0.0,
        "response_b_score": 0.0,
        "score_gap": 0.0,
        "reasoning": (
            "Recovered explicit rating from malformed judge response."
            if recovered else ""
        ),
        "rating": rating or 0,
        "parse_error": not recovered,
        "parse_fallback": recovered,
    }


def _stable_turing_generated_is_b(
    response: str,
    *,
    user_id: Any = "",
    post_id: Any = "",
    target_idx: Any = "",
    seed_material: str = "",
) -> bool:
    """Deterministically randomize whether the generated response is Response B."""
    material = seed_material or f"{user_id}|{post_id}|{target_idx}|{response}"
    digest = hashlib.sha256(str(material).encode("utf-8")).hexdigest()
    return int(digest, 16) % 2 == 0
