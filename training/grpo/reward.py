"""Reward functions for veRL GRPO training."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
import os
import re
import sys
from typing import Any, Optional

try:  # pragma: no cover - exercised in runtime envs
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.prompt_utils import (
    parse_reasoning_and_response,
    response_format_components,
)
from shared.api_client import (
    build_chat_payload,
    get_openai_max_retries,
    post_chat_async,
    resolve_judge_api_key,
)
from shared.judge_prompts import TURING_PROMPT
from shared.judge_utils import (
    _coerce_turing_rating,
    _extract_turing_rating,
    _rating_from_turing_score_gap,
    _stable_turing_generated_is_b,
    _turing_parse_failure_result,
    build_source_copy_warning,
    format_source_copy_watchlist,
)

try:
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except ImportError:  # pragma: no cover - only used outside GRPO envs
    DataProto = Any  # type: ignore[assignment]
    RewardManagerBase = object  # type: ignore[assignment]


JUDGE_MODEL = "qwen/qwen3.5-397b-a17b"
DEFAULT_FORMAT_NONEMPTY_REASONING_BONUS = 0.0
DEFAULT_FORMAT_REASONING_SCHEMA_BONUS = 0.05
DEFAULT_FORMAT_NO_POST_HUMAN_THINKING_BONUS = 0.05
TURING_RAW_REWARD_SCALE = 0.9
TURING_JUDGE_SCORE_CLIP_MAX = 5.0
DEFAULT_TURING_LENGTH_LOWER_RATIO = 0.8
DEFAULT_TURING_LENGTH_UPPER_RATIO = 1.1
DEFAULT_TURING_LENGTH_SHORT_PENALTY_LAMBDA = 0.35
DEFAULT_TURING_LENGTH_LONG_PENALTY_LAMBDA = 0.15
DEFAULT_TURING_LENGTH_PENALTY_CAP = 0.25
_JUDGE_REQUEST_SEMAPHORES: dict[int, Any] = {}
_JUDGE_REQUEST_LIMITS: dict[int, int] = {}
_JUDGE_REQUEST_LOGGED_LOOPS: set[int] = set()
_LOOSE_HUMAN_PREFIX_RE = re.compile(r"(?i)\[\s*human\s*[\]\}](?:\s*:)?")
_EXACT_HUMAN_PREFIX_RE = re.compile(r"(?im)^\s*\[\s*human\s*\]\s*:")
_XML_TAG_RE = re.compile(r"(?s)<[^>]+>")
_SPEAKER_LABEL_RE = re.compile(r"(?is)\[\s*(?:human|other(?:\s*-\s*op)?)\s*\]\s*:?", re.IGNORECASE)
_ALPHABETIC_TOKEN_RE = re.compile(r"(?u)\b[^\W\d_]+\b")


def _reward_judge_request_limit() -> int:
    for env_name in ("TURING_JUDGE_MAX_CONCURRENCY", "PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(1, int(raw_value))
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return 512


def _get_reward_judge_request_semaphore() -> Any:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    limit = _reward_judge_request_limit()
    semaphore = _JUDGE_REQUEST_SEMAPHORES.get(loop_id)
    if semaphore is None or _JUDGE_REQUEST_LIMITS.get(loop_id) != limit:
        semaphore = asyncio.Semaphore(limit)
        _JUDGE_REQUEST_SEMAPHORES[loop_id] = semaphore
        _JUDGE_REQUEST_LIMITS[loop_id] = limit
    if loop_id not in _JUDGE_REQUEST_LOGGED_LOOPS:
        print(f"[reward_judge] max concurrent requests per process: {limit}", flush=True)
        _JUDGE_REQUEST_LOGGED_LOOPS.add(loop_id)
    return semaphore


def _get_response_format_bonus_weights() -> tuple[float, float, float, float]:
    """Return additive format bonuses for the final visible response layout."""
    return (
        float(os.environ.get("FORMAT_HUMAN_PREFIX_BONUS", "0.0")),
        float(os.environ.get(
            "FORMAT_NONEMPTY_REASONING_BONUS",
            str(DEFAULT_FORMAT_NONEMPTY_REASONING_BONUS),
        )),
        float(os.environ.get(
            "FORMAT_REASONING_SCHEMA_BONUS",
            str(DEFAULT_FORMAT_REASONING_SCHEMA_BONUS),
        )),
        float(os.environ.get(
            "FORMAT_NO_POST_HUMAN_THINKING_BONUS",
            str(DEFAULT_FORMAT_NO_POST_HUMAN_THINKING_BONUS),
        )),
    )


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_pre_final_human_text(solution_str: str) -> str:
    """Return visible non-response text before the final human response marker."""
    text = solution_str or ""
    human_matches = list(_LOOSE_HUMAN_PREFIX_RE.finditer(text))
    prefix = text[: human_matches[-1].start()] if human_matches else text
    prefix = _XML_TAG_RE.sub(" ", prefix)
    prefix = _SPEAKER_LABEL_RE.sub(" ", prefix)
    return prefix.strip()


def build_meaningful_thinking_info(
    solution_str: str,
    prompt_mode: str | None,
    response_components: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return hard-zero signals for visible thinking modes."""
    raw_thinking = _extract_pre_final_human_text(solution_str)
    normalized_thinking = raw_thinking.strip()
    has_alphabetic_thinking = bool(_ALPHABETIC_TOKEN_RE.search(normalized_thinking))
    meaningful = has_alphabetic_thinking
    hard_zero = not meaningful
    return {
        "meaningful_thinking_required": 1.0,
        "meaningful_thinking": 1.0 if meaningful else 0.0,
        "thinking_has_alphabetic_token": 1.0 if has_alphabetic_thinking else 0.0,
        "thinking_hard_zero": 1.0 if hard_zero else 0.0,
    }


def empty_format_reward_info() -> dict[str, float]:
    """Return a zeroed format-bonus payload."""
    return {
        "format": 0.0,
        "format_score": 0.0,
        "format_human_prefix": 0.0,
        "format_nonempty_reasoning": 0.0,
        "format_no_post_human_thinking": 0.0,
        "format_reasoning_schema": 0.0,
    }


def build_format_reward_info(
    solution_str: str,
    metric: str,
    prompt_mode: str | None,
    response_components: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return additive format bonuses for reasoning tags and final response layout."""
    info = empty_format_reward_info()
    if response_components is None:
        response_components = response_format_components(solution_str)
    prefix_bonus, nonempty_reasoning_bonus, reasoning_schema_bonus, clean_tail_bonus = _get_response_format_bonus_weights()

    reasoning_schema_pass = (
        int(response_components["reasoning_open_count"]) == 1
        and int(response_components["reasoning_close_count"]) == 1
        and bool(response_components["response_nonempty"])
        and bool(response_components["has_reasoning_schema"])
        and not bool(response_components["placeholder_reasoning_prefix"])
        and not bool(response_components["has_forbidden_xml_tag"])
    )
    response_tail_pass = (
        int(response_components["human_prefix_count"]) == 1
        and bool(response_components["has_exact_human_prefix"])
        and bool(response_components["response_nonempty"])
        and bool(response_components["no_post_human_thinking_trace"])
    )

    if response_tail_pass:
        info["format_human_prefix"] = prefix_bonus
        info["format_no_post_human_thinking"] = clean_tail_bonus
    if reasoning_schema_pass:
        info["format_nonempty_reasoning"] = nonempty_reasoning_bonus
        info["format_reasoning_schema"] = reasoning_schema_bonus

    info["format_score"] = (
        info["format_human_prefix"]
        + info["format_nonempty_reasoning"]
        + info["format_no_post_human_thinking"]
        + info["format_reasoning_schema"]
    )
    info["format"] = info["format_score"]
    return info


def parse_response_for_prompt_mode(solution_str: str, prompt_mode: str | None) -> tuple[str, str]:
    """Parse a reasoning-mode rollout."""
    _ = prompt_mode
    return parse_reasoning_and_response(solution_str)


def _get_logprob_clip_bounds() -> tuple[float, float]:
    """Return clipping bounds for raw mean logprob rewards."""
    clip_min = float(os.environ.get("LOGPROB_CLIP_MIN", "-8.0"))
    clip_max = float(os.environ.get("LOGPROB_CLIP_MAX", "0.0"))
    if clip_min > clip_max:
        raise ValueError(f"Invalid logprob clip bounds: {clip_min=} > {clip_max=}")
    return clip_min, clip_max


def clip_logprob_reward(mean_logprob: float) -> float:
    """Clip raw mean logprob into the bounded reward range used for GRPO."""
    clip_min, clip_max = _get_logprob_clip_bounds()
    return max(clip_min, min(clip_max, float(mean_logprob)))


def build_logprob_reward_result(
    mean_logprob: float,
    *,
    num_tokens: int | None = None,
    logprob_source: str = "current_policy_rollout",
    logprob_failure: str | None = None,
    format_reward_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent reward payload for current-policy logprob runs."""
    clipped_logprob = clip_logprob_reward(mean_logprob)
    normalized_format_info = empty_format_reward_info()
    if format_reward_info:
        for key in normalized_format_info:
            if key in format_reward_info:
                normalized_format_info[key] = float(format_reward_info[key])
    format_score = normalized_format_info["format_score"]
    result = {
        "score": clipped_logprob + format_score,
        "total_score": clipped_logprob + format_score,
        "raw_reward": clipped_logprob,
        "logprob_unclipped": float(mean_logprob),
        "logprob_clipped": clipped_logprob,
        "logprob_num_tokens": float(num_tokens or 0),
        "logprob_source": logprob_source,
        "logprob_failure": logprob_failure,
    }
    result.update(normalized_format_info)
    return result


def adjust_turing_raw_reward(raw_reward: float) -> float:
    """Scale the Turing raw reward before adding format bonuses."""
    return float(raw_reward) * TURING_RAW_REWARD_SCALE


def clip_turing_judge_score(score: float) -> float:
    """Clip the raw Turing judge score before reward normalization for training."""
    return min(float(score), TURING_JUDGE_SCORE_CLIP_MAX)


def compute_turing_length_info(response: str, ground_truth: str) -> dict[str, float]:
    """Return reference-relative length diagnostics and bounded penalty."""
    generated_words = _count_words(response)
    human_words = _count_words(ground_truth)
    denominator = max(human_words, 1)
    length_ratio = generated_words / denominator
    relative_diff = abs(generated_words - human_words) / denominator
    shortfall_relative = max(human_words - generated_words, 0) / denominator
    excess_relative = max(generated_words - human_words, 0) / denominator
    lower_ratio = max(
        0.0,
        float(os.environ.get("TURING_LENGTH_LOWER_RATIO", str(DEFAULT_TURING_LENGTH_LOWER_RATIO))),
    )
    upper_ratio = max(
        lower_ratio,
        float(os.environ.get("TURING_LENGTH_UPPER_RATIO", str(DEFAULT_TURING_LENGTH_UPPER_RATIO))),
    )
    short_penalty_lambda = max(
        0.0,
        float(
            os.environ.get(
                "TURING_LENGTH_SHORT_PENALTY_LAMBDA",
                os.environ.get("TURING_LENGTH_PENALTY_LAMBDA", str(DEFAULT_TURING_LENGTH_SHORT_PENALTY_LAMBDA)),
            )
        ),
    )
    long_penalty_lambda = max(
        0.0,
        float(os.environ.get("TURING_LENGTH_LONG_PENALTY_LAMBDA", str(DEFAULT_TURING_LENGTH_LONG_PENALTY_LAMBDA))),
    )
    penalty_cap = max(
        0.0,
        float(os.environ.get("TURING_LENGTH_PENALTY_CAP", str(DEFAULT_TURING_LENGTH_PENALTY_CAP))),
    )
    short_deadband_violation = max(lower_ratio - length_ratio, 0.0) / max(lower_ratio, 1e-12)
    long_deadband_violation = max(length_ratio - upper_ratio, 0.0) / max(upper_ratio, 1e-12)
    short_length_penalty = short_deadband_violation * short_penalty_lambda
    long_length_penalty = long_deadband_violation * long_penalty_lambda
    length_penalty = min(short_length_penalty + long_length_penalty, penalty_cap)
    return {
        "length_generated_words": float(generated_words),
        "length_human_words": float(human_words),
        "length_ratio": length_ratio,
        "length_relative_diff": relative_diff,
        "length_shortfall_relative": shortfall_relative,
        "length_excess_relative": excess_relative,
        "length_short_deadband_violation": short_deadband_violation,
        "length_long_deadband_violation": long_deadband_violation,
        "length_short_penalty": short_length_penalty,
        "length_long_penalty": long_length_penalty,
        "length_penalty": length_penalty,
        "length_lower_ratio": lower_ratio,
        "length_upper_ratio": upper_ratio,
        "length_penalty_lambda": short_penalty_lambda,
        "length_short_penalty_lambda": short_penalty_lambda,
        "length_long_penalty_lambda": long_penalty_lambda,
        "length_penalty_cap": penalty_cap,
    }


async def _openai_chat(
    session: aiohttp.ClientSession,
    messages: list[dict],
    api_key: str,
    model: str | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str:
    """Make an async OpenRouter chat completion request for reward judging."""
    if response_format and (
        _env_flag("PERSONA_JUDGE_DISABLE_RESPONSE_FORMAT", False)
        or _env_flag("PERSONA_LOCAL_JUDGE_DISABLE_RESPONSE_FORMAT", False)
    ):
        response_format = None
    payload = build_chat_payload(
        model=model or os.environ.get("JUDGE_MODEL", JUDGE_MODEL),
        messages=messages,
        max_completion_tokens=max_tokens or _get_judge_max_completion_tokens(),
        response_format=response_format,
        reasoning=False,
    )
    return await post_chat_async(session, payload, semaphore=_get_reward_judge_request_semaphore())

    raise RuntimeError(f"OpenAI API call failed after {max_retries} retries")


def _get_judge_max_completion_tokens() -> int:
    return int(os.environ.get("PERSONA_JUDGE_MAX_COMPLETION_TOKENS", "8192"))


def _extract_json(text: str | None) -> dict | None:
    """Extract JSON object from response text."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _coerce_json_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_json_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_penalty(value: Any) -> float:
    return max(0.0, min(1.0, _coerce_json_float(value, 0.0)))


def _sanitize_text(text: str) -> str:
    """Remove control characters that break JSON serialization."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


WORD_RE = re.compile(r"\b[\w']+\b")


def _count_words(text: str) -> int:
    """Count approximate words for communication-length metadata."""
    return len(WORD_RE.findall(text))


def _normalize_turing_calibration_domain(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    if "prism" in normalized:
        return "prism"
    if "convokit" in normalized or "reddit" in normalized:
        return "convokit"
    return ""


def _turing_calibration_domain_from_metadata(data_source: Any, extra_info: dict | None) -> str:
    """Prefer explicit GRPO metadata; fall back to context shape inside the scorer."""
    extra_info = extra_info or {}
    candidates = (
        data_source,
        extra_info.get("data_source"),
        extra_info.get("source_name"),
        extra_info.get("dataset_name"),
        extra_info.get("dataset_config"),
        os.environ.get("GRPO_DATASET", ""),
    )
    for candidate in candidates:
        domain = _normalize_turing_calibration_domain(candidate)
        if domain:
            return domain
    return ""


async def _score_pairwise_likert_with_info(
    session: aiohttp.ClientSession,
    api_key: str,
    response: str,
    ground_truth: str,
    user_history: str,
    context: str,
    *,
    prompt_template: str,
    persona: str = "",
    calibration_domain: str = "",
    user_id: Any = "",
    post_id: Any = "",
    target_idx: Any = "",
    randomization_seed_material: str = "",
) -> dict[str, Any]:
    """Shared pairwise Likert scorer used by Turing rewards."""
    response = _sanitize_text(response)
    ground_truth = _sanitize_text(ground_truth)
    user_history = _sanitize_text(user_history)
    context = _sanitize_text(context)
    persona = _sanitize_text(persona)
    response_source_copy_warning = build_source_copy_warning(
        response,
        user_history=user_history,
        thread_context=context,
    )
    ground_truth_source_copy_warning = build_source_copy_warning(
        ground_truth,
        user_history=user_history,
        thread_context=context,
    )

    async def _call(
        resp_a: str,
        resp_b: str,
        source_copy_warning_a: dict[str, Any],
        source_copy_warning_b: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = prompt_template.format(
            persona=persona,
            user_history=user_history,
            context=context,
            response_a=resp_a,
            response_b=resp_b,
            source_copy_watchlist=format_source_copy_watchlist(
                [source_copy_warning_a, source_copy_warning_b],
                item_label="Response",
                labels=["Response A", "Response B"],
            ),
        )
        parse_attempts = get_openai_max_retries()
        data = None
        for parse_attempt in range(parse_attempts):
            text = await _openai_chat(
                session,
                [{"role": "user", "content": prompt}],
                api_key,
                response_format={"type": "json_object"},
            )
            data = _extract_json(text)
            if data is not None:
                break
            recovered_rating = _extract_turing_rating(text)
            if recovered_rating is not None:
                data = _turing_parse_failure_result(rating=recovered_rating, raw_text=text)
                print(
                    f"[turing] recovered rating={recovered_rating} from malformed judge response",
                    flush=True,
                )
                break
            if parse_attempt < parse_attempts - 1:
                print(
                    f"[turing] parse retry {parse_attempt + 1}/{parse_attempts}: judge returned malformed JSON",
                    flush=True,
                )
        if data is None:
            return _turing_parse_failure_result()
        if _coerce_json_bool(data.get("parse_fallback", False)):
            return data
        parse_error = False
        has_score_fields = any(
            key in data
            for key in (
                "immediate_target_score_a",
                "immediate_target_score_b",
                "human_goal_score_a",
                "human_goal_score_b",
                "communication_style_score_a",
                "communication_style_score_b",
            )
        )
        explicit_rating = _coerce_turing_rating(data.get("rating"))
        immediate_target_score_a = _coerce_json_float(data.get("immediate_target_score_a"))
        immediate_target_score_b = _coerce_json_float(data.get("immediate_target_score_b"))
        human_goal_score_a = _coerce_json_float(data.get("human_goal_score_a"))
        human_goal_score_b = _coerce_json_float(data.get("human_goal_score_b"))
        communication_style_score_a = _coerce_json_float(data.get("communication_style_score_a"))
        communication_style_score_b = _coerce_json_float(data.get("communication_style_score_b"))
        source_copy_penalty_a = _coerce_penalty(data.get("source_copy_penalty_a"))
        source_copy_penalty_b = _coerce_penalty(data.get("source_copy_penalty_b"))
        assistant_like_penalty_a = _coerce_penalty(data.get("assistant_like_penalty_a"))
        assistant_like_penalty_b = _coerce_penalty(data.get("assistant_like_penalty_b"))
        wrong_target_or_role_penalty_a = _coerce_penalty(data.get("wrong_target_or_role_penalty_a"))
        wrong_target_or_role_penalty_b = _coerce_penalty(data.get("wrong_target_or_role_penalty_b"))
        unsupported_adversarial_reframing_penalty_a = _coerce_penalty(
            data.get("unsupported_adversarial_reframing_penalty_a")
        )
        unsupported_adversarial_reframing_penalty_b = _coerce_penalty(
            data.get("unsupported_adversarial_reframing_penalty_b")
        )
        base_score_a = immediate_target_score_a + human_goal_score_a + communication_style_score_a
        base_score_b = immediate_target_score_b + human_goal_score_b + communication_style_score_b
        penalty_a = (
            source_copy_penalty_a
            + assistant_like_penalty_a
            + wrong_target_or_role_penalty_a
            + unsupported_adversarial_reframing_penalty_a
        ) / 4.0 * 3.0
        penalty_b = (
            source_copy_penalty_b
            + assistant_like_penalty_b
            + wrong_target_or_role_penalty_b
            + unsupported_adversarial_reframing_penalty_b
        ) / 4.0 * 3.0
        response_a_score = max(0.0, base_score_a - penalty_a)
        response_b_score = max(0.0, base_score_b - penalty_b)
        score_gap = response_b_score - response_a_score
        rating = _rating_from_turing_score_gap(score_gap) if has_score_fields or explicit_rating is None else explicit_rating

        return {
            "response_a_source_copy": str(data.get("response_a_source_copy", "") or ""),
            "response_b_source_copy": str(data.get("response_b_source_copy", "") or ""),
            "source_copy_penalty_a": source_copy_penalty_a,
            "source_copy_penalty_b": source_copy_penalty_b,
            "response_a_assistant_like": str(data.get("response_a_assistant_like", "") or ""),
            "response_b_assistant_like": str(data.get("response_b_assistant_like", "") or ""),
            "assistant_like_penalty_a": assistant_like_penalty_a,
            "assistant_like_penalty_b": assistant_like_penalty_b,
            "response_a_wrong_target_or_role": str(data.get("response_a_wrong_target_or_role", "") or ""),
            "response_b_wrong_target_or_role": str(data.get("response_b_wrong_target_or_role", "") or ""),
            "wrong_target_or_role_penalty_a": wrong_target_or_role_penalty_a,
            "wrong_target_or_role_penalty_b": wrong_target_or_role_penalty_b,
            "response_a_unsupported_adversarial_reframing": str(
                data.get("response_a_unsupported_adversarial_reframing", "") or ""
            ),
            "response_b_unsupported_adversarial_reframing": str(
                data.get("response_b_unsupported_adversarial_reframing", "") or ""
            ),
            "unsupported_adversarial_reframing_penalty_a": unsupported_adversarial_reframing_penalty_a,
            "unsupported_adversarial_reframing_penalty_b": unsupported_adversarial_reframing_penalty_b,
            "immediate_target_score_a": immediate_target_score_a,
            "immediate_target_score_b": immediate_target_score_b,
            "human_goal_score_a": human_goal_score_a,
            "human_goal_score_b": human_goal_score_b,
            "communication_style_score_a": communication_style_score_a,
            "communication_style_score_b": communication_style_score_b,
            "base_score_a": base_score_a,
            "base_score_b": base_score_b,
            "penalty_a": penalty_a,
            "penalty_b": penalty_b,
            "response_a_score": response_a_score,
            "response_b_score": response_b_score,
            "score_gap": score_gap,
            "reasoning": str(data.get("reasoning", "") or ""),
            "rating": rating,
            "parse_error": parse_error,
        }

    generated_is_b = _stable_turing_generated_is_b(
        response,
        user_id=user_id,
        post_id=post_id,
        target_idx=target_idx,
        seed_material=randomization_seed_material,
    )
    if generated_is_b:
        result = await _call(
            ground_truth,
            response,
            ground_truth_source_copy_warning,
            response_source_copy_warning,
        )
        generated_source_copy_penalty = float(result.get("source_copy_penalty_b", 0.0) or 0.0)
        generated_source_copy = generated_source_copy_penalty > 0.0
        generated_wrong_perspective = False
        generated_assistant_like_penalty = float(result.get("assistant_like_penalty_b", 0.0) or 0.0)
        generated_assistant_like = generated_assistant_like_penalty > 0.0
        generated_wrong_target_or_role_penalty = float(result.get("wrong_target_or_role_penalty_b", 0.0) or 0.0)
        generated_wrong_target_or_role = generated_wrong_target_or_role_penalty > 0.0
        generated_unsupported_adversarial_reframing_penalty = float(
            result.get("unsupported_adversarial_reframing_penalty_b", 0.0) or 0.0
        )
        generated_unsupported_adversarial_reframing = generated_unsupported_adversarial_reframing_penalty > 0.0
        judge_parse_error = bool(result.get("parse_error", False))
        generated_hallucination_reasoning = str(
            result.get("response_b_wrong_target_or_role", "")
            or result.get("response_b_unsupported_adversarial_reframing", "")
            or result.get("response_b_logic", "")
        )
        likert_score = 0.0 if judge_parse_error else float(result["rating"])
        rating_gt_first = int(result["rating"])
        rating_gen_first = None
        source_copy_gt_first = generated_source_copy
        source_copy_gen_first = None
        judge_gt_first = result
        judge_gen_first = None
        randomized_order = "gt_first"
    else:
        result = await _call(
            response,
            ground_truth,
            response_source_copy_warning,
            ground_truth_source_copy_warning,
        )
        generated_source_copy_penalty = float(result.get("source_copy_penalty_a", 0.0) or 0.0)
        generated_source_copy = generated_source_copy_penalty > 0.0
        generated_wrong_perspective = False
        generated_assistant_like_penalty = float(result.get("assistant_like_penalty_a", 0.0) or 0.0)
        generated_assistant_like = generated_assistant_like_penalty > 0.0
        generated_wrong_target_or_role_penalty = float(result.get("wrong_target_or_role_penalty_a", 0.0) or 0.0)
        generated_wrong_target_or_role = generated_wrong_target_or_role_penalty > 0.0
        generated_unsupported_adversarial_reframing_penalty = float(
            result.get("unsupported_adversarial_reframing_penalty_a", 0.0) or 0.0
        )
        generated_unsupported_adversarial_reframing = generated_unsupported_adversarial_reframing_penalty > 0.0
        judge_parse_error = bool(result.get("parse_error", False))
        generated_hallucination_reasoning = str(
            result.get("response_a_wrong_target_or_role", "")
            or result.get("response_a_unsupported_adversarial_reframing", "")
            or result.get("response_a_logic", "")
        )
        likert_score = 0.0 if judge_parse_error else float(8 - int(result["rating"]))
        rating_gt_first = None
        rating_gen_first = int(result["rating"])
        source_copy_gt_first = None
        source_copy_gen_first = generated_source_copy
        judge_gt_first = None
        judge_gen_first = result
        randomized_order = "gen_first"
    return {
        "score": likert_score,
        "source_copy": generated_source_copy,
        "source_copy_penalty": generated_source_copy_penalty,
        "wrong_perspective": generated_wrong_perspective,
        "assistant_like": generated_assistant_like,
        "assistant_like_penalty": generated_assistant_like_penalty,
        "wrong_target_or_role": generated_wrong_target_or_role,
        "wrong_target_or_role_penalty": generated_wrong_target_or_role_penalty,
        "unsupported_adversarial_reframing": generated_unsupported_adversarial_reframing,
        "unsupported_adversarial_reframing_penalty": generated_unsupported_adversarial_reframing_penalty,
        "hallucination_reasoning": generated_hallucination_reasoning,
        "rating_gt_first": rating_gt_first,
        "rating_gen_first": rating_gen_first,
        "rating_randomized": int(result["rating"]),
        "generated_is_b": generated_is_b,
        "randomized_order": randomized_order,
        "source_copy_gt_first": source_copy_gt_first,
        "source_copy_gen_first": source_copy_gen_first,
        "judge_gt_first": judge_gt_first,
        "judge_gen_first": judge_gen_first,
        "judge_randomized": result,
    }


async def score_turing_with_info(
    session: aiohttp.ClientSession,
    api_key: str,
    response: str,
    ground_truth: str,
    user_history: str,
    context: str,
    calibration_domain: str = "",
    user_id: Any = "",
    post_id: Any = "",
    target_idx: Any = "",
    randomization_seed_material: str = "",
) -> dict[str, Any]:
    """Turing test with judge-returned source-copy metadata."""
    return await _score_pairwise_likert_with_info(
        session,
        api_key,
        response,
        ground_truth,
        user_history,
        context,
        prompt_template=TURING_PROMPT,
        calibration_domain=calibration_domain,
        user_id=user_id,
        post_id=post_id,
        target_idx=target_idx,
        randomization_seed_material=randomization_seed_material,
    )


_shared_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    """Lazily create a shared session. Reuses TCP connections across calls."""
    if aiohttp is None:
        raise ImportError("OpenAI-backed reward scoring requires aiohttp to be installed")
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        request_limit = _reward_judge_request_limit()
        connector_limit = max(
            request_limit,
            int(os.environ.get("PERSONA_OPENAI_CONNECTION_LIMIT", str(request_limit))),
        )
        timeout_seconds = float(os.environ.get("PERSONA_OPENAI_TIMEOUT_SECONDS", "400"))
        connector = aiohttp.TCPConnector(limit=connector_limit)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        print(
            f"[openai] aiohttp connection limit={connector_limit} timeout_s={timeout_seconds:g}",
            flush=True,
        )
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


_compute_score_call_count = 0
_generation_log_path = os.environ.get("GENERATION_LOG", "")


def _log_generation(record: dict):
    """Append a generation record to the JSONL log file."""
    if not _generation_log_path:
        return
    with open(_generation_log_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Score one veRL sample."""
    global _compute_score_call_count
    _compute_score_call_count += 1
    call_id = _compute_score_call_count

    metric = os.environ.get("REWARD_METRIC", "turing")
    extra_info = extra_info or {}
    prompt_mode = str(extra_info.get("prompt_mode", "") or "")
    cot, response = parse_response_for_prompt_mode(solution_str, prompt_mode)
    response_components = response_format_components(solution_str)
    thinking_info = build_meaningful_thinking_info(solution_str, prompt_mode, response_components)
    logged_thinking_info = {
        key: value for key, value in thinking_info.items() if key != "thinking_hard_zero"
    }
    if bool(thinking_info.get("thinking_hard_zero", 0.0)) and metric != "logprob":
        format_reward_info = empty_format_reward_info()
    else:
        format_reward_info = build_format_reward_info(solution_str, metric, prompt_mode, response_components)
    format_score = format_reward_info["format_score"]
    reward = 0.0
    unadjusted_raw_reward = 0.0
    adjusted_raw_reward = 0.0
    turing_judge_score_raw = 0.0
    turing_judge_score_clipped = 0.0
    length_info = compute_turing_length_info(response, ground_truth)
    length_penalty = float(length_info.get("length_penalty", 0.0)) if metric == "turing" else 0.0
    source_copy = False
    assistant_like_response = False
    wrong_target_or_role_response = False
    unsupported_adversarial_reframing_response = False

    if call_id <= 5:
        print(f"[reward #{call_id}] metric={metric} solution_str_len={len(solution_str)} response_len={len(response)} gt_len={len(ground_truth)} format={format_score}", flush=True)
        print(f"[reward #{call_id}] RAW: {repr(solution_str[:300])}", flush=True)
        print(f"[reward #{call_id}] PARSED: {repr(response[:300])}", flush=True)

    if bool(thinking_info.get("thinking_hard_zero", 0.0)) and metric != "logprob":
        format_reward_info = empty_format_reward_info()
        format_score = 0.0
        if call_id <= 5:
            print(
                f"[reward #{call_id}] hard_zero=no_meaningful_thinking "
                f"thinking_has_alphabetic_token={thinking_info['thinking_has_alphabetic_token']}",
                flush=True,
            )
        _log_generation({
            "call_id": call_id,
            "metric": metric,
            "raw_generation": solution_str,
            "cot": cot,
            "parsed_response": response,
            "ground_truth": ground_truth,
            "reward": 0.0,
            "unadjusted_raw_reward": 0.0,
            "adjusted_raw_reward": 0.0,
            "turing_judge_score_raw": 0.0,
            "turing_judge_score_clipped": 0.0,
            "source_copy": False,
            "assistant_like_response": False,
            "wrong_target_or_role_response": False,
            "unsupported_adversarial_reframing_response": False,
            **length_info,
            **format_reward_info,
            **logged_thinking_info,
            "total_score": 0.0,
            "logprob_failure": "no_meaningful_thinking",
        })
        result = {
            "score": 0.0,
            "total_score": 0.0,
            "raw_reward": 0.0,
            "unadjusted_raw_reward": 0.0,
            "adjusted_raw_reward": 0.0,
            "turing_judge_score_raw": 0.0,
            "turing_judge_score_clipped": 0.0,
            "source_copy": 0.0,
            "assistant_like_response": 0.0,
            "wrong_target_or_role_response": 0.0,
            "unsupported_adversarial_reframing_response": 0.0,
            **length_info,
        }
        result.update(format_reward_info)
        result.update(logged_thinking_info)
        return result

    if not response and metric != "logprob":
        total_score = max(0.0, format_score - length_penalty)
        result = {
            "score": total_score,
            "total_score": total_score,
            "raw_reward": 0.0,
            "unadjusted_raw_reward": 0.0,
            "adjusted_raw_reward": 0.0,
            "turing_judge_score_raw": 0.0,
            "turing_judge_score_clipped": 0.0,
            "source_copy": 0.0,
            "assistant_like_response": 0.0,
            "wrong_target_or_role_response": 0.0,
            "unsupported_adversarial_reframing_response": 0.0,
            **length_info,
        }
        result.update(format_reward_info)
        result.update(logged_thinking_info)
        return result

    context = extra_info.get("context", "")
    user_history = extra_info.get("user_history", "")
    if metric == "turing":
        session = _get_session()
        api_key = resolve_judge_api_key()
        pairwise_result = await score_turing_with_info(
            session,
            api_key,
            response,
            ground_truth,
            user_history,
            context,
            calibration_domain=_turing_calibration_domain_from_metadata(data_source, extra_info),
            user_id=extra_info.get("user_id", ""),
            post_id=extra_info.get("post_id", ""),
            target_idx=extra_info.get("target_idx", extra_info.get("prompt_idx", "")),
        )
        source_copy = bool(pairwise_result.get("source_copy", False))
        assistant_like_response = bool(pairwise_result.get("assistant_like", False))
        wrong_target_or_role_response = bool(pairwise_result.get("wrong_target_or_role", False))
        unsupported_adversarial_reframing_response = bool(
            pairwise_result.get("unsupported_adversarial_reframing", False)
        )
        turing_judge_score_raw = float(pairwise_result["score"])
        turing_judge_score_clipped = clip_turing_judge_score(turing_judge_score_raw)
        unadjusted_raw_reward = (turing_judge_score_clipped - 1.0) / 6.0
        adjusted_raw_reward = adjust_turing_raw_reward(unadjusted_raw_reward)
        reward = adjusted_raw_reward
    elif metric == "logprob":
        raise RuntimeError(
            "REWARD_METRIC=logprob now expects rollout-side current-policy scoring. "
            "The external frozen logprob server path has been retired."
        )
    else:
        raise ValueError(f"Unknown REWARD_METRIC: {metric}")

    total_score = max(0.0, reward + format_score - length_penalty)
    if call_id <= 5:
        print(
            f"[reward #{call_id}] Done. reward={reward:.4f} "
            f"format={format_score:.4f} length_penalty={length_penalty:.4f} "
            f"total_score={total_score:.4f}",
            flush=True,
        )

    _log_generation({
        "call_id": call_id,
        "metric": metric,
        "raw_generation": solution_str,
        "cot": cot,
        "parsed_response": response,
        "ground_truth": ground_truth,
        "reward": reward,
        "unadjusted_raw_reward": unadjusted_raw_reward,
        "adjusted_raw_reward": adjusted_raw_reward,
        "turing_judge_score_raw": turing_judge_score_raw,
        "turing_judge_score_clipped": turing_judge_score_clipped,
        "source_copy": source_copy,
        "assistant_like_response": assistant_like_response,
        "wrong_target_or_role_response": wrong_target_or_role_response,
        "unsupported_adversarial_reframing_response": unsupported_adversarial_reframing_response,
        **length_info,
        **format_reward_info,
        **logged_thinking_info,
        "total_score": total_score,
        "logprob_failure": None,
    })

    result = {
        "score": total_score,
        "total_score": total_score,
        "raw_reward": reward,
        "unadjusted_raw_reward": unadjusted_raw_reward,
        "adjusted_raw_reward": adjusted_raw_reward,
        "turing_judge_score_raw": turing_judge_score_raw,
        "turing_judge_score_clipped": turing_judge_score_clipped,
        "source_copy": 1.0 if source_copy else 0.0,
        "assistant_like_response": 1.0 if assistant_like_response else 0.0,
        "wrong_target_or_role_response": 1.0 if wrong_target_or_role_response else 0.0,
        "unsupported_adversarial_reframing_response": (
            1.0 if unsupported_adversarial_reframing_response else 0.0
        ),
        **length_info,
    }
    result.update(format_reward_info)
    result.update(logged_thinking_info)
    return result
