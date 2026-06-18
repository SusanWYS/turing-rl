"""Judge metrics for heldout generations."""

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from shared.judge_utils import (
    _coerce_turing_rating,
    _extract_turing_rating,
    _rating_from_turing_score_gap,
    _stable_turing_generated_is_b,
    _turing_parse_failure_result,
    build_source_copy_warning,
    format_source_copy_watchlist,
    judge_response_batch,
)

from shared.api_client import get_openai_max_retries, openrouter_request_extras, post_chat_sync
from shared.load_env import get_openai_api_key
from shared.judge_prompts import SPECIFICITY_PROMPT, TURING_PROMPT


def _sanitize_text(text: str) -> str:
    """Return JSON-safe text."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_SIM_EVAL_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"


def get_judge_model() -> str:
    """Return the eval judge model."""
    return os.getenv("PERSONA_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def get_sim_eval_judge_model() -> str:
    """Return the sim judge model."""
    return (
        os.getenv("PERSONA_SIM_EVAL_JUDGE_MODEL")
        or os.getenv("SIM_JUDGE_MODEL")
        or DEFAULT_SIM_EVAL_JUDGE_MODEL
    )


def _extract_json(text: str | None) -> Optional[dict]:
    """Extract one JSON object."""
    if not isinstance(text, str):
        return None
    text = text.strip()

    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("  Warning: Failed to parse judge JSON response")
        return None


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_specificity_response(response_text: str) -> dict:
    """Parse specificity judge JSON."""
    data = _extract_json(response_text)
    if data is None:
        return {
            "scores": {
                "context_specificity": 0.0,
                "user_evidence_compatibility": 0.0,
            },
            "reasons": {
                "context_specificity": "",
                "user_evidence_compatibility": "",
            },
            "reasoning": "",
            "overall": 0.0,
            "parse_error": True,
        }

    scores = {}
    reasons = {}
    for dimension in ("context_specificity", "user_evidence_compatibility"):
        entry = data.get(dimension, {})
        if not isinstance(entry, dict):
            return _specificity_parse_failure_result()
        score = _parse_float(entry.get("score"))
        if score is None:
            return _specificity_parse_failure_result()
        reason = str(entry.get("reason", "") or "")
        score = max(0.0, min(1.0, score))
        scores[dimension] = score
        reasons[dimension] = reason

    computed_overall = (
        scores["context_specificity"] + scores["user_evidence_compatibility"]
    ) / 2.0
    raw_overall = data.get("overall", computed_overall)
    overall = _parse_float(raw_overall)
    if overall is None:
        return _specificity_parse_failure_result()
    overall = max(0.0, min(1.0, overall))

    return {
        "scores": scores,
        "reasons": reasons,
        "reasoning": str(data.get("reasoning", "") or ""),
        "overall": overall,
        "computed_overall": computed_overall,
        "parse_error": False,
    }


def _specificity_parse_failure_result() -> dict:
    return {
        "scores": {
            "context_specificity": 0.0,
            "user_evidence_compatibility": 0.0,
        },
        "reasons": {
            "context_specificity": "",
            "user_evidence_compatibility": "",
        },
        "reasoning": "",
        "overall": 0.0,
        "computed_overall": 0.0,
        "parse_error": True,
    }


def _specificity_api_call(
    *,
    user_history: str,
    context: str,
    candidate_response: str,
    max_tokens: int = 2048,
) -> dict:
    """Score one response with the specificity judge."""
    prompt = SPECIFICITY_PROMPT.format(
        user_history=_sanitize_text(user_history),
        context=_sanitize_text(context),
        candidate_response=_sanitize_text(candidate_response),
    )
    max_tokens = int(os.getenv("PERSONA_SPECIFICITY_MAX_COMPLETION_TOKENS", str(max_tokens)))
    parse_retries = get_openai_max_retries()
    parsed = _specificity_parse_failure_result()
    for _attempt in range(max(0, parse_retries) + 1):
        kwargs = {
            "model": get_judge_model(),
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        kwargs.update(openrouter_request_extras(reasoning=True))
        parsed = _parse_specificity_response(post_chat_sync(kwargs))
        if not parsed.get("parse_error"):
            return parsed
    return parsed


def specificity_judge_generate_results(
    generate_results: dict,
    *,
    user_histories: Optional[dict[str, str]] = None,
    max_workers: int = 100,
) -> dict:
    """Score generations with the specificity judge."""
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Export it before running specificity eval:\n"
            "  export OPENAI_API_KEY='sk-...'"
        )

    jobs = []
    thread_index = {}

    for user_id, user_data in generate_results.items():
        threads = user_data.get("test_threads") or user_data.get("test_targets", [])
        for t_idx, thread_data in enumerate(threads):
            generations = thread_data.get("generations", [])
            if generations and isinstance(generations[0], dict):
                gen_texts = [g["response"] for g in generations]
            elif generations and isinstance(generations[0], str):
                gen_texts = generations
            else:
                gen_texts = []

            if not gen_texts:
                if thread_data.get("generated_response"):
                    gen_texts = [thread_data["generated_response"]]
                else:
                    continue

            ground_truth = thread_data["ground_truth"]
            context = thread_data.get("context", "")
            user_history = str(thread_data.get("user_history", "") or "")
            if not user_history and user_histories is not None:
                user_history = user_histories.get(user_id, "")

            thread_index[(user_id, t_idx)] = {
                "post_id": thread_data["post_id"],
                "target_idx": thread_data["target_idx"],
                "ground_truth": ground_truth,
                "gen_texts": gen_texts,
                "context": context,
                "user_history": user_history,
            }

            for g_idx, gen_text in enumerate(gen_texts):
                jobs.append(("generation", user_id, t_idx, g_idx, user_history, context, gen_text))

    results_map = {}

    def _call(job):
        kind, uid, t_idx, g_idx, user_history, context, candidate_response = job
        result = _specificity_api_call(
            user_history=user_history,
            context=context,
            candidate_response=candidate_response,
        )
        return (uid, t_idx, kind, g_idx), result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call, job): job for job in jobs}
        for future in as_completed(futures):
            key, result = future.result()
            results_map[key] = result

    judged_results = {}
    for (user_id, t_idx), meta in thread_index.items():
        if user_id not in judged_results:
            user_data = generate_results[user_id]
            judged_results[user_id] = {
                "user_id": user_id,
                "test_threads": [],
            }
            if "sparse_thread_ids" in user_data:
                judged_results[user_id]["sparse_thread_ids"] = user_data["sparse_thread_ids"]

        gen_texts = meta["gen_texts"]
        n_gens = len(gen_texts)
        dimension_scores = {
            "context_specificity": [0.0] * n_gens,
            "user_evidence_compatibility": [0.0] * n_gens,
        }
        dimension_reasons = {
            "context_specificity": [""] * n_gens,
            "user_evidence_compatibility": [""] * n_gens,
        }
        reasonings = [""] * n_gens
        overall_scores = [0.0] * n_gens
        parse_errors = [False] * n_gens

        for g_idx in range(n_gens):
            result = results_map.get((user_id, t_idx, "generation", g_idx), _specificity_parse_failure_result())
            overall_scores[g_idx] = float(result.get("overall", 0.0))
            reasonings[g_idx] = str(result.get("reasoning", "") or "")
            parse_errors[g_idx] = bool(result.get("parse_error", False))
            for dimension in ("context_specificity", "user_evidence_compatibility"):
                dimension_scores[dimension][g_idx] = float(result.get("scores", {}).get(dimension, 0.0))
                dimension_reasons[dimension][g_idx] = str(result.get("reasons", {}).get(dimension, "") or "")

        judged_results[user_id]["test_threads"].append({
            "post_id": meta["post_id"],
            "target_idx": meta["target_idx"],
            "ground_truth": meta["ground_truth"],
            "generations": gen_texts,
            "context": meta["context"],
            "user_history": meta["user_history"],
            "specificity_scores": overall_scores,
            "specificity_dimension_scores": dimension_scores,
            "specificity_reasons": dimension_reasons,
            "specificity_reasonings": reasonings,
            "specificity_parse_errors": parse_errors,
        })

    return judged_results


def sim_judge_generate_results(
    generate_results: dict,
    *,
    user_histories: Optional[dict[str, str]] = None,
    max_workers: int = 100,
    include_breakdown: bool = False,
) -> dict:
    """Score generations with the sim judge."""
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Export it before running sim eval:\n"
            "  export OPENAI_API_KEY='sk-...'"
        )

    resolved_judge_model = get_sim_eval_judge_model()
    jobs = []
    thread_index = {}

    for user_id, user_data in generate_results.items():
        threads = user_data.get("test_threads") or user_data.get("test_targets", [])
        for t_idx, thread_data in enumerate(threads):
            generations = thread_data.get("generations", [])
            if generations and isinstance(generations[0], dict):
                gen_texts = [g["response"] for g in generations]
            elif generations and isinstance(generations[0], str):
                gen_texts = generations
            else:
                gen_texts = []

            if not gen_texts:
                if thread_data.get("generated_response"):
                    gen_texts = [thread_data["generated_response"]]
                else:
                    continue

            ground_truth = thread_data["ground_truth"]
            context = thread_data.get("context", "")
            user_history = thread_data.get("user_history", "")
            if not user_history and user_histories is not None:
                user_history = user_histories.get(user_id, "")

            thread_index[(user_id, t_idx)] = {
                "post_id": thread_data["post_id"],
                "target_idx": thread_data["target_idx"],
                "ground_truth": ground_truth,
                "gen_texts": gen_texts,
                "context": context,
                "user_history": user_history,
            }
            jobs.append((user_id, t_idx, user_history, context, ground_truth, gen_texts))

    async def _run_jobs() -> dict[tuple[str, int], list[dict]]:
        semaphore = asyncio.Semaphore(max(1, max_workers))

        async def _call(job: tuple[str, int, str, str, str, list[str]]) -> tuple[tuple[str, int], list[dict]]:
            user_id, t_idx, user_history, context, ground_truth, gen_texts = job
            async with semaphore:
                copy_warnings = [
                    build_source_copy_warning(
                        candidate,
                        user_history=user_history,
                        thread_context=context,
                    )
                    for candidate in gen_texts
                ]
                outputs = await judge_response_batch(
                    user_history=user_history,
                    thread_context=context,
                    ground_truth=ground_truth,
                    candidates=gen_texts,
                    model=resolved_judge_model,
                    copy_warnings=copy_warnings,
                    include_breakdown=include_breakdown,
                    enable_hard_flags=False,
                    label="offline sim eval",
                )
                return (user_id, t_idx), outputs

        tasks = [asyncio.create_task(_call(job)) for job in jobs]
        results_map: dict[tuple[str, int], list[dict]] = {}
        for task in asyncio.as_completed(tasks):
            key, outputs = await task
            results_map[key] = outputs
        return results_map

    results_map = asyncio.run(_run_jobs())

    judged_results = {}
    for (user_id, t_idx), meta in thread_index.items():
        if user_id not in judged_results:
            user_data = generate_results[user_id]
            judged_results[user_id] = {
                "user_id": user_id,
                "test_threads": [],
            }
            if "sparse_thread_ids" in user_data:
                judged_results[user_id]["sparse_thread_ids"] = user_data["sparse_thread_ids"]

        outputs = results_map[(user_id, t_idx)]
        sim_scores = [float(output.get("score", 0.0)) for output in outputs]
        sim_semantic_similarity = [
            float(output.get("semantic_similarity", output.get("score", 0.0)))
            for output in outputs
        ]
        sim_information_completeness = [
            float(output.get("information_completeness", output.get("score", 0.0)))
            for output in outputs
        ]
        metrics_info = [str(output.get("metrics_info", "") or "") for output in outputs]
        thread_result = {
            "post_id": meta["post_id"],
            "target_idx": meta["target_idx"],
            "ground_truth": meta["ground_truth"],
            "generations": meta["gen_texts"],
            "context": meta["context"],
            "user_history": meta["user_history"],
            "sim_scores": sim_scores,
            "sim_metrics_info": metrics_info,
        }
        if include_breakdown:
            thread_result["sim_semantic_similarity"] = sim_semantic_similarity
            thread_result["sim_information_completeness"] = sim_information_completeness
        judged_results[user_id]["test_threads"].append(thread_result)

    return judged_results


def _parse_turing_response(response_text: str) -> dict:
    """Parse Turing judge JSON."""
    data = _extract_json(response_text)
    if data is None:
        return _turing_parse_failure_result(rating=_extract_turing_rating(response_text))

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

    def _required_score_field(key: str) -> float | None:
        return _parse_float(data.get(key))

    def _optional_penalty_field(key: str) -> float | None:
        if key not in data:
            return 0.0
        value = _parse_float(data.get(key))
        return None if value is None else max(0.0, min(1.0, value))

    if has_score_fields:
        immediate_target_score_a = _required_score_field("immediate_target_score_a")
        immediate_target_score_b = _required_score_field("immediate_target_score_b")
        human_goal_score_a = _required_score_field("human_goal_score_a")
        human_goal_score_b = _required_score_field("human_goal_score_b")
        communication_style_score_a = _required_score_field("communication_style_score_a")
        communication_style_score_b = _required_score_field("communication_style_score_b")
        source_copy_penalty_a = _optional_penalty_field("source_copy_penalty_a")
        source_copy_penalty_b = _optional_penalty_field("source_copy_penalty_b")
        assistant_like_penalty_a = _optional_penalty_field("assistant_like_penalty_a")
        assistant_like_penalty_b = _optional_penalty_field("assistant_like_penalty_b")
        wrong_target_or_role_penalty_a = _optional_penalty_field("wrong_target_or_role_penalty_a")
        wrong_target_or_role_penalty_b = _optional_penalty_field("wrong_target_or_role_penalty_b")
        unsupported_adversarial_reframing_penalty_a = _optional_penalty_field(
            "unsupported_adversarial_reframing_penalty_a"
        )
        unsupported_adversarial_reframing_penalty_b = _optional_penalty_field(
            "unsupported_adversarial_reframing_penalty_b"
        )
        parsed_numbers = (
            immediate_target_score_a,
            immediate_target_score_b,
            human_goal_score_a,
            human_goal_score_b,
            communication_style_score_a,
            communication_style_score_b,
            source_copy_penalty_a,
            source_copy_penalty_b,
            assistant_like_penalty_a,
            assistant_like_penalty_b,
            wrong_target_or_role_penalty_a,
            wrong_target_or_role_penalty_b,
            unsupported_adversarial_reframing_penalty_a,
            unsupported_adversarial_reframing_penalty_b,
        )
        if any(value is None for value in parsed_numbers):
            return _turing_parse_failure_result(raw_text=response_text)
    elif explicit_rating is None:
        return _turing_parse_failure_result(raw_text=response_text)
    else:
        immediate_target_score_a = immediate_target_score_b = 0.0
        human_goal_score_a = human_goal_score_b = 0.0
        communication_style_score_a = communication_style_score_b = 0.0
        source_copy_penalty_a = source_copy_penalty_b = 0.0
        assistant_like_penalty_a = assistant_like_penalty_b = 0.0
        wrong_target_or_role_penalty_a = wrong_target_or_role_penalty_b = 0.0
        unsupported_adversarial_reframing_penalty_a = unsupported_adversarial_reframing_penalty_b = 0.0
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
    rating = (
        _rating_from_turing_score_gap(score_gap)
        if has_score_fields or explicit_rating is None
        else explicit_rating
    )
    reasoning = data.get("reasoning", "")
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
        "reasoning": reasoning,
        "rating": rating,
        "parse_error": parse_error,
    }


def _turing_api_call(
    context: str,
    response_a: str,
    response_b: str,
    user_history: Optional[str] = None,
    max_tokens: int = 2048,
    source_copy_warning_a: Optional[dict] = None,
    source_copy_warning_b: Optional[dict] = None,
    return_details: bool = False,
) -> int | dict:
    """Score one Turing comparison."""
    history_text = user_history or ""
    if source_copy_warning_a is None:
        source_copy_warning_a = build_source_copy_warning(
            response_a,
            user_history=history_text,
            thread_context=context,
        )
    if source_copy_warning_b is None:
        source_copy_warning_b = build_source_copy_warning(
            response_b,
            user_history=history_text,
            thread_context=context,
        )
    source_copy_watchlist = format_source_copy_watchlist(
        [source_copy_warning_a, source_copy_warning_b],
        item_label="Response",
        labels=["Response A", "Response B"],
    )
    if not user_history:
        raise ValueError(
            "Turing eval requires non-empty user_history; refusing to score without it."
        )
    prompt = TURING_PROMPT.format(
        user_history=user_history,
        context=context,
        response_a=response_a,
        response_b=response_b,
        source_copy_watchlist=source_copy_watchlist,
    )

    parse_retries = get_openai_max_retries()
    for attempt in range(max(0, parse_retries) + 1):
        kwargs = {
            "model": get_judge_model(),
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        kwargs.update(openrouter_request_extras(reasoning=True))
        response_content = post_chat_sync(kwargs)

        parsed = _parse_turing_response(response_content)
        if return_details:
            parsed["source_copy_warning_a"] = source_copy_warning_a
            parsed["source_copy_warning_b"] = source_copy_warning_b
        if not parsed.get("parse_error"):
            return parsed if return_details else parsed["rating"]

    return parsed if return_details else parsed["rating"]


def turing_test_generate_results(
    generate_results: dict,
    user_histories: Optional[dict[str, str]] = None,
    max_workers: int = 100,
) -> dict:
    """Score generations with the Turing judge."""
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Export it before running eval_turing:\n"
            "  export OPENAI_API_KEY='sk-...'"
        )


    call_jobs = []
    thread_index = {}

    for user_id, user_data in generate_results.items():
        threads = user_data.get("test_threads") or user_data.get("test_targets", [])

        for t_idx, thread_data in enumerate(threads):
            generations = thread_data.get("generations", [])
            if generations and isinstance(generations[0], dict):
                gen_texts = [g["response"] for g in generations]
            elif generations and isinstance(generations[0], str):
                gen_texts = generations
            else:
                gen_texts = []

            if not gen_texts:
                if thread_data.get("generated_response"):
                    gen_texts = [thread_data["generated_response"]]
                else:
                    continue

            ground_truth = _sanitize_text(thread_data["ground_truth"])
            context = _sanitize_text(thread_data.get("context", ""))
            user_history = thread_data.get("user_history", "")
            if not user_history and user_histories is not None:
                user_history = user_histories.get(user_id, "")
            if not user_history:
                raise ValueError(
                    "Turing eval requires non-empty user_history "
                    f"for user_id={user_id!r}, post_id={thread_data.get('post_id')!r}, "
                    f"target_idx={thread_data.get('target_idx')!r}."
                )
            history = _sanitize_text(user_history) if user_history else None

            thread_index[(user_id, t_idx)] = {
                "post_id": thread_data["post_id"],
                "target_idx": thread_data["target_idx"],
                "ground_truth": thread_data["ground_truth"],
                "gen_texts": gen_texts,
                "context": thread_data.get("context", ""),
                "user_history": user_history or "",
            }

            for g_idx, gen_text in enumerate(gen_texts):
                gen_text = _sanitize_text(gen_text)
                generated_is_b = _stable_turing_generated_is_b(
                    gen_text,
                    user_id=user_id,
                    post_id=thread_data["post_id"],
                    target_idx=thread_data["target_idx"],
                )
                if generated_is_b:
                    call_jobs.append(
                        (user_id, t_idx, g_idx, "gt_first", context, ground_truth, gen_text, history)
                    )
                else:
                    call_jobs.append(
                        (user_id, t_idx, g_idx, "gen_first", context, gen_text, ground_truth, history)
                    )

    results_map = {}

    def _call(job):
        uid, t_idx, g_idx, ordering, ctx, resp_a, resp_b, hist = job
        rating = _turing_api_call(ctx, resp_a, resp_b, hist, return_details=True)
        return (uid, t_idx, g_idx, ordering), rating

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call, job): job for job in call_jobs}
        for future in as_completed(futures):
            key, rating = future.result()
            results_map[key] = rating

    turing_results = {}

    def _rating_value(result) -> int:
        if isinstance(result, dict):
            return int(result.get("rating", 1))
        return int(result)

    def _bool_value(result, key: str) -> bool:
        if isinstance(result, dict):
            return bool(result.get(key, False))
        return False

    def _penalty_value(result, key: str) -> float:
        if isinstance(result, dict):
            return float(result.get(key, 0.0) or 0.0)
        return 0.0

    def _logic_value(result, key: str) -> str:
        if isinstance(result, dict):
            return str(result.get(key, "") or "")
        return ""

    for (user_id, t_idx), meta in thread_index.items():
        if user_id not in turing_results:
            turing_results[user_id] = {
                "user_id": user_id,
                "test_threads": [],
            }

        gen_texts = meta["gen_texts"]
        comparisons = []
        for g_idx in range(len(gen_texts)):
            gen_text = _sanitize_text(gen_texts[g_idx])
            generated_is_b = _stable_turing_generated_is_b(
                gen_text,
                user_id=user_id,
                post_id=meta["post_id"],
                target_idx=meta["target_idx"],
            )
            if generated_is_b:
                result_gt_first = results_map.get(
                    (user_id, t_idx, g_idx, "gt_first"),
                    {"rating": 0, "parse_error": True},
                )
                result_gen_first = None
                r_gt_first = _rating_value(result_gt_first)
                r_gen_first = None
                source_copy_penalty = _penalty_value(result_gt_first, "source_copy_penalty_b")
                source_copy = source_copy_penalty > 0.0
                wrong_perspective = False
                assistant_like_penalty = _penalty_value(result_gt_first, "assistant_like_penalty_b")
                assistant_like = assistant_like_penalty > 0.0
                wrong_target_or_role_penalty = _penalty_value(result_gt_first, "wrong_target_or_role_penalty_b")
                wrong_target_or_role = wrong_target_or_role_penalty > 0.0
                unsupported_adversarial_reframing_penalty = _penalty_value(
                    result_gt_first, "unsupported_adversarial_reframing_penalty_b"
                )
                unsupported_adversarial_reframing = unsupported_adversarial_reframing_penalty > 0.0
                parse_error = _bool_value(result_gt_first, "parse_error")
                hallucination_reasoning = (
                    _logic_value(result_gt_first, "response_b_wrong_target_or_role")
                    or _logic_value(result_gt_first, "response_b_unsupported_adversarial_reframing")
                    or _logic_value(result_gt_first, "response_b_logic")
                )
                score = 0.0 if parse_error else float(r_gt_first)
                randomized_rating = r_gt_first
                randomized_order = "gt_first"
                source_copy_gt_first = source_copy
                source_copy_gen_first = None
            else:
                result_gt_first = None
                result_gen_first = results_map.get(
                    (user_id, t_idx, g_idx, "gen_first"),
                    {"rating": 0, "parse_error": True},
                )
                r_gt_first = None
                r_gen_first = _rating_value(result_gen_first)
                source_copy_penalty = _penalty_value(result_gen_first, "source_copy_penalty_a")
                source_copy = source_copy_penalty > 0.0
                wrong_perspective = False
                assistant_like_penalty = _penalty_value(result_gen_first, "assistant_like_penalty_a")
                assistant_like = assistant_like_penalty > 0.0
                wrong_target_or_role_penalty = _penalty_value(result_gen_first, "wrong_target_or_role_penalty_a")
                wrong_target_or_role = wrong_target_or_role_penalty > 0.0
                unsupported_adversarial_reframing_penalty = _penalty_value(
                    result_gen_first, "unsupported_adversarial_reframing_penalty_a"
                )
                unsupported_adversarial_reframing = unsupported_adversarial_reframing_penalty > 0.0
                parse_error = _bool_value(result_gen_first, "parse_error")
                hallucination_reasoning = (
                    _logic_value(result_gen_first, "response_a_wrong_target_or_role")
                    or _logic_value(result_gen_first, "response_a_unsupported_adversarial_reframing")
                    or _logic_value(result_gen_first, "response_a_logic")
                )
                score = 0.0 if parse_error else float(8 - r_gen_first)
                randomized_rating = r_gen_first
                randomized_order = "gen_first"
                source_copy_gt_first = None
                source_copy_gen_first = source_copy
            comparisons.append({
                "rating_gt_first": r_gt_first,
                "rating_gen_first": r_gen_first,
                "rating_randomized": randomized_rating,
                "generated_is_b": generated_is_b,
                "randomized_order": randomized_order,
                "score": score,
                "source_copy": source_copy,
                "source_copy_penalty": source_copy_penalty,
                "wrong_perspective": wrong_perspective,
                "assistant_like": assistant_like,
                "assistant_like_penalty": assistant_like_penalty,
                "wrong_target_or_role": wrong_target_or_role,
                "wrong_target_or_role_penalty": wrong_target_or_role_penalty,
                "unsupported_adversarial_reframing": unsupported_adversarial_reframing,
                "unsupported_adversarial_reframing_penalty": unsupported_adversarial_reframing_penalty,
                "hallucination_reasoning": hallucination_reasoning,
                "source_copy_gt_first": source_copy_gt_first,
                "source_copy_gen_first": source_copy_gen_first,
            })

        mean_score = sum(c["score"] for c in comparisons) / len(comparisons)

        turing_results[user_id]["test_threads"].append({
            "post_id": meta["post_id"],
            "target_idx": meta["target_idx"],
            "ground_truth": meta["ground_truth"],
            "generations": gen_texts,
            "context": meta["context"],
            "user_history": meta["user_history"],
            "comparisons": comparisons,
            "mean_score": mean_score,
        })

    return turing_results
