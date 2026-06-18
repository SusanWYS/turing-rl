"""Generate heldout baseline outputs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.generate_trained import (
    get_domain_generation_defaults,
    load_user_results_from_test_parquet,
    render_prompt_text_for_generation,
)
from shared.api_client import openrouter_request_extras, post_chat_sync, resolve_judge_api_key
from shared.prompt_utils import (
    CONDITIONING_MODE_CHOICES,
    CONDITIONING_MODE_HISTORY,
    build_messages_for_prompt_mode,
    normalize_prompt_messages,
)
from shared.sft_prompt_utils import parse_sft_generation
from shared.load_env import get_openai_api_base
from shared.model_ids import DEFAULT_MODEL_ID, QWEN3_5_MOE_MODEL_ID

MODEL_ALIASES = {
    "qwen3-8b": DEFAULT_MODEL_ID,
    "qwen3.5-397b": QWEN3_5_MOE_MODEL_ID,
}
OPENAI_BASELINE_MODEL = "gpt-5-mini"
GPT_API_PROMPT_MODE = "reasoning"
DEFAULT_QWEN_BATCH_SIZE = 4


def parse_generation(text: str) -> dict[str, str]:
    return parse_sft_generation(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate convokit heldout outputs for retained baselines")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "qwen3-8b",
            "qwen3.5-397b",
            "gpt-5",
        ],
        help="Baseline family to generate",
    )
    parser.add_argument("--user_offset", type=int, default=0, help="Heldout users to skip before selection")
    parser.add_argument("--num_users", type=int, default=None, help="Number of heldout users (default: all)")
    parser.add_argument("--gen_num", type=int, default=1, help="Generations per test target")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens per generation")
    parser.add_argument("--output", type=str, default=None, help="Output pickle path")
    parser.add_argument("--max_workers", type=int, default=100, help="Max parallel API calls for gpt-5 mode")
    parser.add_argument(
        "--openai_model",
        type=str,
        default=OPENAI_BASELINE_MODEL,
        help="OpenAI model for gpt-5 generation",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="",
        choices=["", "none", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort for GPT reasoning models in gpt-5 mode.",
    )
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default=CONDITIONING_MODE_HISTORY,
        choices=CONDITIONING_MODE_CHOICES,
        help="Whether prompts should include history, persona, or both.",
    )
    parser.add_argument(
        "--test_parquet",
        type=str,
        required=True,
        help="GRPO-format convokit heldout test parquet",
    )
    return parser.parse_args()


def load_qwen_pipeline(model_id: str, generation_defaults: dict[str, float | int | None]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    print(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    gen_kwargs = {
        "temperature": generation_defaults["temperature"],
        "top_p": generation_defaults["top_p"],
        "repetition_penalty": generation_defaults["repetition_penalty"],
    }
    return pipe, tokenizer, gen_kwargs


def _results_dict(user_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(user_result["user_id"]): user_result for user_result in user_results}


def _ensure_target_prompt_messages(target_entry: dict[str, Any]) -> list[dict[str, str]]:
    prompt_messages = normalize_prompt_messages(target_entry.get("prompt_messages"))
    if prompt_messages is None:
        prompt_messages = normalize_prompt_messages(target_entry.get("raw_prompt"))
    if prompt_messages is None:
        persona = str(target_entry.get("persona", "") or "")
        prompt_messages = build_messages_for_prompt_mode(
            user_history=str(target_entry.get("user_history", "") or ""),
            thread_context=str(target_entry.get("context", "") or ""),
            prompt_mode=str(target_entry.get("prompt_mode") or GPT_API_PROMPT_MODE),
            persona=persona,
            conditioning_mode=str(target_entry.get("conditioning_mode") or CONDITIONING_MODE_HISTORY),
        )
        target_entry["raw_prompt"] = json.dumps(prompt_messages, ensure_ascii=False)

    target_entry["prompt_messages"] = prompt_messages
    if not target_entry.get("prompt_mode"):
        target_entry["prompt_mode"] = GPT_API_PROMPT_MODE
    return prompt_messages


def _build_jobs_from_user_results(user_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for user_result in user_results:
        for target_entry in user_result.get("test_targets", []):
            jobs.append(
                {
                    "user_id": str(user_result["user_id"]),
                    "target_entry": target_entry,
                    "prompt_messages": _ensure_target_prompt_messages(target_entry),
                }
            )
    return jobs


def generate_qwen_results(user_results: list[dict[str, Any]], args: argparse.Namespace):
    generation_defaults = get_domain_generation_defaults(args.test_parquet)
    pipe, tokenizer, gen_kwargs = load_qwen_pipeline(MODEL_ALIASES[args.model], generation_defaults)
    jobs = _build_jobs_from_user_results(user_results)
    prompts = [render_prompt_text_for_generation(job["target_entry"], tokenizer) for job in jobs]

    output_iter = pipe(
        prompts,
        batch_size=DEFAULT_QWEN_BATCH_SIZE,
        max_new_tokens=args.max_tokens,
        num_return_sequences=args.gen_num,
        do_sample=True,
        return_full_text=False,
        **gen_kwargs,
    )
    total_generations = 0
    empty_responses = 0
    for job, raw_output in zip(jobs, output_iter, strict=True):
        generations = raw_output if isinstance(raw_output, list) else [raw_output]
        parsed = []
        for item in generations:
            generation = parse_generation(item["generated_text"])
            if not generation["response"]:
                empty_responses += 1
            parsed.append(generation)
            total_generations += 1
        job["target_entry"]["generations"] = parsed

    results = _results_dict(user_results)
    return results, len(jobs), total_generations, empty_responses


def generate_api(
    user_results: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
):
    """Generate with an OpenAI-compatible endpoint."""
    api_key = resolve_judge_api_key()
    if not api_key:
        raise ValueError("API key not set for the requested generation endpoint.")
    api_base = get_openai_api_base()
    if model is None:
        model = args.openai_model
    if reasoning_effort is None:
        reasoning_effort = args.reasoning_effort

    jobs = _build_jobs_from_user_results(user_results)
    gen_num = int(args.gen_num)
    if gen_num <= 0:
        raise ValueError(f"gen_num must be > 0, got {gen_num}")
    request_jobs = [(job, generation_idx) for job in jobs for generation_idx in range(gen_num)]
    print(f"Total API calls: {len(request_jobs)}")

    def _api_call(job: dict[str, Any], generation_idx: int) -> tuple[dict[str, Any], int, dict[str, str]]:
        request_kwargs = {
            "model": model,
            "messages": job["prompt_messages"],
            "max_completion_tokens": args.max_tokens,
        }
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
        if "openrouter.ai" in api_base.lower():
            request_kwargs.update(openrouter_request_extras(reasoning=False))
        text = post_chat_sync(request_kwargs, api_base=api_base, api_key=api_key)
        return job, generation_idx, parse_generation(text)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_api_call, job, generation_idx): (job, generation_idx)
            for job, generation_idx in request_jobs
        }
        for job in jobs:
            job["target_entry"]["generations"] = [None] * gen_num
        for future in as_completed(futures):
            job, generation_idx, generation = future.result()
            job["target_entry"]["generations"][generation_idx] = generation
        for job in jobs:
            job["target_entry"]["generations"] = [
                generation for generation in job["target_entry"]["generations"] if generation is not None
            ]

    results = _results_dict(user_results)
    total_targets = sum(len(user_result.get("test_targets", [])) for user_result in user_results)
    total_generations = total_targets * gen_num
    empty_responses = sum(
        1
        for user_result in user_results
        for target_entry in user_result.get("test_targets", [])
        for generation in target_entry.get("generations", [])
        if not generation.get("response")
    )
    return results, total_targets, total_generations, empty_responses


def main():
    args = parse_args()

    if args.max_tokens is None:
        args.max_tokens = 2048 if args.model == "qwen3-8b" else 1024

    print(f"Loading held-out test parquet: {args.test_parquet}")
    user_results = load_user_results_from_test_parquet(
        args.test_parquet,
        user_offset=args.user_offset,
        num_users=args.num_users,
        conditioning_mode=args.conditioning_mode,
    )
    print(f"Loaded {len(user_results)} heldout user profiles (offset={args.user_offset})")

    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join("results", "sft_gen")
        os.makedirs(output_dir, exist_ok=True)
        conditioning_suffix = (
            ""
            if args.conditioning_mode == CONDITIONING_MODE_HISTORY
            else f"_{args.conditioning_mode}"
        )
        filename = f"{args.model}_notrain{conditioning_suffix}_gen{args.gen_num}.pkl"
        output_path = os.path.join(output_dir, filename)

    if args.model == "gpt-5":
        results, total_targets, total_generations, empty_responses = generate_api(user_results, args)
    elif args.model == "qwen3.5-397b":
        results, total_targets, total_generations, empty_responses = generate_api(
            user_results,
            args,
            model=MODEL_ALIASES["qwen3.5-397b"],
            reasoning_effort="",
        )
    else:
        results, total_targets, total_generations, empty_responses = generate_qwen_results(user_results, args)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)

    print(f"Saved to: {output_path}")
    print(f"Targets: {total_targets}")
    print(f"Generations: {total_generations}")
    print(f"Empty responses: {empty_responses}/{max(total_generations, 1)}")


if __name__ == "__main__":
    main()
