"""Generate heldout outputs from GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.prompt_utils import (
    CONDITIONING_MODE_CHOICES,
    CONDITIONING_MODE_HISTORY,
    build_grpo_prompt_payload,
    conditioning_mode_uses_persona,
    get_chat_template_kwargs_for_prompt_mode,
    normalize_prompt_messages,
    parse_reasoning_and_response,
)
from shared.load_personas import get_persona_for_user, load_persona_map
from shared.model_ids import (
    DEFAULT_MODEL_ID,
    load_tokenizer,
    normalize_model_id,
)
DEFAULT_GEN_NUM = 1
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = -1
DEFAULT_MIN_P: float | None = None
DEFAULT_PRESENCE_PENALTY: float | None = 0.5
DEFAULT_REPETITION_PENALTY: float | None = 1.0
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.6
DEFAULT_VLLM_MAX_MODEL_LEN: int | None = None
DEFAULT_VLLM_MAX_NUM_SEQS = 32
DEFAULT_VLLM_ENFORCE_EAGER = False
DEFAULT_VLLM_DISABLE_CUSTOM_ALL_REDUCE = False
REWARD_FAMILY_CHOICES = ("turing", "sim", "logprob")
PROMPT_MODE = "reasoning"
DOMAIN_TEMPERATURES = {
    "convokit": 0.4,
    "prism": 0.6,
}


def infer_heldout_domain(test_parquet: str) -> str:
    """Infer the heldout domain from its path."""
    path_parts = {part.lower() for part in os.path.normpath(test_parquet).split(os.sep)}
    for domain in DOMAIN_TEMPERATURES:
        if domain in path_parts:
            return domain
    raise ValueError(
        "Could not infer heldout domain from --test_parquet. "
        "Expected the path to contain 'convokit' or 'prism'."
    )


def get_domain_generation_defaults(test_parquet: str) -> dict[str, float | int | None]:
    """Return heldout decoding defaults."""
    domain = infer_heldout_domain(test_parquet)
    return {
        "temperature": DOMAIN_TEMPERATURES[domain],
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "min_p": DEFAULT_MIN_P,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
    }


def render_prompt_text_for_generation(target_result: dict[str, Any], tokenizer: Any) -> str:
    """Render the heldout prompt."""
    prompt_mode = str(target_result.get("prompt_mode") or "reasoning")
    prompt_messages = normalize_prompt_messages(target_result.get("prompt_messages"))
    if prompt_messages:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            **get_chat_template_kwargs_for_prompt_mode(prompt_mode),
        )
        target_result["prompt_messages"] = prompt_messages
        target_result["prompt_text"] = prompt_text
        target_result["raw_prompt"] = json.dumps(prompt_messages, ensure_ascii=False)
        return prompt_text

    user_history = str(target_result.get("user_history", "") or "")
    context = str(target_result.get("context", "") or target_result.get("thread_context", "") or "")
    if user_history or context:
        prompt_payload = build_grpo_prompt_payload(
            tokenizer,
            user_history=user_history,
            thread_context=context,
            prompt_mode=prompt_mode,
            persona=str(target_result.get("persona", "") or ""),
            conditioning_mode=str(
                target_result.get("conditioning_mode", CONDITIONING_MODE_HISTORY) or CONDITIONING_MODE_HISTORY
            ),
        )
        target_result["prompt_messages"] = prompt_payload["prompt"]
        target_result["prompt_text"] = prompt_payload["prompt_text"]
        target_result["raw_prompt"] = prompt_payload["raw_prompt"]
        target_result["prompt_mode"] = prompt_payload["prompt_mode"]
        target_result["conditioning_mode"] = prompt_payload["conditioning_mode"]
        return prompt_payload["prompt_text"]

    prompt_text = str(target_result.get("prompt_text", "") or "")
    if not prompt_text:
        raise ValueError(
            f"Generation target post_id={target_result.get('post_id')} target_idx={target_result.get('target_idx')} "
            "has neither prompt messages nor enough structured fields to rebuild the prompt."
        )
    return prompt_text


def parse_generation(text: str) -> dict[str, str]:
    """Parse one model generation."""
    reasoning, response = parse_reasoning_and_response(text)
    return {"reasoning": reasoning, "response": response}


def resolve_adapter_path(path: str | None) -> str | None:
    """Resolve a PEFT adapter directory."""
    if not path:
        return None

    candidates = [
        path,
        os.path.join(path, "lora_adapter"),
        os.path.join(path, "actor", "lora_adapter"),
        os.path.join(path, "final"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            return candidate
    return None


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Resolve the latest checkpoint adapter."""
    if not os.path.isdir(checkpoint_dir):
        return None
    steps = []
    for name in os.listdir(checkpoint_dir):
        if not name.startswith("global_step_"):
            continue
        try:
            steps.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    if not steps:
        return None
    checkpoint_root = os.path.join(checkpoint_dir, f"global_step_{max(steps)}")
    return resolve_adapter_path(checkpoint_root)


def _build_generation_task_list(
    user_results: list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Flatten test targets for batched generation."""
    tasks: list[dict[str, Any]] = []
    for user_result in user_results:
        for target_result in user_result["test_targets"]:
            if tokenizer is None:
                prompt_text = str(target_result.get("prompt_text", "") or "")
            else:
                prompt_text = render_prompt_text_for_generation(target_result, tokenizer)
            tasks.append({"prompt_text": prompt_text, "target_result": target_result})
    return tasks


def load_vllm_engine(
    *,
    model_id: str,
    adapter_path: str | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int,
    enforce_eager: bool,
    disable_custom_all_reduce: bool,
) -> Any:
    """Load the vLLM engine."""
    from vllm import LLM

    engine_kwargs: dict[str, Any] = {
        "model": normalize_model_id(model_id),
        "tokenizer": normalize_model_id(model_id),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "enable_lora": adapter_path is not None,
        "enforce_eager": enforce_eager,
        "disable_custom_all_reduce": disable_custom_all_reduce,
    }
    if max_model_len is not None and max_model_len > 0:
        engine_kwargs["max_model_len"] = max_model_len
    lora_rank = infer_lora_rank(adapter_path)
    if lora_rank is not None:
        engine_kwargs["max_lora_rank"] = lora_rank
    print(
        "Loading vLLM engine: "
        f"model={engine_kwargs['model']}, "
        f"adapter_path={adapter_path or '<none>'}, "
        f"tp={tensor_parallel_size}, "
        f"gpu_memory_utilization={gpu_memory_utilization}, "
        f"max_model_len={engine_kwargs.get('max_model_len')}, "
        f"max_num_seqs={max_num_seqs}, "
        f"max_lora_rank={engine_kwargs.get('max_lora_rank')}, "
        f"enforce_eager={enforce_eager}, "
        f"disable_custom_all_reduce={disable_custom_all_reduce}",
        flush=True,
    )
    return LLM(**engine_kwargs)


def infer_lora_rank(adapter_path: str | None) -> int | None:
    """Infer LoRA rank from adapter_config.json."""
    if not adapter_path:
        return None
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    rank = int(config["r"])
    return rank if rank > 0 else None


def build_vllm_sampling_params(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    max_tokens: int,
    gen_num: int,
) -> Any:
    """Build vLLM sampling params."""
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "n": gen_num,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    if min_p is not None:
        kwargs["min_p"] = min_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return SamplingParams(**kwargs)


def build_vllm_lora_request(adapter_path: str | None) -> Any | None:
    """Build the vLLM LoRA request."""
    if not adapter_path:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest("grpo_adapter", 1, adapter_path)


def generate_for_user_results_vllm(
    *,
    user_results: list[dict[str, Any]],
    model_id: str,
    adapter_path: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    batch_size: int,
    max_tokens: int,
    gen_num: int,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int | None,
    vllm_max_num_seqs: int,
    vllm_enforce_eager: bool,
    vllm_disable_custom_all_reduce: bool,
    vllm_truncate_prompt_tokens: int | None,
) -> dict[str, dict[str, Any]]:
    """Generate with vLLM."""
    if not user_results:
        return {}
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if vllm_tensor_parallel_size <= 0:
        raise ValueError(f"vllm_tensor_parallel_size must be > 0, got {vllm_tensor_parallel_size}")
    if vllm_max_num_seqs <= 0:
        raise ValueError(f"vllm_max_num_seqs must be > 0, got {vllm_max_num_seqs}")
    tokenizer = load_tokenizer(model_id)
    generation_tasks = _build_generation_task_list(user_results, tokenizer=tokenizer)
    target_count = len(generation_tasks)
    print(
        f"[vllm] Starting generation for {len(user_results)} users / {target_count} targets",
        flush=True,
    )
    llm = load_vllm_engine(
        model_id=model_id,
        adapter_path=adapter_path,
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        max_num_seqs=vllm_max_num_seqs,
        enforce_eager=vllm_enforce_eager,
        disable_custom_all_reduce=vllm_disable_custom_all_reduce,
    )
    sampling_params = build_vllm_sampling_params(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        gen_num=gen_num,
    )
    lora_request = build_vllm_lora_request(adapter_path)

    for start in range(0, target_count, batch_size):
        batch_tasks = generation_tasks[start:start + batch_size]
        batch_prompts: list[Any]
        if vllm_truncate_prompt_tokens is not None and vllm_truncate_prompt_tokens > 0:
            previous_truncation_side = getattr(tokenizer, "truncation_side", "right")
            tokenizer.truncation_side = "left"
            try:
                encoded = tokenizer(
                    [task["prompt_text"] for task in batch_tasks],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=vllm_truncate_prompt_tokens,
                )
            finally:
                tokenizer.truncation_side = previous_truncation_side
            batch_prompts = [{"prompt_token_ids": token_ids} for token_ids in encoded["input_ids"]]
        else:
            batch_prompts = [task["prompt_text"] for task in batch_tasks]
        outputs = llm.generate(
            batch_prompts,
            sampling_params,
            lora_request=lora_request,
        )
        if len(outputs) != len(batch_tasks):
            raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(batch_tasks)} prompts")
        for task, request_output in zip(batch_tasks, outputs, strict=True):
            parsed_generations = []
            for output in request_output.outputs:
                raw_completion = output.text
                parsed = parse_generation(raw_completion)
                parsed["raw_completion"] = raw_completion
                parsed["finish_reason"] = getattr(output, "finish_reason", None)
                parsed["stop_reason"] = getattr(output, "stop_reason", None)
                token_ids = getattr(output, "token_ids", None)
                parsed["output_token_count"] = len(token_ids) if token_ids is not None else None
                parsed_generations.append(parsed)
            task["target_result"]["generations"] = parsed_generations

    print(f"[vllm] Finished generation for {target_count} targets", flush=True)
    return {user_result["user_id"]: user_result for user_result in user_results}


def load_user_results_from_test_parquet(
    test_parquet: str,
    *,
    user_offset: int = 0,
    num_users: int | None = None,
    target_offset: int = 0,
    max_targets: int | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
    reward_family: str | None = None,
) -> list[dict[str, Any]]:
    """Load heldout targets."""
    import pandas as pd

    if user_offset < 0:
        raise ValueError(f"user_offset must be >= 0, got {user_offset}")
    if target_offset < 0:
        raise ValueError(f"target_offset must be >= 0, got {target_offset}")
    rows = pd.read_parquet(test_parquet).to_dict("records")
    if target_offset:
        if target_offset >= len(rows):
            raise ValueError(f"target_offset {target_offset} is out of range for {len(rows)} target rows")
        rows = rows[target_offset:]
    if max_targets is not None:
        if max_targets <= 0:
            raise ValueError(f"max_targets must be > 0, got {max_targets}")
        rows = rows[:max_targets]
    user_results_by_id: dict[str, dict[str, Any]] = {}
    user_order: list[str] = []

    for row in rows:
        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        row_prompt_mode = extra_info.get("prompt_mode")
        row_conditioning_mode = extra_info.get("conditioning_mode", CONDITIONING_MODE_HISTORY)
        if row_prompt_mode and row_prompt_mode != PROMPT_MODE:
            raise ValueError(
                f"test parquet prompt_mode={row_prompt_mode!r} does not match requested prompt_mode={PROMPT_MODE!r}"
            )
        if row_conditioning_mode != conditioning_mode:
            raise ValueError(
                "test parquet conditioning_mode="
                f"{row_conditioning_mode!r} does not match requested conditioning_mode={conditioning_mode!r}"
            )

        user_id = str(extra_info.get("user_id") or "")
        if not user_id:
            raise ValueError("test parquet row is missing extra_info.user_id")
        if user_id not in user_results_by_id:
            user_order.append(user_id)
            user_results_by_id[user_id] = {
                "user_id": user_id,
                "raw_user_id": extra_info.get("raw_user_id", user_id),
                "source_name": extra_info.get("source_name", ""),
                "split": extra_info.get("split", "test"),
                "history_thread_ids": extra_info.get("history_thread_ids", []),
                "test_targets": [],
            }

        prompt_messages = normalize_prompt_messages(row.get("prompt")) or normalize_prompt_messages(
            extra_info.get("raw_prompt")
        )
        prompt_text = str(extra_info.get("prompt_text", "") or "")
        if not prompt_text and not prompt_messages:
            raise ValueError(
                f"test parquet row for user_id={user_id} post_id={extra_info.get('post_id')} "
                "lacks both prompt messages and prompt_text"
            )
        target_result = {
            "post_id": extra_info["post_id"],
            "target_idx": extra_info["target_idx"],
            "ground_truth": reward_model.get("ground_truth", ""),
            "context": extra_info.get("context", extra_info.get("thread_context", "")),
            "user_history": extra_info.get("user_history", ""),
            "persona": extra_info.get("persona", ""),
            "raw_user_id": extra_info.get("raw_user_id", user_id),
            "source_name": extra_info.get("source_name", ""),
            "subreddit": extra_info.get("target_subreddit", ""),
            "reward_family": reward_family,
            "prompt_mode": row_prompt_mode or PROMPT_MODE,
            "conditioning_mode": row_conditioning_mode,
            "prompt_text": prompt_text,
            "prompt_messages": prompt_messages,
            "raw_prompt": extra_info.get(
                "raw_prompt",
                json.dumps(prompt_messages, ensure_ascii=False) if prompt_messages else prompt_text,
            ),
            "generations": [],
        }
        user_results_by_id[user_id]["test_targets"].append(target_result)

    total_users = len(user_order)
    if total_users == 0:
        raise ValueError(f"No users found in test parquet: {test_parquet}")
    if user_offset >= total_users:
        raise ValueError(f"user_offset {user_offset} is out of range for {total_users} test users")

    selected_user_ids = user_order[user_offset:]
    if num_users is not None:
        selected_user_ids = selected_user_ids[:num_users]
    return [user_results_by_id[user_id] for user_id in selected_user_ids]


def apply_persona_map_to_user_results(
    user_results: list[dict[str, Any]],
    persona_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Apply external personas."""
    if not persona_map:
        raise ValueError("Persona conditioning requires a non-empty external persona map.")

    for user_result in user_results:
        user_id = str(user_result.get("user_id") or "")
        raw_user_id = str(user_result.get("raw_user_id") or "")
        persona_text = get_persona_for_user(persona_map, user_id, raw_user_id)
        if not persona_text:
            raise ValueError(
                "Missing external persona for heldout generation user "
                f"user_id={user_id!r} raw_user_id={raw_user_id!r}"
            )
        for target_result in user_result.get("test_targets") or []:
            target_result["persona"] = persona_text
    return user_results


def apply_generation_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply decoding defaults."""
    reward_family = getattr(args, "metric", None)
    if reward_family not in REWARD_FAMILY_CHOICES:
        if reward_family is not None:
            raise ValueError(
                f"Unknown reward_family={reward_family!r}. Expected one of {', '.join(REWARD_FAMILY_CHOICES)}."
            )
    args.reward_family = reward_family

    if args.gen_num is None:
        args.gen_num = DEFAULT_GEN_NUM
    if args.max_tokens is None:
        args.max_tokens = DEFAULT_MAX_TOKENS
    normalize_model_id(getattr(args, "model_id", DEFAULT_MODEL_ID))
    model_defaults = get_domain_generation_defaults(args.test_parquet)
    args.temperature = model_defaults["temperature"]
    args.top_p = model_defaults["top_p"]
    args.top_k = model_defaults["top_k"]
    args.min_p = model_defaults["min_p"]
    args.presence_penalty = model_defaults["presence_penalty"]
    args.repetition_penalty = DEFAULT_REPETITION_PENALTY
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate on held-out test data using retained GRPO checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to a veRL checkpoint directory containing global_step_*",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["turing", "sim", "logprob"],
        help="GRPO reward metric for output metadata",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        help=f"HF model id to load (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument("--user_offset", type=int, default=0, help="Test users to skip before selecting users")
    parser.add_argument("--num_users", type=int, default=None, help="Number of test users (default: all)")
    parser.add_argument(
        "--target_offset",
        type=int,
        default=0,
        help="Parquet target rows to skip before applying --max_targets and grouping by user",
    )
    parser.add_argument(
        "--max_targets",
        type=int,
        default=None,
        help="Maximum number of parquet target rows to load before grouping by user",
    )
    parser.add_argument(
        "--test_parquet",
        type=str,
        required=True,
        help="GRPO-format held-out test parquet",
    )
    parser.add_argument("--gen_num", type=int, default=None, help="Generations per target")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts per generation batch")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens per generation")
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
        help="Tensor-parallel size",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=DEFAULT_VLLM_MAX_MODEL_LEN,
        help="Optional vLLM max_model_len",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=DEFAULT_VLLM_MAX_NUM_SEQS,
        help="vLLM max_num_seqs",
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        default=DEFAULT_VLLM_ENFORCE_EAGER,
        help="Pass enforce_eager=True to vLLM for heldout generation",
    )
    parser.add_argument(
        "--vllm_disable_custom_all_reduce",
        action="store_true",
        default=DEFAULT_VLLM_DISABLE_CUSTOM_ALL_REDUCE,
        help="Pass disable_custom_all_reduce=True to vLLM for heldout generation",
    )
    parser.add_argument(
        "--vllm_truncate_prompt_tokens",
        type=int,
        default=None,
        help=(
            "If set, left-truncate prompts to this many tokens "
            "before passing token ids to vLLM. Useful when retained histories exceed max_model_len."
        ),
    )
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default=CONDITIONING_MODE_HISTORY,
        choices=CONDITIONING_MODE_CHOICES,
        help="Whether prompts should include history, persona, or both.",
    )
    parser.add_argument(
        "--persona_path",
        type=str,
        default=None,
        help="Optional JSON/JSONL/pickle file mapping user_id to persona text.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_id = normalize_model_id(args.model_id)

    args.adapter_path = find_latest_checkpoint(args.checkpoint_dir)
    if args.adapter_path is None:
        args.adapter_path = resolve_adapter_path(args.checkpoint_dir)
    if args.adapter_path is None:
        raise ValueError(f"No LoRA adapter found under {args.checkpoint_dir}")
    print(f"Using checkpoint/adapter: {args.adapter_path}")

    args = apply_generation_defaults(args)
    print(
        "Heldout decoding: "
        f"reward_family={args.reward_family}, "
        f"temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"top_k={args.top_k}, "
        f"min_p={args.min_p}, "
        f"presence_penalty={args.presence_penalty}, "
        f"repetition_penalty={args.repetition_penalty}, "
        f"max_tokens={args.max_tokens}, "
        f"vllm_tp={args.vllm_tensor_parallel_size}, "
        f"vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}, "
        f"vllm_max_model_len={args.vllm_max_model_len}, "
        f"vllm_max_num_seqs={args.vllm_max_num_seqs}, "
        f"vllm_enforce_eager={args.vllm_enforce_eager}, "
        f"vllm_disable_custom_all_reduce={args.vllm_disable_custom_all_reduce}, "
        f"vllm_truncate_prompt_tokens={args.vllm_truncate_prompt_tokens}",
        flush=True,
    )
    print(
        "Heldout prompt rendering: re-render from GRPO prompt messages/fields with "
        f"apply_chat_template_kwargs={get_chat_template_kwargs_for_prompt_mode(PROMPT_MODE)}",
        flush=True,
    )
    if conditioning_mode_uses_persona(args.conditioning_mode) and not args.persona_path:
        raise ValueError(f"--persona_path is required when --conditioning_mode={args.conditioning_mode}")
    persona_map = load_persona_map(args.persona_path)

    if args.output:
        output_path = args.output
    else:
        conditioning_suffix = (
            ""
            if args.conditioning_mode == CONDITIONING_MODE_HISTORY
            else f"_{args.conditioning_mode}"
        )
        if args.metric:
            tag = f"grpo_{args.metric}"
        else:
            adapter_clean = args.adapter_path.rstrip("/")
            parts = adapter_clean.split("/")
            tag = parts[-2] if parts[-1] in ("actor", "final") else parts[-1]
        output_dir = os.path.join("results", "grpo_gen")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{tag}{conditioning_suffix}_gen{args.gen_num}.pkl")

    print(f"Loading held-out test parquet: {args.test_parquet}")
    user_results = load_user_results_from_test_parquet(
        args.test_parquet,
        user_offset=args.user_offset,
        num_users=args.num_users,
        target_offset=args.target_offset,
        max_targets=args.max_targets,
        conditioning_mode=args.conditioning_mode,
        reward_family=args.reward_family,
    )
    print(
        f"Loaded {len(user_results)} test user profiles from parquet "
        f"(user_offset={args.user_offset}, target_offset={args.target_offset})"
    )
    if conditioning_mode_uses_persona(args.conditioning_mode):
        user_results = apply_persona_map_to_user_results(
            user_results=user_results,
            persona_map=persona_map,
        )

    results = generate_for_user_results_vllm(
        user_results=user_results,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        gen_num=args.gen_num,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_disable_custom_all_reduce=args.vllm_disable_custom_all_reduce,
        vllm_truncate_prompt_tokens=args.vllm_truncate_prompt_tokens,
    )

    generation_tasks = _build_generation_task_list(user_results)

    total_gens = sum(
        len(target["generations"])
        for user_result in results.values()
        for target in user_result["test_targets"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)

    print(f"Saved {len(results)} users / {len(generation_tasks)} targets / {total_gens} generations")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
