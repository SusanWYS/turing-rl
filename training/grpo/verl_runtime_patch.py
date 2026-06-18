from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from collections.abc import Mapping
from functools import lru_cache
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)
_SINGLE_TURN_SAMPLING_PARAMS_LOGGED = False
_SINGLE_TURN_PROMPT_PREFILL_STRIP_LOGGED = False
_SINGLE_TURN_THINK_CLOSE_STRIP_LOGGED = False

_PROPAGATED_RUNTIME_ENV_VARS = (
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_RAS_ENABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_PXN_DISABLE",
    "NCCL_CUMEM_ENABLE",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_PROVIDER_ORDER",
    "OPENROUTER_ALLOW_FALLBACKS",
    "OPENROUTER_REASONING_ENABLED",
    "JUDGE_MODEL",
    "SIM_JUDGE_MODEL",
    "PERSONA_EVAL_JUDGE_MODEL",
    "PERSONA_SIM_EVAL_JUDGE_MODEL",
    "PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY",
    "PERSONA_OPENAI_MAX_RETRIES",
    "PERSONA_OPENAI_RETRY_SLEEP_SECONDS",
    "SIM_JUDGE_MAX_CONCURRENCY",
    "WANDB_API_KEY",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "REWARD_METRIC",
    "GENERATION_LOG",
    "GRPO_DATASET",
    "CONDITIONING_MODE",
    "PERSONA_PATH",
    "PERSONA_ENABLE_RUNTIME_PROMPT_OVERRIDE",
    "PERSONA_RUNTIME_CONDITIONING_MODE",
    "PERSONA_VLLM_PRESENCE_PENALTY",
    "TURING_LENGTH_LOWER_RATIO",
    "TURING_LENGTH_UPPER_RATIO",
    "TURING_LENGTH_SHORT_PENALTY_LAMBDA",
    "TURING_LENGTH_LONG_PENALTY_LAMBDA",
    "TURING_LENGTH_PENALTY_LAMBDA",
    "TURING_LENGTH_PENALTY_CAP",
    "PERSONA_ENABLE_CURRENT_POLICY_LOGPROB",
    "PERSONA_LOGPROB_SCORE_CONCURRENCY_PER_SERVER",
    "PERSONA_LOGPROB_SCORE_BATCH_SIZE",
    "PERSONA_LOGPROB_SCORE_BATCH_TIMEOUT_MS",
    "LOGPROB_CLIP_MIN",
    "LOGPROB_CLIP_MAX",
    "PERSONA_ELBO_SFT_MAX_LENGTH",
)
_PERSONA_WORKER_PROCESS_SETUP_HOOK = "training.grpo.ray_worker_setup.persona_worker_process_setup"
_UPSTREAM_WORKER_PROCESS_SETUP_HOOK_ENV = "PERSONA_UPSTREAM_WORKER_PROCESS_SETUP_HOOK"
_ELBO_SFT_TENSOR_KEYS = (
    "elbo_input_ids",
    "elbo_attention_mask",
    "elbo_position_ids",
    "elbo_responses",
    "elbo_response_mask",
)


def _int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _float_env(name: str, default: float | None = None) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _patch_actor_elbo_sft_source() -> None:
    """Patch veRL actor config for optional ELBO/SFT loss."""
    spec = importlib.util.find_spec("verl.workers.actor.dp_actor")
    if spec is None or spec.origin is None:
        return
    path = Path(spec.origin)
    try:
        text = path.read_text()
    except OSError:
        return
    if "_persona_elbo_sft_weight" in text:
        return

    text = text.replace(
        '        metrics = {\n            "actor/pg_loss": 0.0,\n            "actor/kl_loss": 0.0,\n        }\n',
        '        metrics = {\n            "actor/pg_loss": 0.0,\n            "actor/kl_loss": 0.0,\n        }\n'
        '        _persona_elbo_sft_weight = float((self.config.get("elbo_sft_weight", 0.0) if hasattr(self.config, "get") else getattr(self.config, "elbo_sft_weight", 0.0)) or 0.0)\n'
        '        if _persona_elbo_sft_weight > 0:\n'
        '            metrics["actor/elbo_sft_loss"] = 0.0\n',
    )
    text = text.replace(
        '        if "rollout_log_probs" in data.batch.keys():\n            select_keys.append("rollout_log_probs")\n',
        '        if "rollout_log_probs" in data.batch.keys():\n            select_keys.append("rollout_log_probs")\n'
        '        _persona_elbo_sft_weight = float((self.config.get("elbo_sft_weight", 0.0) if hasattr(self.config, "get") else getattr(self.config, "elbo_sft_weight", 0.0)) or 0.0)\n'
        '        if _persona_elbo_sft_weight > 0:\n'
        '            for _persona_elbo_key in (\n'
        '                "elbo_input_ids",\n'
        '                "elbo_attention_mask",\n'
        '                "elbo_position_ids",\n'
        '                "elbo_responses",\n'
        '                "elbo_response_mask",\n'
        '            ):\n'
        '                if _persona_elbo_key in data.batch.keys():\n'
        '                    select_keys.append(_persona_elbo_key)\n',
    )
    text = text.replace(
        '                    policy_loss = pg_loss\n'
        '                    if calculate_entropy and entropy is not None:\n',
        '                    policy_loss = pg_loss\n'
        '                    if _persona_elbo_sft_weight > 0 and "elbo_input_ids" in model_inputs:\n'
        '                        _persona_elbo_inputs = {\n'
        '                            "input_ids": model_inputs["elbo_input_ids"],\n'
        '                            "attention_mask": model_inputs["elbo_attention_mask"],\n'
        '                            "position_ids": model_inputs["elbo_position_ids"],\n'
        '                            "responses": model_inputs["elbo_responses"],\n'
        '                            "pad_token_id": pad_token_id,\n'
        '                        }\n'
        '                        _persona_elbo_outputs = self._forward_micro_batch(\n'
        '                            _persona_elbo_inputs,\n'
        '                            temperature=temperature,\n'
        '                            calculate_entropy=False,\n'
        '                        )\n'
        '                        _persona_elbo_mask = model_inputs["elbo_response_mask"].to(_persona_elbo_outputs["log_probs"].dtype)\n'
        '                        _persona_elbo_denom = _persona_elbo_mask.sum().clamp_min(1.0)\n'
        '                        _persona_elbo_sft_loss = -(\n'
        '                            _persona_elbo_outputs["log_probs"] * _persona_elbo_mask\n'
        '                        ).sum() / _persona_elbo_denom\n'
        '                        policy_loss = policy_loss + _persona_elbo_sft_loss * _persona_elbo_sft_weight\n'
        '                        metrics["actor/elbo_sft_loss"] += _persona_elbo_sft_loss.detach().item() * loss_scale_factor\n'
        '                        micro_batch_metrics["actor/elbo_sft_weight"] = _persona_elbo_sft_weight\n'
        '                    if calculate_entropy and entropy is not None:\n',
    )
    try:
        path.write_text(text)
    except OSError:
        logger.warning("Failed to patch veRL actor ELBO/SFT source at %s", path, exc_info=True)


def _patch_actor_config_elbo_sft_source() -> None:
    spec = importlib.util.find_spec("verl.workers.config.actor")
    if spec is None or spec.origin is None:
        return
    path = Path(spec.origin)
    try:
        text = path.read_text()
    except OSError:
        return
    if "elbo_sft_weight" in text:
        return
    anchor = "    loss_scale_factor: Optional[int] = None\n"
    if anchor not in text:
        return
    text = text.replace(
        anchor,
        anchor
        + "    # Optional Persona patch: auxiliary teacher-forced GT SFT/ELBO loss.\n"
        + "    elbo_sft_weight: float = 0.0\n",
    )
    try:
        path.write_text(text)
    except OSError:
        logger.warning("Failed to patch veRL actor config ELBO/SFT source at %s", path, exc_info=True)


def _patch_peft_meta_adapter_load_source() -> None:
    """Patch PEFT adapter loading on meta tensors."""
    spec = importlib.util.find_spec("verl.workers.fsdp_workers")
    if spec is None or spec.origin is None:
        return
    path = Path(spec.origin)
    try:
        text = path.read_text()
    except OSError:
        return
    if "PERSONA_PEFT_LOAD_LOW_CPU_MEM_USAGE" in text:
        return

    old_actor = "                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)\n"
    new_actor = (
        "                actor_module = PeftModel.from_pretrained(\n"
        "                    actor_module,\n"
        "                    local_adapter_path,\n"
        "                    is_trainable=True,\n"
        "                    low_cpu_mem_usage=os.environ.get(\n"
        '                        "PERSONA_PEFT_LOAD_LOW_CPU_MEM_USAGE", "1"\n'
        '                    ).strip().lower() in {"1", "true", "yes", "on"},\n'
        "                )\n"
    )
    old_critic = "                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)\n"
    new_critic = (
        "                critic_module = PeftModel.from_pretrained(\n"
        "                    critic_module,\n"
        "                    local_adapter_path,\n"
        "                    is_trainable=True,\n"
        "                    low_cpu_mem_usage=os.environ.get(\n"
        '                        "PERSONA_PEFT_LOAD_LOW_CPU_MEM_USAGE", "1"\n'
        '                    ).strip().lower() in {"1", "true", "yes", "on"},\n'
        "                )\n"
    )
    if old_actor not in text and old_critic not in text:
        return

    text = text.replace(old_actor, new_actor).replace(old_critic, new_critic)
    try:
        path.write_text(text)
    except OSError:
        logger.warning("Failed to patch veRL PEFT adapter load source at %s", path, exc_info=True)


def _apply_presence_penalty_to_sampling_params(sampling_params: Any) -> Any:
    presence_penalty = _float_env("PERSONA_VLLM_PRESENCE_PENALTY", 0.5)
    if presence_penalty is None:
        return sampling_params
    if isinstance(sampling_params, dict):
        patched = dict(sampling_params)
        patched["presence_penalty"] = presence_penalty
        return patched
    setattr(sampling_params, "presence_penalty", presence_penalty)
    return sampling_params


def _runtime_env_propagation_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_RUNTIME_ENV_PROPAGATION", False)


def _worker_process_setup_hook_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_WORKER_PROCESS_SETUP_HOOK", True)


def _ray_loopback_advertise_enabled() -> bool:
    return _bool_env("PERSONA_FORCE_RAY_LOOPBACK_ADVERTISE", True)


def _grouped_sim_route_patch_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_GROUPED_SIM_ROUTE_PATCH", False)


def _log_single_turn_sampling_params_once(sampling_params: Any) -> None:
    """Print the first rollout sampling params payload seen in this container."""
    global _SINGLE_TURN_SAMPLING_PARAMS_LOGGED
    if _SINGLE_TURN_SAMPLING_PARAMS_LOGGED:
        return
    _SINGLE_TURN_SAMPLING_PARAMS_LOGGED = True

    run_key = (
        os.environ.get("WANDB_RUN_ID")
        or os.environ.get("WANDB_NAME")
        or os.environ.get("SLURM_JOB_ID")
        or "default"
    )
    safe_run_key = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_key)[:120]
    sentinel_path = f"/tmp/persona_sampling_params_logged_{safe_run_key}"
    try:
        fd = os.open(sentinel_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        return
    except OSError:
        # Fall back to per-process one-shot logging if the cross-process sentinel cannot be created.
        pass

    try:
        rendered = json.dumps(sampling_params, default=str, sort_keys=True)
    except Exception:
        rendered = repr(sampling_params)
    if len(rendered) > 2000:
        rendered = rendered[:2000] + "...<truncated>"
    print(f"PERSONA_DEBUG_AGENT_LOOP: single_turn_sampling_params={rendered}", flush=True)


def _full_gpu_colocated_training_enabled() -> bool:
    return _bool_env("PERSONA_RESERVE_FULL_GPU_FOR_COLOCATED_TRAINING", True)


def _local_rank_visible_device_fallback_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_LOCAL_RANK_VISIBLE_DEVICE_FALLBACK", True)


def _current_policy_logprob_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_CURRENT_POLICY_LOGPROB", True)


def _runtime_prompt_override_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_RUNTIME_PROMPT_OVERRIDE", False)


def _epoch_end_checkpointing_enabled() -> bool:
    return _bool_env("PERSONA_ENABLE_EPOCH_END_CHECKPOINTING", True)


def _resolve_auto_rollout_data_parallel_size(config: Any) -> int | None:
    actor_rollout_ref = getattr(config, "actor_rollout_ref", None)
    rollout_cfg = getattr(actor_rollout_ref, "rollout", None)
    trainer_cfg = getattr(config, "trainer", None)
    if rollout_cfg is None or trainer_cfg is None:
        return None

    configured_dp = _config_get(rollout_cfg, "data_parallel_size", None)
    try:
        configured_dp = int(configured_dp)
    except (TypeError, ValueError):
        return None

    if configured_dp > 0:
        return None

    try:
        tp = int(_config_get(rollout_cfg, "tensor_model_parallel_size", 1))
        pp = int(_config_get(rollout_cfg, "pipeline_model_parallel_size", 1))
        trainer_gpus_per_node = int(_config_get(trainer_cfg, "n_gpus_per_node", 0))
        trainer_nnodes = int(_config_get(trainer_cfg, "nnodes", 0))
    except (TypeError, ValueError):
        return None

    total_gpus = trainer_gpus_per_node * max(trainer_nnodes, 1)
    infer_width = tp * pp
    if total_gpus <= 0 or infer_width <= 0:
        return None
    if total_gpus % infer_width != 0:
        return None

    return total_gpus // infer_width


def _maybe_normalize_rollout_parallelism(config: Any) -> None:
    resolved_dp = _resolve_auto_rollout_data_parallel_size(config)
    if resolved_dp is None:
        return

    rollout_cfg = config.actor_rollout_ref.rollout
    original_dp = _config_get(rollout_cfg, "data_parallel_size", None)
    rollout_cfg.data_parallel_size = resolved_dp

    print(
        "PERSONA: resolved actor_rollout_ref.rollout.data_parallel_size "
        f"from {original_dp} to {resolved_dp} "
        f"with tp={_config_get(rollout_cfg, 'tensor_model_parallel_size', None)} "
        f"pp={_config_get(rollout_cfg, 'pipeline_model_parallel_size', None)} "
        f"trainer_gpus={_config_get(config.trainer, 'n_gpus_per_node', None)}x{_config_get(config.trainer, 'nnodes', None)}",
        flush=True,
    )


def _config_get(config_obj: Any, key: str, default: Any = None) -> Any:
    getter = getattr(config_obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)
    return getattr(config_obj, key, default)


def _disable_critic_for_retained_reward_modes(trainer: Any) -> None:
    metric = os.environ.get("REWARD_METRIC", "").strip().lower()
    if metric not in {"turing", "sim", "logprob"}:
        return

    critic_cfg = getattr(getattr(trainer, "config", None), "critic", None)
    prior_enable = _config_get(critic_cfg, "enable", None) if critic_cfg is not None else None
    prior_use_critic = bool(getattr(trainer, "use_critic", False))

    if critic_cfg is not None:
        critic_cfg.enable = False
    trainer.use_critic = False

    if prior_enable is not False or prior_use_critic:
        print(
            "PERSONA: forcing critic.disable for retained reward mode "
            f"{metric}; prior critic.enable={prior_enable!r} prior use_critic={prior_use_critic}",
            flush=True,
        )


def _resolve_epoch_aligned_save_freq(config: Any, steps_per_epoch: int) -> int | None:
    trainer_cfg = getattr(config, "trainer", None)
    if trainer_cfg is None:
        return None

    total_epochs = _config_get(trainer_cfg, "total_epochs", 1)
    try:
        total_epochs = int(total_epochs)
    except (TypeError, ValueError):
        return None

    if total_epochs <= 1:
        return None

    return max(1, int(steps_per_epoch))


def _should_save_epoch_end_checkpoint(config: Any, global_step: int, steps_per_epoch: int) -> bool:
    if not _epoch_end_checkpointing_enabled():
        return False

    forced_save_freq = _resolve_epoch_aligned_save_freq(config, steps_per_epoch)
    if forced_save_freq is None:
        return False

    try:
        normalized_global_step = int(global_step)
    except (TypeError, ValueError):
        return False

    return normalized_global_step > 0 and normalized_global_step % forced_save_freq == 0


def _normalize_loopback_ray_node_ip(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"127.0.0.1", "localhost"}:
        return "127.0.0.1"
    if stripped == "::1":
        return "::1"
    return None


def _patch_ray_loopback_advertise_modules(ray_services_mod: Any, ray_util_mod: Any) -> bool:
    """Keep Ray from rewriting an explicitly requested loopback node IP."""
    desired_ip = _normalize_loopback_ray_node_ip(os.environ.get("RAY_NODE_IP_ADDRESS"))
    if desired_ip is None:
        logger.warning(
            "PERSONA_FORCE_RAY_LOOPBACK_ADVERTISE is enabled, but RAY_NODE_IP_ADDRESS=%r is not loopback; "
            "leaving Ray localhost resolution unchanged",
            os.environ.get("RAY_NODE_IP_ADDRESS"),
        )
        return False

    if getattr(ray_services_mod, "_persona_loopback_advertise_patch_applied", False):
        return True

    original_resolve_ip_for_localhost = ray_services_mod.resolve_ip_for_localhost
    original_get_node_ip_address = ray_services_mod.get_node_ip_address

    def patched_resolve_ip_for_localhost(host: str):
        normalized = _normalize_loopback_ray_node_ip(host)
        if normalized is not None:
            return normalized
        return original_resolve_ip_for_localhost(host)

    def patched_get_node_ip_address(address=None):
        if address is None:
            return desired_ip
        return original_get_node_ip_address(address)

    ray_services_mod.resolve_ip_for_localhost = patched_resolve_ip_for_localhost
    ray_services_mod.get_node_ip_address = patched_get_node_ip_address
    ray_services_mod._persona_loopback_advertise_patch_applied = True
    ray_services_mod._persona_original_resolve_ip_for_localhost = original_resolve_ip_for_localhost
    ray_services_mod._persona_original_get_node_ip_address = original_get_node_ip_address

    if ray_util_mod is not None:
        ray_util_mod.get_node_ip_address = patched_get_node_ip_address

    logger.info("Configured Ray to advertise loopback node IP %s", desired_ip)
    return True


def _patch_ray_loopback_advertise() -> bool:
    if not _ray_loopback_advertise_enabled():
        return False

    try:
        import ray
        import ray._private.services as ray_services_mod
    except ImportError:
        logger.exception("Unable to import Ray while enabling loopback advertise patch")
        return False

    if ray.is_initialized():
        logger.warning("Ray is already initialized; loopback advertise patch is too late to affect this run")
        return False

    return _patch_ray_loopback_advertise_modules(ray_services_mod, ray.util)


def _get_server_semaphore(server: Any, attr_name: str, limit_attr_name: str, env_name: str, default: int):
    limit = _int_env(env_name, default)
    if limit <= 0:
        return None

    semaphore = getattr(server, attr_name, None)
    if semaphore is None or getattr(server, limit_attr_name, None) != limit:
        semaphore = asyncio.Semaphore(limit)
        setattr(server, attr_name, semaphore)
        setattr(server, limit_attr_name, limit)
        logger.info("Configured %s=%s for %s", env_name, limit, type(server).__name__)
    return semaphore


class _LogprobScoreBatcher:
    def __init__(
        self,
        batch_size: int,
        batch_timeout_ms: int,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.batch_timeout_s = max(0, int(batch_timeout_ms)) / 1000.0
        self._lock = asyncio.Lock()
        self._pending_by_key: dict[str, list[tuple[Any, dict[str, Any], asyncio.Future]]] = {}
        self._flush_task_by_key: dict[str, asyncio.Task] = {}

    async def enqueue(self, key: str, server: Any, payload: dict[str, Any]) -> dict[str, float]:
        if self.batch_size <= 1:
            return await server.score_prompt_tokens.remote(**payload)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        flush_now = False
        flush_task: asyncio.Task | None = None

        async with self._lock:
            pending = self._pending_by_key.setdefault(key, [])
            pending.append((server, payload, future))
            if len(pending) >= self.batch_size:
                flush_now = True
                flush_task = self._flush_task_by_key.pop(key, None)
            elif key not in self._flush_task_by_key or self._flush_task_by_key[key].done():
                self._flush_task_by_key[key] = loop.create_task(self._delayed_flush(key))

        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
        if flush_now:
            loop.create_task(self._flush(key))

        return await future

    async def _delayed_flush(self, key: str) -> None:
        try:
            if self.batch_timeout_s > 0:
                await asyncio.sleep(self.batch_timeout_s)
            else:
                await asyncio.sleep(0)
            await self._flush(key)
        except asyncio.CancelledError:
            raise

    async def _flush(self, key: str) -> None:
        while True:
            async with self._lock:
                pending = self._pending_by_key.get(key)
                if not pending:
                    self._flush_task_by_key.pop(key, None)
                    return

                batch = pending[: self.batch_size]
                remaining = pending[self.batch_size :]
                if remaining:
                    self._pending_by_key[key] = remaining
                else:
                    self._pending_by_key.pop(key, None)
                    self._flush_task_by_key.pop(key, None)

            server = batch[0][0]
            payloads = [payload for _, payload, _ in batch]
            futures = [future for _, _, future in batch]

            try:
                if len(payloads) == 1:
                    results = [await server.score_prompt_tokens.remote(**payloads[0])]
                else:
                    results = await server.score_prompt_tokens_batch.remote(requests=payloads)
                    if len(results) != len(payloads):
                        raise RuntimeError(
                            f"score_prompt_tokens_batch returned {len(results)} results for {len(payloads)} payloads"
                        )
            except Exception as exc:
                for future in futures:
                    if not future.done():
                        future.set_exception(exc)
            else:
                for result, future in zip(results, futures):
                    if not future.done():
                        future.set_result(result)


def _get_logprob_score_batcher(owner: Any) -> _LogprobScoreBatcher:
    batch_size = _int_env("PERSONA_LOGPROB_SCORE_BATCH_SIZE", 8)
    batch_timeout_ms = _int_env("PERSONA_LOGPROB_SCORE_BATCH_TIMEOUT_MS", 5)
    batcher = getattr(owner, "_persona_logprob_score_batcher", None)
    if (
        batcher is None
        or not isinstance(batcher, _LogprobScoreBatcher)
        or batcher.batch_size != max(1, batch_size)
        or int(round(batcher.batch_timeout_s * 1000)) != max(0, batch_timeout_ms)
    ):
        batcher = _LogprobScoreBatcher(
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
        )
        setattr(owner, "_persona_logprob_score_batcher", batcher)
    return batcher


def _normalize_prompt_token_ids(prompt_token_ids: Any) -> list[int]:
    return [int(token_id) for token_id in prompt_token_ids]


async def _drain_request_output_collector(collector: Any) -> Any:
    final_output = None
    finished = False
    while not finished:
        output = collector.get_nowait()
        if output is None:
            output = await collector.get()
        final_output = output
        finished = bool(getattr(output, "finished", False))

    if final_output is None:
        raise RuntimeError("vLLM returned no request output for current-policy logprob scoring")
    return final_output


def _build_logprob_score_result_from_output(
    final_res: Any,
    *,
    prompt_token_ids: list[int],
    prompt_token_count: int,
) -> dict[str, float]:
    prompt_logprobs = getattr(final_res, "prompt_logprobs", None)
    if prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt logprobs for current-policy logprob scoring")

    total_logprob = 0.0
    num_tokens = 0
    for pos in range(prompt_token_count, len(prompt_token_ids)):
        if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            continue
        token_id = prompt_token_ids[pos]
        token_logprob = prompt_logprobs[pos].get(token_id)
        if token_logprob is None:
            continue
        total_logprob += float(token_logprob.logprob)
        num_tokens += 1

    if num_tokens <= 0:
        raise RuntimeError("Current-policy logprob scorer produced zero target tokens")

    return {
        "mean_logprob": total_logprob / num_tokens,
        "num_tokens": float(num_tokens),
    }


async def _run_logprob_score_batch_via_engine(
    engine: Any,
    requests: list[dict[str, Any]],
    *,
    lora_request: Any = None,
    default_priority: int = 0,
    sampling_params_factory: Any | None = None,
    prompt_factory: Any | None = None,
) -> list[dict[str, float]]:
    if not requests:
        return []

    if sampling_params_factory is None or prompt_factory is None:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        if sampling_params_factory is None:
            def sampling_params_factory():
                params = SamplingParams(
                    # vLLM validates max_tokens >= 1 even when we only need prompt_logprobs.
                    # Request one continuation token and ignore it below.
                    max_tokens=1,
                    prompt_logprobs=1,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    repetition_penalty=1.0,
                )
                params.output_kind = RequestOutputKind.FINAL_ONLY
                return params

        if prompt_factory is None:
            prompt_factory = lambda token_ids: TokensPrompt(prompt_token_ids=token_ids)

    pending: list[tuple[Any, list[int], int]] = []
    active_internal_request_ids: list[str] = []

    try:
        for request in requests:
            prompt_token_ids = _normalize_prompt_token_ids(request["prompt_token_ids"])
            prompt_token_count = int(request["prompt_token_count"])
            if prompt_token_count < 0 or prompt_token_count >= len(prompt_token_ids):
                raise ValueError(
                    f"Invalid prompt_token_count={prompt_token_count} for {len(prompt_token_ids)} prompt tokens"
                )

            external_request_id = str(request.get("request_id") or uuid4().hex)
            priority = int(request.get("priority", default_priority))
            collector = await engine.add_request(
                request_id=external_request_id,
                prompt=prompt_factory(prompt_token_ids),
                params=sampling_params_factory(),
                lora_request=lora_request,
                priority=priority,
            )
            pending.append((collector, prompt_token_ids, prompt_token_count))
            active_internal_request_ids.append(str(getattr(collector, "request_id", external_request_id)))

        final_outputs = await asyncio.gather(
            *[_drain_request_output_collector(collector) for collector, _, _ in pending]
        )
        return [
            _build_logprob_score_result_from_output(
                final_output,
                prompt_token_ids=prompt_token_ids,
                prompt_token_count=prompt_token_count,
            )
            for final_output, (_, prompt_token_ids, prompt_token_count) in zip(final_outputs, pending, strict=True)
        ]
    except Exception:
        if active_internal_request_ids:
            try:
                await engine.abort(active_internal_request_ids, internal=True)
            except Exception:
                logger.exception(
                    "Failed to abort current-policy logprob batch requests: %s",
                    active_internal_request_ids,
                )
        raise
    finally:
        for collector, _, _ in pending:
            collector.close()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_reward_extra_info_keys(inputs: list[Any]) -> None:
    reward_keys: set[str] = set()
    for input_item in inputs:
        extra_fields = _coerce_mapping(getattr(input_item, "extra_fields", {}))
        reward_info = extra_fields.get("reward_extra_info")
        if isinstance(reward_info, Mapping):
            reward_keys.update(str(key) for key in reward_info.keys())

    if not reward_keys:
        return

    normalized = False
    for input_item in inputs:
        extra_fields = _coerce_mapping(getattr(input_item, "extra_fields", {}))
        reward_info = extra_fields.get("reward_extra_info")
        normalized_info = dict(reward_info) if isinstance(reward_info, Mapping) else {}
        for key in reward_keys:
            if key not in normalized_info:
                normalized_info[key] = 0.0
                normalized = True
        extra_fields["reward_extra_info"] = normalized_info
        input_item.extra_fields = extra_fields

    if normalized:
        logger.warning(
            "Normalized reward_extra_info keys across %s rollout outputs: %s",
            len(inputs),
            ",".join(sorted(reward_keys)),
        )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_logprob_ground_truth(kwargs: Mapping[str, Any], extra_info: Mapping[str, Any]) -> str:
    reward_model = _coerce_mapping(kwargs.get("reward_model"))
    candidates = (
        reward_model.get("ground_truth"),
        kwargs.get("ground_truth"),
        extra_info.get("ground_truth"),
    )
    for candidate in candidates:
        text = _coerce_text(candidate).strip()
        if text:
            return text
    return ""


def _actor_elbo_sft_weight(config: Any) -> float:
    actor_cfg = _config_get(_config_get(config, "actor_rollout_ref", None), "actor", None)
    raw_weight = _config_get(actor_cfg, "elbo_sft_weight", 0.0) if actor_cfg is not None else 0.0
    try:
        return float(raw_weight or 0.0)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid actor_rollout_ref.actor.elbo_sft_weight=%r", raw_weight)
        return 0.0


def _decode_token_ids(tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _elbo_sft_max_length(config: Any) -> int:
    env_value = _int_env("PERSONA_ELBO_SFT_MAX_LENGTH", 0)
    if env_value and env_value > 0:
        return env_value

    rollout_cfg = _config_get(_config_get(config, "actor_rollout_ref", None), "rollout", None)
    max_model_len = _config_get(rollout_cfg, "max_model_len", None) if rollout_cfg is not None else None
    try:
        if max_model_len:
            return int(max_model_len)
    except (TypeError, ValueError):
        pass

    data_cfg = _config_get(config, "data", None)
    max_prompt_length = _config_get(data_cfg, "max_prompt_length", 0) if data_cfg is not None else 0
    max_response_length = _config_get(data_cfg, "max_response_length", 0) if data_cfg is not None else 0
    try:
        total_length = int(max_prompt_length or 0) + int(max_response_length or 0)
        if total_length > 0:
            return total_length
    except (TypeError, ValueError):
        pass

    return 14336


def _build_elbo_sft_tensor_batch(trainer: Any, batch: Any) -> dict[str, Any]:
    """Build teacher-forced GT tensors for optional auxiliary actor SFT loss."""

    import torch

    from shared.prompt_utils import build_response_prefill, tokenize_with_prefix_boundary
    from training.grpo.reward import parse_response_for_prompt_mode

    tokenizer = trainer.tokenizer
    tensor_batch = batch.batch
    non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
    input_ids = tensor_batch["input_ids"]
    attention_mask = tensor_batch["attention_mask"]
    responses = tensor_batch["responses"]
    response_mask = tensor_batch.get("response_mask")
    prompts = tensor_batch.get("prompts")
    prompt_len = int(prompts.shape[1]) if prompts is not None else int(input_ids.shape[1] - responses.shape[1])
    max_length = _elbo_sft_max_length(trainer.config)

    extra_infos = list(non_tensor_batch.get("extra_info", []))
    reward_models = list(non_tensor_batch.get("reward_model", []))
    device = input_ids.device
    batch_size = int(input_ids.shape[0])
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)

    prefix_id_rows: list[list[int]] = []
    target_id_rows: list[list[int]] = []
    missing_ground_truth: list[int] = []
    empty_target: list[int] = []

    for idx in range(batch_size):
        extra_info = _coerce_mapping(extra_infos[idx] if idx < len(extra_infos) else {})
        reward_model = _coerce_mapping(reward_models[idx] if idx < len(reward_models) else {})
        ground_truth = _extract_logprob_ground_truth({"reward_model": reward_model}, extra_info)
        if not ground_truth:
            missing_ground_truth.append(idx)
            continue

        valid_prompt_mask = attention_mask[idx, :prompt_len].bool()
        prompt_token_ids = input_ids[idx, :prompt_len][valid_prompt_mask].detach().cpu().tolist()
        valid_response_mask = (
            response_mask[idx].bool()
            if response_mask is not None
            else attention_mask[idx, prompt_len : prompt_len + responses.shape[1]].bool()
        )
        response_token_ids = responses[idx][valid_response_mask].detach().cpu().tolist()
        prompt_text = _decode_token_ids(tokenizer, prompt_token_ids, skip_special_tokens=False)
        solution_str = _decode_token_ids(tokenizer, response_token_ids, skip_special_tokens=True)
        prompt_mode = str(extra_info.get("prompt_mode", "reasoning") or "reasoning")
        cot, _ = parse_response_for_prompt_mode(solution_str, prompt_mode)
        prefix_text = prompt_text + build_response_prefill(cot)
        full_text = prefix_text + ground_truth
        full_token_ids, prefix_token_count = tokenize_with_prefix_boundary(
            tokenizer,
            prefix_text=prefix_text,
            full_text=full_text,
        )
        target_token_ids = list(full_token_ids[prefix_token_count:])
        prefix_token_ids = list(full_token_ids[:prefix_token_count])
        if not target_token_ids:
            empty_target.append(idx)
            continue

        if len(prefix_token_ids) + len(target_token_ids) > max_length:
            max_target_len = min(len(target_token_ids), max_length)
            target_token_ids = target_token_ids[:max_target_len]
            remaining_prefix_len = max(0, max_length - len(target_token_ids))
            prefix_token_ids = prefix_token_ids[-remaining_prefix_len:] if remaining_prefix_len else []

        prefix_id_rows.append(prefix_token_ids)
        target_id_rows.append(target_token_ids)

    if missing_ground_truth or empty_target:
        raise RuntimeError(
            "Cannot build ELBO/SFT tensors: "
            f"missing_ground_truth_indices={missing_ground_truth} "
            f"empty_target_indices={empty_target}"
        )

    if len(prefix_id_rows) != batch_size or len(target_id_rows) != batch_size:
        raise RuntimeError(
            f"Cannot build ELBO/SFT tensors for all rows: built={len(prefix_id_rows)} batch_size={batch_size}"
        )

    max_prefix_len = max((len(row) for row in prefix_id_rows), default=0)
    max_target_len = max((len(row) for row in target_id_rows), default=1)
    elbo_input_ids = []
    elbo_attention_mask = []
    elbo_responses = []
    elbo_response_mask = []

    for prefix_token_ids, target_token_ids in zip(prefix_id_rows, target_id_rows, strict=True):
        left_pad = max_prefix_len - len(prefix_token_ids)
        right_pad = max_target_len - len(target_token_ids)
        elbo_input_ids.append(
            [pad_token_id] * left_pad
            + prefix_token_ids
            + target_token_ids
            + [pad_token_id] * right_pad
        )
        elbo_attention_mask.append(
            [0] * left_pad
            + [1] * len(prefix_token_ids)
            + [1] * len(target_token_ids)
            + [0] * right_pad
        )
        elbo_responses.append(target_token_ids + [pad_token_id] * right_pad)
        elbo_response_mask.append([1] * len(target_token_ids) + [0] * right_pad)

    attention_tensor = torch.tensor(elbo_attention_mask, dtype=attention_mask.dtype, device=device)
    position_ids = attention_tensor.long().cumsum(dim=-1) - 1
    position_ids = position_ids.clamp_min(0)
    return {
        "elbo_input_ids": torch.tensor(elbo_input_ids, dtype=input_ids.dtype, device=device),
        "elbo_attention_mask": attention_tensor,
        "elbo_position_ids": position_ids,
        "elbo_responses": torch.tensor(elbo_responses, dtype=responses.dtype, device=device),
        "elbo_response_mask": torch.tensor(elbo_response_mask, dtype=torch.bool, device=device),
    }


def _maybe_attach_elbo_sft_tensors(trainer: Any, batch: Any) -> None:
    if _actor_elbo_sft_weight(getattr(trainer, "config", None)) <= 0:
        return
    for tensor_key in _ELBO_SFT_TENSOR_KEYS:
        if tensor_key in batch.batch:
            return
    batch.batch.update(_build_elbo_sft_tensor_batch(trainer, batch))


@lru_cache(maxsize=4)
def _load_runtime_persona_map(persona_path: str) -> dict[str, str]:
    from shared.load_personas import load_persona_map

    return load_persona_map(persona_path)


def _requested_runtime_conditioning_mode(extra_info: Mapping[str, Any] | None = None) -> str | None:
    if not _runtime_prompt_override_enabled():
        return None

    requested_mode = (
        os.environ.get("PERSONA_RUNTIME_CONDITIONING_MODE")
        or os.environ.get("CONDITIONING_MODE")
        or (extra_info or {}).get("conditioning_mode")
    )
    if not requested_mode:
        return None

    from shared.prompt_utils import CONDITIONING_MODE_CHOICES

    requested_mode = str(requested_mode)
    if requested_mode not in CONDITIONING_MODE_CHOICES:
        raise ValueError(
            f"Unknown conditioning_mode={requested_mode!r}. "
            f"Expected one of {', '.join(CONDITIONING_MODE_CHOICES)}."
        )
    return requested_mode


def _maybe_override_prompt_messages_for_runtime_conditioning(
    extra_info: Mapping[str, Any] | None,
) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    normalized_extra_info = _coerce_mapping(extra_info)
    requested_conditioning_mode = _requested_runtime_conditioning_mode(normalized_extra_info)
    if requested_conditioning_mode is None:
        return None, normalized_extra_info

    from shared.prompt_utils import build_messages_for_prompt_mode, conditioning_mode_uses_persona
    from shared.load_personas import get_persona_for_user

    row_conditioning_mode = str(normalized_extra_info.get("conditioning_mode", "history") or "history")
    row_persona = str(
        normalized_extra_info.get("persona", normalized_extra_info.get("persona_memory", "")) or ""
    ).strip()
    requires_persona = conditioning_mode_uses_persona(requested_conditioning_mode)
    needs_rebuild = requested_conditioning_mode != row_conditioning_mode or (requires_persona and not row_persona)
    if not needs_rebuild:
        return None, normalized_extra_info

    user_history = str(normalized_extra_info.get("user_history", "") or "")
    thread_context = str(
        normalized_extra_info.get("context", normalized_extra_info.get("thread_context", "")) or ""
    )
    prompt_mode = str(normalized_extra_info.get("prompt_mode", "reasoning") or "reasoning")
    persona = row_persona

    if requires_persona and not persona:
        persona_path = os.environ.get("PERSONA_PATH")
        if not persona_path:
            raise ValueError(
                "PERSONA_ENABLE_RUNTIME_PROMPT_OVERRIDE=1 with persona-backed conditioning requires PERSONA_PATH"
            )
        persona_map = _load_runtime_persona_map(persona_path)
        user_id = str(normalized_extra_info.get("user_id", "") or "")
        raw_user_id = str(normalized_extra_info.get("raw_user_id", user_id) or "")
        persona = get_persona_for_user(persona_map, user_id, raw_user_id)
        if not persona:
            raise ValueError(
                "Runtime prompt override could not find a persona for "
                f"user_id={user_id!r} raw_user_id={raw_user_id!r} using PERSONA_PATH={persona_path!r}"
            )

    prompt_messages = build_messages_for_prompt_mode(
        user_history=user_history,
        thread_context=thread_context,
        prompt_mode=prompt_mode,
        persona=persona,
        conditioning_mode=requested_conditioning_mode,
    )
    rewritten_extra_info = dict(normalized_extra_info)
    rewritten_extra_info.update(
        {
            "persona": persona,
            "conditioning_mode": requested_conditioning_mode,
            # Force logprob reward scoring to rebuild prompt_text from the rewritten fields.
            "prompt_text": "",
            "raw_prompt": json.dumps(prompt_messages, ensure_ascii=False),
        }
    )
    return prompt_messages, rewritten_extra_info


def _tokenizer_terminal_token_ids(tokenizer: Any) -> set[int]:
    terminal_ids: set[int] = set()
    for attr_name in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr_name, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            terminal_ids.update(int(token_id) for token_id in value if token_id is not None)
        else:
            terminal_ids.add(int(value))
    return terminal_ids


def _collect_propagated_runtime_env_vars() -> dict[str, str]:
    if not _runtime_env_propagation_enabled():
        return {}
    return {
        name: os.environ[name]
        for name in _PROPAGATED_RUNTIME_ENV_VARS
        if os.environ.get(name) is not None
    }


def _worker_process_setup_hook_path() -> str:
    return _PERSONA_WORKER_PROCESS_SETUP_HOOK


def _with_repo_root_pythonpath(env_vars: dict[str, str]) -> dict[str, str]:
    repo_root = os.environ.get("REPO_ROOT", "").strip()
    if not repo_root:
        return env_vars

    existing_pythonpath = env_vars.get("PYTHONPATH")
    if existing_pythonpath is None:
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in existing_pythonpath.split(":") if entry]
    if repo_root not in pythonpath_entries:
        pythonpath_entries.insert(0, repo_root)
    env_vars["PYTHONPATH"] = ":".join(pythonpath_entries) if pythonpath_entries else repo_root
    return env_vars


def _merge_propagated_runtime_env_vars(runtime_env: Mapping[str, Any] | None) -> dict[str, Any]:
    merged_runtime_env = dict(runtime_env or {})
    propagated_env_vars = _collect_propagated_runtime_env_vars()
    if propagated_env_vars:
        env_vars = dict(merged_runtime_env.get("env_vars") or {})
        env_vars.update(propagated_env_vars)
        merged_runtime_env["env_vars"] = env_vars

    if _worker_process_setup_hook_enabled():
        env_vars = dict(merged_runtime_env.get("env_vars") or {})
        existing_hook = merged_runtime_env.get("worker_process_setup_hook")
        if existing_hook and existing_hook != _worker_process_setup_hook_path():
            if isinstance(existing_hook, str):
                env_vars[_UPSTREAM_WORKER_PROCESS_SETUP_HOOK_ENV] = existing_hook
            else:
                logger.warning(
                    "Skipping unsupported existing worker_process_setup_hook of type %s while installing persona hook",
                    type(existing_hook).__name__,
                )
        merged_runtime_env["env_vars"] = _with_repo_root_pythonpath(env_vars)
        merged_runtime_env["worker_process_setup_hook"] = _worker_process_setup_hook_path()

    return merged_runtime_env


def _patch_ppo_ray_runtime_env(constants_ppo_mod: Any) -> None:
    if getattr(constants_ppo_mod, "_persona_runtime_env_vars_patch_applied", False):
        return

    original_get_ppo_ray_runtime_env = constants_ppo_mod.get_ppo_ray_runtime_env

    def patched_get_ppo_ray_runtime_env():
        return _merge_propagated_runtime_env_vars(original_get_ppo_ray_runtime_env())

    constants_ppo_mod.get_ppo_ray_runtime_env = patched_get_ppo_ray_runtime_env
    constants_ppo_mod._persona_runtime_env_vars_patch_applied = True


def _patch_ray_worker_group_runtime_env(ray_base_mod: Any) -> None:
    ray_class_with_init = getattr(ray_base_mod, "RayClassWithInitArgs", None)
    if ray_class_with_init is None:
        return

    if getattr(ray_class_with_init, "_persona_runtime_env_update_patch_applied", False):
        return

    original_update_options = ray_class_with_init.update_options

    def patched_update_options(self, options: dict):
        merged_options = dict(options or {})
        runtime_env = merged_options.get("runtime_env")
        if runtime_env is not None:
            merged_options["runtime_env"] = _merge_propagated_runtime_env_vars(runtime_env)
        return original_update_options(self, merged_options)

    ray_class_with_init.update_options = patched_update_options
    ray_class_with_init._persona_runtime_env_update_patch_applied = True


def _patch_reward_loop_worker_runtime_env(reward_loop_mod: Any) -> None:
    reward_loop_manager = getattr(reward_loop_mod, "RewardLoopManager", None)
    if reward_loop_manager is None:
        return

    if getattr(reward_loop_manager, "_persona_runtime_env_patch_applied", False):
        return

    original_init_reward_loop_workers = reward_loop_manager._init_reward_loop_workers

    def patched_init_reward_loop_workers(self):
        self.reward_loop_workers_class = _ServerClassRuntimeEnvProxy(self.reward_loop_workers_class)
        return original_init_reward_loop_workers(self)

    reward_loop_manager._init_reward_loop_workers = patched_init_reward_loop_workers
    reward_loop_manager._persona_runtime_env_patch_applied = True


def _colocated_worker_base_class_name(class_dict: Mapping[str, Any], ray_base_mod: Any) -> str | None:
    determine_base_class = getattr(ray_base_mod, "_determine_fsdp_megatron_base_class", None)
    if not callable(determine_base_class):
        return None

    try:
        mros = [class_with_init.cls.__ray_actor_class__.__mro__ for class_with_init in class_dict.values()]
        base_class = determine_base_class(mros)
    except Exception:
        logger.debug("Unable to determine colocated worker base class", exc_info=True)
        return None

    return getattr(base_class, "__name__", None)


def _inject_local_rank_visible_device_fallback(
    env_vars: Mapping[str, str] | None,
    *,
    use_gpu: bool,
    device_name: str,
    local_rank: int,
) -> dict[str, str]:
    merged_env_vars = dict(env_vars or {})
    if (
        _local_rank_visible_device_fallback_enabled()
        and use_gpu
        and device_name == "cuda"
        and "PERSONA_FALLBACK_VISIBLE_DEVICE_ORDINAL" not in merged_env_vars
    ):
        merged_env_vars["PERSONA_FALLBACK_VISIBLE_DEVICE_ORDINAL"] = str(local_rank)
    return merged_env_vars


def _patch_ray_colocated_worker_gpu_reservation(ray_base_mod: Any, ray_trainer_mod: Any) -> None:
    if getattr(ray_base_mod, "_persona_colocated_worker_gpu_patch_applied", False):
        return

    original_create_colocated_worker_cls = getattr(ray_base_mod, "create_colocated_worker_cls", None)
    ray_worker_group = getattr(ray_base_mod, "RayWorkerGroup", None)
    if original_create_colocated_worker_cls is None or ray_worker_group is None:
        return

    original_create_worker = ray_worker_group._create_worker

    def patched_create_colocated_worker_cls(class_dict: dict[str, Any]):
        ray_cls_with_init = original_create_colocated_worker_cls(class_dict)
        if _full_gpu_colocated_training_enabled():
            if _colocated_worker_base_class_name(class_dict, ray_base_mod) == "Worker":
                setattr(ray_cls_with_init, "_persona_reserve_full_gpu", True)
        return ray_cls_with_init

    def patched_create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached):
        reserve_full_gpu = (
            _full_gpu_colocated_training_enabled()
            and getattr(ray_cls_with_init, "_persona_reserve_full_gpu", False)
            and getattr(resource_pool, "max_colocate_count", 1) > 1
            and getattr(self, "use_gpu", True)
            and getattr(self, "device_name", "cuda") == "cuda"
        )
        worker_env = _inject_local_rank_visible_device_fallback(
            worker_env,
            use_gpu=getattr(self, "use_gpu", True) and getattr(resource_pool, "use_gpu", True),
            device_name=getattr(self, "device_name", "cuda"),
            local_rank=local_rank,
        )
        if not reserve_full_gpu:
            return original_create_worker(
                self,
                rank,
                pg_idx,
                pg,
                local_rank,
                resource_pool,
                ray_cls_with_init,
                worker_env,
                detached,
            )

        original_max_colocate_count = resource_pool.max_colocate_count
        if not getattr(self, "_persona_logged_full_gpu_reservation", False):
            logger.info(
                "Reserving one full GPU per colocated FSDP worker for %s by overriding max_colocate_count=%s -> 1",
                getattr(self, "name_prefix", type(self).__name__),
                original_max_colocate_count,
            )
            self._persona_logged_full_gpu_reservation = True

        resource_pool.max_colocate_count = 1
        try:
            return original_create_worker(
                self,
                rank,
                pg_idx,
                pg,
                local_rank,
                resource_pool,
                ray_cls_with_init,
                worker_env,
                detached,
            )
        finally:
            resource_pool.max_colocate_count = original_max_colocate_count

    ray_base_mod.create_colocated_worker_cls = patched_create_colocated_worker_cls
    if ray_trainer_mod is not None and hasattr(ray_trainer_mod, "create_colocated_worker_cls"):
        ray_trainer_mod.create_colocated_worker_cls = patched_create_colocated_worker_cls
    ray_worker_group._create_worker = patched_create_worker
    ray_base_mod._persona_colocated_worker_gpu_patch_applied = True


class _ServerClassRuntimeEnvProxy:
    def __init__(self, server_class: Any) -> None:
        self._server_class = server_class

    def __getattr__(self, name: str) -> Any:
        return getattr(self._server_class, name)

    def options(self, *args, **kwargs):
        kwargs["runtime_env"] = _merge_propagated_runtime_env_vars(kwargs.get("runtime_env"))
        return self._server_class.options(*args, **kwargs)


def _clean_fsdp_state_key(name: str) -> str:
    return name.replace("._fsdp_wrapped_module.", ".").removeprefix("_fsdp_wrapped_module.")


def _lora_checkpoint_key(name: str, adapter_name: str) -> str:
    return _clean_fsdp_state_key(name).replace(f".{adapter_name}.", ".")


def _lora_checkpoint_lookup_keys(name: str, adapter_name: str) -> list[str]:
    clean_key = _clean_fsdp_state_key(name)
    adapterless_key = _lora_checkpoint_key(name, adapter_name)
    keys = [adapterless_key, clean_key]
    if adapter_name and f".{adapter_name}." not in clean_key:
        for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."):
            if marker in clean_key:
                keys.append(clean_key.replace(marker, f"{marker}{adapter_name}.", 1))
    return list(dict.fromkeys(keys))


def _fsdp1_has_unsharded_handle(model: Any) -> bool:
    from verl.utils.fsdp_utils import fsdp_version

    if fsdp_version(model) != 1:
        return False
    handle = getattr(model, "_handle", None)
    return handle is not None and not getattr(handle, "uses_sharded_strategy", True)


def _get_peft_model(model: Any) -> Any | None:
    wrapped_model = getattr(model, "_fsdp_wrapped_module", model)
    return wrapped_model if hasattr(wrapped_model, "peft_config") else None


def _is_fsdp1_peft_model(model: Any) -> bool:
    from verl.utils.fsdp_utils import fsdp_version

    return fsdp_version(model) == 1 and _get_peft_model(model) is not None


def _collect_unsharded_lora_state_dict(model: Any) -> dict[str, Any]:
    peft_model = _get_peft_model(model)
    if peft_model is None:
        raise RuntimeError("Expected a PEFT model while collecting LoRA checkpoint state")

    adapter_name = next(iter(peft_model.peft_config.keys()), "default")
    try:
        named_parameters = peft_model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = peft_model.named_parameters()

    state_dict = {}
    for name, param in named_parameters:
        key = _lora_checkpoint_key(name, adapter_name)
        if "lora_" not in key:
            continue
        state_dict[key] = param.detach().cpu().clone()
    return state_dict


def _load_unsharded_lora_state_dict(model: Any, state_dict: Mapping[str, Any]) -> tuple[int, list[str]]:
    peft_model = _get_peft_model(model)
    if peft_model is None:
        raise RuntimeError("LoRA-only checkpoint found, but the current model is not a PEFT model")

    adapter_name = next(iter(peft_model.peft_config.keys()), "default")
    try:
        named_parameters = peft_model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = peft_model.named_parameters()

    loaded = 0
    missing = []
    for name, param in named_parameters:
        key = _lora_checkpoint_key(name, adapter_name)
        if "lora_" not in key:
            continue
        tensor = None
        for lookup_key in _lora_checkpoint_lookup_keys(name, adapter_name):
            tensor = state_dict.get(lookup_key)
            if tensor is not None:
                break
        if tensor is None:
            missing.append(key)
            continue
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        loaded += 1
    return loaded, missing


def _should_use_fsdp1_lora_only_checkpoint(model: Any) -> bool:
    return (
        _bool_env("PERSONA_FSDP1_LORA_ONLY_CHECKPOINT", True)
        and _is_fsdp1_peft_model(model)
    )


def _patch_fsdp1_lora_checkpointing() -> bool:
    try:
        import torch
        import verl.workers.fsdp_workers as fsdp_workers_mod
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        from verl.utils.fs import local_mkdir_safe
        from verl.utils.logger import log_with_rank
    except ImportError:
        logger.debug("Unable to import VERL FSDP checkpoint modules for runtime patch", exc_info=True)
        return False

    if getattr(FSDPCheckpointManager, "_persona_fsdp1_lora_checkpoint_patch_applied", False):
        return True

    original_save_checkpoint = FSDPCheckpointManager.save_checkpoint
    original_load_checkpoint = FSDPCheckpointManager.load_checkpoint
    original_layered_summon_lora_params = fsdp_workers_mod.layered_summon_lora_params

    def collect_fsdp1_lora_state_dict(model: Any) -> dict[str, Any]:
        if _fsdp1_has_unsharded_handle(model):
            return _collect_unsharded_lora_state_dict(model)
        return dict(original_layered_summon_lora_params(model))

    def load_fsdp1_lora_state_dict(model: Any, state_dict: Mapping[str, Any]) -> tuple[int, list[str]]:
        if _fsdp1_has_unsharded_handle(model):
            return _load_unsharded_lora_state_dict(model, state_dict)

        peft_model = _get_peft_model(model)
        if peft_model is None:
            raise RuntimeError("LoRA-only checkpoint found, but the current model is not a PEFT model")

        with FSDP.summon_full_params(model, writeback=True):
            return _load_unsharded_lora_state_dict(model, state_dict)

    def patched_save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        if local_path is None or not (self.should_save_model and _should_use_fsdp1_lora_only_checkpoint(self.model)):
            return original_save_checkpoint(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)

        original_save_contents = self.checkpoint_save_contents
        self.checkpoint_save_contents = [item for item in original_save_contents if item != "model"]
        try:
            original_save_checkpoint(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)
        finally:
            self.checkpoint_save_contents = original_save_contents

        local_path = local_mkdir_safe(local_path)
        model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        torch.save(
            {
                "__persona_lora_only__": True,
                "state_dict": collect_fsdp1_lora_state_dict(self.model),
            },
            model_path,
        )
        log_with_rank(
            f"PERSONA: saved LoRA-only model checkpoint to {os.path.abspath(model_path)}",
            rank=self.rank,
            logger=logger,
        )
        return None

    def patched_load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is None or not (self.should_load_model and _should_use_fsdp1_lora_only_checkpoint(self.model)):
            return original_load_checkpoint(self, local_path, hdfs_path, del_local_after_load)

        model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        if not os.path.exists(model_path):
            return original_load_checkpoint(self, local_path, hdfs_path, del_local_after_load)

        model_state_dict = torch.load(model_path, weights_only=False)
        if not (isinstance(model_state_dict, dict) and model_state_dict.get("__persona_lora_only__")):
            return original_load_checkpoint(self, local_path, hdfs_path, del_local_after_load)

        original_load_contents = self.checkpoint_load_contents
        self.checkpoint_load_contents = [item for item in original_load_contents if item != "model"]
        try:
            original_load_checkpoint(self, local_path, hdfs_path, del_local_after_load)
        finally:
            self.checkpoint_load_contents = original_load_contents

        loaded, missing = load_fsdp1_lora_state_dict(self.model, model_state_dict["state_dict"])
        log_with_rank(
            f"PERSONA: loaded LoRA-only model checkpoint from {model_path}: {loaded} tensors"
            + (f", missing {len(missing)} tensors" if missing else ""),
            rank=self.rank,
            logger=logger,
        )
        return None

    def patched_layered_summon_lora_params(fsdp_module):
        if _should_use_fsdp1_lora_only_checkpoint(fsdp_module) and _fsdp1_has_unsharded_handle(fsdp_module):
            return _collect_unsharded_lora_state_dict(fsdp_module)
        return original_layered_summon_lora_params(fsdp_module)

    FSDPCheckpointManager.save_checkpoint = patched_save_checkpoint
    FSDPCheckpointManager.load_checkpoint = patched_load_checkpoint
    fsdp_workers_mod.layered_summon_lora_params = patched_layered_summon_lora_params
    FSDPCheckpointManager._persona_fsdp1_lora_checkpoint_patch_applied = True
    return True


def _build_response_mask(
    tokenizer: Any,
    token_ids: list[int],
    *,
    metric: str | None,
    prompt_mode: str | None,
) -> list[int]:
    """Return the rollout loss mask, matching TRL-style truncated masking for logprob."""
    _ = prompt_mode
    if metric == "logprob" and token_ids:
        terminal_ids = _tokenizer_terminal_token_ids(tokenizer)
        if terminal_ids and int(token_ids[-1]) not in terminal_ids:
            return [0] * len(token_ids)
    return [1] * len(token_ids)


def _tokenize_without_specials(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _suppress_fast_tokenizer_pad_advisory(tokenizer: Any) -> None:
    """Silence the Hugging Face fast-tokenizer encode+pad performance advisory."""
    warnings = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(warnings, dict):
        warnings["Asking-to-pad-a-fast-tokenizer"] = True


def _strip_hidden_thinking_close_token_prefix(
    tokenizer: Any,
    token_ids: list[int],
    log_probs: list[Any] | None,
) -> tuple[list[int], list[Any] | None, int]:
    """Drop a generated Qwen close-think boundary that belongs to prompt plumbing."""
    if not token_ids:
        return token_ids, log_probs, 0

    normalized_token_ids = [int(token_id) for token_id in token_ids]
    for prefix_text in ("</think>\n\n", "</think>\n", "</think>"):
        prefix_ids = _tokenize_without_specials(tokenizer, prefix_text)
        if prefix_ids and normalized_token_ids[: len(prefix_ids)] == prefix_ids:
            stripped_count = len(prefix_ids)
            stripped_log_probs = log_probs[stripped_count:] if log_probs else log_probs
            return normalized_token_ids[stripped_count:], stripped_log_probs, stripped_count

    return normalized_token_ids, log_probs, 0


def _render_text_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    prompt_mode: str | None,
) -> tuple[list[int], bool]:
    from shared.prompt_utils import get_chat_template_kwargs_for_prompt_mode

    prompt_text = tokenizer.apply_chat_template(
        messages,
        **get_chat_template_kwargs_for_prompt_mode(prompt_mode),
    )
    return _tokenize_without_specials(tokenizer, prompt_text), False


def _log_prompt_prefill_preserve_once(stripped: bool) -> None:
    global _SINGLE_TURN_PROMPT_PREFILL_STRIP_LOGGED
    if _SINGLE_TURN_PROMPT_PREFILL_STRIP_LOGGED:
        return
    _SINGLE_TURN_PROMPT_PREFILL_STRIP_LOGGED = True
    print(
        "PERSONA_DEBUG_AGENT_LOOP: preserved_empty_hidden_thinking_prompt_prefill=1",
        flush=True,
    )


def _log_hidden_thinking_close_strip_once(stripped_count: int) -> None:
    global _SINGLE_TURN_THINK_CLOSE_STRIP_LOGGED
    if _SINGLE_TURN_THINK_CLOSE_STRIP_LOGGED:
        return
    _SINGLE_TURN_THINK_CLOSE_STRIP_LOGGED = True
    print(
        "PERSONA_DEBUG_AGENT_LOOP: stripped_leading_hidden_thinking_close_tokens="
        f"{stripped_count}",
        flush=True,
    )


def _stable_bucket(key: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_buckets


def _partition_indices_by_group_keys(group_keys: list[str], num_buckets: int) -> list[list[int]]:
    buckets = [[] for _ in range(num_buckets)]
    grouped_indices: dict[str, list[int]] = {}
    for idx, group_key in enumerate(group_keys):
        grouped_indices.setdefault(group_key, []).append(idx)
    for group_key, indices in grouped_indices.items():
        buckets[_stable_bucket(group_key, num_buckets)].extend(indices)
    return buckets


def _reward_manager_name(config: Any) -> str:
    reward_cfg = getattr(config, "reward", None)
    reward_manager_cfg = getattr(reward_cfg, "reward_manager", None)
    return str(getattr(reward_manager_cfg, "name", "") or "")


def _should_route_grouped_sim_rewards(config: Any, num_workers: int) -> bool:
    if num_workers <= 1:
        return False
    if not _grouped_sim_route_patch_enabled():
        return False
    return _reward_manager_name(config) == "GroupedSimRewardManager"


def apply_verl_runtime_patch() -> bool:
    _patch_ray_loopback_advertise()
    _patch_peft_meta_adapter_load_source()
    _patch_fsdp1_lora_checkpointing()
    _patch_actor_config_elbo_sft_source()
    _patch_actor_elbo_sft_source()

    try:
        import verl.trainer.ppo.ray_trainer as ray_trainer_mod
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer import constants_ppo as constants_ppo_mod
        from verl.single_controller.ray import base as ray_base_mod
        from verl.experimental.agent_loop import agent_loop as agent_loop_mod
        from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
        from verl.experimental.reward_loop import reward_loop as reward_loop_mod
        from verl.experimental.reward_loop.reward_loop import RewardLoopManager
        from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
        from verl.utils.profiler import simple_timer
        from verl.workers.rollout.replica import TokenOutput
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
        from verl.workers.rollout.vllm_rollout.utils import (
            VLLM_LORA_INT_ID,
            VLLM_LORA_NAME,
            VLLM_LORA_PATH,
        )
        from vllm.lora.request import LoRARequest
    except ImportError:
        return False

    enable_runtime_env_propagation = _runtime_env_propagation_enabled()
    enable_worker_process_setup_hook = _worker_process_setup_hook_enabled()
    enable_grouped_sim_route_patch = _grouped_sim_route_patch_enabled()
    enable_full_gpu_colocated_training = _full_gpu_colocated_training_enabled()

    if enable_runtime_env_propagation or enable_worker_process_setup_hook:
        _patch_ppo_ray_runtime_env(constants_ppo_mod)
        _patch_ray_worker_group_runtime_env(ray_base_mod)
        _patch_reward_loop_worker_runtime_env(reward_loop_mod)

    if enable_full_gpu_colocated_training:
        _patch_ray_colocated_worker_gpu_reservation(ray_base_mod, ray_trainer_mod)

    if getattr(agent_loop_mod, "_persona_runtime_patch_applied", False):
        return True

    # vLLM server actors are launched with their own Ray runtime_env, so shell
    # exports from the driver are not guaranteed to be visible inside those
    # remote processes. Capture the intended limits here and ship them with the
    # patched methods/class instead of re-reading only remote env state.
    request_limit_default = _int_env("PERSONA_VLLM_REQUEST_CONCURRENCY_PER_SERVER", 0)
    score_limit_default = _int_env("PERSONA_LOGPROB_SCORE_CONCURRENCY_PER_SERVER", 1)

    original_compute_score = agent_loop_mod.AgentLoopWorker._compute_score
    original_generate = vLLMHttpServer.generate
    original_launch_servers = vLLMReplica.launch_servers
    original_reward_compute_rm_score = RewardLoopManager.compute_rm_score
    original_score_prompt_tokens = getattr(vLLMHttpServer, "score_prompt_tokens", None)
    original_postprocess = agent_loop_mod.AgentLoopWorker._postprocess
    original_trainer_init = RayPPOTrainer.__init__
    original_init_workers = RayPPOTrainer.init_workers
    original_update_actor = RayPPOTrainer._update_actor

    def _install_epoch_end_checkpointing_hooks(trainer: Any) -> None:
        if getattr(trainer, "_persona_epoch_ckpt_hook_installed", False):
            return

        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if checkpoint_manager is None:
            return

        steps_per_epoch = getattr(trainer, "_persona_steps_per_epoch", None)
        if not steps_per_epoch:
            return

        forced_save_freq = getattr(trainer, "_persona_epoch_aligned_save_freq", None)
        if forced_save_freq is None or not _epoch_end_checkpointing_enabled():
            return

        trainer_cfg = trainer.config.trainer
        configured_save_freq = _config_get(trainer_cfg, "save_freq", None)
        total_epochs = _config_get(trainer_cfg, "total_epochs", None)

        original_save_checkpoint = trainer._save_checkpoint

        def patched_save_checkpoint(*save_args, **save_kwargs):
            result = original_save_checkpoint(*save_args, **save_kwargs)
            trainer._persona_last_saved_global_step = getattr(trainer, "global_steps", None)
            return result

        trainer._save_checkpoint = patched_save_checkpoint

        original_update_weights = checkpoint_manager.update_weights

        def patched_update_weights(global_step, *update_args, **update_kwargs):
            result = original_update_weights(global_step, *update_args, **update_kwargs)
            if not _should_save_epoch_end_checkpoint(trainer.config, global_step, steps_per_epoch):
                return result
            if getattr(trainer, "_persona_last_saved_global_step", None) == int(global_step):
                return result

            print(
                "PERSONA: saving additional epoch-end checkpoint at "
                f"global_step={global_step} with save_freq={configured_save_freq} "
                f"and total_epochs={total_epochs}",
                flush=True,
            )
            trainer._save_checkpoint()
            return result

        checkpoint_manager.update_weights = patched_update_weights
        trainer._persona_epoch_ckpt_hook_installed = True

        print(
            "PERSONA: enabling additional epoch-end checkpointing every "
            f"{forced_save_freq} steps while preserving trainer.save_freq={configured_save_freq} "
            f"because total_epochs={total_epochs}",
            flush=True,
        )

    def patched_trainer_init(self, *args, **kwargs):
        original_trainer_init(self, *args, **kwargs)

        steps_per_epoch = len(self.train_dataloader)
        self._persona_steps_per_epoch = steps_per_epoch
        self._persona_last_saved_global_step = None

        forced_save_freq = _resolve_epoch_aligned_save_freq(self.config, steps_per_epoch)
        if forced_save_freq is None or not _epoch_end_checkpointing_enabled():
            return

        trainer_cfg = self.config.trainer
        configured_save_freq = _config_get(trainer_cfg, "save_freq", None)
        total_epochs = _config_get(trainer_cfg, "total_epochs", None)
        self._persona_epoch_aligned_save_freq = forced_save_freq
        _install_epoch_end_checkpointing_hooks(self)

    def patched_init_workers(self, *args, **kwargs):
        _disable_critic_for_retained_reward_modes(self)
        _maybe_normalize_rollout_parallelism(self.config)
        result = original_init_workers(self, *args, **kwargs)
        _install_epoch_end_checkpointing_hooks(self)
        return result

    def patched_update_actor(self, batch):
        _maybe_attach_elbo_sft_tensors(self, batch)
        return original_update_actor(self, batch)

    async def _score_prompt_tokens_batch_impl(self, requests: list[dict[str, Any]], default_priority: int = 0):
        if not requests:
            return []

        lora_request = None
        if self.lora_as_adapter:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                )

        return await _run_logprob_score_batch_via_engine(
            self.engine,
            requests,
            lora_request=lora_request,
            default_priority=default_priority,
        )

    def _coerce_score_prompt_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], int]:
        payload = dict(kwargs)
        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                payload.update(dict(args[0]))
            else:
                field_names = ("prompt_token_ids", "prompt_token_count", "request_id")
                if len(args) > len(field_names):
                    raise TypeError(
                        "score_prompt_tokens fallback received too many positional arguments: "
                        f"expected at most {len(field_names)}, got {len(args)}"
                    )
                for field_name, value in zip(field_names, args, strict=True):
                    payload.setdefault(field_name, value)

        if "prompt_token_ids" not in payload or "prompt_token_count" not in payload:
            raise TypeError(
                "score_prompt_tokens fallback requires prompt_token_ids and prompt_token_count"
            )

        payload["prompt_token_ids"] = _normalize_prompt_token_ids(payload["prompt_token_ids"])
        payload["prompt_token_count"] = int(payload["prompt_token_count"])
        payload["request_id"] = str(payload.get("request_id") or uuid4().hex)
        priority = int(payload.pop("priority", payload.pop("default_priority", 0)))
        return payload, priority

    async def _call_score_prompt_tokens(self, *args, **kwargs):
        if original_score_prompt_tokens is not None:
            return await original_score_prompt_tokens(self, *args, **kwargs)

        payload, priority = _coerce_score_prompt_request(args, kwargs)
        results = await _score_prompt_tokens_batch_impl(self, [payload], default_priority=priority)
        if len(results) != 1:
            raise RuntimeError(f"Expected one logprob score result, got {len(results)}")
        return results[0]

    async def patched_single_turn_run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        sampling_params = _apply_presence_penalty_to_sampling_params(sampling_params)
        _log_single_turn_sampling_params_once(sampling_params)
        _suppress_fast_tokenizer_pad_advisory(self.tokenizer)
        extra_info = _coerce_mapping(kwargs.get("extra_info"))
        rewritten_messages, extra_info = _maybe_override_prompt_messages_for_runtime_conditioning(extra_info)
        messages = rewritten_messages if rewritten_messages is not None else list(kwargs["raw_prompt"])
        metrics = {}

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        if images or videos:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
            )
        else:
            try:
                prompt_ids, prompt_prefill_stripped = _render_text_prompt_ids(
                    self.tokenizer,
                    messages,
                    prompt_mode=extra_info.get("prompt_mode"),
                )
                _log_prompt_prefill_preserve_once(prompt_prefill_stripped)
            except Exception:
                logger.exception("Failed to render text prompt ids; falling back to veRL apply_chat_template")
                prompt_ids = await self.apply_chat_template(
                    messages,
                    images=images,
                    videos=videos,
                )

        server_sticky_request_id = uuid4().hex
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=server_sticky_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_token_ids, response_log_probs, stripped_think_close_tokens = _strip_hidden_thinking_close_token_prefix(
            self.tokenizer,
            output.token_ids,
            output.log_probs if output.log_probs else None,
        )
        if stripped_think_close_tokens:
            metrics["hidden_thinking_close_prefix_tokens_stripped"] = stripped_think_close_tokens
            _log_hidden_thinking_close_strip_once(stripped_think_close_tokens)
        metric = os.environ.get("REWARD_METRIC", "turing")
        response_mask = _build_response_mask(
            self.tokenizer,
            response_token_ids,
            metric=metric,
            prompt_mode=extra_info.get("prompt_mode"),
        )
        routed_experts = output.routed_experts
        if stripped_think_close_tokens and routed_experts is not None:
            routed_expert_prefix = routed_experts[: len(prompt_ids)]
            routed_expert_suffix = routed_experts[len(prompt_ids) + stripped_think_close_tokens :]
            try:
                routed_experts = routed_expert_prefix + routed_expert_suffix
            except Exception:
                try:
                    import torch

                    routed_experts = torch.cat([routed_expert_prefix, routed_expert_suffix], dim=0)
                except Exception:
                    routed_experts = output.routed_experts

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_log_probs[: self.response_length] if response_log_probs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length]
                if routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=dict(output.extra_fields or {}),
        )
        result.extra_fields["server_sticky_request_id"] = server_sticky_request_id
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result

    async def patched_score_prompt_tokens(self, *args, **kwargs):
        request_limit = getattr(type(self), "_persona_vllm_request_limit_default", request_limit_default)
        score_limit = getattr(type(self), "_persona_logprob_score_limit_default", score_limit_default)
        request_semaphore = _get_server_semaphore(
            self,
            attr_name="_persona_vllm_request_semaphore",
            limit_attr_name="_persona_vllm_request_limit",
            env_name="PERSONA_VLLM_REQUEST_CONCURRENCY_PER_SERVER",
            default=request_limit,
        )
        score_semaphore = _get_server_semaphore(
            self,
            attr_name="_persona_logprob_score_semaphore",
            limit_attr_name="_persona_logprob_score_limit",
            env_name="PERSONA_LOGPROB_SCORE_CONCURRENCY_PER_SERVER",
            default=score_limit,
        )

        if request_semaphore is None:
            if score_semaphore is None:
                return await _call_score_prompt_tokens(self, *args, **kwargs)
            async with score_semaphore:
                return await _call_score_prompt_tokens(self, *args, **kwargs)

        async with request_semaphore:
            if score_semaphore is None:
                return await _call_score_prompt_tokens(self, *args, **kwargs)
            async with score_semaphore:
                return await _call_score_prompt_tokens(self, *args, **kwargs)

    async def patched_score_prompt_tokens_batch(self, requests: list[dict[str, Any]], priority: int = 0):
        request_limit = getattr(type(self), "_persona_vllm_request_limit_default", request_limit_default)
        score_limit = getattr(type(self), "_persona_logprob_score_limit_default", score_limit_default)
        request_semaphore = _get_server_semaphore(
            self,
            attr_name="_persona_vllm_request_semaphore",
            limit_attr_name="_persona_vllm_request_limit",
            env_name="PERSONA_VLLM_REQUEST_CONCURRENCY_PER_SERVER",
            default=request_limit,
        )
        score_semaphore = _get_server_semaphore(
            self,
            attr_name="_persona_logprob_score_semaphore",
            limit_attr_name="_persona_logprob_score_limit",
            env_name="PERSONA_LOGPROB_SCORE_CONCURRENCY_PER_SERVER",
            default=score_limit,
        )

        if request_semaphore is None:
            if score_semaphore is None:
                return await _score_prompt_tokens_batch_impl(self, requests, default_priority=priority)
            async with score_semaphore:
                return await _score_prompt_tokens_batch_impl(self, requests, default_priority=priority)

        async with request_semaphore:
            if score_semaphore is None:
                return await _score_prompt_tokens_batch_impl(self, requests, default_priority=priority)
            async with score_semaphore:
                return await _score_prompt_tokens_batch_impl(self, requests, default_priority=priority)

    async def patched_generate(self, *args, **kwargs):
        request_limit = getattr(type(self), "_persona_vllm_request_limit_default", request_limit_default)
        request_semaphore = _get_server_semaphore(
            self,
            attr_name="_persona_vllm_request_semaphore",
            limit_attr_name="_persona_vllm_request_limit",
            env_name="PERSONA_VLLM_REQUEST_CONCURRENCY_PER_SERVER",
            default=request_limit,
        )
        if request_semaphore is None:
            return await original_generate(self, *args, **kwargs)

        async with request_semaphore:
            return await original_generate(self, *args, **kwargs)

    async def patched_launch_servers(self, *args, **kwargs):
        original_server_class = self.server_class
        self.server_class = _ServerClassRuntimeEnvProxy(original_server_class)
        try:
            return await original_launch_servers(self, *args, **kwargs)
        finally:
            self.server_class = original_server_class

    async def patched_compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        metric = os.environ.get("REWARD_METRIC", "turing")
        output.extra_fields = dict(output.extra_fields or {})

        if output.reward_score is None and metric == "logprob" and _current_policy_logprob_enabled():
            from shared.prompt_utils import (
                build_messages_for_prompt_mode,
                get_chat_template_kwargs_for_prompt_mode,
                build_response_prefill,
                tokenize_with_prefix_boundary,
            )
            from training.grpo.reward import (
                build_format_reward_info,
                build_logprob_reward_result,
                parse_response_for_prompt_mode,
            )

            extra_info = _coerce_mapping(kwargs.get("extra_info"))
            _rewritten_messages, extra_info = _maybe_override_prompt_messages_for_runtime_conditioning(extra_info)
            ground_truth = _extract_logprob_ground_truth(kwargs, extra_info)
            context = str(extra_info.get("context", "") or "")
            user_history = str(extra_info.get("user_history", "") or "")
            persona = str(extra_info.get("persona", extra_info.get("persona_memory", "")) or "")
            prompt_mode = str(extra_info.get("prompt_mode", "reasoning") or "reasoning")
            conditioning_mode = str(extra_info.get("conditioning_mode", "history") or "history")
            prompt_text = extra_info.get("prompt_text")
            user_id = extra_info.get("user_id", "unknown_user")
            post_id = extra_info.get("post_id", "unknown_post")

            if not ground_truth:
                raise RuntimeError(
                    f"Missing ground_truth for current-policy logprob scoring "
                    f"(user_id={user_id} post_id={post_id})"
                )

            apply_kwargs = get_chat_template_kwargs_for_prompt_mode(prompt_mode)
            if not prompt_text or apply_kwargs.get("enable_thinking", False):
                messages = build_messages_for_prompt_mode(
                    user_history=user_history,
                    thread_context=context,
                    prompt_mode=prompt_mode,
                    persona=persona,
                    conditioning_mode=conditioning_mode,
                )
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    **apply_kwargs,
                )

            solution_str = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
            format_reward_info = build_format_reward_info(solution_str, metric, prompt_mode)
            cot, _ = parse_response_for_prompt_mode(solution_str, prompt_mode)
            prefix_text = prompt_text + build_response_prefill(cot)
            full_text = prefix_text + ground_truth
            full_token_ids, prompt_token_count = tokenize_with_prefix_boundary(
                self.tokenizer,
                prefix_text=prefix_text,
                full_text=full_text,
            )
            if prompt_token_count >= len(full_token_ids):
                raise RuntimeError(
                    f"Target tokenization is empty for user_id={user_id} post_id={post_id}"
                )

            sticky_request_id = str(
                (output.extra_fields or {}).get("server_sticky_request_id")
                or extra_info.get("server_sticky_request_id")
                or uuid4().hex
            )
            server_id, server = await self.server_manager._acquire_server(sticky_request_id)
            try:
                score_result = await _get_logprob_score_batcher(self).enqueue(
                    str(server_id),
                    server,
                    {
                        "prompt_token_ids": full_token_ids,
                        "prompt_token_count": prompt_token_count,
                        "request_id": uuid4().hex,
                    },
                )
            finally:
                self.server_manager._release_server(server_id)

            reward_extra_info = build_logprob_reward_result(
                float(score_result["mean_logprob"]),
                num_tokens=int(score_result["num_tokens"]),
                logprob_source="current_policy_rollout",
                format_reward_info=format_reward_info,
            )
            output.reward_score = float(reward_extra_info["score"])
            output.extra_fields["reward_extra_info"] = reward_extra_info
            output.extra_fields.setdefault("server_sticky_request_id", sticky_request_id)
            return

        return await original_compute_score(
            self,
            output,
            prompts,
            responses,
            attention_mask,
            input_ids,
            position_ids,
            kwargs,
        )

    def patched_compute_rm_score(self, data):
        if not _should_route_grouped_sim_rewards(self.config, len(self.reward_loop_workers)):
            return original_reward_compute_rm_score(self, data)

        from training.grpo.sim_reward_manager import stable_group_key_from_extra_info

        global_step = int(data.meta_info.get("global_steps", 0) or 0)
        extra_infos = data.non_tensor_batch.get("extra_info")
        if extra_infos is None or len(extra_infos) != len(data):
            return original_reward_compute_rm_score(self, data)

        group_keys = [stable_group_key_from_extra_info(extra_info, global_step) for extra_info in extra_infos]
        bucket_indices = _partition_indices_by_group_keys(group_keys, len(self.reward_loop_workers))
        active_assignments = [
            (worker_idx, self.reward_loop_workers[worker_idx], indices)
            for worker_idx, indices in enumerate(bucket_indices)
            if indices
        ]
        if not active_assignments:
            return original_reward_compute_rm_score(self, data)

        if not hasattr(self, "_persona_grouped_reward_route_log_count"):
            self._persona_grouped_reward_route_log_count = 0
        if self._persona_grouped_reward_route_log_count < 3:
            logger.info(
                "Grouped sim reward routing: workers=%s bucket_sizes=%s groups=%s",
                len(self.reward_loop_workers),
                [len(indices) for indices in bucket_indices],
                len({key for key in group_keys}),
            )
            self._persona_grouped_reward_route_log_count += 1

        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        try:
            outputs_by_worker = reward_loop_mod.ray.get(
                [worker.compute_score_batch.remote(data[indices]) for _, worker, indices in active_assignments]
            )
            outputs_flat: list[dict[str, Any] | None] = [None] * len(data)
            for (_, _, indices), worker_outputs in zip(active_assignments, outputs_by_worker, strict=True):
                if len(worker_outputs) != len(indices):
                    raise RuntimeError(
                        "Reward worker returned mismatched output size: "
                        f"expected {len(indices)} got {len(worker_outputs)}"
                    )
                for item_idx, output in zip(indices, worker_outputs, strict=True):
                    outputs_flat[item_idx] = output

            if any(output is None for output in outputs_flat):
                missing = [idx for idx, output in enumerate(outputs_flat) if output is None]
                raise RuntimeError(f"Grouped reward routing left outputs unset for indices {missing}")

            scores = [float(item["reward_score"]) for item in outputs_flat]
            prompt_length = data.batch["prompts"].size(1)
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
            rm_scores = reward_loop_mod.torch.zeros_like(data.batch["responses"], dtype=reward_loop_mod.torch.float32)
            rm_scores[reward_loop_mod.torch.arange(rm_scores.size(0)), valid_response_length - 1] = reward_loop_mod.torch.tensor(
                scores,
                dtype=reward_loop_mod.torch.float32,
            )
            batch = reward_loop_mod.TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

            reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
            reward_extra_keys = list(reward_extra_infos[0].keys())
            non_tensor_batch = {}
            for key in reward_extra_keys:
                non_tensor_batch[key] = reward_loop_mod.np.array([info[key] for info in reward_extra_infos])

            return reward_loop_mod.DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                meta_info={"reward_extra_keys": reward_extra_keys},
            )
        finally:
            if self.reward_model_manager is not None:
                self.reward_model_manager.sleep()

    def patched_postprocess(self, inputs, input_non_tensor_batch=None):
        _normalize_reward_extra_info_keys(inputs)
        return original_postprocess(self, inputs, input_non_tensor_batch=input_non_tensor_batch)

    SingleTurnAgentLoop.run = patched_single_turn_run
    RayPPOTrainer.__init__ = patched_trainer_init
    RayPPOTrainer.init_workers = patched_init_workers
    RayPPOTrainer._update_actor = patched_update_actor
    agent_loop_mod.AgentLoopWorker._compute_score = patched_compute_score
    agent_loop_mod.AgentLoopWorker._postprocess = patched_postprocess
    if enable_grouped_sim_route_patch:
        RewardLoopManager.compute_rm_score = patched_compute_rm_score
    vLLMHttpServer._persona_vllm_request_limit_default = request_limit_default
    vLLMHttpServer._persona_logprob_score_limit_default = score_limit_default
    vLLMHttpServer.generate = patched_generate
    vLLMHttpServer.score_prompt_tokens = patched_score_prompt_tokens
    vLLMHttpServer.score_prompt_tokens_batch = patched_score_prompt_tokens_batch
    if enable_runtime_env_propagation:
        vLLMReplica.launch_servers = patched_launch_servers
    agent_loop_mod._persona_runtime_patch_applied = True
    return True
