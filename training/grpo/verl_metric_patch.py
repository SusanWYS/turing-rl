from __future__ import annotations

import importlib.util
import os
import re
from typing import Any

import numpy as np

_BASE_TRACKING_METRICS = [
    "reward/score/mean",
    "reward/raw_reward/mean",
    "reward/unadjusted_raw_reward/mean",
    "reward/adjusted_raw_reward/mean",
    "reward/length_generated_words/mean",
    "reward/length_human_words/mean",
    "reward/length_ratio/mean",
    "reward/length_relative_diff/mean",
    "reward/length_shortfall_relative/mean",
    "reward/length_excess_relative/mean",
    "reward/length_short_deadband_violation/mean",
    "reward/length_long_deadband_violation/mean",
    "reward/length_short_penalty/mean",
    "reward/length_long_penalty/mean",
    "reward/length_penalty/mean",
    "reward/meaningful_thinking_rate/mean",
    "reward/sim_response/mean",
    "reward/turing_response/mean",
    "reward/mixed_response/mean",
    "reward/format/mean",
    "reward/format_score/mean",
    "reward/format_bonus/mean",
    "reward/format_human_prefix/mean",
    "reward/format_human_prefix/rate",
    "reward/format_nonempty_reasoning/mean",
    "reward/format_nonempty_reasoning/rate",
    "reward/format_no_post_human_thinking/mean",
    "reward/format_no_post_human_thinking/rate",
    "reward/format_reasoning_schema/mean",
    "reward/format_reasoning_schema/rate",
    "reward/total_score/mean",
    "reward/source_copy/mean",
    "reward/source_copy/rate",
    "reward/source_copy_rate/mean",
    "reward/wrong_perspective_rate/mean",
    "reward/assistant_like_response_rate/mean",
    "reward/unjustified_code_switching_response_rate/mean",
    "reward/wrong_target_or_role_response_rate/mean",
    "reward/unsupported_adversarial_reframing_response_rate/mean",
    "reward/train/response/score/mean",
    "reward/train/turing_response/score/mean",
    "reward/train/sim_response/score/mean",
    "reward/train/mixed_response/score/mean",
    "critic/rewards/mean",
    "critic/score/mean",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/grad_norm",
    "actor/elbo_sft_loss",
    "critic/advantages/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "training/global_step",
    "training/epoch",
]

_TIMING_TRACKING_METRICS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/reward",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/values",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/save_checkpoint",
    "timing_s/update_critic",
    "timing_per_token_ms/gen",
    "timing_per_token_ms/ref",
    "timing_per_token_ms/values",
    "timing_per_token_ms/adv",
    "timing_per_token_ms/update_actor",
    "timing_per_token_ms/update_critic",
]

_PERF_TRACKING_METRICS = [
    "perf/total_num_tokens",
    "perf/time_per_step",
    "perf/throughput",
]

_TRAIN_SCORE_KEY_RE = re.compile(r"^train/(?P<name>[^:]+):score$")
_REWARD_COMPONENT_LOG_EMITTED = False

_REWARD_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "score": ("score",),
    "raw_reward": ("raw_reward",),
    "unadjusted_raw_reward": ("unadjusted_raw_reward",),
    "adjusted_raw_reward": ("adjusted_raw_reward",),
    "length_generated_words": ("length_generated_words",),
    "length_human_words": ("length_human_words",),
    "length_ratio": ("length_ratio",),
    "length_relative_diff": ("length_relative_diff",),
    "length_shortfall_relative": ("length_shortfall_relative",),
    "length_excess_relative": ("length_excess_relative",),
    "length_short_deadband_violation": ("length_short_deadband_violation",),
    "length_long_deadband_violation": ("length_long_deadband_violation",),
    "length_short_penalty": ("length_short_penalty",),
    "length_long_penalty": ("length_long_penalty",),
    "length_penalty": ("length_penalty",),
    "meaningful_thinking": ("meaningful_thinking",),
    "sim_response": ("sim_response",),
    "turing_response": ("turing_response",),
    "mixed_response": ("mixed_response",),
    "format": ("format",),
    "format_score": ("format_score",),
    "format_bonus": ("format_bonus", "format_score", "format"),
    "format_human_prefix": ("format_human_prefix",),
    "format_nonempty_reasoning": ("format_nonempty_reasoning",),
    "format_no_post_human_thinking": ("format_no_post_human_thinking",),
    "format_reasoning_schema": ("format_reasoning_schema",),
    "total_score": ("total_score", "score"),
    "source_copy": ("source_copy",),
    "wrong_perspective": ("wrong_perspective",),
    "assistant_like_response": ("assistant_like_response",),
    "unjustified_code_switching_response": ("unjustified_code_switching_response",),
    "wrong_target_or_role_response": ("wrong_target_or_role_response",),
    "unsupported_adversarial_reframing_response": ("unsupported_adversarial_reframing_response",),
}

_RATE_METRIC_NAMES = {
    "meaningful_thinking",
    "source_copy",
    "wrong_perspective",
    "assistant_like_response",
    "unjustified_code_switching_response",
    "wrong_target_or_role_response",
    "unsupported_adversarial_reframing_response",
    "format_human_prefix",
    "format_nonempty_reasoning",
    "format_no_post_human_thinking",
    "format_reasoning_schema",
}

_LEGACY_RATE_MEAN_NAMES = {
    "meaningful_thinking": "meaningful_thinking_rate",
    "source_copy": "source_copy_rate",
    "wrong_perspective": "wrong_perspective_rate",
    "assistant_like_response": "assistant_like_response_rate",
    "unjustified_code_switching_response": "unjustified_code_switching_response_rate",
    "wrong_target_or_role_response": "wrong_target_or_role_response_rate",
    "unsupported_adversarial_reframing_response": "unsupported_adversarial_reframing_response_rate",
}


def build_tracking_metric_allowlist() -> list[str]:
    return list(dict.fromkeys(_BASE_TRACKING_METRICS + _TIMING_TRACKING_METRICS + _PERF_TRACKING_METRICS))


def _coerce_numeric_array(values: Any) -> np.ndarray | None:
    values = np.asarray(values)
    if values.size == 0:
        return None

    try:
        return values.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return None


def _extract_reward_extra_info_values(reward_extra_info: Any, key: str) -> np.ndarray | None:
    if reward_extra_info is None:
        return None

    if isinstance(reward_extra_info, dict):
        return _coerce_numeric_array(reward_extra_info.get(key))

    if isinstance(reward_extra_info, np.ndarray):
        reward_extra_info = reward_extra_info.tolist()

    if isinstance(reward_extra_info, (list, tuple)):
        collected: list[float] = []
        for item in reward_extra_info:
            if not isinstance(item, dict) or key not in item:
                return None
            collected.append(item[key])
        return _coerce_numeric_array(collected)

    return None


def _extract_reward_metric_values(non_tensor_batch: dict[str, Any], key: str) -> np.ndarray | None:
    if key in non_tensor_batch:
        return _coerce_numeric_array(non_tensor_batch[key])

    reward_extra_info = non_tensor_batch.get("reward_extra_info")
    return _extract_reward_extra_info_values(reward_extra_info, key)


def _collect_reward_metric_names(non_tensor_batch: dict[str, Any]) -> set[str]:
    metric_names = set(_REWARD_KEY_ALIASES)
    for key in non_tensor_batch:
        key = str(key)
        if key.startswith("format_") or key.startswith("length_"):
            metric_names.add(key)
    reward_extra_info = non_tensor_batch.get("reward_extra_info")
    if isinstance(reward_extra_info, dict):
        for key in reward_extra_info:
            key = str(key)
            if key.startswith("format_") or key.startswith("length_"):
                metric_names.add(key)
    return metric_names


def append_custom_reward_metrics(metrics: dict[str, Any], batch: Any) -> None:
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if non_tensor_batch is None or not hasattr(non_tensor_batch, "get"):
        return

    score_mean: float | None = None
    total_score_mean: float | None = None
    raw_reward_mean: float | None = None

    for metric_name in sorted(_collect_reward_metric_names(non_tensor_batch)):
        candidate_keys = _REWARD_KEY_ALIASES.get(metric_name, (metric_name,))
        values = None
        for key in candidate_keys:
            values = _extract_reward_metric_values(non_tensor_batch, key)
            if values is not None:
                break
        if values is None:
            continue

        metric_mean = float(np.mean(values))
        metrics[f"reward/{metric_name}/mean"] = metric_mean

        if metric_name == "raw_reward":
            raw_reward_mean = metric_mean
        elif metric_name == "score":
            score_mean = metric_mean
        elif metric_name == "total_score":
            total_score_mean = metric_mean

        if metric_name in _RATE_METRIC_NAMES:
            metrics[f"reward/{metric_name}/rate"] = metric_mean
            legacy_metric_name = _LEGACY_RATE_MEAN_NAMES.get(metric_name)
            if legacy_metric_name is not None:
                metrics[f"reward/{legacy_metric_name}/mean"] = metric_mean

    if raw_reward_mean is not None:
        metrics["critic/rewards/mean"] = raw_reward_mean
    if total_score_mean is not None:
        metrics["critic/score/mean"] = total_score_mean
    elif score_mean is not None:
        metrics["critic/score/mean"] = score_mean

    for key, values in non_tensor_batch.items():
        match = _TRAIN_SCORE_KEY_RE.match(str(key))
        if not match:
            continue

        values = _coerce_numeric_array(values)
        if values is None:
            continue

        stage_name = match.group("name")
        active_key = f"train/active/{stage_name}"
        active_values = non_tensor_batch.get(active_key)
        if active_values is not None:
            active_values = _coerce_numeric_array(active_values)
            if active_values is not None and active_values.shape == values.shape:
                active_mask = active_values > 0.0
                if np.any(active_mask):
                    values = values[active_mask]
                else:
                    continue

        metrics[f"reward/train/{stage_name}/score/mean"] = float(np.mean(values))


def maybe_emit_reward_component_log(metrics: dict[str, Any]) -> None:
    global _REWARD_COMPONENT_LOG_EMITTED

    if _REWARD_COMPONENT_LOG_EMITTED:
        return
    if os.environ.get("PERSONA_LOG_REWARD_COMPONENT_METRICS", "1") == "0":
        return

    keys = [
        "reward/sim_response/mean",
        "reward/turing_response/mean",
        "reward/mixed_response/mean",
        "reward/format_score/mean",
        "reward/format_bonus/mean",
        "reward/source_copy/rate",
        "reward/length_penalty/mean",
    ]
    available_parts = [f"{key}={float(metrics[key]):.6f}" for key in keys if key in metrics]
    if not available_parts:
        return

    print("[persona reward metric patch] " + " ".join(available_parts), flush=True)
    _REWARD_COMPONENT_LOG_EMITTED = True


def wrap_compute_data_metrics(compute_data_metrics_fn: Any) -> Any:
    def _wrapped_compute_data_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        metrics = compute_data_metrics_fn(*args, **kwargs)
        batch = kwargs.get("batch")
        if batch is None and args:
            batch = args[0]
        if batch is not None and hasattr(batch, "non_tensor_batch"):
            append_custom_reward_metrics(metrics, batch)
            maybe_emit_reward_component_log(metrics)
        return metrics

    return _wrapped_compute_data_metrics


def wrap_reward_metric_appender(append_reward_metrics_fn: Any) -> Any:
    def _wrapped_append_reward_metrics(metrics: dict[str, Any], batch: Any, *args: Any, **kwargs: Any) -> Any:
        result = append_reward_metrics_fn(metrics, batch, *args, **kwargs)
        if batch is not None and hasattr(batch, "non_tensor_batch"):
            append_custom_reward_metrics(metrics, batch)
            maybe_emit_reward_component_log(metrics)
        return result

    return _wrapped_append_reward_metrics


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _emit_metric_patch_debug(line: str) -> None:
    if not _bool_env("PERSONA_LOG_METRIC_PATCH_DEBUG", False):
        return
    print(line, flush=True)


def persona_metric_patch_after_compute_data_metrics(metrics: dict[str, Any], batch: Any) -> dict[str, Any]:
    append_custom_reward_metrics(metrics, batch)
    metrics["persona_debug/custom_metric_patch_applied"] = 1.0
    _emit_metric_patch_debug(
        "[persona metric patch] compute_data_metrics "
        + ",".join(sorted(key for key in metrics if key.startswith("reward/")))
    )
    return metrics


def persona_metric_patch_before_tracking_log(data: dict[str, Any], step: Any) -> dict[str, Any]:
    _emit_metric_patch_debug(
        f"[persona metric patch] tracking.log step={step} "
        + ",".join(sorted(key for key in data if key.startswith("reward/")))
    )
    return data


def _patch_installed_trainer_source(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return False

    path = spec.origin
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return False

    if "persona_metric_patch_after_compute_data_metrics" in text:
        return True

    import_anchor = "from verl.utils.rollout_skip import RolloutSkip\n"
    import_patch = (
        import_anchor
        + "from training.grpo.verl_metric_patch import "
        + "persona_metric_patch_after_compute_data_metrics, persona_metric_patch_before_tracking_log\n"
    )
    if import_anchor in text:
        text = text.replace(import_anchor, import_patch, 1)

    compute_pattern = "\n                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))"
    if compute_pattern in text:
        text = text.replace(
            compute_pattern,
            compute_pattern
            + "\n                persona_metric_patch_after_compute_data_metrics(metrics=metrics, batch=batch)",
            1,
        )

    compute_pattern_self = "\n        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))"
    if compute_pattern_self in text:
        text = text.replace(
            compute_pattern_self,
            compute_pattern_self
            + "\n        persona_metric_patch_after_compute_data_metrics(metrics=metrics, batch=batch)",
            1,
        )

    self_logger_pattern = "\n        self.logger.log(data=metrics, step=self.global_steps)"
    if self_logger_pattern in text:
        text = text.replace(
            self_logger_pattern,
            "\n        metrics = persona_metric_patch_before_tracking_log(data=metrics, step=self.global_steps)"
            + self_logger_pattern,
            1,
        )

    logger_pattern = "\n                logger.log(data=metrics, step=self.global_steps)"
    if logger_pattern in text:
        text = text.replace(
            logger_pattern,
            "\n                metrics = persona_metric_patch_before_tracking_log(data=metrics, step=self.global_steps)"
            + logger_pattern,
            1,
        )

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        return False
    return True


def _patch_installed_verl_metric_sources() -> None:
    if os.environ.get("PERSONA_PATCH_VERL_INSTALLED_SOURCE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    patched = []
    for module_name in (
        "verl.trainer.ppo.ray_trainer",
        "verl.experimental.separation.ray_trainer",
    ):
        if _patch_installed_trainer_source(module_name):
            patched.append(module_name)

    if patched:
        _emit_metric_patch_debug("[persona metric patch] installed_source_patched=" + ",".join(patched))


def apply_verl_metric_patch() -> bool:
    try:
        from verl.trainer.ppo import metric_utils
        from verl.trainer.ppo import ray_trainer
        from verl.utils import tracking as tracking_mod
    except ImportError:
        return False

    if getattr(ray_trainer, "_persona_metric_patch_applied", False):
        return True

    _patch_installed_verl_metric_sources()

    original_compute_data_metrics = ray_trainer.compute_data_metrics
    wrapped_compute_data_metrics = wrap_compute_data_metrics(original_compute_data_metrics)
    ray_trainer.compute_data_metrics = wrapped_compute_data_metrics
    metric_utils.compute_data_metrics = wrapped_compute_data_metrics

    if hasattr(ray_trainer, "_append_custom_reward_metrics"):
        original_append_reward_metrics = ray_trainer._append_custom_reward_metrics
        ray_trainer._append_custom_reward_metrics = wrap_reward_metric_appender(original_append_reward_metrics)
        ray_trainer._persona_original_append_custom_reward_metrics = original_append_reward_metrics

    original_tracking_log = tracking_mod.Tracking.log

    def patched_tracking_log(self: Any, data: dict[str, Any], step: Any, backend=None) -> None:
        data = persona_metric_patch_before_tracking_log(data=data, step=step)
        return original_tracking_log(self, data=data, step=step, backend=backend)

    tracking_mod.Tracking.log = patched_tracking_log
    ray_trainer._persona_metric_patch_applied = True
    ray_trainer._persona_original_compute_data_metrics = original_compute_data_metrics
    _emit_metric_patch_debug("[persona metric patch] apply_verl_metric_patch applied=1")
    return True


def install_verl_metric_patch_in_ray_worker() -> None:
    patched = apply_verl_metric_patch()
    if os.environ.get("PERSONA_LOG_RAY_WORKER_SETUP", "1") != "0":
        print(
            f"[persona ray worker setup] pid={os.getpid()} metric_patch_applied={patched}",
            flush=True,
        )


def main() -> int:
    print(",".join(build_tracking_metric_allowlist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
