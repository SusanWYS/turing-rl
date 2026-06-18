from __future__ import annotations

import importlib
import os
from typing import Any

from training.grpo.hf_compat_patches import apply_hf_compat_patches
from training.grpo.verl_runtime_patch import apply_verl_runtime_patch
from training.grpo.verl_metric_patch import install_verl_metric_patch_in_ray_worker

_UPSTREAM_SETUP_HOOK_ENV = "PERSONA_UPSTREAM_WORKER_PROCESS_SETUP_HOOK"
_FALLBACK_VISIBLE_DEVICE_ORDINAL_ENV = "PERSONA_FALLBACK_VISIBLE_DEVICE_ORDINAL"
_STRICT_DEVICE_SET_ENV = "PERSONA_STRICT_RAY_WORKER_SETUP_DEVICE_SET"


def _run_worker_process_setup_hook_from_string(hook_path: str) -> None:
    module_name, separator, attr_name = hook_path.rpartition(".")
    if not module_name or not separator or not attr_name:
        raise ValueError(f"worker_process_setup_hook must be importable as module.attr, got {hook_path!r}")
    module = importlib.import_module(module_name)
    getattr(module, attr_name)()


def _parse_visible_devices(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def _normalize_device_token(token: str) -> str:
    normalized = token.strip()
    try:
        return str(int(float(normalized)))
    except ValueError:
        return normalized


def _resolve_logical_local_rank(visible_devices: list[str], assigned_accelerator_id: str) -> int | None:
    if not visible_devices:
        return None

    normalized_assigned_id = _normalize_device_token(assigned_accelerator_id)
    normalized_visible_devices = [_normalize_device_token(token) for token in visible_devices]

    for logical_rank, token in enumerate(normalized_visible_devices):
        if token == normalized_assigned_id:
            return logical_rank

    if len(visible_devices) == 1:
        return 0

    return None


def _worker_setup_logging_enabled() -> bool:
    return os.environ.get("PERSONA_LOG_RAY_WORKER_SETUP", "1") != "0"


def _strict_device_set_enabled() -> bool:
    return os.environ.get(_STRICT_DEVICE_SET_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _set_local_rank_and_device(
    logical_local_rank: int,
    local_world_size: str | None,
    visible_devices_key: str,
    *,
    set_torch_device: bool = True,
) -> dict[str, Any]:
    os.environ["LOCAL_RANK"] = str(logical_local_rank)
    if os.environ.get("LOCAL_WORLD_SIZE") is None and local_world_size is not None:
        os.environ["LOCAL_WORLD_SIZE"] = local_world_size

    result = {"torch_device_set": False, "torch_device_error": None}
    if not set_torch_device:
        return result

    try:
        from verl.utils.device import get_torch_device
    except ImportError:
        return result

    torch_device = get_torch_device()
    set_device = getattr(torch_device, "set_device", None)
    if callable(set_device):
        try:
            set_device(logical_local_rank)
            result["torch_device_set"] = True
        except Exception as exc:
            result["torch_device_error"] = f"{type(exc).__name__}: {exc}"
            if _strict_device_set_enabled():
                raise
            if _worker_setup_logging_enabled():
                print(
                    "[persona ray worker pin] "
                    f"pid={os.getpid()} "
                    f"skipped torch device set for {visible_devices_key}={os.environ.get(visible_devices_key)} "
                    f"local_rank={logical_local_rank}: {result['torch_device_error']}",
                    flush=True,
                )
    return result


def _apply_fallback_visible_device_ordinal(
    *,
    device_name: str,
    accelerator_ids: dict[str, list[Any]],
    fallback_visible_device_ordinal: str,
    visible_devices_key: str,
    local_world_size: str | None,
) -> dict[str, Any]:
    normalized_visible_device = _normalize_device_token(fallback_visible_device_ordinal)
    os.environ[visible_devices_key] = normalized_visible_device
    # In fallback mode each Ray worker is isolated to one physical GPU via
    # CUDA_VISIBLE_DEVICES. Keep the local namespace consistent with that view;
    # leaving LOCAL_WORLD_SIZE=8 here has caused fragile NCCL/FSDP init behavior.
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    device_set_state = _set_local_rank_and_device(
        logical_local_rank=0,
        local_world_size="1",
        visible_devices_key=visible_devices_key,
        set_torch_device=False,
    )
    return {
        "device_name": device_name,
        "accelerator_ids": accelerator_ids,
        "assigned_accelerator_id": None,
        "visible_devices_key": visible_devices_key,
        "visible_devices": [normalized_visible_device],
        "logical_local_rank": 0,
        "local_world_size": os.environ.get("LOCAL_WORLD_SIZE") or "1",
        "assignment_source": "fallback_visible_device_ordinal",
        **device_set_state,
    }


def _pin_worker_to_assigned_accelerator() -> dict[str, Any] | None:
    try:
        import ray
        from verl.utils.device import get_torch_device, get_visible_devices_keyword, is_npu_available
    except ImportError:
        return None

    device_name = "NPU" if is_npu_available else "GPU"
    accelerator_ids: dict[str, list[Any]] = ray.get_runtime_context().get_accelerator_ids()
    assigned_ids = accelerator_ids.get(device_name) or []
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("RAY_LOCAL_WORLD_SIZE")
    if not assigned_ids:
        visible_devices_key = get_visible_devices_keyword()
        fallback_visible_device_ordinal = os.environ.get(_FALLBACK_VISIBLE_DEVICE_ORDINAL_ENV)
        if fallback_visible_device_ordinal:
            return _apply_fallback_visible_device_ordinal(
                device_name=device_name,
                accelerator_ids=accelerator_ids,
                fallback_visible_device_ordinal=fallback_visible_device_ordinal,
                visible_devices_key=visible_devices_key,
                local_world_size=local_world_size,
            )
        return {
            "device_name": device_name,
            "accelerator_ids": accelerator_ids,
            "assigned_accelerator_id": None,
            "visible_devices_key": None,
            "visible_devices": [],
            "logical_local_rank": None,
            "local_world_size": local_world_size,
            "assignment_source": "unassigned",
        }

    assigned_accelerator_id = str(assigned_ids[0])
    visible_devices_key = get_visible_devices_keyword()
    visible_devices = _parse_visible_devices(os.environ.get(visible_devices_key))
    logical_local_rank = _resolve_logical_local_rank(visible_devices, assigned_accelerator_id)

    if logical_local_rank is None:
        os.environ[visible_devices_key] = assigned_accelerator_id
        visible_devices = [assigned_accelerator_id]
        logical_local_rank = 0

    device_set_state = _set_local_rank_and_device(
        logical_local_rank=logical_local_rank,
        local_world_size=local_world_size,
        visible_devices_key=visible_devices_key,
    )

    return {
        "device_name": device_name,
        "accelerator_ids": accelerator_ids,
        "assigned_accelerator_id": assigned_accelerator_id,
        "visible_devices_key": visible_devices_key,
        "visible_devices": visible_devices,
        "logical_local_rank": logical_local_rank,
        "local_world_size": os.environ.get("LOCAL_WORLD_SIZE") or local_world_size,
        "assignment_source": "ray_accelerator_ids",
        **device_set_state,
    }


def _log_worker_assignment_state(state: dict[str, Any] | None) -> None:
    if not _worker_setup_logging_enabled():
        return
    if state is None:
        print(f"[persona ray worker pin] pid={os.getpid()} state=unavailable", flush=True)
        return
    print(
        "[persona ray worker pin] "
        f"pid={os.getpid()} "
        f"rank={os.environ.get('RANK')} "
        f"world_size={os.environ.get('WORLD_SIZE')} "
        f"local_rank={os.environ.get('LOCAL_RANK')} "
        f"local_world_size={state.get('local_world_size')} "
        f"device_name={state.get('device_name')} "
        f"source={state.get('assignment_source')} "
        f"assigned_accelerator_id={state.get('assigned_accelerator_id')} "
        f"torch_device_set={state.get('torch_device_set')} "
        f"torch_device_error={state.get('torch_device_error')} "
        f"accelerator_ids={state.get('accelerator_ids')} "
        f"{state.get('visible_devices_key')}={','.join(state.get('visible_devices') or [])}",
        flush=True,
    )


def persona_worker_process_setup() -> None:
    apply_hf_compat_patches()
    initial_state = _pin_worker_to_assigned_accelerator()
    upstream_hook = os.environ.get(_UPSTREAM_SETUP_HOOK_ENV, "").strip()
    if upstream_hook and upstream_hook != f"{__name__}.persona_worker_process_setup":
        _run_worker_process_setup_hook_from_string(upstream_hook)
    final_state = _pin_worker_to_assigned_accelerator()
    _log_worker_assignment_state(final_state or initial_state)
    install_verl_metric_patch_in_ray_worker()
    apply_verl_runtime_patch()
