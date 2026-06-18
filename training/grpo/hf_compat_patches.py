"""Hugging Face / PEFT compatibility patches."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any

_PATCH_ATTR = "_persona_init_on_device_parameter_compat_patch_applied"
_ORIGINAL_ATTR = "_persona_original_init_on_device"


def _split_parameter_attrs_for_constructor(
    param_cls: type[Any],
    attrs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        signature = inspect.signature(param_cls)
    except (TypeError, ValueError):
        return {}, dict(attrs)

    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    allowed_kwargs = {
        name
        for name, parameter in signature.parameters.items()
        if name != "data" and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }

    constructor_kwargs: dict[str, Any] = {}
    post_init_attrs: dict[str, Any] = {}
    for name, value in attrs.items():
        if accepts_var_keyword or name in allowed_kwargs:
            constructor_kwargs[name] = value
        else:
            post_init_attrs[name] = value
    return constructor_kwargs, post_init_attrs


def _rebuild_parameter_on_device(param: Any, *, device: Any, requires_grad: bool) -> Any:
    param_cls = type(param)
    constructor_kwargs, post_init_attrs = _split_parameter_attrs_for_constructor(
        param_cls,
        dict(getattr(param, "__dict__", {})),
    )
    constructor_kwargs["requires_grad"] = requires_grad
    rebuilt_param = param_cls(param.to(device), **constructor_kwargs)
    for attr_name, attr_value in post_init_attrs.items():
        setattr(rebuilt_param, attr_name, attr_value)
    return rebuilt_param


def apply_accelerate_init_on_device_parameter_compat_patch() -> bool:
    """Patch accelerate.init_on_device for custom Parameter subclasses."""
    try:
        import accelerate
        import accelerate.big_modeling as big_modeling
    except ImportError:
        return False

    if getattr(big_modeling, _PATCH_ATTR, False):
        return True

    @contextmanager
    def patched_init_on_device(device, include_buffers=None):
        if include_buffers is None:
            include_buffers = big_modeling.parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

        if include_buffers:
            with device:
                yield
            return

        old_register_parameter = big_modeling.nn.Module.register_parameter

        def register_empty_parameter(module, name, param):
            old_register_parameter(module, name, param)
            if param is not None:
                module._parameters[name] = _rebuild_parameter_on_device(
                    module._parameters[name],
                    device=device,
                    requires_grad=param.requires_grad,
                )

        try:
            big_modeling.nn.Module.register_parameter = register_empty_parameter
            yield
        finally:
            big_modeling.nn.Module.register_parameter = old_register_parameter

    setattr(big_modeling, _ORIGINAL_ATTR, big_modeling.init_on_device)
    big_modeling.init_on_device = patched_init_on_device
    accelerate.init_on_device = patched_init_on_device
    setattr(big_modeling, _PATCH_ATTR, True)
    return True


def apply_peft_tensor_parallel_compat_patch() -> None:
    """Keep PEFT 0.19 adapter loading compatible with Transformers builds lacking EmbeddingParallel."""
    try:
        from transformers.integrations import tensor_parallel as tensor_parallel
    except ImportError:
        return

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return

    class _UnavailableEmbeddingParallel:
        pass

    tensor_parallel.EmbeddingParallel = _UnavailableEmbeddingParallel


def apply_hf_compat_patches() -> None:
    """Apply all Hugging Face / PEFT compatibility shims."""
    apply_accelerate_init_on_device_parameter_compat_patch()
    apply_peft_tensor_parallel_compat_patch()
