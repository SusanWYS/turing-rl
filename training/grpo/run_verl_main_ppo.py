from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy

from training.grpo.hf_compat_patches import apply_accelerate_init_on_device_parameter_compat_patch
from training.grpo.verl_metric_patch import apply_verl_metric_patch
from training.grpo.verl_runtime_patch import apply_verl_runtime_patch


def _patch_verl_main_ppo_secret_logging() -> None:
    if os.environ.get("PERSONA_PATCH_VERL_INSTALLED_SOURCE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    spec = importlib.util.find_spec("verl.trainer.main_ppo")
    if spec is None or spec.origin is None:
        return
    path = Path(spec.origin)
    try:
        text = path.read_text()
    except OSError:
        return

    old = '        print(f"ray init kwargs: {ray_init_kwargs}")\n'
    if old not in text or "_persona_redact_ray_init_kwargs" in text:
        return

    new = """        def _persona_redact_ray_init_kwargs(value):
            if isinstance(value, dict):
                redacted = {}
                for key, item in value.items():
                    key_text = str(key).upper()
                    if any(secret in key_text for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                        redacted[key] = "<redacted>"
                    else:
                        redacted[key] = _persona_redact_ray_init_kwargs(item)
                return redacted
            if isinstance(value, list):
                return [_persona_redact_ray_init_kwargs(item) for item in value]
            return value

        print(f"ray init kwargs: {_persona_redact_ray_init_kwargs(OmegaConf.to_container(ray_init_kwargs, resolve=True))}")
"""
    try:
        path.write_text(text.replace(old, new))
    except OSError:
        return


def main() -> None:
    apply_accelerate_init_on_device_parameter_compat_patch()
    apply_verl_metric_patch()
    apply_verl_runtime_patch()
    _patch_verl_main_ppo_secret_logging()
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
