"""Helpers for loading local environment variables."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def _candidate_env_paths(env_path: str | os.PathLike[str] | None = None) -> list[Path]:
    if env_path is not None:
        return [Path(env_path).expanduser()]
    repo_root = Path(__file__).resolve().parents[1]
    env_file = os.environ.get("ENV_FILE")
    if env_file:
        return [Path(env_file).expanduser()]
    candidates = [repo_root / ".env", Path.home() / ".env"]

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _format_env_paths(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def _missing_api_key_message(env_names: tuple[str, ...]) -> str:
    env_paths = _candidate_env_paths()
    key_label = "/".join(env_names)
    return (
        f"Missing {key_label} in .env file. Expected one of: "
        f"{_format_env_paths(env_paths)}"
    )


def load_local_env(
    env_path: str | os.PathLike[str] | None = None,
    *,
    override_keys: Iterable[str] = (),
) -> None:
    """Load local .env entries."""
    override_key_set = set(override_keys)
    env_paths = _candidate_env_paths(env_path)
    env_path = next((path for path in env_paths if path.is_file()), None)

    if env_path is None:
        raise FileNotFoundError(
            "Missing .env file. Expected one of: "
            f"{_format_env_paths(env_paths)}"
        )

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if key in os.environ and key not in override_key_set:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def _first_env_value(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def get_openai_api_key(*, extra_env_names: tuple[str, ...] = ()) -> str:
    """Load an API key."""
    env_names = ("OPENAI_API_KEY", *extra_env_names)
    try:
        load_local_env(override_keys=env_names)
    except FileNotFoundError as exc:
        raise FileNotFoundError(_missing_api_key_message(env_names)) from exc
    api_key = _first_env_value(env_names)
    if not api_key:
        raise RuntimeError(_missing_api_key_message(env_names))
    return api_key


def get_openai_api_base(
    default: str = DEFAULT_OPENAI_API_BASE,
    *,
    extra_env_names: tuple[str, ...] = (),
) -> str:
    """Return the API base URL."""
    env_names = ("OPENAI_API_BASE", *extra_env_names)
    api_base = _first_env_value(env_names)
    if not api_base:
        load_local_env()
        api_base = _first_env_value(env_names)
    return (api_base or default).rstrip("/")
