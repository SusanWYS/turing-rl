"""OpenAI-compatible chat transport."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

try:  # pragma: no cover - exercised in runtime envs
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

from shared.load_env import get_openai_api_base, get_openai_api_key

_OPENAI_MAX_RETRIES_CAP = 3


def get_openai_max_retries(
    *,
    default: int = _OPENAI_MAX_RETRIES_CAP,
    cap: int = _OPENAI_MAX_RETRIES_CAP,
) -> int:
    """Return the bounded retry count."""
    raw_value = os.environ.get("PERSONA_OPENAI_MAX_RETRIES", str(default))
    try:
        configured = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PERSONA_OPENAI_MAX_RETRIES must be an integer, got {raw_value!r}") from exc
    return max(1, min(cap, configured))


def resolve_judge_api_key() -> str:
    """Resolve the judge API key."""
    get_openai_api_key(extra_env_names=("OPENROUTER_API_KEY",))
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def chat_url(api_base: str | None = None) -> str:
    """Return the chat-completions endpoint."""
    return f"{(api_base or get_openai_api_base()).rstrip('/')}/chat/completions"


def openrouter_chat_url() -> str:
    return chat_url()


def openrouter_request_extras(*, reasoning: bool) -> dict[str, Any]:
    """Build OpenRouter routing extras."""
    extras: dict[str, Any] = {}
    provider_order = [
        provider.strip()
        for provider in os.environ.get("OPENROUTER_PROVIDER_ORDER", "Morph").split(",")
        if provider.strip()
    ]
    if provider_order:
        allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "0").strip().lower()
        extras["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks in {"1", "true", "yes", "on"},
        }
    if reasoning:
        extras["reasoning"] = {"enabled": True}
    return extras


def build_chat_payload(
    *,
    model: str,
    messages: list[dict],
    max_completion_tokens: int,
    response_format: dict | None = None,
    reasoning: bool,
) -> dict[str, Any]:
    """Build a chat-completions payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": int(max_completion_tokens),
    }
    if response_format:
        payload["response_format"] = response_format
    payload.update(openrouter_request_extras(reasoning=reasoning))
    return payload


def _compact_openai_error_body(body: str, max_chars: int = 1000) -> str:
    compact = " ".join((body or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def _extract_chat_content(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    raise ValueError("OpenAI response missing choices[0].message.content")


async def post_chat_async(
    session,
    payload: dict,
    *,
    semaphore,
    max_retries: int | None = None,
) -> str:
    """Post a chat request with retries."""
    if aiohttp is None:
        raise ImportError("OpenRouter judge scoring requires aiohttp to be installed")
    api_key = resolve_judge_api_key()
    url = openrouter_chat_url()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if max_retries is None:
        max_retries = get_openai_max_retries()
    retry_sleep_seconds = max(0.0, float(os.environ.get("PERSONA_OPENAI_RETRY_SLEEP_SECONDS", "5")))
    for attempt in range(max_retries):
        try:
            retry_after = None
            async with semaphore:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        body = await resp.text()
                        retry_after_header = resp.headers.get("Retry-After", "")
                        retry_after = retry_sleep_seconds
                        print(
                            "[openai] HTTP 429 rate limited "
                            f"on attempt {attempt+1}/{max_retries}; "
                            f"retry_after={retry_after_header or retry_after}; "
                            f"body={_compact_openai_error_body(body)}",
                            flush=True,
                        )
                    else:
                        if resp.status >= 400:
                            body = await resp.text()
                            print(
                                "[openai] HTTP error "
                                f"status={resp.status} reason={resp.reason!r} "
                                f"on attempt {attempt+1}/{max_retries}; "
                                f"body={_compact_openai_error_body(body)}",
                                flush=True,
                            )
                        resp.raise_for_status()
                        return _extract_chat_content(await resp.json())
            if retry_after is not None:
                await asyncio.sleep(retry_after)
                continue
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            print(f"[openai] {type(exc).__name__} on attempt {attempt+1}/{max_retries}", flush=True)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_sleep_seconds)
    raise RuntimeError(f"OpenAI API call failed after {max_retries} retries")


def post_chat_sync(
    payload: dict,
    *,
    max_retries: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """Post a chat request synchronously."""
    resolved_api_key = api_key or resolve_judge_api_key()
    url = chat_url(api_base)
    headers = {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}
    if max_retries is None:
        max_retries = get_openai_max_retries()
    retry_sleep_seconds = max(0.0, float(os.environ.get("PERSONA_OPENAI_RETRY_SLEEP_SECONDS", "5")))
    timeout = float(os.getenv("PERSONA_OPENAI_TIMEOUT_SECONDS", "180"))
    data_bytes = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return _extract_chat_content(json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retry_after = (
                exc.headers.get("Retry-After", "")
                if exc.headers else retry_sleep_seconds
            )
            print(
                f"[openai] HTTP error status={exc.code} on attempt {attempt+1}/{max_retries}; "
                f"retry_after={retry_after}; body={_compact_openai_error_body(body)}",
                flush=True,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_sleep_seconds)
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            print(f"[openai] {type(exc).__name__} on attempt {attempt+1}/{max_retries}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"OpenAI API call failed after {max_retries} retries")
