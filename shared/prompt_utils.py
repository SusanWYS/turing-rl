"""Shared prompt and parsing utilities for retained GRPO and baseline paths."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONDITIONING_MODE_HISTORY = "history"
CONDITIONING_MODE_PERSONA = "persona"
CONDITIONING_MODE_HISTORY_PERSONA = "history_persona"
CONDITIONING_MODE_CHOICES = (
    CONDITIONING_MODE_HISTORY,
    CONDITIONING_MODE_PERSONA,
    CONDITIONING_MODE_HISTORY_PERSONA,
)

SYSTEM_PROMPT_DIR = Path(__file__).resolve().parent / "system_prompts"
ENABLE_THINKING_OVERRIDE_ENV = "PERSONA_ENABLE_THINKING"

_HUMAN_TAG_RE = re.compile(r"\[\s*human\s*\]\s*:", re.IGNORECASE)
_EXACT_HUMAN_PREFIX_RE = re.compile(r"(?im)^\s*\[HUMAN\]:\s*")
_RESPONSE_MARKER_RE = re.compile(
    r"(?im)^\s*(?:(?:\(\s*response\s*\)|<\s*human\s*>)(?:\s*:)?|"
    r"\[\s*[a-z][a-z0-9 _-]{0,40}\s*[\]\}](?:\s*:)?|human\s*:)\s*"
)
_POST_HUMAN_REASONING_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:\*{0,2}\s*reasoning\s*\*{0,2}\s*:|"
    r"##\s*(?:verbal style|personal connection|stance|response type)\b)"
)
_REASONING_TAG_RE = re.compile(r"(?is)<\s*/?\s*reasoning\b")
_ANY_REASONING_TAG_RE = re.compile(r"(?is)<\s*/?\s*(?:reasoning|think)\b")
_REASONING_OPEN_TAG_RE = re.compile(r"(?is)<\s*reasoning\s*>")
_REASONING_CLOSE_TAG_RE = re.compile(r"(?is)</\s*reasoning\s*>")
_HIDDEN_THINKING_CLOSE_TAG_RE = re.compile(r"(?is)</\s*think\s*>")
_XML_TAG_RE = re.compile(r"(?is)</?\s*[a-zA-Z][^>]*>")
_XML_TAG_NAME_RE = re.compile(r"(?is)</?\s*([a-zA-Z][a-zA-Z0-9_-]*)")
_ALPHABETIC_TOKEN_RE = re.compile(r"(?u)\b[^\W\d_]+\b")
_THREAD_SOURCE_BLOCK_RE = re.compile(r"(?m)^\[THREAD SOURCE\]\s*\n[^\n]*(?:\n+|$)")
HUMAN_PREFIX = "[HUMAN]:"
HUMAN_PREFIX_WITH_SPACE = f"{HUMAN_PREFIX} "

_SHARED_SYSTEM_PROMPT = (
    (SYSTEM_PROMPT_DIR / "shared_system.txt").resolve().read_text(encoding="utf-8").strip()
)
_RESPONSE_ONLY_INSTRUCTION = (
    "Format your output exactly like this:\n"
    "<reasoning>...</reasoning>\n"
    "[HUMAN]: your response\n"
    "Write any reasoning before `[HUMAN]:` enclosed by the reasoning tags, "
    "and write only the final response after `[HUMAN]:`.\n"
)

RETAINED_TASK_PROMPT = (
    "[TASK]\n"
    "Your task is to predict [HUMAN]'s next message, matching [HUMAN]'s "
    "writing intentions, style, vocabulary, and tone. Use all provided "
    "information about [HUMAN] and the current context to predict the next "
    "message. Target your message to [OTHER] or [OTHER - OP], not to people "
    "who are described in the context but are not participants in the thread "
    "or conversation. Use second-person words like you and your only for the "
    "participant [HUMAN] is replying to, never for someone who is only "
    "described inside another participant's message. Do not restate or "
    "rewrite another participant's message as [HUMAN]'s own story. "
    "Do not answer as an assistant, analyst, or narrator.\n\n"
    "Before outputting the final answer, briefly reason about what [HUMAN] "
    "would say and think step by step. "
    "The generated response should naturally follow from your reasoning.\n"
    f"{_RESPONSE_ONLY_INSTRUCTION}"
    "Do not include reasoning after `[HUMAN]:`"
)


def conditioning_mode_uses_history(mode: str) -> bool:
    return mode in {CONDITIONING_MODE_HISTORY, CONDITIONING_MODE_HISTORY_PERSONA}


def conditioning_mode_uses_persona(mode: str) -> bool:
    return mode in {CONDITIONING_MODE_PERSONA, CONDITIONING_MODE_HISTORY_PERSONA}


def _check_conditioning_mode(mode: str) -> None:
    if mode not in CONDITIONING_MODE_CHOICES:
        raise ValueError(
            f"Unknown conditioning_mode={mode!r}. Expected one of {', '.join(CONDITIONING_MODE_CHOICES)}."
        )


def _build_persona_section(persona: str) -> str:
    stripped = (persona or "").strip()
    if not stripped:
        raise ValueError("persona must be non-empty when persona conditioning is enabled")
    return f"[PERSONA]\n{stripped}"


def build_shared_system_prompt(
    *,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> str:
    """Build the shared system prompt."""
    _check_conditioning_mode(conditioning_mode)
    if not conditioning_mode_uses_persona(conditioning_mode):
        return _SHARED_SYSTEM_PROMPT
    return f"{_SHARED_SYSTEM_PROMPT}\n\n{_build_persona_section(persona or '')}"


def build_grpo_system_prompt(
    task_prompt: str,
    *,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> str:
    """Build the GRPO system turn."""
    shared_system_prompt = build_shared_system_prompt(
        persona=persona,
        conditioning_mode=conditioning_mode,
    )
    sections = [shared_system_prompt, task_prompt.strip()]
    return "\n\n".join(section for section in sections if section.strip())


def build_reasoning_task_prompt(prompt_mode: str) -> str:
    """Build the reasoning task prompt."""
    if prompt_mode == "reasoning":
        return RETAINED_TASK_PROMPT
    raise ValueError(f"Unknown reasoning prompt mode: {prompt_mode}")


def resolve_chat_template_thinking_override(default: bool = False) -> bool:
    """Resolve Qwen hidden-thinking mode."""
    override = str(os.environ.get(ENABLE_THINKING_OVERRIDE_ENV, "") or "").strip().lower()
    if override:
        if override in {"1", "true", "yes", "on"}:
            return True
        if override in {"0", "false", "no", "off"}:
            return False
        raise ValueError(
            f"Invalid {ENABLE_THINKING_OVERRIDE_ENV}={override!r}. "
            "Expected one of 1/true/yes/on or 0/false/no/off."
        )
    return default


def prompt_mode_uses_chat_template_thinking() -> bool:
    """Check whether Qwen hidden thinking is enabled."""
    return resolve_chat_template_thinking_override(default=False)


def get_chat_template_kwargs_for_prompt_mode(prompt_mode: str | None) -> dict[str, Any]:
    """Build chat-template kwargs."""
    return {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": prompt_mode_uses_chat_template_thinking(),
    }


def tokenize_with_prefix_boundary(
    tokenizer: Any,
    prefix_text: str,
    full_text: str,
) -> tuple[list[int], int]:
    """Tokenize text and locate the prefix boundary."""
    tokenize_kwargs = {"add_special_tokens": False}

    try:
        full_tokenized = tokenizer(full_text, return_offsets_mapping=True, **tokenize_kwargs)
    except (TypeError, NotImplementedError, ValueError):
        full_token_ids = tokenizer(full_text, **tokenize_kwargs)["input_ids"]
        prefix_token_ids = tokenizer(prefix_text, **tokenize_kwargs)["input_ids"]
        return full_token_ids, len(prefix_token_ids)

    full_token_ids = full_tokenized["input_ids"]
    offsets = full_tokenized.get("offset_mapping")
    if offsets:
        prefix_chars = len(prefix_text)
        for token_idx, (start, end) in enumerate(offsets):
            if start >= prefix_chars or end > prefix_chars:
                return full_token_ids, token_idx
        return full_token_ids, len(full_token_ids)

    prefix_token_ids = tokenizer(prefix_text, **tokenize_kwargs)["input_ids"]
    return full_token_ids, len(prefix_token_ids)


def build_response_prefill(reasoning: str | None = None) -> str:
    """Build the assistant-side response prefix."""
    reasoning = (reasoning or "").strip()
    if reasoning:
        return f"{reasoning}\n{HUMAN_PREFIX_WITH_SPACE}"
    return HUMAN_PREFIX_WITH_SPACE


def _build_past_messages_block(user_history: str) -> str:
    history_block = (user_history or "").strip()
    if not history_block:
        return ""
    history_block = _THREAD_SOURCE_BLOCK_RE.sub("", history_block).strip()
    if history_block.startswith("[USER HISTORY]"):
        return history_block
    if history_block.startswith("[HISTORICAL CONTEXT]"):
        return "[USER HISTORY]" + history_block[len("[HISTORICAL CONTEXT]"):]
    return f"[USER HISTORY]\n{history_block}"


def _strip_human_suffix(text: str) -> str:
    stripped = (text or "").rstrip()
    if stripped.endswith(HUMAN_PREFIX_WITH_SPACE):
        return stripped[:-len(HUMAN_PREFIX_WITH_SPACE)].rstrip()
    if stripped.endswith(HUMAN_PREFIX):
        return stripped[:-len(HUMAN_PREFIX)].rstrip()
    return stripped


def _build_current_thread_block(thread_context: str) -> str:
    return "[CURRENT CONTEXT]\n" f"{_strip_human_suffix(thread_context)}"


def build_grpo_combined_user_message(
    user_history: str,
    thread_context: str,
    *,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> str:
    """Build the GRPO user turn."""
    _check_conditioning_mode(conditioning_mode)
    sections: list[str] = []
    if conditioning_mode_uses_history(conditioning_mode):
        history_block = _build_past_messages_block(user_history)
        if history_block:
            sections.append(history_block)
    sections.append(_build_current_thread_block(thread_context))
    return "\n\n".join(section for section in sections if section.strip())


def build_split_prompt_messages(
    user_history: str,
    thread_context: str,
    task_prompt: str,
    *,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> list[dict[str, str]]:
    """Build split system/user prompt messages."""
    return [
        {
            "role": "system",
            "content": build_grpo_system_prompt(
                task_prompt,
                persona=persona,
                conditioning_mode=conditioning_mode,
            ),
        },
        {
            "role": "user",
            "content": build_grpo_combined_user_message(
                user_history=user_history,
                thread_context=thread_context,
                conditioning_mode=conditioning_mode,
            ),
        },
    ]


def build_reasoning_messages(
    user_history: str,
    thread_context: str,
    prompt_mode: str,
    *,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> list[dict[str, str]]:
    """Build reasoning prompt messages."""
    task_prompt = build_reasoning_task_prompt(prompt_mode)
    return build_split_prompt_messages(
        user_history=user_history,
        thread_context=thread_context,
        task_prompt=task_prompt,
        persona=persona,
        conditioning_mode=conditioning_mode,
    )


def build_messages_for_prompt_mode(
    user_history: str,
    thread_context: str,
    prompt_mode: str,
    *,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> list[dict[str, str]]:
    """Build prompt messages."""
    return build_reasoning_messages(
        user_history=user_history,
        thread_context=thread_context,
        prompt_mode=prompt_mode,
        persona=persona,
        conditioning_mode=conditioning_mode,
    )


def build_prompt_messages_and_text(
    tokenizer: Any,
    *,
    user_history: str,
    thread_context: str,
    prompt_mode: str,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> tuple[list[dict[str, str]], str]:
    """Build messages and rendered text."""
    prompt_messages = build_messages_for_prompt_mode(
        user_history=user_history,
        thread_context=thread_context,
        prompt_mode=prompt_mode,
        persona=persona,
        conditioning_mode=conditioning_mode,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        **get_chat_template_kwargs_for_prompt_mode(prompt_mode),
    )
    return prompt_messages, prompt_text


def build_grpo_prompt_payload(
    tokenizer: Any,
    *,
    user_history: str,
    thread_context: str,
    prompt_mode: str,
    persona: str | None = None,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> dict[str, Any]:
    """Build a GRPO prompt payload."""
    prompt_messages, prompt_text = build_prompt_messages_and_text(
        tokenizer,
        user_history=user_history,
        thread_context=thread_context,
        prompt_mode=prompt_mode,
        persona=persona,
        conditioning_mode=conditioning_mode,
    )
    return {
        "prompt": prompt_messages,
        "prompt_text": prompt_text,
        "prompt_mode": prompt_mode,
        "conditioning_mode": conditioning_mode,
        "raw_prompt": json.dumps(prompt_messages, ensure_ascii=False),
    }


def parse_reasoning_and_response(text: str) -> tuple[str, str]:
    """Split reasoning from the final response."""
    text = (text or "").strip()
    text, _ = _strip_hidden_thinking_close_prefill(text)
    if not text:
        return "", ""

    reasoning, response = _split_on_human_tag(text)
    if response:
        return reasoning, response

    reasoning, response, found_tag = _extract_tagged_response(text, "response")
    if found_tag:
        return reasoning, response

    reasoning, response = _split_on_response_marker(text)
    if response:
        return reasoning, response

    return "", text


def _strip_hidden_thinking_close_prefill(text: str) -> tuple[str, bool]:
    """Remove a generated close-only Qwen thinking boundary."""
    stripped = (text or "").strip()
    if not stripped:
        return "", False

    first_human = _HUMAN_TAG_RE.search(stripped)
    prefix_end = first_human.start() if first_human else len(stripped)
    prefix = stripped[:prefix_end]
    if _REASONING_OPEN_TAG_RE.search(prefix):
        return stripped, False

    close_matches = list(_HIDDEN_THINKING_CLOSE_TAG_RE.finditer(prefix))
    if len(close_matches) != 1:
        return stripped, False

    close_match = close_matches[0]
    normalized = (
        prefix[: close_match.start()]
        + prefix[close_match.end() :]
        + stripped[prefix_end:]
    ).strip()
    return normalized, True


def _split_on_human_tag(text: str) -> tuple[str, str]:
    matches = list(_HUMAN_TAG_RE.finditer(text or ""))
    if not matches:
        return "", ""

    marker = matches[-1]
    reasoning = text[:marker.start()].strip()
    reasoning = re.sub(r"<response\s*>$", "", reasoning, flags=re.IGNORECASE).strip()
    response = text[marker.end():].strip()
    response = re.sub(r"</response\s*>$", "", response, flags=re.IGNORECASE).strip()
    response = normalize_state_output_content("response", response)
    return reasoning, response


def _split_on_response_marker(text: str) -> tuple[str, str]:
    matches = list(_RESPONSE_MARKER_RE.finditer(text))
    if not matches:
        return "", text

    marker = matches[-1]
    reasoning = text[:marker.start()].strip()
    response = text[marker.end():].strip()
    response = re.sub(r"^\*{1,2}\s*", "", response)
    return reasoning, response


def normalize_state_output_content(state_name: str, content: str) -> str:
    """Normalize parsed state content."""
    normalized = (content or "").strip()
    if state_name.strip().lower() != "response":
        return normalized

    normalized = re.sub(r"^\*{1,2}\s*", "", normalized)
    normalized = re.sub(
        r"^(?:(?:\[\s*human\s*\]|<\s*human\s*>)(?:\s*:)?|human\s*:)\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.strip()


def _extract_tagged_response(text: str, tag_name: str) -> tuple[str, str, bool]:
    stripped = (text or "").strip()
    if not stripped:
        return "", "", False

    tag = re.escape(tag_name)
    full_pattern = re.compile(
        rf"(?is)^(?P<prefix>.*?)<\s*{tag}\s*>(?P<content>.*?)</\s*{tag}\s*>\s*$"
    )
    full_match = full_pattern.match(stripped)
    if full_match:
        return (
            full_match.group("prefix").strip(),
            normalize_state_output_content(tag_name, full_match.group("content")),
            True,
        )

    inline_pattern = re.compile(rf"(?is)<\s*{tag}\s*>(?P<content>.*?)</\s*{tag}\s*>")
    inline_match = inline_pattern.search(stripped)
    if inline_match:
        return (
            stripped[:inline_match.start()].strip(),
            normalize_state_output_content(tag_name, inline_match.group("content")),
            True,
        )

    open_pattern = re.compile(rf"(?is)<\s*{tag}\s*>")
    open_match = open_pattern.search(stripped)
    if open_match:
        suffix = stripped[open_match.end():]
        suffix = re.sub(rf"(?is)</\s*{tag}\s*>$", "", suffix).strip()
        return (
            stripped[:open_match.start()].strip(),
            normalize_state_output_content(tag_name, suffix),
            True,
        )

    return "", "", False


def response_format_components(text: str) -> dict[str, bool | int]:
    """Inspect response-tail format."""
    stripped = (text or "").strip()
    stripped, _ = _strip_hidden_thinking_close_prefill(stripped)
    if not stripped:
        return {
            "has_exact_human_prefix": False,
            "reasoning_nonempty": False,
            "has_reasoning_schema": False,
            "placeholder_reasoning_prefix": False,
            "response_nonempty": False,
            "no_post_human_thinking_trace": False,
            "reasoning_open_count": 0,
            "reasoning_close_count": 0,
            "human_prefix_count": 0,
            "has_duplicate_reasoning_block": False,
            "has_duplicate_human_prefix": False,
            "has_unmatched_reasoning_tags": False,
            "has_forbidden_xml_tag": False,
            "format_hard_fail": False,
        }

    reasoning_open_count = len(list(_REASONING_OPEN_TAG_RE.finditer(stripped)))
    reasoning_close_count = len(list(_REASONING_CLOSE_TAG_RE.finditer(stripped)))
    human_prefix_count = len(list(_HUMAN_TAG_RE.finditer(stripped)))
    has_duplicate_reasoning_block = reasoning_open_count > 1 or reasoning_close_count > 1
    has_duplicate_human_prefix = human_prefix_count > 1
    has_unmatched_reasoning_tags = (
        reasoning_open_count != reasoning_close_count
        and (reasoning_open_count > 0 or reasoning_close_count > 0)
    )
    has_forbidden_xml_tag = False
    for match in _XML_TAG_RE.finditer(stripped):
        name_match = _XML_TAG_NAME_RE.match(match.group(0))
        if not name_match:
            continue
        tag_name = name_match.group(1).lower()
        if tag_name not in {"reasoning"}:
            has_forbidden_xml_tag = True
            break
    format_hard_fail = (
        has_duplicate_reasoning_block
        or has_duplicate_human_prefix
        or has_unmatched_reasoning_tags
        or has_forbidden_xml_tag
    )

    reasoning, _ = parse_reasoning_and_response(stripped)
    matches = list(_EXACT_HUMAN_PREFIX_RE.finditer(stripped))
    if not matches:
        return {
            "has_exact_human_prefix": False,
            "reasoning_nonempty": bool((reasoning or "").strip()),
            "has_reasoning_schema": False,
            "placeholder_reasoning_prefix": False,
            "response_nonempty": False,
            "no_post_human_thinking_trace": False,
            "reasoning_open_count": reasoning_open_count,
            "reasoning_close_count": reasoning_close_count,
            "human_prefix_count": human_prefix_count,
            "has_duplicate_reasoning_block": has_duplicate_reasoning_block,
            "has_duplicate_human_prefix": has_duplicate_human_prefix,
            "has_unmatched_reasoning_tags": has_unmatched_reasoning_tags,
            "has_forbidden_xml_tag": has_forbidden_xml_tag,
            "format_hard_fail": format_hard_fail,
        }

    marker = matches[-1]
    prefix = stripped[:marker.start()]
    response_tail = stripped[marker.end():].strip()
    response_nonempty = bool(response_tail)
    has_post_human_reasoning = False
    if response_nonempty:
        has_post_human_reasoning = bool(
            _ANY_REASONING_TAG_RE.search(response_tail)
            or _POST_HUMAN_REASONING_RE.search(response_tail)
        )
    stripped_prefix = _XML_TAG_RE.sub(" ", prefix).strip()
    prefix_has_alphabetic_token = bool(_ALPHABETIC_TOKEN_RE.search(stripped_prefix))
    placeholder_reasoning_prefix = not prefix_has_alphabetic_token
    close_tag_matches = list(_REASONING_CLOSE_TAG_RE.finditer(prefix))
    if close_tag_matches:
        tail_after_close = prefix[close_tag_matches[-1].end():]
        tail_after_close = _XML_TAG_RE.sub(" ", tail_after_close).strip()
        has_reasoning_schema = not bool(_ALPHABETIC_TOKEN_RE.search(tail_after_close))
    else:
        has_reasoning_schema = False

    return {
        "has_exact_human_prefix": True,
        "reasoning_nonempty": bool((reasoning or "").strip()),
        "has_reasoning_schema": has_reasoning_schema,
        "placeholder_reasoning_prefix": placeholder_reasoning_prefix,
        "response_nonempty": response_nonempty,
        "no_post_human_thinking_trace": response_nonempty and not has_post_human_reasoning,
        "reasoning_open_count": reasoning_open_count,
        "reasoning_close_count": reasoning_close_count,
        "human_prefix_count": human_prefix_count,
        "has_duplicate_reasoning_block": has_duplicate_reasoning_block,
        "has_duplicate_human_prefix": has_duplicate_human_prefix,
        "has_unmatched_reasoning_tags": has_unmatched_reasoning_tags,
        "has_forbidden_xml_tag": has_forbidden_xml_tag,
        "format_hard_fail": format_hard_fail,
    }


def normalize_prompt_messages(prompt_value: Any) -> list[dict[str, str]] | None:
    """Normalize stored prompt messages."""
    if prompt_value is None:
        return None
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, str):
        try:
            prompt_value = json.loads(prompt_value)
        except json.JSONDecodeError:
            return None
    if isinstance(prompt_value, tuple):
        prompt_value = list(prompt_value)
    if not isinstance(prompt_value, list):
        return None

    messages: list[dict[str, str]] = []
    for message in prompt_value:
        if not isinstance(message, Mapping):
            return None
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            return None
        messages.append({"role": str(role), "content": str(content)})
    return messages
