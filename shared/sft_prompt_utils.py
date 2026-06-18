"""SFT formatting and parsing."""

from __future__ import annotations

import re

from shared.prompt_utils import (
    HUMAN_PREFIX_WITH_SPACE,
    normalize_state_output_content,
    parse_reasoning_and_response,
)

START_THOUGHTS = "<reasoning>"
END_THOUGHTS = "</reasoning>"

_REASONING_RE = re.compile(
    r"<\s*(?:reasoning|think)\b[^>]*>\s*(.*?)\s*</\s*(?:reasoning|think)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_NON_RESPONSE_PAIR_RE = re.compile(
    r"<\s*(?:reasoning|think)\b[^>]*>.*?</\s*(?:reasoning|think)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_NON_RESPONSE_TAG_RE = re.compile(
    r"</?\s*(?:reasoning|think)\b[^>]*>",
    re.IGNORECASE,
)


def format_sft_assistant_content(response: str, cot: str) -> str:
    """Format one SFT assistant turn: visible reasoning then the [HUMAN]: response.

    SFT is always CoT-conditioned, so a reasoning trace is required.
    """
    normalized_response = normalize_state_output_content("response", response)
    reasoning = format_sft_reasoning_content(cot)
    return f"{reasoning}\n{HUMAN_PREFIX_WITH_SPACE}{normalized_response}"


def format_sft_reasoning_content(cot: str) -> str:
    """Normalize one SFT reasoning trace and wrap it in visible reasoning tags."""
    cot = (cot or "").strip()
    if not cot:
        return ""
    match = _REASONING_RE.fullmatch(cot)
    if match:
        cot = match.group(1).strip()
    else:
        cot = _NON_RESPONSE_TAG_RE.sub("", cot).strip()
    return f"{START_THOUGHTS}{cot}{END_THOUGHTS}"


def parse_sft_generation(text: str) -> dict[str, str]:
    """Parse legacy SFT generations into reasoning text and final response."""
    stripped = (text or "").strip()
    reasoning_blob, response = parse_reasoning_and_response(stripped)
    reasoning_blob = (reasoning_blob or "").strip()
    reasoning = reasoning_blob
    for pattern in (_REASONING_RE,):
        match = pattern.search(reasoning_blob)
        if match:
            reasoning = match.group(1).strip()
            break
    else:
        for pattern in (_REASONING_RE,):
            match = pattern.search(stripped)
            if match:
                reasoning = match.group(1).strip()
                break

    if stripped and response == stripped:
        # Strip raw reasoning blocks before the loose final-answer parse.
        cleaned = stripped
        prev = None
        while prev != cleaned:
            prev = cleaned
            cleaned = _NON_RESPONSE_PAIR_RE.sub("", cleaned)
        cleaned = _NON_RESPONSE_TAG_RE.sub("", cleaned).strip()
        _, response = parse_reasoning_and_response(cleaned)

    return {
        "reasoning": reasoning.strip(),
        "response": normalize_state_output_content("response", response),
    }
