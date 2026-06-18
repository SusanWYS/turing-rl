"""Persona-memory map loading and prompt-ready persona text formatting."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any


def _format_persona_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item or "").strip()]


def _format_persona_dict(persona: dict[str, Any]) -> str:
    """Format a persona dict as readable prompt text."""
    if "background" not in persona and "other_personal_cues" in persona:
        persona = dict(persona)
        persona["background"] = persona.get("other_personal_cues")

    original_field_order = (
        "values",
        "verbal_quirks",
        "expression_style",
        "length_prior",
        "background",
    )
    if any(key in persona for key in original_field_order):
        lines = []
        for key in original_field_order:
            value = persona.get(key)
            text = str(value or "").strip()
            if text:
                lines.append(f"{key}: {text}")
        if lines:
            return "\n".join(lines).strip()

    lines: list[str] = []
    demographics = persona.get("demographics")
    if isinstance(demographics, dict):
        demo_lines = [
            f"  {key}: {str(value).strip()}"
            for key, value in demographics.items()
            if str(value or "").strip() and str(value).strip() != "NA"
        ]
        if demo_lines:
            lines.append("Demographics:")
            lines.extend(demo_lines)

    for key in ("interests", "values", "communication", "statistics"):
        items = _format_persona_list(persona.get(key))
        if items:
            lines.append(f"{key.capitalize()}:")
            lines.extend(f"  {item}" for item in items)

    if lines:
        return "\n".join(lines).strip()

    fallback_lines = []
    for key, value in persona.items():
        if isinstance(value, (dict, list)):
            continue
        text = str(value or "").strip()
        if text:
            fallback_lines.append(f"{key}: {text}")
    return "\n".join(fallback_lines).strip()


def extract_persona_text(value: Any) -> str:
    """Recover prompt-ready persona text from common JSON/Python container shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if re.match(r"(?is)^<\s*persona\s*>.*</\s*persona\s*>\s*$", stripped):
            return stripped
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return extract_persona_text(parsed)
        return stripped
    if isinstance(value, dict):
        for key in ("persona", "persona_memory", "conditioning_summary", "summary"):
            candidate = value.get(key)
            persona_text = extract_persona_text(candidate)
            if persona_text:
                return persona_text
        return _format_persona_dict(value)
    return ""


def _normalize_persona_payload(payload: Any) -> dict[str, str]:
    if isinstance(payload, dict):
        normalized: dict[str, str] = {}
        for user_id, value in payload.items():
            persona_text = extract_persona_text(value)
            if persona_text:
                normalized[str(user_id)] = persona_text
        return normalized

    if isinstance(payload, list):
        normalized = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            user_id = item.get("user_id")
            if user_id is None:
                continue
            persona_text = extract_persona_text(item)
            if persona_text:
                normalized[str(user_id)] = persona_text
        return normalized

    raise ValueError(
        "Unsupported persona payload type. Expected a dict mapping user ids to persona strings, "
        "or a list of {'user_id', 'persona'} objects."
    )


def load_persona_map(path: str | None) -> dict[str, str]:
    """Load one user_id -> persona-memory mapping from JSON, JSONL, or pickle."""
    if not path:
        return {}

    persona_path = Path(path)
    if not persona_path.is_file():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")

    suffix = persona_path.suffix.lower()
    if suffix == ".json":
        with persona_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return _normalize_persona_payload(payload)

    if suffix == ".jsonl":
        rows = []
        with persona_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        return _normalize_persona_payload(rows)

    if suffix in {".pkl", ".pickle"}:
        with persona_path.open("rb") as handle:
            payload = pickle.load(handle)
        return _normalize_persona_payload(payload)

    raise ValueError(
        f"Unsupported persona file extension for {persona_path}. "
        "Expected .json, .jsonl, .pkl, or .pickle."
    )


def get_persona_for_user(persona_map: dict[str, str], user_id: str, *aliases: str) -> str:
    """Look up persona text."""
    if not persona_map:
        return ""
    for candidate in (user_id, *aliases):
        persona = str(persona_map.get(str(candidate), "") or "").strip()
        if persona:
            return persona
    return ""
