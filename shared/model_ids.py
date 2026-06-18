"""Model ids and tokenizer loading."""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
QWEN3_5_MOE_MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
SUPPORTED_MODEL_IDS = {DEFAULT_MODEL_ID, QWEN3_5_MOE_MODEL_ID}
MODEL_ID_ALIASES = {
    "qwen3-8b": DEFAULT_MODEL_ID,
    "qwen3.5-397b": QWEN3_5_MOE_MODEL_ID,
}


def normalize_model_id(model_id: str) -> str:
    """Resolve supported model aliases."""
    normalized = MODEL_ID_ALIASES.get(model_id, model_id)
    if normalized != model_id:
        print(f"Model id alias detected: {model_id} -> {normalized}")
    if normalized not in SUPPORTED_MODEL_IDS:
        raise ValueError(
            f"Unsupported model_id {model_id!r}. Retained generation supports "
            f"{sorted(SUPPORTED_MODEL_IDS)!r} and known aliases."
        )
    return normalized


def load_tokenizer(model_id: str) -> Any:
    """Load the tokenizer."""
    from transformers import AutoTokenizer

    model_id = normalize_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer
