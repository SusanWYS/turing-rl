"""SFT training with reasoning traces using TRL + LoRA."""

import argparse
import os
import time
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from shared.prompt_utils import tokenize_with_prefix_boundary


MODEL_MAP = {
    "qwen3-8b": "Qwen/Qwen3-8B",
}

LORA_TARGET_MODULES = {
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

QWEN_EMPTY_THINK_PREFILLS = (
    "<think>\n\n</think>\n\n",
    "<think>\n</think>\n\n",
    "<think></think>\n\n",
    "<think>\n\n</think>",
    "<think></think>",
)
CHAT_TEMPLATE_END_TOKENS = ("<|im_end|>",)


def get_lora_targets(model_name: str) -> list[str]:
    """Get LoRA target modules for the model family."""
    if "qwen" in model_name:
        return LORA_TARGET_MODULES["qwen3"]
    if "llama" in model_name:
        return LORA_TARGET_MODULES["llama"]
    raise ValueError(f"Unknown model family for: {model_name}")


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def strip_empty_think_prefill(prompt_text: str) -> tuple[str, str]:
    """Strip a tokenizer-injected empty Qwen think block from the assistant prompt."""
    for prefill in QWEN_EMPTY_THINK_PREFILLS:
        if prompt_text.endswith(prefill):
            return prompt_text[: -len(prefill)], prefill
    return prompt_text, ""


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError(f"Expected messages to be a list, got {type(messages).__name__}")

    normalized = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Expected messages[{idx}] to be a dict, got {type(message).__name__}")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if not role:
            raise ValueError(f"messages[{idx}] is missing a role")
        normalized.append({"role": role, "content": content})
    return normalized


def _assistant_target_end_char(full_text: str, target_start_char: int) -> int:
    """Return the char offset before the assistant end marker, if the template has one."""
    candidate_offsets = [
        full_text.rfind(end_token)
        for end_token in CHAT_TEMPLATE_END_TOKENS
        if full_text.rfind(end_token) >= target_start_char
    ]
    if not candidate_offsets:
        return len(full_text)
    return min(candidate_offsets)


def build_chat_template_sft_features(tokenizer: Any, messages: Any) -> dict[str, list[int]]:
    """Tokenize one SFT row and mask the assistant target span."""
    normalized_messages = _normalize_messages(messages)
    if not normalized_messages:
        raise ValueError("messages must be non-empty")
    if normalized_messages[-1]["role"] != "assistant":
        raise ValueError("The last SFT message must be the assistant target")

    prompt_messages = normalized_messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    masked_prefix_text, _think_prefill = strip_empty_think_prefill(prompt_text)
    if not full_text.startswith(masked_prefix_text):
        raise ValueError(
            "Rendered full chat template does not start with the rendered prompt prefix. "
            "Cannot build a reliable SFT completion mask."
        )

    target_start_char = len(masked_prefix_text)
    target_end_char = _assistant_target_end_char(full_text, target_start_char)
    if target_end_char <= target_start_char:
        raise ValueError("Computed an empty assistant target span")

    input_ids, target_start_token = tokenize_with_prefix_boundary(
        tokenizer,
        masked_prefix_text,
        full_text,
    )
    _, target_end_token = tokenize_with_prefix_boundary(
        tokenizer,
        full_text[:target_end_char],
        full_text,
    )
    if target_end_token <= target_start_token:
        raise ValueError("Computed an empty assistant target token span")

    completion_mask = [0] * len(input_ids)
    for token_idx in range(target_start_token, target_end_token):
        completion_mask[token_idx] = 1
    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training with reasoning traces")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model to finetune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=5120,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        default=True,
        help="Enable sequence packing (default: on)",
    )
    parser.add_argument(
        "--no_packing",
        action="store_true",
        help="Disable sequence packing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override training data JSONL path",
    )
    parser.add_argument(
        "--base_adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter to merge before training (for iterative SFT)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device train batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--no_torch_compile",
        action="store_true",
        help="Disable torch.compile for training",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="Override LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="Override LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=None,
        help="Override LoRA dropout",
    )
    parser.add_argument(
        "--no_qlora",
        action="store_true",
        help="Disable 4-bit QLoRA loading",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Use only the first N training examples, for quick distributed smoke tests",
    )
    parser.add_argument(
        "--exit_after_trainer_build",
        action="store_true",
        help="Exit successfully after SFTTrainer construction, before trainer.train()",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))

    def log(message: str) -> None:
        elapsed = time.strftime("%H:%M:%S")
        print(f"[{elapsed} rank={rank}] {message}", flush=True)

    config_name = args.model.replace("-", "_").replace(".", "_") + "_lora.yaml"
    config_path = os.path.join("training", "sft", "configs", config_name)
    config = {}
    if os.path.exists(config_path):
        config = load_config(config_path)
        log(f"Loaded config from: {config_path}")
    else:
        log(f"Warning: config not found at {config_path}, using defaults")

    model_id = MODEL_MAP[args.model]

    if args.data_path:
        dataset_path = args.data_path
    else:
        dataset_path = os.path.join("data", "sft", f"{args.model}_sft_cot.jsonl")

    output_dir = args.output_dir or config.get("output_dir")
    if output_dir is None:
        output_dir = os.path.join("results", "sft", f"{args.model}_sft_cot")

    log(f"Model: {model_id}")
    log(f"Dataset: {dataset_path}")
    log(f"Output: {output_dir}")
    if args.base_adapter:
        log(f"Base adapter: {args.base_adapter}")

    log("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("Tokenizer loaded")

    log("Loading JSON dataset")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if args.max_train_examples is not None:
        if args.max_train_examples <= 0:
            raise ValueError("--max_train_examples must be positive")
        original_len = len(dataset)
        dataset = dataset.select(range(min(args.max_train_examples, original_len)))
        log(f"Selected {len(dataset)} / {original_len} examples for smoke run")
    log(f"Training examples: {len(dataset)}")

    def tokenize_with_completion_mask(example):
        return build_chat_template_sft_features(tokenizer, example["messages"])

    log("Tokenizing chat templates with explicit assistant completion masks")
    dataset = dataset.map(
        tokenize_with_completion_mask,
        remove_columns=dataset.column_names,
    )
    log("Tokenized chat templates with explicit assistant completion masks")

    lora_r = args.lora_r or config.get("lora_r", 16)
    lora_alpha = args.lora_alpha or config.get("lora_alpha", 32)
    lora_dropout = (
        args.lora_dropout
        if args.lora_dropout is not None
        else config.get("lora_dropout", 0.05)
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=get_lora_targets(args.model),
    )

    use_qlora = config.get("use_qlora", True) and not args.no_qlora
    bnb_config = None
    model_kwargs = {}
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            model_kwargs["device_map"] = {"": local_rank}

    log("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        **model_kwargs,
    )
    log("Model loaded")

    if args.base_adapter:
        from peft import PeftModel

        log(f"Merging base adapter: {args.base_adapter}")
        model = PeftModel.from_pretrained(model, args.base_adapter)
        model = model.merge_and_unload()
        log("Base adapter merged")

    num_epochs = args.num_epochs or config.get("num_epochs", 3)
    train_batch_size = args.batch_size or config.get("batch_size", 4)
    grad_accum_steps = (
        args.gradient_accumulation_steps
        or config.get("gradient_accumulation_steps", 4)
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=float(config.get("learning_rate", 2e-4)),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        bf16=True,
        logging_steps=config.get("logging_steps", 10),
        save_strategy="epoch",
        max_length=args.max_seq_length,
        packing=not args.no_packing,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=config.get("report_to", "none"),
        completion_only_loss=True,
        ddp_find_unused_parameters=False,
        seed=42,
        torch_compile=not args.no_torch_compile,
    )

    log("Building SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    log("SFTTrainer built")
    if args.exit_after_trainer_build:
        log("Exiting after SFTTrainer build as requested")
        return

    log("Starting training")
    log(f"  LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    log(f"  QLoRA: {use_qlora}")
    log(f"  Epochs: {training_args.num_train_epochs}")
    log(
        "  Batch size: "
        f"{training_args.per_device_train_batch_size} x "
        f"{training_args.gradient_accumulation_steps} accum"
    )
    log(f"  Learning rate: {training_args.learning_rate}")
    log(f"  Max seq length: {args.max_seq_length}")

    trainer.train()

    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    log(f"Training complete. Model saved to: {os.path.join(output_dir, 'final')}")


if __name__ == "__main__":
    main()
