from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.prompt_utils import (
    get_chat_template_kwargs_for_prompt_mode,
    resolve_chat_template_thinking_override,
)

REPO_TRAINING_TEMPERATURE = 0.6
REPO_TRAINING_TOP_P = 1.0
REPO_TRAINING_TOP_K = -1
REPO_TRAINING_PRESENCE_PENALTY = 0.5
REPO_OPENAI_MAX_RETRIES_CAP = 3


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _default_learning_rate() -> float:
    return 1e-5


@dataclass(frozen=True)
class TuringGRPOConfig:
    num_users: int = 600
    rollout_n: int = 4
    train_batch_size: int = 128
    ppo_mini_batch_size: int = 128
    total_epochs: int = 3
    run_suffix: str | None = None
    resume_mode: str = "auto"
    fresh_start: bool = False
    gpu_memory_utilization: float = 0.6
    rollout_temperature: float = REPO_TRAINING_TEMPERATURE
    rollout_presence_penalty: float = REPO_TRAINING_PRESENCE_PENALTY
    learning_rate: float = 1e-5
    kl_loss_coef: float = 1e-3
    max_num_batched_tokens: int = 12288
    max_num_seqs: int = 32
    rollout_agent_num_workers: int = 8
    judge_max_concurrency: int = 512
    openai_max_retries: int = 3
    rollout_tensor_model_parallel_size: int = 1
    rollout_layered_summon: bool = False
    actor_fsdp_size: int = -1
    save_freq: int = 50
    model_name: str = "qwen3-8b"

    @classmethod
    def from_env(cls) -> "TuringGRPOConfig":
        fresh_start = _env_bool("FRESH_START", False)
        resume_mode = os.environ.get("RESUME_MODE", "auto")
        if fresh_start:
            resume_mode = "disable"
        return cls(
            num_users=_env_int("NUM_USERS", 600),
            rollout_n=_env_int("ROLLOUT_N", 4),
            train_batch_size=_env_int("TRAIN_BATCH_SIZE", 128),
            ppo_mini_batch_size=_env_int("PPO_MINI_BATCH_SIZE", 128),
            total_epochs=_env_int("TOTAL_EPOCHS", 3),
            run_suffix=os.environ.get("RUN_SUFFIX"),
            resume_mode=resume_mode,
            fresh_start=fresh_start,
            gpu_memory_utilization=_env_float("ROLLOUT_GPU_MEMORY_UTILIZATION", 0.6),
            rollout_temperature=_env_float("ROLLOUT_TEMPERATURE", REPO_TRAINING_TEMPERATURE),
            rollout_presence_penalty=_env_float("PERSONA_VLLM_PRESENCE_PENALTY", REPO_TRAINING_PRESENCE_PENALTY),
            learning_rate=_env_float("LEARNING_RATE", _default_learning_rate()),
            kl_loss_coef=_env_float("TURING_KL_LOSS_COEF", 1e-3),
            max_num_batched_tokens=_env_int("ROLLOUT_MAX_NUM_BATCHED_TOKENS", 12288),
            max_num_seqs=_env_int("ROLLOUT_MAX_NUM_SEQS", 32),
            rollout_agent_num_workers=_env_int("ROLLOUT_AGENT_NUM_WORKERS", 8),
            judge_max_concurrency=_env_int("PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY", 512),
            openai_max_retries=min(_env_int("PERSONA_OPENAI_MAX_RETRIES", 3), REPO_OPENAI_MAX_RETRIES_CAP),
            rollout_tensor_model_parallel_size=_env_int("ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE", 1),
            rollout_layered_summon=_env_bool("ROLLOUT_LAYERED_SUMMON", False),
            actor_fsdp_size=_env_int("ACTOR_FSDP_SIZE", -1),
            save_freq=_env_int("SAVE_FREQ", 50),
            model_name=os.environ.get("MODEL_NAME", "qwen3-8b"),
        )

    @property
    def prompt_mode(self) -> str:
        return "reasoning"

    @property
    def resolved_run_suffix(self) -> str:
        if self.run_suffix:
            return self.run_suffix
        return f"{self.num_users}u_r{self.rollout_n}_reasoning"

    @property
    def checkpoint_dir(self) -> str:
        return f"results/grpo/checkpoints_turing_{self.resolved_run_suffix}"

    @property
    def data_dir(self) -> str:
        return f"data/convokit/runs/turing_{self.resolved_run_suffix}"

    @property
    def train_file(self) -> str:
        return f"{self.data_dir}/train.parquet"

    @property
    def val_file(self) -> str:
        return f"{self.data_dir}/val.parquet"

    @property
    def test_file(self) -> str:
        return f"{self.data_dir}/test.parquet"

    @property
    def gen_output(self) -> str:
        return f"results/grpo_gen/grpo_turing_{self.resolved_run_suffix}_gen1.pkl"

    @property
    def experiment_name(self) -> str:
        return f"{self.model_name}-grpo-turing-{self.resolved_run_suffix}-8gpu"

    @property
    def enable_chat_template_thinking(self) -> bool:
        return resolve_chat_template_thinking_override(default=False)

    def shell_exports(self) -> dict[str, str]:
        return {
            "REWARD_METRIC": "turing",
            "NUM_USERS": str(self.num_users),
            "ROLLOUT_N": str(self.rollout_n),
            "TRAIN_BATCH_SIZE": str(self.train_batch_size),
            "PPO_MINI_BATCH_SIZE": str(self.ppo_mini_batch_size),
            "TOTAL_EPOCHS": str(self.total_epochs),
            "RUN_SUFFIX": self.resolved_run_suffix,
            "CHECKPOINT_DIR": self.checkpoint_dir,
            "DATA_DIR": self.data_dir,
            "TRAIN_FILE": self.train_file,
            "VAL_FILE": self.val_file,
            "TEST_FILE": self.test_file,
            "GEN_OUTPUT": self.gen_output,
            "RESUME_MODE": self.resume_mode,
            "FRESH_START": "1" if self.fresh_start else "0",
            "GRPO_PROMPT_MODE": self.prompt_mode,
            "TURING_PROMPT_MODE": self.prompt_mode,
            "ROLLOUT_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "ROLLOUT_TEMPERATURE": str(self.rollout_temperature),
            "PERSONA_VLLM_PRESENCE_PENALTY": str(self.rollout_presence_penalty),
            "LEARNING_RATE": str(self.learning_rate),
            "TURING_KL_LOSS_COEF": str(self.kl_loss_coef),
            "ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(self.max_num_batched_tokens),
            "ROLLOUT_MAX_NUM_SEQS": str(self.max_num_seqs),
            "ROLLOUT_AGENT_NUM_WORKERS": str(self.rollout_agent_num_workers),
            "PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY": str(self.judge_max_concurrency),
            "PERSONA_OPENAI_MAX_RETRIES": str(min(self.openai_max_retries, REPO_OPENAI_MAX_RETRIES_CAP)),
            "ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE": str(self.rollout_tensor_model_parallel_size),
            "ROLLOUT_LAYERED_SUMMON": str(self.rollout_layered_summon).lower(),
            "ACTOR_FSDP_SIZE": str(self.actor_fsdp_size),
            "SAVE_FREQ": str(self.save_freq),
        }

    def verl_overrides(self) -> list[str]:
        return [
            "trainer.n_gpus_per_node=8",
            "trainer.nnodes=1",
            f"trainer.total_epochs={self.total_epochs}",
            "trainer.project_name=grpo-user-sim",
            f"trainer.default_local_dir={self.checkpoint_dir}",
            f"trainer.experiment_name={self.experiment_name}",
            f"trainer.resume_mode={self.resume_mode}",
            f"trainer.save_freq={self.save_freq}",
            f"data.train_files={self.train_file}",
            f"data.val_files={self.val_file}",
            f"data.train_batch_size={self.train_batch_size}",
            "data.dataloader_num_workers=0",
            f"data.apply_chat_template_kwargs.enable_thinking={str(self.enable_chat_template_thinking).lower()}",
            "critic.enable=false",
            "actor_rollout_ref.actor.use_kl_loss=true",
            f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
            "actor_rollout_ref.actor.loss_agg_mode=token-mean",
            f"critic.optim.lr={self.learning_rate}",
            f"actor_rollout_ref.actor.optim.lr={self.learning_rate}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.ppo_mini_batch_size}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
            f"actor_rollout_ref.rollout.n={self.rollout_n}",
            f"actor_rollout_ref.rollout.temperature={self.rollout_temperature}",
            f"actor_rollout_ref.rollout.top_p={REPO_TRAINING_TOP_P}",
            f"actor_rollout_ref.rollout.top_k={REPO_TRAINING_TOP_K}",
            f"actor_rollout_ref.rollout.presence_penalty={self.rollout_presence_penalty}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.rollout_tensor_model_parallel_size}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            "actor_rollout_ref.rollout.max_model_len=12288",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={self.max_num_batched_tokens}",
            f"actor_rollout_ref.rollout.max_num_seqs={self.max_num_seqs}",
            f"actor_rollout_ref.rollout.agent.num_workers={self.rollout_agent_num_workers}",
            "actor_rollout_ref.rollout.load_format=safetensors",
            f"actor_rollout_ref.rollout.layered_summon={str(self.rollout_layered_summon).lower()}",
            f"actor_rollout_ref.actor.fsdp_config.fsdp_size={self.actor_fsdp_size}",
            f"actor_rollout_ref.ref.fsdp_config.fsdp_size={self.actor_fsdp_size}",
        ]


def _cmd_shell_exports() -> int:
    config = TuringGRPOConfig.from_env()
    for key, value in config.shell_exports().items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


def _cmd_verl_overrides() -> int:
    config = TuringGRPOConfig.from_env()
    for item in config.verl_overrides():
        print(item)
    return 0


def _cmd_summary() -> int:
    config = TuringGRPOConfig.from_env()
    print(f"run_suffix={config.resolved_run_suffix}")
    print(f"checkpoint_dir={config.checkpoint_dir}")
    print(f"data_dir={config.data_dir}")
    print(f"experiment_name={config.experiment_name}")
    print(f"prompt_mode={config.prompt_mode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared turing GRPO launcher config helper.")
    parser.add_argument("command", choices=["shell-exports", "verl-overrides", "summary"])
    args = parser.parse_args()

    if args.command == "shell-exports":
        return _cmd_shell_exports()
    if args.command == "verl-overrides":
        return _cmd_verl_overrides()
    return _cmd_summary()


if __name__ == "__main__":
    raise SystemExit(main())
