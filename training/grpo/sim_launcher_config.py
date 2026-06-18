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

from shared.prompt_utils import resolve_chat_template_thinking_override

REPO_TRAINING_TEMPERATURE = 0.6
REPO_TRAINING_TOP_P = 1.0
REPO_TRAINING_TOP_K = -1
REPO_TRAINING_PRESENCE_PENALTY = 0.5


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
class SimGRPOConfig:
    num_users: int = 600
    rollout_n: int = 4
    train_batch_size: int = 128
    ppo_mini_batch_size: int = 128
    learning_rate: float = 1e-5
    kl_loss_coef: float = 1e-3
    total_epochs: int = 3
    reward_num_workers: int = 1
    group_timeout_s: int = 180
    judge_max_concurrency: int = 400
    grpo_data_mode: str = "reasoning"
    run_suffix: str | None = None
    resume_mode: str = "auto"
    fresh_start: bool = False
    trainer_n_gpus_per_node: int = 8
    trainer_nnodes: int = 1
    tensor_model_parallel_size: int = 8
    gpu_memory_utilization: float = 0.6
    rollout_presence_penalty: float = REPO_TRAINING_PRESENCE_PENALTY
    max_model_len: int = 12288
    max_num_seqs: int = 32
    rollout_agent_num_workers: int = 8
    actor_ppo_micro_batch_size_per_gpu: int = 4
    max_response_length: int = 1024
    save_freq: int = 50
    project_name: str = "grpo-user-sim"
    model_name: str = "qwen3-8b"
    judge_model: str = "qwen/qwen3.5-397b-a17b"
    reward_manager_name: str = "GroupedSimRewardManager"
    reward_manager_module_path: str = str(REPO_ROOT / "finetuning" / "grpo" / "sim_reward_manager.py")

    @classmethod
    def from_env(cls) -> "SimGRPOConfig":
        fresh_start = _env_bool("FRESH_START", False)
        resume_mode = os.environ.get("RESUME_MODE", "auto")
        if fresh_start:
            resume_mode = "disable"

        trainer_n_gpus = _env_int("TRAINER_N_GPUS_PER_NODE", 8)
        return cls(
            num_users=_env_int("NUM_USERS", 600),
            rollout_n=_env_int("ROLLOUT_N", 4),
            train_batch_size=_env_int("TRAIN_BATCH_SIZE", 128),
            ppo_mini_batch_size=_env_int("PPO_MINI_BATCH_SIZE", 128),
            learning_rate=_env_float("LEARNING_RATE", _default_learning_rate()),
            kl_loss_coef=_env_float("SIM_KL_LOSS_COEF", 1e-3),
            total_epochs=_env_int("TOTAL_EPOCHS", 3),
            reward_num_workers=_env_int("SIM_REWARD_NUM_WORKERS", 1),
            group_timeout_s=_env_int("SIM_GROUP_TIMEOUT_S", 180),
            judge_max_concurrency=_env_int("SIM_JUDGE_MAX_CONCURRENCY", 400),
            grpo_data_mode=os.environ.get("GRPO_DATA_MODE", "reasoning"),
            run_suffix=os.environ.get("RUN_SUFFIX"),
            resume_mode=resume_mode,
            fresh_start=fresh_start,
            trainer_n_gpus_per_node=trainer_n_gpus,
            trainer_nnodes=_env_int("TRAINER_NNODES", 1),
            tensor_model_parallel_size=_env_int("TENSOR_MODEL_PARALLEL_SIZE", 8),
            gpu_memory_utilization=_env_float("ROLLOUT_GPU_MEMORY_UTILIZATION", 0.6),
            rollout_presence_penalty=_env_float("PERSONA_VLLM_PRESENCE_PENALTY", REPO_TRAINING_PRESENCE_PENALTY),
            max_model_len=_env_int("ROLLOUT_MAX_MODEL_LEN", 12288),
            max_num_seqs=_env_int("ROLLOUT_MAX_NUM_SEQS", 32),
            rollout_agent_num_workers=_env_int("ROLLOUT_AGENT_NUM_WORKERS", 8),
            actor_ppo_micro_batch_size_per_gpu=_env_int("PPO_MICRO_BATCH_SIZE_PER_GPU", 4),
            max_response_length=_env_int("MAX_RESPONSE_LENGTH", 1024),
            save_freq=_env_int("SAVE_FREQ", 50),
            model_name=os.environ.get("MODEL_NAME", "qwen3-8b"),
            judge_model=os.environ.get("SIM_JUDGE_MODEL", "qwen/qwen3.5-397b-a17b"),
        )

    @property
    def prompt_mode(self) -> str:
        return "reasoning"

    @property
    def resolved_run_suffix(self) -> str:
        if self.run_suffix:
            return self.run_suffix
        return f"{self.num_users}u_r{self.rollout_n}_sim_reasoning"

    @property
    def checkpoint_dir(self) -> str:
        return f"results/grpo/checkpoints_sim_{self.resolved_run_suffix}"

    @property
    def data_dir(self) -> str:
        return f"data/convokit/runs/sim_{self.resolved_run_suffix}"

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
        return f"results/grpo_gen/grpo_sim_{self.resolved_run_suffix}_gen1.pkl"

    @property
    def experiment_name(self) -> str:
        return f"{self.model_name}-grpo-sim-{self.resolved_run_suffix}-8gpu"

    @property
    def rollout_batch_size(self) -> int:
        return self.train_batch_size * self.rollout_n

    @property
    def enable_chat_template_thinking(self) -> bool:
        return resolve_chat_template_thinking_override(default=False)

    def shell_exports(self) -> dict[str, str]:
        return {
            "REWARD_METRIC": "sim",
            "SIM_JUDGE_MODEL": self.judge_model,
            "GRPO_DATA_MODE": self.grpo_data_mode,
            "NUM_USERS": str(self.num_users),
            "ROLLOUT_N": str(self.rollout_n),
            "TRAIN_BATCH_SIZE": str(self.train_batch_size),
            "PPO_MINI_BATCH_SIZE": str(self.ppo_mini_batch_size),
            "LEARNING_RATE": str(self.learning_rate),
            "SIM_KL_LOSS_COEF": str(self.kl_loss_coef),
            "TOTAL_EPOCHS": str(self.total_epochs),
            "SIM_REWARD_NUM_WORKERS": str(self.reward_num_workers),
            "SIM_GROUP_TIMEOUT_S": str(self.group_timeout_s),
            "SIM_JUDGE_MAX_CONCURRENCY": str(self.judge_max_concurrency),
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
            "TRAINER_N_GPUS_PER_NODE": str(self.trainer_n_gpus_per_node),
            "TRAINER_NNODES": str(self.trainer_nnodes),
            "TENSOR_MODEL_PARALLEL_SIZE": str(self.tensor_model_parallel_size),
            "ROLLOUT_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "PERSONA_VLLM_PRESENCE_PENALTY": str(self.rollout_presence_penalty),
            "ROLLOUT_MAX_MODEL_LEN": str(self.max_model_len),
            "ROLLOUT_MAX_NUM_SEQS": str(self.max_num_seqs),
            "ROLLOUT_AGENT_NUM_WORKERS": str(self.rollout_agent_num_workers),
            "PPO_MICRO_BATCH_SIZE_PER_GPU": str(self.actor_ppo_micro_batch_size_per_gpu),
            "MAX_RESPONSE_LENGTH": str(self.max_response_length),
            "SAVE_FREQ": str(self.save_freq),
            "SIM_PROMPT_MODE": self.prompt_mode,
        }

    def verl_overrides(self) -> list[str]:
        return [
            f"trainer.n_gpus_per_node={self.trainer_n_gpus_per_node}",
            f"trainer.nnodes={self.trainer_nnodes}",
            f"trainer.total_epochs={self.total_epochs}",
            f"trainer.project_name={self.project_name}",
            f"trainer.default_local_dir={self.checkpoint_dir}",
            f"trainer.experiment_name={self.experiment_name}",
            f"trainer.resume_mode={self.resume_mode}",
            f"trainer.save_freq={self.save_freq}",
            f"reward.num_workers={self.reward_num_workers}",
            "reward.reward_manager.source=importlib",
            f"reward.reward_manager.name={self.reward_manager_name}",
            f"reward.reward_manager.module.path={self.reward_manager_module_path}",
            f"+reward_model.reward_kwargs.n_rollouts={self.rollout_n}",
            f"+reward_model.reward_kwargs.judge_model={self.judge_model}",
            f"data.train_files={self.train_file}",
            f"data.val_files={self.val_file}",
            f"data.train_batch_size={self.train_batch_size}",
            "data.dataloader_num_workers=0",
            f"data.max_response_length={self.max_response_length}",
            f"data.apply_chat_template_kwargs.enable_thinking={str(self.enable_chat_template_thinking).lower()}",
            "critic.enable=false",
            "actor_rollout_ref.actor.use_kl_loss=true",
            f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
            "actor_rollout_ref.actor.loss_agg_mode=token-mean",
            f"critic.optim.lr={self.learning_rate}",
            f"actor_rollout_ref.actor.optim.lr={self.learning_rate}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={self.actor_ppo_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.rollout.n={self.rollout_n}",
            f"actor_rollout_ref.rollout.temperature={REPO_TRAINING_TEMPERATURE}",
            f"actor_rollout_ref.rollout.top_p={REPO_TRAINING_TOP_P}",
            f"actor_rollout_ref.rollout.top_k={REPO_TRAINING_TOP_K}",
            f"actor_rollout_ref.rollout.presence_penalty={self.rollout_presence_penalty}",
            f"actor_rollout_ref.rollout.response_length={self.max_response_length}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.tensor_model_parallel_size}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            f"actor_rollout_ref.rollout.max_model_len={self.max_model_len}",
            f"actor_rollout_ref.rollout.max_num_seqs={self.max_num_seqs}",
            f"actor_rollout_ref.rollout.agent.num_workers={self.rollout_agent_num_workers}",
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.layered_summon=true",
        ]


def _cmd_shell_exports() -> int:
    config = SimGRPOConfig.from_env()
    for key, value in config.shell_exports().items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


def _cmd_verl_overrides() -> int:
    config = SimGRPOConfig.from_env()
    for item in config.verl_overrides():
        print(item)
    return 0


def _cmd_summary() -> int:
    config = SimGRPOConfig.from_env()
    print(f"run_suffix={config.resolved_run_suffix}")
    print(f"checkpoint_dir={config.checkpoint_dir}")
    print(f"data_dir={config.data_dir}")
    print(f"rollout_batch_size={config.rollout_batch_size}")
    print(f"experiment_name={config.experiment_name}")
    print(f"prompt_mode={config.prompt_mode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared response-similarity GRPO launcher config helper.")
    parser.add_argument("command", choices=["shell-exports", "verl-overrides", "summary"])
    args = parser.parse_args()

    if args.command == "shell-exports":
        return _cmd_shell_exports()
    if args.command == "verl-overrides":
        return _cmd_verl_overrides()
    return _cmd_summary()


if __name__ == "__main__":
    raise SystemExit(main())
