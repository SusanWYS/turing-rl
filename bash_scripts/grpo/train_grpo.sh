#!/usr/bin/env bash
# Usage: bash_scripts/grpo/train_grpo.sh <reward> <data> <condition> <persona_inductor>

set -euo pipefail

REWARD="${1:-}"; DATA="${2:-}"; CONDITION="${3:-}"; PERSONA_INDUCTOR="${4:-}"
if [[ -z "$REWARD" || -z "$DATA" || -z "$CONDITION" || -z "$PERSONA_INDUCTOR" ]]; then
  echo "usage: $0 <turing|sim|logprob> <convokit|prism> <history|persona|history_persona> <persona_inductor>" >&2
  exit 2
fi
case "$REWARD" in turing|sim|logprob) ;; *) echo "bad reward: $REWARD (turing|sim|logprob)" >&2; exit 2 ;; esac
case "$DATA" in convokit|prism) ;; *) echo "bad dataset: $DATA (convokit|prism)" >&2; exit 2 ;; esac
case "$CONDITION" in history|persona|history_persona) ;; *) echo "bad condition: $CONDITION" >&2; exit 2 ;; esac
case "$PERSONA_INDUCTOR" in
  gpt-5.4-nano|opus4.8|qwen3-8b) ;;
  *) echo "bad persona_inductor: $PERSONA_INDUCTOR (gpt-5.4-nano|opus4.8|qwen3-8b)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"
SEED="${SEED:-42}"

INDUCTOR_SLUG="$(echo "$PERSONA_INDUCTOR" | tr '/:.' '___' | tr -cd '[:alnum:]_-')"
case "$CONDITION" in
  history)         BUILD="${DATA}_history_s${SEED}_sft40_grpo60" ;;
  history_persona) BUILD="${DATA}_history_persona_${INDUCTOR_SLUG}_s${SEED}_sft40_grpo60" ;;
  persona)         BUILD="${DATA}_persona_${INDUCTOR_SLUG}_s${SEED}_sft40_grpo60" ;;
esac
TRAIN_FILE="data/${DATA}/${BUILD}/grpo/train.parquet"
VAL_FILE="data/${DATA}/${BUILD}/grpo/val.parquet"

CONFIG_NAME="qwen3_8b_grpo_${REWARD}"
EXP="qwen3_8b_grpo_${REWARD}_${DATA}_${CONDITION}_${INDUCTOR_SLUG}"
CHECKPOINT_DIR="results/grpo/checkpoints_${EXP}"
SFT_OUTPUT_DIR="results/sft/qwen3-8b_${DATA}_${CONDITION}_${INDUCTOR_SLUG}_sft_cot"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${SFT_OUTPUT_DIR}/final}"

detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n; n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
    [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]] && { echo "$n"; return; }
  fi
  echo 8
}
N_GPUS="${N_GPUS_PER_NODE:-$(detect_gpus)}"
NNODES="${NNODES:-1}"

export REWARD_METRIC="$REWARD"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-397b-a17b}"
export PERSONA_EVAL_JUDGE_MODEL="${PERSONA_EVAL_JUDGE_MODEL:-$JUDGE_MODEL}"
export PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY="${PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY:-512}"
export PERSONA_OPENAI_MAX_RETRIES="${PERSONA_OPENAI_MAX_RETRIES:-3}"
[[ "$REWARD" == "sim" ]] && export SIM_JUDGE_MODEL="${SIM_JUDGE_MODEL:-$JUDGE_MODEL}"

run() { echo "+ $*" >&2; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$TRAIN_FILE" ]]; then
  echo "error: GRPO train parquet not found: $TRAIN_FILE" >&2
  echo "  build it first: bash_scripts/data/generate_data.sh $DATA $PERSONA_INDUCTOR" >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" && ! -f "$SFT_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "error: SFT adapter not found: $SFT_ADAPTER_PATH" >&2
  echo "  train it first: bash_scripts/sft/train_sft.sh $DATA $CONDITION $PERSONA_INDUCTOR" >&2
  echo "  or override with: SFT_ADAPTER_PATH=<adapter_dir> $0 $REWARD $DATA $CONDITION $PERSONA_INDUCTOR" >&2
  exit 1
fi

echo ">> GRPO $REWARD | qwen3-8b | $DATA/$CONDITION | inductor=$PERSONA_INDUCTOR | gpus=${N_GPUS}x${NNODES}" >&2
echo ">> config=$CONFIG_NAME  data=$TRAIN_FILE  sft_adapter=$SFT_ADAPTER_PATH  ->  $CHECKPOINT_DIR" >&2
run "$PY" -m training.grpo.run_verl_main_ppo \
  --config-dir training/grpo/configs \
  --config-name "$CONFIG_NAME" \
  actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER_PATH" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE"

echo ">> done. checkpoints: $CHECKPOINT_DIR" >&2
