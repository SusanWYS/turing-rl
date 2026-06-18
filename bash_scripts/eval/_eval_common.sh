# Shared eval argument parsing and path derivation.

set -euo pipefail

MODEL="${1:-}"
DATA="${2:-}"
CONDITION="${3:-}"
TRAIN_MODE="${4:-}"
PERSONA_INDUCTOR="${5:-}"

_usage() {
  cat >&2 <<USAGE
usage: $(basename "$0") <model> <data> <condition> <train_mode> <persona_inductor> <metric>
  model             qwen3-8b | qwen3.5-397b | gpt-5
  data              convokit | prism
  condition         history | persona | history_persona
  train_mode        none | sft | turing | sim | logprob   (none for qwen3.5-397b / gpt-5)
  persona_inductor  inductor slug, required for persona/history_persona
  <metric>          turing | sim | specificity | all       (score_test.sh only; default all)
USAGE
  exit 2
}

[[ -z "$MODEL" || -z "$DATA" || -z "$CONDITION" || -z "$TRAIN_MODE" ]] && _usage
case "$MODEL" in qwen3-8b|qwen3.5-397b|gpt-5) ;; *) echo "bad model: $MODEL" >&2; _usage ;; esac
case "$DATA" in convokit|prism) ;; *) echo "bad data: $DATA" >&2; _usage ;; esac
case "$CONDITION" in history|persona|history_persona) ;; *) echo "bad condition: $CONDITION" >&2; _usage ;; esac
case "$TRAIN_MODE" in none|sft|turing|sim|logprob) ;; *) echo "bad train_mode: $TRAIN_MODE" >&2; _usage ;; esac
if [[ "$CONDITION" != "history" && -z "$PERSONA_INDUCTOR" ]]; then
  echo "error: persona_inductor is required for condition=$CONDITION" >&2
  _usage
fi
if [[ -n "$PERSONA_INDUCTOR" ]]; then
  case "$PERSONA_INDUCTOR" in
    gpt-5.4-nano|opus4.8|qwen3-8b) ;;
    *) echo "bad persona_inductor: $PERSONA_INDUCTOR" >&2; _usage ;;
  esac
fi

if [[ "$TRAIN_MODE" != "none" && "$MODEL" != "qwen3-8b" ]]; then
  echo "error: train_mode=$TRAIN_MODE is only valid for model qwen3-8b; $MODEL is baseline-only (use train_mode=none)." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
GEN_NUM="${GEN_NUM:-1}"
MAX_WORKERS="${MAX_WORKERS:-100}"
case "$MODEL" in
  qwen3-8b) DEFAULT_MAX_TOKENS=2048 ;;
  qwen3.5-397b|gpt-5) DEFAULT_MAX_TOKENS=1024 ;;
esac

if [[ "$CONDITION" == "history" ]]; then
  INDUCTOR_SLUG=""
else
  INDUCTOR_SLUG="$(echo "$PERSONA_INDUCTOR" | tr '/:.' '___' | tr -cd '[:alnum:]_-')"
fi

case "$CONDITION" in
  history)         BUILD="${DATA}_history_s${SEED}_sft40_grpo60" ;;
  history_persona) BUILD="${DATA}_history_persona_${INDUCTOR_SLUG}_s${SEED}_sft40_grpo60" ;;
  persona)         BUILD="${DATA}_persona_${INDUCTOR_SLUG}_s${SEED}_sft40_grpo60" ;;
esac
BUILD_DIR="data/${DATA}/${BUILD}"
TEST_PARQUET="${BUILD_DIR}/test.parquet"
PERSONA_MAP="data/${DATA}/personas_${INDUCTOR_SLUG}_s${SEED}.jsonl"

if [[ "$TRAIN_MODE" == "none" ]]; then
  IS_TRAINED=0; TRAIN_TAG="notrain"; GEN_DIR="results/sft_gen"; EVAL_ROOT="results/sft_eval"
elif [[ "$TRAIN_MODE" == "sft" ]]; then
  IS_TRAINED=1; TRAIN_TAG="sft"; GEN_DIR="results/sft_gen"; EVAL_ROOT="results/sft_eval"
  DEFAULT_CHECKPOINT_DIR="results/sft/qwen3-8b_${DATA}_${CONDITION}_${INDUCTOR_SLUG}_sft_cot"
else
  IS_TRAINED=1; TRAIN_TAG="$TRAIN_MODE"; GEN_DIR="results/grpo_gen"; EVAL_ROOT="results/grpo_eval"
  DEFAULT_CHECKPOINT_DIR="results/grpo/checkpoints_qwen3_8b_grpo_${TRAIN_MODE}_${DATA}_${CONDITION}_${INDUCTOR_SLUG}"
fi
TAG="${MODEL}_${TRAIN_TAG}_${DATA}_${CONDITION}"
[[ -n "$INDUCTOR_SLUG" ]] && TAG="${TAG}_${INDUCTOR_SLUG}"
GEN_PKL="${GEN_DIR}/${TAG}_gen${GEN_NUM}.pkl"
EVAL_DIR="${EVAL_ROOT}/${TAG}_gen${GEN_NUM}"

run() { echo "+ $*" >&2; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }
