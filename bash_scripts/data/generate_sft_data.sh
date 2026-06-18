#!/usr/bin/env bash
# Usage: bash_scripts/data/generate_sft_data.sh <convokit|prism> <persona_inductor>

set -euo pipefail

DATASET="${1:-}"
PERSONA_INDUCTOR="${2:-}"
if [[ -z "$DATASET" || -z "$PERSONA_INDUCTOR" ]]; then
  echo "usage: $0 <convokit|prism> <persona_inductor_model>" >&2
  exit 2
fi

case "$PERSONA_INDUCTOR" in
  gpt-5.4-nano|opus4.8|qwen3-8b) ;;
  *)
    echo "bad persona_inductor: $PERSONA_INDUCTOR (expected gpt-5.4-nano|opus4.8|qwen3-8b)" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

SEED="${SEED:-42}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
COT_MODEL="${COT_MODEL:-qwen/qwen3-8b}"
COT_MAX_COMPLETION_TOKENS="${COT_MAX_COMPLETION_TOKENS:-4096}"
COT_MAX_REGEN_ATTEMPTS="${COT_MAX_REGEN_ATTEMPTS:-10}"
COT_LEAKAGE_NGRAM_SIZE="${COT_LEAKAGE_NGRAM_SIZE:-5}"
COT_LEAKAGE_MAX_MATCH_TOKENS="${COT_LEAKAGE_MAX_MATCH_TOKENS:-5}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-data/sft}"

INDUCTOR_SLUG="$(echo "$PERSONA_INDUCTOR" | tr '/:.' '___' | tr -cd '[:alnum:]_-')"
HP_BASE="${3:-${BUILD_NAME:-${DATASET}_history_persona_${INDUCTOR_SLUG}_s${SEED}}}"

DATA_DIR="data/${DATASET}"
HIST_SPLIT="${DATA_DIR}/${DATASET}_history_s${SEED}_sft40_grpo60"
HP_SPLIT="${DATA_DIR}/${HP_BASE}_sft40_grpo60"
PERSONA_SPLIT="${DATA_DIR}/${DATASET}_persona_${INDUCTOR_SLUG}_s${SEED}_sft40_grpo60"

run() { echo "+ $*" >&2; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

source_dir_for_mode() {
  case "$1" in
    history) echo "$HIST_SPLIT" ;;
    history_persona) echo "$HP_SPLIT" ;;
    persona) echo "$PERSONA_SPLIT" ;;
    *)
      echo "unknown conditioning mode: $1" >&2
      exit 2
      ;;
  esac
}

output_base_for_mode() {
  case "$1" in
    history) echo "${DATASET}_history_s${SEED}" ;;
    history_persona) echo "$HP_BASE" ;;
    persona) echo "${DATASET}_persona_${INDUCTOR_SLUG}_s${SEED}" ;;
  esac
}

generate_for_mode() {
  local mode="$1"
  local split_dir source_parquet base cot_parquet cot_jsonl

  split_dir="$(source_dir_for_mode "$mode")"
  source_parquet="${split_dir}/sft/train.parquet"
  if [[ ! -f "$source_parquet" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "missing source SFT parquet: $source_parquet" >&2
    echo "run bash_scripts/data/generate_data.sh $DATASET $PERSONA_INDUCTOR first" >&2
    exit 1
  fi

  base="$(output_base_for_mode "$mode")"
  cot_parquet="${SFT_OUTPUT_DIR}/${base}_sft_cot.parquet"
  cot_jsonl="${SFT_OUTPUT_DIR}/${base}_sft_cot.jsonl"

  echo ">> [$mode] source: $source_parquet" >&2
  run mkdir -p "$SFT_OUTPUT_DIR"
  run "$PY" -m data.sft.generate_cot \
    --input "$source_parquet" \
    --output "$cot_parquet" \
    --model "$COT_MODEL" \
    --max_completion_tokens "$COT_MAX_COMPLETION_TOKENS" \
    --max_regen_attempts "$COT_MAX_REGEN_ATTEMPTS" \
    --leakage_ngram_size "$COT_LEAKAGE_NGRAM_SIZE" \
    --leakage_max_match_tokens "$COT_LEAKAGE_MAX_MATCH_TOKENS"

  run "$PY" -m data.sft.build_sft_jsonl \
    --input_parquet "$cot_parquet" \
    --output_jsonl "$cot_jsonl"

  echo ">> [$mode] wrote:" >&2
  echo ">>   $cot_parquet" >&2
  echo ">>   $cot_jsonl" >&2
}

if [[ -n "${CONDITIONING_MODES:-}" ]]; then
  read -r -a MODES <<< "$CONDITIONING_MODES"
else
  MODES=(history history_persona persona)
fi

case "$DATASET" in
  convokit|prism) ;;
  *)
    echo "unknown dataset: $DATASET (expected convokit|prism)" >&2
    exit 2
    ;;
esac

echo ">> dataset=$DATASET inductor=$PERSONA_INDUCTOR seed=$SEED tokenizer=$TOKENIZER" >&2
echo ">> cot_model=$COT_MODEL output_dir=$SFT_OUTPUT_DIR" >&2
echo ">> conditioning_modes=${MODES[*]}" >&2

for mode in "${MODES[@]}"; do
  generate_for_mode "$mode"
done

echo ">> done. SFT outputs are under $SFT_OUTPUT_DIR" >&2
