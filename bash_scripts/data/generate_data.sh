#!/usr/bin/env bash
# Usage: bash_scripts/data/generate_data.sh <convokit|prism> <persona_inductor>

set -euo pipefail

DATASET="${1:-}"
PERSONA_INDUCTOR="${2:-}"
if [[ -z "$DATASET" || -z "$PERSONA_INDUCTOR" ]]; then
  echo "usage: $0 <convokit|prism> <persona_inductor_model>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

SEED="${SEED:-42}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
PROMPT_MODE="reasoning"

case "$PERSONA_INDUCTOR" in
  qwen3-8b)
    PERSONA_TEMPERATURE="${PERSONA_TEMPERATURE:-0.6}"
    PERSONA_MAX_COMPLETION_TOKENS="${PERSONA_MAX_COMPLETION_TOKENS:-2048}"
    PERSONA_TOP_P="${PERSONA_TOP_P:-0.95}"
    PERSONA_TOP_K="${PERSONA_TOP_K:-20}"
    PERSONA_MIN_P="${PERSONA_MIN_P:-0.0}"
    ;;
  opus4.8)
    PERSONA_TEMPERATURE="${PERSONA_TEMPERATURE:-0.0}"
    PERSONA_MAX_COMPLETION_TOKENS="${PERSONA_MAX_COMPLETION_TOKENS:-1024}"
    ;;
  gpt-5.4-nano)
    PERSONA_TEMPERATURE="${PERSONA_TEMPERATURE:-0.0}"
    PERSONA_MAX_COMPLETION_TOKENS="${PERSONA_MAX_COMPLETION_TOKENS:-1024}"
    ;;
  *)
    echo "bad persona_inductor: $PERSONA_INDUCTOR (expected gpt-5.4-nano|opus4.8|qwen3-8b)" >&2
    exit 2
    ;;
esac
PERSONA_MAX_HISTORY_WORDS="${PERSONA_MAX_HISTORY_WORDS:-8192}"
PERSONA_WORKERS="${PERSONA_WORKERS:-24}"
PERSONA_ATTEMPTS="${PERSONA_ATTEMPTS:-3}"
if [[ -z "${PERSONA_REASONING:-}" ]]; then
  if [[ "$PERSONA_INDUCTOR" == "qwen3-8b" ]]; then
    PERSONA_REASONING=1
  else
    PERSONA_REASONING=0
  fi
fi
REASONING_FLAG=()
[[ "$PERSONA_REASONING" == "1" ]] && REASONING_FLAG=(--reasoning)
SAMPLING_ARGS=()
[[ -n "${PERSONA_TOP_P:-}" ]] && SAMPLING_ARGS+=(--top_p "$PERSONA_TOP_P")
[[ -n "${PERSONA_TOP_K:-}" ]] && SAMPLING_ARGS+=(--top_k "$PERSONA_TOP_K")
[[ -n "${PERSONA_MIN_P:-}" ]] && SAMPLING_ARGS+=(--min_p "$PERSONA_MIN_P")

INDUCTOR_SLUG="$(echo "$PERSONA_INDUCTOR" | tr '/:.' '___' | tr -cd '[:alnum:]_-')"
HP_BASE="${3:-${BUILD_NAME:-${DATASET}_history_persona_${INDUCTOR_SLUG}_s${SEED}}}"

DATA_DIR="data/${DATASET}"
HIST_DIR="${DATA_DIR}/${DATASET}_history_s${SEED}"
HIST_SPLIT="${DATA_DIR}/${DATASET}_history_s${SEED}_sft40_grpo60"
HP_SPLIT="${DATA_DIR}/${HP_BASE}_sft40_grpo60"
PERSONA_SPLIT="${DATA_DIR}/${DATASET}_persona_${INDUCTOR_SLUG}_s${SEED}_sft40_grpo60"
PERSONA_MAP="${DATA_DIR}/personas_${INDUCTOR_SLUG}_s${SEED}.jsonl"

CONVOKIT_TRAIN_SUBREDDITS=(AmItheAsshole AskMen AskWomen business changemyview Frugal news relationship_advice tifu worldnews)
CONVOKIT_TEST_SUBREDDITS=(Economics TrueReddit relationships MaliciousCompliance)
CONVOKIT_CORPUS_DIR="data/convokit/subreddit_corpora"

run() { echo "+ $*" >&2; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

build_history() {
  local out="$1"
  mkdir -p "$out"
  case "$DATASET" in
    convokit)
      run "$PY" -m data.convokit.build \
        --download --corpus_zip_dir "$CONVOKIT_CORPUS_DIR" \
        --train_subreddits "${CONVOKIT_TRAIN_SUBREDDITS[@]}" \
        --test_subreddits "${CONVOKIT_TEST_SUBREDDITS[@]}" \
        --min_conversations 8 --val_frac 0.3 \
        --random_history_count_min 2 --random_history_count_max 6 --history_count_seed "$SEED" \
        --conditioning_mode history --mode "$PROMPT_MODE" \
        --tokenizer "$TOKENIZER" --data_source reddit_user_sim_mixed \
        --shuffle_rows --shuffle_seed "$SEED" \
        --output "$out/train.parquet" \
        --val_output "$out/val.parquet" \
        --test_output "$out/test.parquet"
      ;;
    prism)
      run "$PY" -m data.prism.build \
        --dataset_name HannahRoseKirk/prism-alignment --config_name conversations --split train \
        --min_conversations 6 --heldout_user_frac 0.4 \
        --history_count_min 2 --history_count_max 4 \
        --eval_thread_frac 0.3 --prediction_start_turn 1 --split_seed "$SEED" \
        --conditioning_mode history --mode "$PROMPT_MODE" \
        --tokenizer "$TOKENIZER" --data_source prism_alignment_user_sim \
        --shuffle_rows --shuffle_seed "$SEED" \
        --output "$out/train.parquet" \
        --val_output "$out/val.parquet" \
        --test_output "$out/test.parquet"
      ;;
    *)
      echo "unknown dataset: $DATASET (expected convokit|prism)" >&2
      exit 2
      ;;
  esac
}

induce_personas() {
  run "$PY" -m data.induce_personas \
    --input_parquet "$HIST_DIR/train.parquet" "$HIST_DIR/val.parquet" "$HIST_DIR/test.parquet" \
    --output_jsonl "$PERSONA_MAP" \
    --model "$PERSONA_INDUCTOR" \
    ${REASONING_FLAG[@]+"${REASONING_FLAG[@]}"} \
    --temperature "$PERSONA_TEMPERATURE" \
    --max_completion_tokens "$PERSONA_MAX_COMPLETION_TOKENS" \
    --max_history_words "$PERSONA_MAX_HISTORY_WORDS" \
    ${SAMPLING_ARGS[@]+"${SAMPLING_ARGS[@]}"} \
    --workers "$PERSONA_WORKERS" --attempts "$PERSONA_ATTEMPTS"
}

split_data() {
  local in="$1" out="$2"
  case "$DATASET" in
    convokit)
      run "$PY" -m data.convokit.split_data \
        --input-dir "$in" --output-dir "$out" \
        --heldout-subreddits r/tifu r/worldnews \
        --grpo-frac 0.6 --grpo-val-frac 0.3 --seed "$SEED"
      ;;
    prism)
      run "$PY" -m data.prism.split_data \
        --input-dir "$in" --output-dir "$out" \
        --heldout-user-frac 0.1 --grpo-frac 0.6 --grpo-val-frac 0.1 --seed "$SEED"
      ;;
  esac
}

derive_mode() {
  local mode="$1" out="$2"
  run "$PY" -m data.conditioning_variants \
    --input-dir "$HIST_SPLIT" --output-dir "$out" \
    --conditioning-mode "$mode" \
    --persona-path "$PERSONA_MAP" \
    --tokenizer "$TOKENIZER"
}

echo ">> dataset=$DATASET inductor=$PERSONA_INDUCTOR (reasoning=$PERSONA_REASONING) seed=$SEED" >&2
echo ">> [1/5] build history (only full build): $HIST_DIR" >&2
build_history "$HIST_DIR"

echo ">> [2/5] induce personas -> $PERSONA_MAP" >&2
induce_personas

echo ">> [3/5] split history into sft/grpo/heldout: $HIST_SPLIT" >&2
split_data "$HIST_DIR" "$HIST_SPLIT"

echo ">> [4/5] derive history_persona (attach personas, re-render): $HP_SPLIT" >&2
derive_mode history_persona "$HP_SPLIT"

echo ">> [5/5] derive persona (attach personas, re-render): $PERSONA_SPLIT" >&2
derive_mode persona "$PERSONA_SPLIT"

echo ">> done. builds (identical splits):" >&2
echo ">>   history          $HIST_SPLIT" >&2
echo ">>   history_persona   $HP_SPLIT" >&2
echo ">>   persona           $PERSONA_SPLIT" >&2
