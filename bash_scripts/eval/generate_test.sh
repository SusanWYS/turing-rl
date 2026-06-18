#!/usr/bin/env bash
# Usage: bash_scripts/eval/generate_test.sh <model> <data> <condition> <train_mode> <persona_inductor>

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

MAX_TOKENS="${MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
USER_OFFSET="${USER_OFFSET:-0}"
NUM_USERS="${NUM_USERS:-}"

DRY="${DRY_RUN:-0}"
if [[ "$DRY" != "1" && ! -f "$TEST_PARQUET" ]]; then
  echo "error: heldout test parquet not found: $TEST_PARQUET" >&2
  echo "  build it first: bash_scripts/data/generate_data.sh $DATA $PERSONA_INDUCTOR" >&2
  exit 1
fi
[[ "$DRY" != "1" ]] && mkdir -p "$GEN_DIR"

if [[ "$IS_TRAINED" == "1" ]]; then
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"
  if [[ "$DRY" == "1" ]]; then
    :
  else
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
      echo "error: checkpoint dir not found: $CHECKPOINT_DIR" >&2
      echo "  inferred from: model=$MODEL data=$DATA condition=$CONDITION train_mode=$TRAIN_MODE persona_inductor=$PERSONA_INDUCTOR" >&2
      echo "  override with: CHECKPOINT_DIR=/path/to/checkpoint_dir" >&2
      exit 1
    fi
  fi

  extra=()
  [[ "$CONDITION" != "history" ]] && extra+=(--persona_path "$PERSONA_MAP")
  [[ "$TRAIN_MODE" != "sft" ]] && extra+=(--metric "$TRAIN_MODE")
  [[ -n "$NUM_USERS" ]] && extra+=(--num_users "$NUM_USERS")
  [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]] && extra+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
  [[ -n "${VLLM_TRUNCATE_PROMPT_TOKENS:-}" ]] && extra+=(--vllm_truncate_prompt_tokens "$VLLM_TRUNCATE_PROMPT_TOKENS")

  echo ">> generate (trained $TRAIN_MODE, $MODEL): $GEN_PKL" >&2
  run "$PY" -m eval.generate_trained \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_id "$BASE_MODEL" \
    --conditioning_mode "$CONDITION" \
    --test_parquet "$TEST_PARQUET" \
    --gen_num "$GEN_NUM" --max_tokens "$MAX_TOKENS" --user_offset "$USER_OFFSET" \
    --vllm_tensor_parallel_size "${VLLM_TP:-1}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM:-0.6}" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-32}" \
    ${extra[@]+"${extra[@]}"} \
    --output "$GEN_PKL"
else
  extra=()
  [[ -n "$NUM_USERS" ]] && extra+=(--num_users "$NUM_USERS")
  if [[ "$MODEL" == "gpt-5" ]]; then
    extra+=(--openai_model "${OPENAI_MODEL:-gpt-5-mini}")
    [[ -n "${REASONING_EFFORT:-}" ]] && extra+=(--reasoning_effort "$REASONING_EFFORT")
  fi

  echo ">> generate (baseline $MODEL): $GEN_PKL" >&2
  run "$PY" -m eval.generate_baselines \
    --model "$MODEL" \
    --conditioning_mode "$CONDITION" \
    --test_parquet "$TEST_PARQUET" \
    --gen_num "$GEN_NUM" --max_tokens "$MAX_TOKENS" --user_offset "$USER_OFFSET" \
    --max_workers "$MAX_WORKERS" \
    ${extra[@]+"${extra[@]}"} \
    --output "$GEN_PKL"
fi

echo ">> done. generations: $GEN_PKL" >&2
