#!/usr/bin/env bash
# Usage: bash_scripts/eval/score_test.sh <model> <data> <condition> <train_mode> <persona_inductor> <metric>

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

if [[ "$CONDITION" == "history" ]]; then
  case "${5:-}" in
    ""|turing|sim|specificity|all) EVAL="${5:-all}" ;;
    *) EVAL="${6:-all}" ;;
  esac
else
  EVAL="${6:-all}"
fi
case "$EVAL" in turing|sim|specificity|all) ;; *) echo "bad eval: $EVAL (turing|sim|specificity|all)" >&2; exit 2 ;; esac

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$GEN_PKL" ]]; then
  echo "error: generations not found: $GEN_PKL" >&2
  echo "  run generation first:" >&2
  generate_cmd=(bash_scripts/eval/generate_test.sh "$MODEL" "$DATA" "$CONDITION" "$TRAIN_MODE")
  [[ -n "$PERSONA_INDUCTOR" ]] && generate_cmd+=("$PERSONA_INDUCTOR")
  printf '  %q' "${generate_cmd[@]}" >&2
  printf '\n' >&2
  exit 1
fi
[[ "${DRY_RUN:-0}" != "1" ]] && mkdir -p "$EVAL_DIR"

extra=()
[[ -n "${EVAL_NUM:-}" ]] && extra+=(--eval_num "$EVAL_NUM")

if [[ "$IS_TRAINED" == "1" ]]; then
  echo ">> score (trained $TRAIN_MODE, eval=$EVAL): $EVAL_DIR" >&2
  run "$PY" -m eval.score_trained \
    --input "$GEN_PKL" --eval "$EVAL" \
    --output_dir "$EVAL_DIR" --max_workers "$MAX_WORKERS" \
    ${extra[@]+"${extra[@]}"}
else
  echo ">> score (baseline $MODEL, eval=$EVAL): $EVAL_DIR" >&2
  run "$PY" -m eval.score_baselines \
    --model "$MODEL" --input "$GEN_PKL" --eval "$EVAL" \
    --output_dir "$EVAL_DIR" --max_workers "$MAX_WORKERS" \
    ${extra[@]+"${extra[@]}"}
fi

echo ">> done. eval results: $EVAL_DIR" >&2
