#!/usr/bin/env bash
# Usage: bash_scripts/sft/train_sft.sh <data> <condition> <persona_inductor>

set -euo pipefail

DATA="${1:-}"; CONDITION="${2:-}"; PERSONA_INDUCTOR="${3:-}"
if [[ -z "$DATA" || -z "$CONDITION" || -z "$PERSONA_INDUCTOR" ]]; then
  echo "usage: $0 <convokit|prism> <history|persona|history_persona> <persona_inductor>" >&2
  exit 2
fi
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
  history)         BASE="${DATA}_history_s${SEED}" ;;
  history_persona) BASE="${DATA}_history_persona_${INDUCTOR_SLUG}_s${SEED}" ;;
  persona)         BASE="${DATA}_persona_${INDUCTOR_SLUG}_s${SEED}" ;;
esac
SFT_JSONL="${SFT_DATA_DIR:-data/sft}/${BASE}_sft_cot.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-results/sft/qwen3-8b_${DATA}_${CONDITION}_${INDUCTOR_SLUG}_sft_cot}"

run() { echo "+ $*" >&2; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$SFT_JSONL" ]]; then
  echo "error: CoT SFT jsonl not found: $SFT_JSONL" >&2
  echo "  build it first: bash_scripts/data/generate_sft_data.sh $DATA $PERSONA_INDUCTOR" >&2
  exit 1
fi

extra=()
[[ -n "${MAX_SEQ_LENGTH:-}" ]] && extra+=(--max_seq_length "$MAX_SEQ_LENGTH")
[[ -n "${BASE_ADAPTER:-}" ]] && extra+=(--base_adapter "$BASE_ADAPTER")

echo ">> SFT (CoT) | qwen3-8b | $DATA/$CONDITION | inductor=$PERSONA_INDUCTOR" >&2
echo ">> data=$SFT_JSONL  ->  $OUTPUT_DIR" >&2
run "$PY" -m training.sft.lora_sft \
  --model qwen3-8b \
  --data_path "$SFT_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  ${extra[@]+"${extra[@]}"}

echo ">> done. checkpoints: $OUTPUT_DIR" >&2
