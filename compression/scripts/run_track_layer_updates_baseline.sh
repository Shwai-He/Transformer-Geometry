#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B"
PROMPTS_FILE="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/prompts.txt"
MAX_PROMPTS=32
MAX_LENGTH=512
OUTPUT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/compression/outputs/layerwise_para_perp"
METHOD_NAME="dense_baseline_track"

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/track_layer_updates_baseline.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/baseline_track"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/${STAMP}_${METHOD_NAME}.log"
PID_PATH="$LOG_DIR/${STAMP}_${METHOD_NAME}.pid"

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --model_name "$MODEL_NAME"
  --prompts_file "$PROMPTS_FILE"
  --max_prompts "$MAX_PROMPTS"
  --max_length "$MAX_LENGTH"
  --output_dir "$OUTPUT_DIR"
  --method_name "$METHOD_NAME"
  --baseline_mode aligned_drop
  --ref_mode baseline_update
)

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"
echo "Run cmd: ${CMD[*]}"

if [[ "$BACKGROUND" == "true" ]]; then
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
  echo "Started in background."
  echo "tail -f $LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
fi
