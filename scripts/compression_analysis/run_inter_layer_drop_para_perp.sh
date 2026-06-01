#!/usr/bin/env bash
set -euo pipefail

# Inter-layer drop analysis (dense vs dropped-mask model)
MODEL_NAME="/path/to/dense_model"
MODEL_TAG=""                      # optional; empty => auto from model_name in python
METHOD_NAME="inter_attn_drop8"
DROPPED_ROOT_PATH="/path/to/dropped_results"
TARGET_LAYER="attn"                # attn | mlp | all
DROP_N=8

EFFECT_SCOPE="global"              # global | local
FOCUS_LAYER=12                       # used only when EFFECT_SCOPE=local
PROMPTS_FILE="/path/to/prompts.txt"
MAX_PROMPTS=32
MAX_LENGTH=512
OUTPUT_DIR="representation-analysis/outputs/layerwise_para_perp"

PYTHON_BIN="python3"
BACKGROUND=true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/compression_analysis/inter_layer"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${METHOD_NAME}_${EFFECT_SCOPE}_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/layerwise_para_perp_compare.py"
  --model_name "$MODEL_NAME"
  --analysis_mode dropped
  --compression_type prune
  --effect_scope "$EFFECT_SCOPE"
  --method_name "$METHOD_NAME"
  --dropped_root_path "$DROPPED_ROOT_PATH"
  --target_layer "$TARGET_LAYER"
  --drop_n "$DROP_N"
  --prompts_file "$PROMPTS_FILE"
  --max_prompts "$MAX_PROMPTS"
  --max_length "$MAX_LENGTH"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MODEL_TAG" ]]; then
  CMD+=(--model_tag "$MODEL_TAG")
fi
if [[ "$EFFECT_SCOPE" == "local" ]]; then
  CMD+=(--focus_layer "$FOCUS_LAYER")
fi

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"
echo "Run cmd: ${CMD[*]}"

cd "$REPO_ROOT"
if [[ "$BACKGROUND" == "true" ]]; then
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
  echo "Started in background."
  echo "tail -f $LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
fi
