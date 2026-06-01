#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
MODEL_NAME="/path/to/dense_model"
ANALYSIS_MODE="pruned"          # dropped | pruned
COMPRESSION_TYPE="prune"        # prune | quant
EFFECT_SCOPE="global"           # global | local
FOCUS_LAYER=12                   # used when EFFECT_SCOPE=local
METHOD_NAME="wanda_global"
PRUNED_MODEL_NAME="/path/to/compressed_model"
DROPPED_ROOT_PATH="/path/to/dropped_results"
TARGET_LAYER="attn"             # attn | mlp | all
DROP_N=8
PROMPTS_FILE="/path/to/prompts.txt"
MAX_PROMPTS=32
MAX_LENGTH=512
OUTPUT_DIR=""

# Quantization options (used when COMPRESSION_TYPE=quant)
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
BNB_4BIT_QUANT_TYPE="nf4"       # nf4 | fp4
BNB_4BIT_COMPUTE_DTYPE="float16" # float16 | bfloat16 | float32

# Runtime options
PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/../code/layerwise_para_perp_compare.py"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$WORKSPACE_ROOT/outputs/layerwise_para_perp"
fi
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${METHOD_NAME}_${EFFECT_SCOPE}_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --model_name "$MODEL_NAME"
  --analysis_mode "$ANALYSIS_MODE"
  --compression_type "$COMPRESSION_TYPE"
  --effect_scope "$EFFECT_SCOPE"
  --method_name "$METHOD_NAME"
  --pruned_model_name "$PRUNED_MODEL_NAME"
  --dropped_root_path "$DROPPED_ROOT_PATH"
  --target_layer "$TARGET_LAYER"
  --drop_n "$DROP_N"
  --prompts_file "$PROMPTS_FILE"
  --max_prompts "$MAX_PROMPTS"
  --max_length "$MAX_LENGTH"
  --output_dir "$OUTPUT_DIR"
  --bnb_4bit_quant_type "$BNB_4BIT_QUANT_TYPE"
  --bnb_4bit_compute_dtype "$BNB_4BIT_COMPUTE_DTYPE"
)

if [[ "$EFFECT_SCOPE" == "local" ]]; then
  CMD+=(--focus_layer "$FOCUS_LAYER")
fi

if [[ "$LOAD_IN_4BIT" == "true" ]]; then
  CMD+=(--load_in_4bit)
fi
if [[ "$LOAD_IN_8BIT" == "true" ]]; then
  CMD+=(--load_in_8bit)
fi

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"
echo "Run cmd: ${CMD[*]}"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

cd "$WORKSPACE_ROOT"

if [[ "$BACKGROUND" == "true" ]]; then
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
  echo "Started in background."
  echo "tail -f $LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
fi
