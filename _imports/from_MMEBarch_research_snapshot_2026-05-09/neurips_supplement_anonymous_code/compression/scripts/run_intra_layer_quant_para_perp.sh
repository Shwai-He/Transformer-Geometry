#!/usr/bin/env bash
set -euo pipefail

# Intra-layer quant analysis (dense vs quantized model)
MODEL_NAME="/path/to/resource"
QUANT_MODEL_NAME="/path/to/resource"
METHOD_NAME="intra_bnb4_nf4"

EFFECT_SCOPE="global"              # global | local
FOCUS_LAYER=12                       # used only when EFFECT_SCOPE=local
PROMPTS_FILE="/path/to/resource"
MAX_PROMPTS=32
MAX_LENGTH=512
OUTPUT_DIR="/path/to/resource"

# Quant config
LOAD_IN_4BIT=true
LOAD_IN_8BIT=false
BNB_4BIT_QUANT_TYPE="nf4"          # nf4 | fp4
BNB_4BIT_COMPUTE_DTYPE="float16"   # float16 | bfloat16 | float32
STRICT_QUANT_LOADING=true

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/../code/layerwise_para_perp_compare.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/intra_layer_quant"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${METHOD_NAME}_${EFFECT_SCOPE}_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --model_name "$MODEL_NAME"
  --analysis_mode pruned
  --compare_mode dual_model
  --compression_type quant
  --effect_scope "$EFFECT_SCOPE"
  --method_name "$METHOD_NAME"
  --pruned_model_name "$QUANT_MODEL_NAME"
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
if [[ "$STRICT_QUANT_LOADING" == "true" ]]; then
  CMD+=(--strict_quant_loading)
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
