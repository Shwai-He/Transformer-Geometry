#!/usr/bin/env bash
set -euo pipefail

# Intra-layer quant analysis (dense vs quantized model)
MODEL_NAME="${MODEL_NAME:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
QUANT_MODEL_NAME="${QUANT_MODEL_NAME:-$MODEL_NAME}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"
METHOD_NAME="${METHOD_NAME:-fake_int4_g128_global}"

EFFECT_SCOPE="${EFFECT_SCOPE:-global}"              # global | local
FOCUS_LAYER="${FOCUS_LAYER:-12}"                    # used only when EFFECT_SCOPE=local
PROMPTS_FILE="${PROMPTS_FILE:-results/prompts.txt}"
MAX_PROMPTS="${MAX_PROMPTS:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-compression/outputs/layerwise_para_perp}"

# Quant config
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"          # nf4 | fp4
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-float16}" # float16 | bfloat16 | float32
STRICT_QUANT_LOADING="${STRICT_QUANT_LOADING:-true}"
FAKE_QUANT_BITS="${FAKE_QUANT_BITS:-4}"
FAKE_QUANT_GROUP_SIZE="${FAKE_QUANT_GROUP_SIZE:-128}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKGROUND="${BACKGROUND:-false}"

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
  --model_tag "$MODEL_TAG"
  --pruned_model_name "$QUANT_MODEL_NAME"
  --prompts_file "$PROMPTS_FILE"
  --max_prompts "$MAX_PROMPTS"
  --max_length "$MAX_LENGTH"
  --output_dir "$OUTPUT_DIR"
  --bnb_4bit_quant_type "$BNB_4BIT_QUANT_TYPE"
  --bnb_4bit_compute_dtype "$BNB_4BIT_COMPUTE_DTYPE"
  --fake_quant_bits "$FAKE_QUANT_BITS"
  --fake_quant_group_size "$FAKE_QUANT_GROUP_SIZE"
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
