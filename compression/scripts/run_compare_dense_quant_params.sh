#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
DENSE_MODEL="/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B"
QUANT_MODEL="/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B-AWQ"
DTYPE="float16"  # float16 | bfloat16 | float32
TAG="qwen3_4b_vs_awq_param_sanity"
OUT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/focused-compression-analysis/outputs/dense_quant_params"

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/compare_dense_quant_params.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/dense_quant_params"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="dense_quant_params_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --dense_model "$DENSE_MODEL"
  --quant_model "$QUANT_MODEL"
  --dtype "$DTYPE"
  --out_dir "$OUT_DIR"
  --tag "$TAG"
)

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"
echo "Run cmd: ${CMD[*]}"

cd "$WORKSPACE_ROOT"
if [[ "$BACKGROUND" == "true" ]]; then
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
  echo "Started in background."
  echo "tail -f $LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
fi
