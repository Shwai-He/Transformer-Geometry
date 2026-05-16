#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
DENSE_MODEL="/path/to/resource"
WANDA_MODEL="/path/to/resource"
DTYPE="float16"  # float16 | bfloat16 | float32
TAG="qwen3_4b_vs_wanda_param_sanity"
OUT_DIR="/path/to/resource"

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/compare_dense_quant_params.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/dense_wanda_params"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="dense_wanda_params_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --dense_model "$DENSE_MODEL"
  --quant_model "$WANDA_MODEL"
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
