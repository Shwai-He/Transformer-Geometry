#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
AWQ_MODEL="/path/to/resource"
OUT_DIR="/path/to/resource"
DTYPE="float16"  # float16 | bfloat16 | float32
SAVE_SAFETENSORS=false

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/export_awq_dequantized_dense.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/export_awq_dequantized_dense"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="export_awq_dequantized_dense_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --awq_model "$AWQ_MODEL"
  --out_dir "$OUT_DIR"
  --dtype "$DTYPE"
)
if [[ "$SAVE_SAFETENSORS" == "true" ]]; then
  CMD+=(--save_safetensors)
fi

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
