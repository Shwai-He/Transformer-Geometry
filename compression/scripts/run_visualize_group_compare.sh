#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
# Fill with concrete run directories (each contains layerwise_metrics.csv).
RUN_DIRS=""
LABELS=""
COMPONENT="block_out"  # block_out | attn_out | mlp_out
OUT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/focused-compression-analysis/outputs/viz_group_compare"

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/visualize_group_compare.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/viz_group_compare"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/${STAMP}_${COMPONENT}.log"
PID_PATH="$LOG_DIR/${STAMP}_${COMPONENT}.pid"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi
if [[ -z "$RUN_DIRS" ]]; then
  echo "[ERROR] RUN_DIRS is empty. Please set comma-separated run dirs." >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --run_dirs "$RUN_DIRS"
  --component "$COMPONENT"
  --out_dir "$OUT_DIR/$COMPONENT"
)
if [[ -n "$LABELS" ]]; then
  CMD+=(--labels "$LABELS")
fi

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
