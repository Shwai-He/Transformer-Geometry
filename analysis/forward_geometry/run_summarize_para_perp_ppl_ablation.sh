#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/summarize_para_perp_ppl_ablation.py"

LOG_DIR="$SCRIPT_DIR/logs/ppl_ablation_summary"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/${STAMP}_summarize_para_perp_ppl_ablation.log"
PID_PATH="$LOG_DIR/${STAMP}_summarize_para_perp_ppl_ablation.pid"

CMD=("$PYTHON_BIN" "$PY_SCRIPT")

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"
echo "Run cmd: ${CMD[*]}"

"${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo $! > "$PID_PATH"

echo "Started in background."
echo "tail -f $LOG_PATH"
