#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="/path/to/resource"
OUT_DIR="$RUNS_ROOT/_summaries"
PYTHON_BIN="python3"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SUMMARY_PY="$WORKSPACE_ROOT/code/summarize_local_setting_no_plot.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/summarize_local_all"
mkdir -p "$LOG_DIR" "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/summarize_local_all_${STAMP}.log"

echo "Logs:"
echo "  $LOG_PATH"
echo "[INFO] runs_root=$RUNS_ROOT" | tee "$LOG_PATH"
echo "[INFO] out_dir=$OUT_DIR" | tee -a "$LOG_PATH"

if [[ ! -f "$SUMMARY_PY" ]]; then
  echo "[ERROR] Python script not found: $SUMMARY_PY" | tee -a "$LOG_PATH" >&2
  exit 1
fi

mapfile -t METHOD_PREFIXES < <(
  find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*__*' -printf '%f\n' \
    | sed -E 's/^[^_]*__//' \
    | sed -E 's/_l[0-9]+$//' \
    | sort -u
)

if [[ ${#METHOD_PREFIXES[@]} -eq 0 ]]; then
  echo "[ERROR] No local run dirs found under $RUNS_ROOT" | tee -a "$LOG_PATH" >&2
  exit 1
fi

rc=0
for method_prefix in "${METHOD_PREFIXES[@]}"; do
  echo "[INFO] summarize method_prefix=$method_prefix" | tee -a "$LOG_PATH"
  if ! "$PYTHON_BIN" "$SUMMARY_PY" \
    --runs_root "$RUNS_ROOT" \
    --method_prefix "$method_prefix" \
    --out_dir "$OUT_DIR" >> "$LOG_PATH" 2>&1; then
    rc=1
    echo "[WARN] summary failed for method_prefix=$method_prefix" | tee -a "$LOG_PATH"
  fi
done

MASTER_TSV="$OUT_DIR/all_settings_master_v2.tsv"
echo "[INFO] building master table: $MASTER_TSV" | tee -a "$LOG_PATH"
if ! "$PYTHON_BIN" "$WORKSPACE_ROOT/code/aggregate_all_local_results.py" \
  --runs_root "$RUNS_ROOT" \
  --out_tsv "$MASTER_TSV" >> "$LOG_PATH" 2>&1; then
  rc=1
  echo "[WARN] aggregate_all_local_results failed" | tee -a "$LOG_PATH"
fi

echo "[INFO] done. tail -f $LOG_PATH" | tee -a "$LOG_PATH"
exit "$rc"
