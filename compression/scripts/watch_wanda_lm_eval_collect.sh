#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
JOBS_CSV="${JOBS_CSV:-59571,59572,59573,59574,59575,59576,59577,59578}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/quality_eval/by_model/baseline_full_compression/collector_local}"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

has_active_jobs() {
  local jobs_arg
  jobs_arg="${JOBS_CSV//,/ }"
  if [[ -z "${jobs_arg// }" ]]; then
    return 1
  fi
  squeue -h -j "$JOBS_CSV" >/tmp/wanda_collect_squeue.$$ 2>/dev/null || true
  if [[ -s /tmp/wanda_collect_squeue.$$ ]]; then
    rm -f /tmp/wanda_collect_squeue.$$
    return 0
  fi
  rm -f /tmp/wanda_collect_squeue.$$
  return 1
}

run_collect() {
  date -u +"[COLLECT] %Y-%m-%dT%H:%M:%SZ"
  "$PYTHON_BIN" "$REPO_ROOT/compression/scripts/collect_wanda_lm_eval_results.py"
}

run_collect
while has_active_jobs; do
  sleep "$INTERVAL_SEC"
  run_collect
done
run_collect
echo "[DONE] watched jobs finished or left queue: $JOBS_CSV"
