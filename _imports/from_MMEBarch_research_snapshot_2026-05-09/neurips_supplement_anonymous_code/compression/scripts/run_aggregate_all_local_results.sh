#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="/path/to/resource"
OUT_TSV="/path/to/resource"

PYTHON_BIN="python3"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/aggregate_all_local_results.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

"$PYTHON_BIN" "$PY_SCRIPT" \
  --runs_root "$RUNS_ROOT" \
  --out_tsv "$OUT_TSV"
