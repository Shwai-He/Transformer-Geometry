#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SEARCH_ROOTS="${SEARCH_ROOTS:-$HARNESS_DIR/../nanoGPT/out:$HARNESS_DIR/nanoGPT/out:$HARNESS_DIR/outputs}" \
OUTPUT_DIR="${OUTPUT_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_summary}" \
"$PYTHON_BIN" "$SCRIPT_DIR/collect_nanogpt_lm_eval_results.py"
