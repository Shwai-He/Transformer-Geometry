#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_ROOTS="${INPUT_ROOTS:-$HARNESS_DIR/outputs/xsa_lm_eval:$HARNESS_DIR/outputs/xsa_lm_eval_alpha_sweep:$HARNESS_DIR/outputs/xsa_lm_eval_ruler}" \
OUTPUT_DIR="${OUTPUT_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_summary}" \
"$PYTHON_BIN" "$SCRIPT_DIR/collect_xsa_lm_eval_results.py"
