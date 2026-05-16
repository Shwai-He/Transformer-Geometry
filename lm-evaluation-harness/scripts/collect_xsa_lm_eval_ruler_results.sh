#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_ROOT="${INPUT_ROOT:-$HARNESS_DIR/outputs/xsa_lm_eval_ruler}" \
OUTPUT_DIR="${OUTPUT_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_ruler_summary}" \
"$PYTHON_BIN" "$SCRIPT_DIR/collect_xsa_lm_eval_ruler_results.py"
