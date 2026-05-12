#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SETTING="${SETTING:-none}"
export ATTN_DIAG_ENABLED="${ATTN_DIAG_ENABLED:-true}"
export ATTN_DIAG_MODE="${ATTN_DIAG_MODE:-zero_renorm}"
export ATTN_DIAG_KEEP_FIRST="${ATTN_DIAG_KEEP_FIRST:-true}"

bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh" "$@"
