#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SETTING="${SETTING:-xsa_middle_multihead}"
export SETTING_LABEL="${SETTING_LABEL:-xsa_add_parallel}"
export XSA_FORWARD_OP="${XSA_FORWARD_OP:-add_parallel}"
export XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-1.0}"

bash "$SCRIPT_DIR/run_lm_eval_xsa_multihead_batch.sh" "$@"
