#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compression eval uses the same hf-xsa model wrapper only as a neutral HF
# launcher. No XSA intervention is applied.
export SETTING="${SETTING:-none}"
export XSA_FORWARD_OP="${XSA_FORWARD_OP:-remove_parallel}"
export XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-1.0}"
export XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
export XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"

bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh" "$@"
