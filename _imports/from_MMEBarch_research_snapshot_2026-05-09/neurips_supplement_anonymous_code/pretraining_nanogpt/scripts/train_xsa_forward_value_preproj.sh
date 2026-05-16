#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned XSA: remove the self-value projection before c_proj/o_proj,
# independently in each attention head's value space.
export XSA_FORWARD_SPACE="${XSA_FORWARD_SPACE:-pre_o_proj}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_xsa_forward_value.sh" "$@"
