#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for setting in xsa_middle xsa_middle_multihead; do
  echo "[INFO] Running setting=$setting"
  SETTING="$setting" bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
done
