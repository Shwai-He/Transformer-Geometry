#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETTINGS="${SETTINGS:-none,xsa_middle,xsa_middle_multihead,residual_attn,residual_mlp,residual_both}"

IFS=',' read -r -a settings <<< "$SETTINGS"
for setting in "${settings[@]}"; do
  setting="$(echo "$setting" | xargs)"
  [[ -z "$setting" ]] && continue
  echo "[INFO] Running setting=$setting"
  SETTING="$setting" bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
done
