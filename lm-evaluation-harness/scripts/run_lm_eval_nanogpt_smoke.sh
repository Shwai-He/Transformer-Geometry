#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NANOGPT_CKPT="${NANOGPT_CKPT:-}"
if [[ -z "$NANOGPT_CKPT" ]]; then
  echo "[ERROR] NANOGPT_CKPT is required." >&2
  exit 1
fi

# Small, fast sanity run
TASKS="${TASKS:-piqa,hellaswag,winogrande}"
LIMIT="${LIMIT:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
BACKGROUND="${BACKGROUND:-false}"

NANOGPT_CKPT="$NANOGPT_CKPT" \
TASKS="$TASKS" \
LIMIT="$LIMIT" \
BATCH_SIZE="$BATCH_SIZE" \
DEVICE="$DEVICE" \
DTYPE="$DTYPE" \
BACKGROUND="$BACKGROUND" \
bash "$SCRIPT_DIR/run_lm_eval_nanogpt_setting.sh"
