#!/usr/bin/env bash
set -euo pipefail

# Low-cost diagonal attention suppression baseline. This modifies attention logits only:
#   ATTN_DIAG_MODE=hard: mask self edge a_ii for i>0 by default
#   ATTN_DIAG_MODE=bias: subtract ATTN_DIAG_BIAS from self-edge logits

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ATTN_DIAG_MODE="${ATTN_DIAG_MODE:-hard}"
ATTN_DIAG_BIAS="${ATTN_DIAG_BIAS:-4.0}"
ATTN_DIAG_KEEP_FIRST="${ATTN_DIAG_KEEP_FIRST:-true}"
MODEL_SIZE_TAG="${MODEL_SIZE_TAG:-}"
SEED="${SEED:-1337}"
XSA_SELF_LOSS_OBJECTIVE="${XSA_SELF_LOSS_OBJECTIVE:-none}"

RUN_NAME_PREFIX="${MODEL_SIZE_TAG:+${MODEL_SIZE_TAG}-}"
DIAG_TAG="diag-${ATTN_DIAG_MODE}"
if [[ "$ATTN_DIAG_MODE" == "bias" ]]; then
  BIAS_TAG="${ATTN_DIAG_BIAS//./p}"
  DIAG_TAG="${DIAG_TAG}-b${BIAS_TAG}"
fi
DIAG_TAG="${DIAG_TAG}-keepfirst${ATTN_DIAG_KEEP_FIRST}"

export ATTN_DIAG_MODE ATTN_DIAG_BIAS ATTN_DIAG_KEEP_FIRST
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME_PREFIX}${DIAG_TAG}-xobj${XSA_SELF_LOSS_OBJECTIVE}-s${SEED}}"
export OUT_DIR="${OUT_DIR:-out/${RUN_NAME_PREFIX}${DIAG_TAG}-xobj${XSA_SELF_LOSS_OBJECTIVE}-s${SEED}}"
export SEED XSA_SELF_LOSS_OBJECTIVE MODEL_SIZE_TAG

bash "$SCRIPT_DIR/train_baseline.sh" "${1:-config/train_gpt2.py}"
