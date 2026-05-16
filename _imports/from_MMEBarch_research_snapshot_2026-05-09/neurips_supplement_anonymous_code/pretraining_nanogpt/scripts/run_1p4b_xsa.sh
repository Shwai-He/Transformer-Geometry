#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit this file to define one concrete experiment. Values are set directly here.

export SIZE_TAG="1p4b"
export METHOD="xsa"

# Data / runtime.
export DATASET="${DATASET:-fineweb100bt}"
export PREPARE_DATA="${PREPARE_DATA:-false}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}"
export NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
export NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
# Leave MASTER_PORT empty to let train_* scripts use METIS_WORKER_0_PORT or 29500.
export MASTER_PORT="${MASTER_PORT:-}"
export COMPILE="${COMPILE:-false}"
export SEED="${SEED:-1337}"
export ENABLE_WANDB="${ENABLE_WANDB:-false}"
export ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Training schedule. Defaults keep ~0.5M tokens/iter:
# BATCH_SIZE * BLOCK_SIZE * GRADIENT_ACCUMULATION_STEPS = 524288.
export BATCH_SIZE="2"
export BLOCK_SIZE="2048"
export GRADIENT_ACCUMULATION_STEPS="128"
export MAX_ITERS="200000"
export LR_DECAY_ITERS="200000"
export WARMUP_ITERS="2000"
export LEARNING_RATE="4e-4"
export MIN_LR="4e-5"

# Logging / output.
export WANDB_PROJECT="${WANDB_PROJECT:-anonymous-supplement}"

bash "$SCRIPT_DIR/train_xsa_forward_value_preproj.sh" "$SCRIPT_DIR/../config/train_xsa_1p4b.py"
