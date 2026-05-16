#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export DEVICE="${DEVICE:-cuda}"
export SAMPLE_IDX="${SAMPLE_IDX:--1}"

echo "[INFO] Single-GPU mode for qwen_xsa_single_layer_flip_viz"
echo "[INFO] GPU_ID=$GPU_ID"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] SAMPLE_IDX=$SAMPLE_IDX"

exec bash "$SCRIPT_DIR/run_qwen_xsa_single_layer_flip_viz.sh"
