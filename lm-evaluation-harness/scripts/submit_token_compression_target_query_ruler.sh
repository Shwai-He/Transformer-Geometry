#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${MODEL_NAME:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
TASKS="${TASKS:-niah_single_1}"
RULER_LENGTHS="${RULER_LENGTHS:-4096,8192}"
LIMIT="${LIMIT:-20}"
KEEP_RATIOS_CSV="${KEEP_RATIOS_CSV:-0.7,0.8,0.9}"
MODES_CSV="${MODES_CSV:-uniform,target_value_norm_keep,target_value_perp_only_norm_keep}"
TOKEN_COMPRESSION_LAYERS="${TOKEN_COMPRESSION_LAYERS:-all}"
TOKEN_COMPRESSION_PARALLEL_WEIGHT="${TOKEN_COMPRESSION_PARALLEL_WEIGHT:-0.25}"
PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
TIME="${TIME:-3:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"

IFS=',' read -r -a lengths <<< "$RULER_LENGTHS"
for length in "${lengths[@]}"; do
  length="${length// /}"
  [[ -z "$length" ]] && continue
  MODEL_NAME="$MODEL_NAME" \
  TASKS="$TASKS" \
  RULER_LENGTHS="$length" \
  LIMIT="$LIMIT" \
  KEEP_RATIOS_CSV="$KEEP_RATIOS_CSV" \
  MODES_CSV="$MODES_CSV" \
  TOKEN_COMPRESSION_LAYERS="$TOKEN_COMPRESSION_LAYERS" \
  TOKEN_COMPRESSION_PARALLEL_WEIGHT="$TOKEN_COMPRESSION_PARALLEL_WEIGHT" \
  PARTITION="$PARTITION" \
  QOS="$QOS" \
  GRES="$GRES" \
  TIME="$TIME" \
  CPUS_PER_TASK="$CPUS_PER_TASK" \
  MEM="$MEM" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/submit_token_compression_ruler_slurm.sh"
done
