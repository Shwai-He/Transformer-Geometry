#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SLURM_DIR="${SLURM_DIR:-$REPO_ROOT/lm-evaluation-harness/outputs/geo_prune_lm_eval_slurm}"
mkdir -p "$SLURM_DIR"

if [[ -z "${SLURM_JOB_ID:-}" && "${INSIDE_WANDA_LAYERWISE_SLURM:-0}" != "1" ]]; then
  PARTITION="${PARTITION:-scavenger}"
  QOS="${QOS:-scavenger}"
  GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
  MEM="${MEM:-96G}"
  TIME="${TIME:-1-00:00:00}"
  JOB_NAME="${JOB_NAME:-wanda-layerwise-prune}"
  export SLURM_DIR
  sbatch_args=(
    --parsable
    --partition="$PARTITION"
    --qos="$QOS"
    --job-name="$JOB_NAME"
    --gres="$GRES"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$TIME"
    --output="$SLURM_DIR/%x-%j.out"
    --error="$SLURM_DIR/%x-%j.err"
    --export=ALL,INSIDE_WANDA_LAYERWISE_SLURM=1
  )
  if [[ -n "${DEPENDENCY:-}" ]]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
  fi
  job_id="$(
    sbatch "${sbatch_args[@]}" "$SCRIPT_PATH"
  )"
  echo "[SUBMITTED] job_id=$job_id"
  echo "[INFO] logs=$SLURM_DIR/${JOB_NAME}-${job_id}.out"
  exit 0
fi

cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
MODEL_PATH="${MODEL_PATH:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
CALIB_FILE="${CALIB_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/qwen3_0p6b/unstructured_s0p5_c4_ns128_seq2048}"
LOCAL_FILES_ONLY_ARGS=()
case "${LOCAL_FILES_ONLY:-true}" in
  1|true|TRUE|yes|YES|y|Y|on|ON)
    LOCAL_FILES_ONLY_ARGS=(--local_files_only)
    ;;
  0|false|FALSE|no|NO|n|N|off|OFF)
    LOCAL_FILES_ONLY_ARGS=(--no-local_files_only)
    ;;
  *)
    echo "[ERROR] Invalid LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" "$REPO_ROOT/compression/code/save_wanda_layerwise_pruned_model.py" \
  --model_name_or_path "$MODEL_PATH" \
  --calib_file "$CALIB_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --nsamples "${NSAMPLES:-128}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --sparsity_ratio "${SPARSITY_RATIO:-0.5}" \
  --sparsity_type "${SPARSITY_TYPE:-unstructured}" \
  --threshold_scope "${THRESHOLD_SCOPE:-global}" \
  --targets "${TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
  --geometry_scores "${GEOMETRY_SCORES:-}" \
  --geometry_strategy "${GEOMETRY_STRATEGY:-none}" \
  --geometry_mode "${GEOMETRY_MODE:-residual_error}" \
  --geometry_alpha "${GEOMETRY_ALPHA:-1.0}" \
  --geometry_targets "${GEOMETRY_TARGETS:-o_proj,down_proj,v_proj}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  "${LOCAL_FILES_ONLY_ARGS[@]}"
