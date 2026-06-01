#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_l40s:4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEM="${MEM:-96G}"
TIME="${TIME:-1-00:00:00}"

PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
MODEL_NAME="${MODEL_NAME:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"

MAX_SAMPLES="${MAX_SAMPLES:-2048}"
COMPARE_MAX_SAMPLES="${COMPARE_MAX_SAMPLES:-$MAX_SAMPLES}"
GENERATION_MAX_SAMPLES="${GENERATION_MAX_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
DTYPE="${DTYPE:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-512}"

RUN_COMPARE_STAGE="${RUN_COMPARE_STAGE:-true}"
RUN_GENERATION_STAGE="${RUN_GENERATION_STAGE:-false}"
SKIP_EXISTING_COMPARE="${SKIP_EXISTING_COMPARE:-true}"
SKIP_EXISTING_GENERATION="${SKIP_EXISTING_GENERATION:-true}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-false}"
XSA_LAYER_SCALE_MODE="${XSA_LAYER_SCALE_MODE:-none}"
XSA_LAYER_SCALE_SEED="${XSA_LAYER_SCALE_SEED:-0}"
XSA_LAYER_PARA_SCALE_MIN="${XSA_LAYER_PARA_SCALE_MIN:--20.0}"
XSA_LAYER_PARA_SCALE_MAX="${XSA_LAYER_PARA_SCALE_MAX:-20.0}"
XSA_LAYER_PERP_SCALE_MIN="${XSA_LAYER_PERP_SCALE_MIN:--1.5}"
XSA_LAYER_PERP_SCALE_MAX="${XSA_LAYER_PERP_SCALE_MAX:-1.5}"
XSA_INTERVENTION_SITES="${XSA_INTERVENTION_SITES:-xsa_middle_multihead,residual_output}"
XSA_PRIMARY_SITE="${XSA_PRIMARY_SITE:-xsa_middle_multihead}"
XSA_VALUE_REF_EXPANSION="${XSA_VALUE_REF_EXPANSION:-head_aware}"
XSA_ATTN_ATTR_SOURCE="${XSA_ATTN_ATTR_SOURCE:-head_aware}"
XSA_MIDDLE_TARGETS="${XSA_MIDDLE_TARGETS:-attn}"
RESIDUAL_OUTPUT_TARGETS="${RESIDUAL_OUTPUT_TARGETS:-none,attn,mlp,both}"

DATASET_NAME="${DATASET_NAME:-allenai/c4}"
DATASET_CONFIG="${DATASET_CONFIG:-en}"
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
DATASET_TEXT_KEY="${DATASET_TEXT_KEY:-text}"
DATASET_MIN_CHARS="${DATASET_MIN_CHARS:-160}"
DATASET_MAX_CHARS="${DATASET_MAX_CHARS:-1200}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-1337}"
PROMPTS_PATH="${PROMPTS_PATH:-}"

SUBMIT_SWEEPS="${SUBMIT_SWEEPS:-para,perp}"
PARA_SCALES="${PARA_SCALES:--20 -17.5 -15 -12.5 -10 -9 -8 -7 -6 -5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10 12.5 15 17.5 20}"
PERP_SCALES="${PERP_SCALES:--1.5 -1.375 -1.25 -1.125 -1.0 -0.875 -0.75 -0.625 -0.5 -0.375 -0.25 -0.125 0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5}"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/representation-analysis/outputs/$MODEL_TAG}"
SLURM_DIR="${SLURM_DIR:-$OUT_ROOT/slurm}"
mkdir -p "$SLURM_DIR"

submit_sweep() {
  local sweep="$1"
  local script_path
  local out_dir
  local job_name
  case "$sweep" in
    para)
      script_path="$SCRIPT_DIR/run_xsa_para_ablation.sh"
      out_dir="$OUT_ROOT/para_ablation"
      job_name="xsa-${MODEL_TAG}-para"
      ;;
    perp)
      script_path="$SCRIPT_DIR/run_xsa_perp_ablation.sh"
      out_dir="$OUT_ROOT/perp_ablation"
      job_name="xsa-${MODEL_TAG}-perp"
      ;;
    *)
      echo "[WARN] Unknown sweep '$sweep'; skipping" >&2
      return 0
      ;;
  esac

  mkdir -p "$out_dir" "$out_dir/logs"

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="$job_name" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$SLURM_DIR/%x-%j.out" \
    --error="$SLURM_DIR/%x-%j.err" \
    --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHON_BIN="$PYTHON_BIN"
export MODEL_NAME="$MODEL_NAME"
export MAX_SAMPLES="$MAX_SAMPLES"
export COMPARE_MAX_SAMPLES="$COMPARE_MAX_SAMPLES"
export GENERATION_MAX_SAMPLES="$GENERATION_MAX_SAMPLES"
export BATCH_SIZE="$BATCH_SIZE"
export NUM_GPUS="$NUM_GPUS"
export GPU_LIST="$GPU_LIST"
export DTYPE="$DTYPE"
export MAX_LENGTH="$MAX_LENGTH"
export RUN_COMPARE_STAGE="$RUN_COMPARE_STAGE"
export RUN_GENERATION_STAGE="$RUN_GENERATION_STAGE"
export SKIP_EXISTING_COMPARE="$SKIP_EXISTING_COMPARE"
export SKIP_EXISTING_GENERATION="$SKIP_EXISTING_GENERATION"
export USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE"
export XSA_LAYER_SCALE_MODE="$XSA_LAYER_SCALE_MODE"
export XSA_LAYER_SCALE_SEED="$XSA_LAYER_SCALE_SEED"
export XSA_LAYER_PARA_SCALE_MIN="$XSA_LAYER_PARA_SCALE_MIN"
export XSA_LAYER_PARA_SCALE_MAX="$XSA_LAYER_PARA_SCALE_MAX"
export XSA_LAYER_PERP_SCALE_MIN="$XSA_LAYER_PERP_SCALE_MIN"
export XSA_LAYER_PERP_SCALE_MAX="$XSA_LAYER_PERP_SCALE_MAX"
export XSA_INTERVENTION_SITES="$XSA_INTERVENTION_SITES"
export XSA_PRIMARY_SITE="$XSA_PRIMARY_SITE"
export XSA_VALUE_REF_EXPANSION="$XSA_VALUE_REF_EXPANSION"
export XSA_ATTN_ATTR_SOURCE="$XSA_ATTN_ATTR_SOURCE"
export XSA_MIDDLE_TARGETS="$XSA_MIDDLE_TARGETS"
export RESIDUAL_OUTPUT_TARGETS="$RESIDUAL_OUTPUT_TARGETS"
export DATASET_NAME="$DATASET_NAME"
export DATASET_CONFIG="$DATASET_CONFIG"
export DATASET_SPLIT="$DATASET_SPLIT"
export DATASET_TEXT_KEY="$DATASET_TEXT_KEY"
export DATASET_MIN_CHARS="$DATASET_MIN_CHARS"
export DATASET_MAX_CHARS="$DATASET_MAX_CHARS"
export DATASET_SHUFFLE_SEED="$DATASET_SHUFFLE_SEED"
export PROMPTS_PATH="$PROMPTS_PATH"
export OUT_DIR="$out_dir"
export PROMPT_CACHE_DIR="$out_dir/prompts"
export COMPARE_OUT_DIR="$out_dir/compare"
export GENERATE_OUT_DIR="$out_dir/generate"
export LOCAL_RUN_DIR="/tmp/\${USER:-xsa}/${MODEL_TAG}_${sweep}_\${SLURM_JOB_ID}"
export TMP_BASE_DIR="\$LOCAL_RUN_DIR/tmp"
export LOG_DIR="$out_dir/logs"
export XSA_LOG_DIR="$out_dir/logs"
export PPL_SUMMARY_TSV="$out_dir/ppl_summary.tsv"
export GEN_SUMMARY_JSONL="$out_dir/generation_paths.jsonl"
export GEN_COMBINED_JSONL="$out_dir/generation_summary.jsonl"
export PARA_SCALES="$PARA_SCALES"
export PERP_SCALES="$PERP_SCALES"
echo "[INFO] sweep=$sweep"
echo "[INFO] job_id=\$SLURM_JOB_ID"
echo "[INFO] hostname=\$(hostname)"
echo "[INFO] python=\$PYTHON_BIN"
"\$PYTHON_BIN" - <<'PY'
import torch, transformers
print("[INFO] torch", torch.__version__, "cuda", torch.cuda.is_available(), "n", torch.cuda.device_count(), flush=True)
print("[INFO] transformers", transformers.__version__, flush=True)
PY
bash "$script_path"
EOF
)"
}

IFS=',' read -r -a sweeps <<< "$SUBMIT_SWEEPS"
for sweep in "${sweeps[@]}"; do
  sweep="$(echo "$sweep" | xargs)"
  [[ -z "$sweep" ]] && continue
  job_id="$(submit_sweep "$sweep")"
  echo "[SUBMITTED] sweep=$sweep job_id=$job_id"
done

echo "[INFO] Outputs: $OUT_ROOT"
echo "[INFO] Slurm logs: $SLURM_DIR"
