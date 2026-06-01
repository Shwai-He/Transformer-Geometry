#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Submit geometry-aware pruning lm-eval to Slurm.
#
# Defaults follow the full baseline/XSA variant task surface.
# The job keeps Hugging Face/model/dataset access offline by default and uses the
# cached Qwen3-0.6B-Base checkpoint unless MODEL_PATH is overridden.
##############################################################################

if [[ -n "${GEO_PRUNE_SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$GEO_PRUNE_SCRIPT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ -n "${GEO_PRUNE_HARNESS_DIR:-}" ]]; then
  HARNESS_DIR="$GEO_PRUNE_HARNESS_DIR"
else
  HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ -n "${GEO_PRUNE_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$GEO_PRUNE_REPO_ROOT"
else
  REPO_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
fi

SLURM_DIR="${SLURM_DIR:-$HARNESS_DIR/outputs/geo_prune_lm_eval_slurm}"
mkdir -p "$SLURM_DIR"

if [[ -z "${SLURM_JOB_ID:-}" && "${INSIDE_GEO_PRUNE_SLURM:-0}" != "1" ]]; then
  PARTITION="${PARTITION:-beacon}"
  QOS="${QOS:-medium}"
  GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
  MEM="${MEM:-64G}"
  TIME="${TIME:-1-00:00:00}"
  JOB_NAME="${JOB_NAME:-geo-prune-lmeval}"
  export GEO_PRUNE_SCRIPT_DIR="$SCRIPT_DIR"
  export GEO_PRUNE_HARNESS_DIR="$HARNESS_DIR"
  export GEO_PRUNE_REPO_ROOT="$REPO_ROOT"
  export SLURM_DIR

  job_id="$(
    sbatch --parsable \
      --partition="$PARTITION" \
      --qos="$QOS" \
      --job-name="$JOB_NAME" \
      --gres="$GRES" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --output="$SLURM_DIR/%x-%j.out" \
      --error="$SLURM_DIR/%x-%j.err" \
      --export=ALL,INSIDE_GEO_PRUNE_SLURM=1 \
      "$SCRIPT_DIR/submit_lm_eval_geo_prune_slurm.sh"
  )"
  echo "[SUBMITTED] job_id=$job_id"
  echo "[INFO] partition=$PARTITION qos=$QOS gres=$GRES mem=$MEM time=$TIME"
  echo "[INFO] logs=$SLURM_DIR/${JOB_NAME}-${job_id}.out"
  exit 0
fi

cd "$REPO_ROOT"

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
export MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_geo_nm_s0p5_full_tasks}"
export TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu,gsm8k_cot,humaneval,nq_open,drop,mbpp,bbh_cot_zeroshot}"
export INCLUDE_PATH="${INCLUDE_PATH:-$REPO_ROOT/compression/lm_eval_tasks}"

export DEFAULT_DEVICE="${DEFAULT_DEVICE:-cuda}"
export DEFAULT_DTYPE="${DEFAULT_DTYPE:-bfloat16}"
export DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-1}"
export DEFAULT_TRUST_REMOTE_CODE="${DEFAULT_TRUST_REMOTE_CODE:-true}"
export DEFAULT_APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE:-false}"
export DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-4096}"

export GEO_PRUNE_METHOD="${GEO_PRUNE_METHOD:-wanda}"
export GEO_PRUNE_TARGETS="${GEO_PRUNE_TARGETS:-q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj}"
export GEO_GEOMETRY_MODE="${GEO_GEOMETRY_MODE:-residual_error}"
export GEO_GEOMETRY_ALPHA="${GEO_GEOMETRY_ALPHA:-1.0}"
export GEO_GEOMETRY_TARGETS="${GEO_GEOMETRY_TARGETS:-o_proj+down_proj+v_proj}"
export GEO_SPARSITY_RATIO="${GEO_SPARSITY_RATIO:-0.5}"
export GEO_THRESHOLD_SCOPE="${GEO_THRESHOLD_SCOPE:-global}"
export GEO_TOKEN_SCOPE="${GEO_TOKEN_SCOPE:-last}"
export GEO_MAX_PROMPTS="${GEO_MAX_PROMPTS:-128}"
export GEO_MAX_LENGTH="${GEO_MAX_LENGTH:-2048}"

export GEO_SETTINGS="${GEO_SETTINGS:-dense|false|none|unstructured;wanda_unstructured|true|none|unstructured;wanda_unstructured_residual_perp|true|residual_perp|unstructured;wanda_unstructured_residual_para|true|residual_para|unstructured;wanda_unstructured_value_perp|true|value_perp|unstructured;wanda_2_4|true|none|2:4;wanda_4_8|true|none|4:8}"

echo "[INFO] Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Repo: $REPO_ROOT"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Tasks: $TASKS_CSV"
echo "[INFO] Include path: $INCLUDE_PATH"
echo "[INFO] Model tag: $MODEL_TAG"
echo "[INFO] Settings: $GEO_SETTINGS"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] Offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

bash "$SCRIPT_DIR/run_lm_eval_geo_prune_batch.sh"
