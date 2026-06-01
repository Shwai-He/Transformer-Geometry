#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# 用途
# - 把多组 lm-eval 实验分发到不同单卡并行运行
# - 每个子任务固定使用 1 张 GPU
# - 默认不走 accelerate 多进程，避免同节点端口/进程冲突
# - 适合 baseline / multihead xsa / attn removal 同时跑
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_ROOT"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_ROOT/$(date +"%Y-%m-%dT%H-%M-%S")-gpu-farm.log"
fi
if [[ "${LM_EVAL_LOGGING_INITIALIZED:-0}" != "1" ]]; then
  export LM_EVAL_LOGGING_INITIALIZED=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

##############################################################################
# GPU 配置：直接在这里改
##############################################################################
if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPUS <<< "$GPU_LIST"
else
  GPUS=(0 1 2 3 4 5 6 7)
fi

##############################################################################
# 方法配置：直接在这里改
# 可选值：
#   baseline
#   xsa_multihead
#   xsa_keep_parallel
#   xsa_add_parallel
#   attn_removal
##############################################################################
if [[ -n "${METHOD_LIST:-}" ]]; then
  read -r -a METHODS <<< "$METHOD_LIST"
else
  METHODS=(
    baseline
    xsa_multihead
    xsa_keep_parallel
    xsa_add_parallel
    attn_removal
  )
fi

##############################################################################
# 模型配置：直接在这里改
##############################################################################
if [[ -n "${MODEL_POSTFIX_LIST:-}" ]]; then
  read -r -a MODELS <<< "$MODEL_POSTFIX_LIST"
else
  MODELS=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-30B-A3B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-14B"
  )
fi

##############################################################################
# 通用实验参数
##############################################################################
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/hdfs/shwai.he/models}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-auto}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bfloat16}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-causal}"
DEFAULT_TRUST_REMOTE_CODE="${DEFAULT_TRUST_REMOTE_CODE:-true}"
DEFAULT_APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE:-false}"
DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-4096}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
REVERSE_TASKS="${REVERSE_TASKS:-false}"
LIMIT="${LIMIT:-}"
CONTINUE_ON_TASK_ERROR="${CONTINUE_ON_TASK_ERROR:-true}"
CONTINUE_ON_JOB_ERROR="${CONTINUE_ON_JOB_ERROR:-true}"

timestamp() {
  date +"%Y-%m-%dT%H-%M-%S"
}

script_for_method() {
  case "$1" in
    baseline)
      echo "$SCRIPT_DIR/run_lm_eval_baseline_batch.sh"
      ;;
    xsa_multihead)
      echo "$SCRIPT_DIR/run_lm_eval_xsa_multihead_batch.sh"
      ;;
    xsa_keep_parallel)
      echo "$SCRIPT_DIR/run_lm_eval_xsa_keep_parallel_batch.sh"
      ;;
    xsa_add_parallel)
      echo "$SCRIPT_DIR/run_lm_eval_xsa_add_parallel_batch.sh"
      ;;
    attn_removal)
      echo "$SCRIPT_DIR/run_lm_eval_attn_removal_batch.sh"
      ;;
    *)
      echo ""
      ;;
  esac
}

sanitize_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  echo "$s"
}

refresh_active_jobs() {
  local pid
  local new_active=()
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_active+=("$pid")
    else
      local gpu="${PID_TO_GPU[$pid]:-}"
      local status=0
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      if [[ -n "$gpu" ]]; then
        FREE_GPUS+=("$gpu")
      fi
      if [[ $status -eq 0 ]]; then
        echo "[DONE] pid=$pid gpu=${gpu:-?}"
      else
        echo "[FAIL] pid=$pid gpu=${gpu:-?} exit_code=$status" >&2
        FAILED_JOBS+=("$pid")
        if [[ "$CONTINUE_ON_JOB_ERROR" != "true" ]]; then
          echo "[ERROR] CONTINUE_ON_JOB_ERROR=false, stopping GPU farm early." >&2
          exit "$status"
        fi
      fi
      unset 'PID_TO_GPU[$pid]'
    fi
  done
  ACTIVE_PIDS=("${new_active[@]}")
}

wait_for_free_gpu() {
  while (( ${#FREE_GPUS[@]} == 0 )); do
    refresh_active_jobs
    if (( ${#FREE_GPUS[@]} == 0 )); then
      sleep 5
    fi
  done
}

declare -a FREE_GPUS=("${GPUS[@]}")
declare -a ACTIVE_PIDS=()
declare -a FAILED_JOBS=()
declare -A PID_TO_GPU=()

echo "[INFO] GPU farm start"
echo "[INFO] GPUs: ${GPUS[*]}"
echo "[INFO] Methods: ${METHODS[*]}"
echo "[INFO] Models: ${MODELS[*]}"
echo "[INFO] Logs: $LOG_ROOT"

for method in "${METHODS[@]}"; do
  runner="$(script_for_method "$method")"
  if [[ -z "$runner" || ! -f "$runner" ]]; then
    echo "[ERROR] Unknown or missing method runner for: $method" >&2
    exit 1
  fi

  for model_postfix in "${MODELS[@]}"; do
    wait_for_free_gpu

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    model_tag="$(sanitize_name "$model_postfix")"
    log_path="$LOG_ROOT/$(timestamp)-${method}-${model_tag}-gpu${gpu}.log"

    echo "[LAUNCH] method=$method model=$model_postfix gpu=$gpu"
    echo "         Logs:"
    echo "           $log_path"

    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export LAUNCH_MODE="single"
      export MODEL_POSTFIX_LIST="$model_postfix"
      export DEFAULT_BATCH_SIZE="$DEFAULT_BATCH_SIZE"
      export DEFAULT_DTYPE="$DEFAULT_DTYPE"
      export DEFAULT_BACKEND="$DEFAULT_BACKEND"
      export DEFAULT_TRUST_REMOTE_CODE="$DEFAULT_TRUST_REMOTE_CODE"
      export DEFAULT_APPLY_CHAT_TEMPLATE="$DEFAULT_APPLY_CHAT_TEMPLATE"
      export DEFAULT_MAX_LENGTH="$DEFAULT_MAX_LENGTH"
      export XSA_TRACK_STATS="$XSA_TRACK_STATS"
      export XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS"
      export XSA_START_LAYER="$XSA_START_LAYER"
      export XSA_END_LAYER="$XSA_END_LAYER"
      export XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N"
      export XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N"
      export CONFIRM_RUN_UNSAFE_CODE="$CONFIRM_RUN_UNSAFE_CODE"
      export CONTINUE_ON_TASK_ERROR="$CONTINUE_ON_TASK_ERROR"
      export REVERSE_TASKS="$REVERSE_TASKS"
      export BASE_MODEL_DIR="$BASE_MODEL_DIR"
      export LIMIT="$LIMIT"
      bash "$runner"
    ) >"$log_path" 2>&1 &

    pid=$!
    ACTIVE_PIDS+=("$pid")
    PID_TO_GPU["$pid"]="$gpu"
  done
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  refresh_active_jobs
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    sleep 5
  fi
done

if (( ${#FAILED_JOBS[@]} > 0 )); then
  echo "[ERROR] Some jobs failed: ${FAILED_JOBS[*]}" >&2
  exit 1
fi

echo "[OK] All GPU farm jobs completed."
