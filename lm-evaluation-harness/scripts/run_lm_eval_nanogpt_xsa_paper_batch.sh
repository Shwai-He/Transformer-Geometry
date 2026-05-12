#!/usr/bin/env bash
set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/scripts"

HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SINGLE_SCRIPT="$SCRIPT_DIR/run_lm_eval_nanogpt_setting.sh"

CKPT_ROOT="${CKPT_ROOT:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/nanoGPT/out}"
PATH_FILTER="${PATH_FILTER:-xsa-paper}"
CKPT_NAME="${CKPT_NAME:-ckpt_best.pt}"
SORT_RESULTS="${SORT_RESULTS:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-true}"
BACKGROUND="${BACKGROUND:-false}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

DEFAULT_TASKS="arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
TASKS="${TASKS:-$DEFAULT_TASKS}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
NANOGPT_REPO_ROOT="${NANOGPT_REPO_ROOT:-$HARNESS_DIR/nanoGPT}"

XSA_FORWARD_ONLY="${XSA_FORWARD_ONLY:-}"
XSA_FORWARD_TARGET="${XSA_FORWARD_TARGET:-}"
XSA_FORWARD_REF="${XSA_FORWARD_REF:-}"
XSA_FORWARD_SPACE="${XSA_FORWARD_SPACE:-}"
XSA_FORWARD_OP="${XSA_FORWARD_OP:-}"
XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-}"

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_logs}"
PID_DIR="${PID_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_pids}"
mkdir -p "$LOG_DIR" "$PID_DIR"

timestamp="$(date +"%Y-%m-%dT%H-%M-%S")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${timestamp}-nanogpt-xsa-paper-batch.log}"
PID_FILE="${PID_FILE:-$PID_DIR/${timestamp}-nanogpt-xsa-paper-batch.pid}"

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

trim_spaces() {
  echo "${1:-}" | xargs
}

launch_eval_job() {
  local gpu_id="$1"
  local ckpt_path="$2"
  local task="$3"
  local fewshot="$4"
  local output_path="$5"
  local model_log_file="$6"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
  NANOGPT_CKPT="$ckpt_path" \
  NANOGPT_REPO_ROOT="$NANOGPT_REPO_ROOT" \
  TASKS="$task" \
  BATCH_SIZE="$BATCH_SIZE" \
  DEVICE="$DEVICE" \
  DTYPE="$DTYPE" \
  NUM_FEWSHOT="$fewshot" \
  LIMIT="$LIMIT" \
  MAX_LENGTH="$MAX_LENGTH" \
  XSA_FORWARD_ONLY="$XSA_FORWARD_ONLY" \
  XSA_FORWARD_TARGET="$XSA_FORWARD_TARGET" \
  XSA_FORWARD_REF="$XSA_FORWARD_REF" \
  XSA_FORWARD_SPACE="$XSA_FORWARD_SPACE" \
  XSA_FORWARD_OP="$XSA_FORWARD_OP" \
  XSA_FORWARD_ALPHA="$XSA_FORWARD_ALPHA" \
  OUTPUT_PATH="$output_path" \
  LOG_FILE="$model_log_file" \
  BACKGROUND="false" \
  bash "$SINGLE_SCRIPT"
}

if [[ "$BACKGROUND" == "true" && "${_NANOGPT_XSA_PAPER_BATCH_BG_CHILD:-0}" != "1" ]]; then
  export _NANOGPT_XSA_PAPER_BATCH_BG_CHILD=1
  export LOG_FILE PID_FILE
  nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  bg_pid="$!"
  echo "$bg_pid" > "$PID_FILE"
  echo "[INFO] Running in background."
  echo "[INFO] Logs:"
  echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
  echo "[INFO] PID file:"
  echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"
  echo "[INFO] Tail command: tail -f $(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
  exit 0
fi

exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "[ERROR] CKPT_ROOT not found: $CKPT_ROOT"
  exit 1
fi
if [[ ! -f "$SINGLE_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $SINGLE_SCRIPT"
  exit 1
fi

echo "[INFO] CKPT_ROOT=$CKPT_ROOT"
echo "[INFO] PATH_FILTER=$PATH_FILTER"
echo "[INFO] CKPT_NAME=$CKPT_NAME"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] SKIP_EXISTING_RESULTS=$SKIP_EXISTING_RESULTS"
echo "[INFO] Logs:"
echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
echo "[INFO] PID file:"
echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"

IFS=',' read -r -a TASK_LIST <<< "$TASKS"
if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] TASKS is empty."
  exit 1
fi

IFS=',' read -r -a GPU_ITEMS <<< "$GPU_LIST"
GPU_IDS=()
for raw_gpu in "${GPU_ITEMS[@]}"; do
  gpu="$(trim_spaces "$raw_gpu")"
  [[ -n "$gpu" ]] && GPU_IDS+=("$gpu")
done
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty."
  exit 1
fi

declare -A DEFAULT_FEWSHOT_MAP=(
  [arc_easy]=0
  [boolq]=0
  [hellaswag]=10
  [lambada_openai]=0
  [openbookqa]=0
  [piqa]=0
  [rte]=0
  [winogrande]=5
)

ckpt_paths=()
while IFS= read -r -d '' ckpt; do
  ckpt_paths+=("$ckpt")
done < <(find "$CKPT_ROOT" -type f -name "$CKPT_NAME" -path "*${PATH_FILTER}*" -print0)

if [[ "$SORT_RESULTS" == "true" && ${#ckpt_paths[@]} -gt 1 ]]; then
  mapfile -t ckpt_paths < <(printf '%s\n' "${ckpt_paths[@]}" | sort)
fi

if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
  echo "[WARN] No checkpoint matched filter under: $CKPT_ROOT"
  exit 0
fi

echo "[INFO] Found ${#ckpt_paths[@]} checkpoint(s) matching filter."

fail_count=0
skip_count=0
run_count=0

JOB_PIDS=()
JOB_GPU=()
JOB_CKPT=()
JOB_TASK=()

poll_jobs() {
  local idx
  for idx in "${!JOB_PIDS[@]}"; do
    local pid="${JOB_PIDS[$idx]}"
    if [[ -z "$pid" ]]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    if wait "$pid"; then
      echo "[DONE] gpu=${JOB_GPU[$idx]} ckpt=${JOB_CKPT[$idx]} task=${JOB_TASK[$idx]}"
    else
      echo "[FAIL] gpu=${JOB_GPU[$idx]} ckpt=${JOB_CKPT[$idx]} task=${JOB_TASK[$idx]}"
      fail_count=$((fail_count + 1))
      if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
        echo "[ERROR] Stopping due to CONTINUE_ON_ERROR=false"
        exit 1
      fi
    fi
    JOB_PIDS[$idx]=""
    JOB_GPU[$idx]=""
    JOB_CKPT[$idx]=""
    JOB_TASK[$idx]=""
  done
}

count_active_jobs() {
  local c=0
  local idx
  for idx in "${!JOB_PIDS[@]}"; do
    [[ -n "${JOB_PIDS[$idx]}" ]] && c=$((c + 1))
  done
  echo "$c"
}

find_free_gpu() {
  local gpu
  local idx
  for gpu in "${GPU_IDS[@]}"; do
    local busy=0
    for idx in "${!JOB_PIDS[@]}"; do
      if [[ -n "${JOB_PIDS[$idx]}" && "${JOB_GPU[$idx]}" == "$gpu" ]]; then
        busy=1
        break
      fi
    done
    if (( busy == 0 )); then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

acquire_free_gpu() {
  local gpu
  while true; do
    poll_jobs
    if gpu="$(find_free_gpu)"; then
      echo "$gpu"
      return 0
    fi
    sleep 1
  done
}

for raw_task in "${TASK_LIST[@]}"; do
  task="$(trim_spaces "$raw_task")"
  if [[ -z "$task" ]]; then
    continue
  fi

  fewshot="${NUM_FEWSHOT:-${DEFAULT_FEWSHOT_MAP[$task]:-0}}"
  echo "=================================================="
  echo "[TASK] dispatching task=$task fewshot=$fewshot across GPUs"

  for ckpt_path in "${ckpt_paths[@]}"; do
    ckpt_dir="$(cd "$(dirname "$ckpt_path")" && pwd)"
    ckpt_tag="$(sanitize_tag "$ckpt_path")"
    model_log_file="${LOG_DIR}/${timestamp}-${ckpt_tag}.log"
    output_root="${ckpt_dir}/lm_eval_results"
    output_path="${output_root}/${ckpt_tag}-$(sanitize_tag "$task").json"

    if [[ "$SKIP_EXISTING_RESULTS" == "true" && -f "$output_path" ]]; then
      echo "[SKIP] existing result: $output_path"
      skip_count=$((skip_count + 1))
      continue
    fi

    echo "--------------------------------------------------"
    gpu_id="$(acquire_free_gpu)"
    echo "[RUN] gpu=$gpu_id ckpt=$ckpt_path task=$task fewshot=$fewshot"
    run_count=$((run_count + 1))
    launch_eval_job "$gpu_id" "$ckpt_path" "$task" "$fewshot" "$output_path" "$model_log_file" &
    pid=$!
    JOB_PIDS+=("$pid")
    JOB_GPU+=("$gpu_id")
    JOB_CKPT+=("$ckpt_path")
    JOB_TASK+=("$task")
  done

  while (( $(count_active_jobs) > 0 )); do
    poll_jobs
    sleep 1
  done
done

echo "--------------------------------------------------"
echo "[DONE] runs=$run_count skipped=$skip_count failed=$fail_count"
if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
