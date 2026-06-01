#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SINGLE_SCRIPT="$SCRIPT_DIR/run_lm_eval_nanogpt_setting.sh"

CKPT_ROOT="${CKPT_ROOT:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/nanoGPT/out}"
PATH_FILTER="${PATH_FILTER:-xsa-paper}"
CKPT_NAME="${CKPT_NAME:-ckpt_best.pt}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
BACKGROUND="${BACKGROUND:-false}"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-0}"             # 0 -> equals number of GPUs
MAX_CONCURRENT_LOADS="${MAX_CONCURRENT_LOADS:-1}"   # limit concurrent ckpt loading bursts
LOAD_WINDOW_SEC="${LOAD_WINDOW_SEC:-90}"            # a job is in "loading window" for N seconds after launch
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-8}"       # pause between launches to reduce remote FS spikes
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"

DEFAULT_TASKS="arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
TASKS="${TASKS:-$DEFAULT_TASKS}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
NANOGPT_REPO_ROOT="${NANOGPT_REPO_ROOT:-$HARNESS_DIR/nanoGPT}"

XSA_FORWARD_ONLY="${XSA_FORWARD_ONLY:-}"
XSA_FORWARD_TARGET="${XSA_FORWARD_TARGET:-}"
XSA_FORWARD_REF="${XSA_FORWARD_REF:-}"
XSA_FORWARD_SPACE="${XSA_FORWARD_SPACE:-}"
XSA_FORWARD_OP="${XSA_FORWARD_OP:-}"
XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-}"

NANOGPT_STAGE_CKPT_LOCAL="${NANOGPT_STAGE_CKPT_LOCAL:-false}"
NANOGPT_TORCH_LOAD_RETRIES="${NANOGPT_TORCH_LOAD_RETRIES:-12}"
NANOGPT_TORCH_LOAD_RETRY_SLEEP="${NANOGPT_TORCH_LOAD_RETRY_SLEEP:-3}"

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_logs}"
PID_DIR="${PID_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_pids}"
mkdir -p "$LOG_DIR" "$PID_DIR"

timestamp="$(date +"%Y-%m-%dT%H-%M-%S")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${timestamp}-nanogpt-xsa-paper-stable-batch.log}"
PID_FILE="${PID_FILE:-$PID_DIR/${timestamp}-nanogpt-xsa-paper-stable-batch.pid}"

sanitize_tag() {
  local s="$1"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

ckpt_tag_from_path() {
  local ckpt_path="$1"
  local rel="$ckpt_path"
  if [[ "$ckpt_path" == "$CKPT_ROOT/"* ]]; then
    rel="${ckpt_path#"$CKPT_ROOT/"}"
  fi
  # Remove the trailing "/ckpt.pt" for cleaner, unique tags.
  rel="${rel%/$CKPT_NAME}"
  sanitize_tag "$rel"
}

trim_spaces() {
  echo "${1:-}" | xargs
}

launch_job() {
  local gpu_id="$1"
  local ckpt_path="$2"
  local output_path="$3"
  local batch_log_file="$4"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
  NANOGPT_CKPT="$ckpt_path" \
  NANOGPT_REPO_ROOT="$NANOGPT_REPO_ROOT" \
  TASKS="$TASKS" \
  BATCH_SIZE="$BATCH_SIZE" \
  DEVICE="$DEVICE" \
  DTYPE="$DTYPE" \
  NUM_FEWSHOT="$NUM_FEWSHOT" \
  LIMIT="$LIMIT" \
  MAX_LENGTH="$MAX_LENGTH" \
  XSA_FORWARD_ONLY="$XSA_FORWARD_ONLY" \
  XSA_FORWARD_TARGET="$XSA_FORWARD_TARGET" \
  XSA_FORWARD_REF="$XSA_FORWARD_REF" \
  XSA_FORWARD_SPACE="$XSA_FORWARD_SPACE" \
  XSA_FORWARD_OP="$XSA_FORWARD_OP" \
  XSA_FORWARD_ALPHA="$XSA_FORWARD_ALPHA" \
  NANOGPT_STAGE_CKPT_LOCAL="$NANOGPT_STAGE_CKPT_LOCAL" \
  NANOGPT_TORCH_LOAD_RETRIES="$NANOGPT_TORCH_LOAD_RETRIES" \
  NANOGPT_TORCH_LOAD_RETRY_SLEEP="$NANOGPT_TORCH_LOAD_RETRY_SLEEP" \
  NANOGPT_DISABLE_INNER_TEE="1" \
  OUTPUT_PATH="$output_path" \
  LOG_FILE="$batch_log_file" \
  BACKGROUND="false" \
  bash "$SINGLE_SCRIPT"
}

if [[ "$BACKGROUND" == "true" && "${_NANOGPT_XSA_STABLE_BG_CHILD:-0}" != "1" ]]; then
  export _NANOGPT_XSA_STABLE_BG_CHILD=1
  export LOG_FILE PID_FILE
  nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  bg_pid="$!"
  echo "$bg_pid" > "$PID_FILE"
  echo "[INFO] Running in background."
  echo "[INFO] Logs:"
  echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
  echo "[INFO] PID file:"
  echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"
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
if [[ "$MAX_ACTIVE_JOBS" == "0" ]]; then
  MAX_ACTIVE_JOBS="${#GPU_IDS[@]}"
fi

echo "[INFO] CKPT_ROOT=$CKPT_ROOT"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] GPUs=${GPU_IDS[*]}"
echo "[INFO] MAX_ACTIVE_JOBS=$MAX_ACTIVE_JOBS MAX_CONCURRENT_LOADS=$MAX_CONCURRENT_LOADS"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] NANOGPT_STAGE_CKPT_LOCAL=$NANOGPT_STAGE_CKPT_LOCAL"
echo "[INFO] Logs:"
echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
echo "[INFO] PID file:"
echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"

ckpt_paths=()
while IFS= read -r -d '' ckpt; do
  ckpt_paths+=("$ckpt")
done < <(find "$CKPT_ROOT" -type f -name "$CKPT_NAME" -path "*${PATH_FILTER}*" -print0)
if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
  echo "[WARN] No checkpoint matched filter."
  exit 0
fi
mapfile -t ckpt_paths < <(printf '%s\n' "${ckpt_paths[@]}" | sort)
echo "[INFO] Found ${#ckpt_paths[@]} checkpoint(s)."

JOB_PIDS=()
JOB_GPU=()
JOB_CKPT=()
JOB_START_TS=()
LAST_HEARTBEAT_TS="$(date +%s)"

poll_jobs() {
  local idx
  for idx in "${!JOB_PIDS[@]}"; do
    local pid="${JOB_PIDS[$idx]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    if wait "$pid"; then
      echo "[DONE] gpu=${JOB_GPU[$idx]} ckpt=${JOB_CKPT[$idx]}"
    else
      echo "[FAIL] gpu=${JOB_GPU[$idx]} ckpt=${JOB_CKPT[$idx]}"
      if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
        echo "[ERROR] CONTINUE_ON_ERROR=false, abort."
        exit 1
      fi
    fi
    JOB_PIDS[$idx]=""
    JOB_GPU[$idx]=""
    JOB_CKPT[$idx]=""
    JOB_START_TS[$idx]=""
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

count_loading_window_jobs() {
  local now
  now="$(date +%s)"
  local c=0
  local idx
  for idx in "${!JOB_PIDS[@]}"; do
    [[ -z "${JOB_PIDS[$idx]}" ]] && continue
    local st="${JOB_START_TS[$idx]}"
    [[ -z "$st" ]] && continue
    if (( now - st < LOAD_WINDOW_SEC )); then
      c=$((c + 1))
    fi
  done
  echo "$c"
}

emit_heartbeat_if_needed() {
  local now
  now="$(date +%s)"
  if (( now - LAST_HEARTBEAT_TS < HEARTBEAT_SEC )); then
    return 0
  fi
  LAST_HEARTBEAT_TS="$now"
  local active loading
  active="$(count_active_jobs)"
  loading="$(count_loading_window_jobs)"
  echo "[HEARTBEAT] active_jobs=$active loading_window_jobs=$loading"
  local idx
  for idx in "${!JOB_PIDS[@]}"; do
    [[ -z "${JOB_PIDS[$idx]}" ]] && continue
    local age=$(( now - ${JOB_START_TS[$idx]} ))
    echo "[HEARTBEAT] gpu=${JOB_GPU[$idx]} age_sec=$age ckpt=${JOB_CKPT[$idx]}"
  done
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

wait_for_launch_slot() {
  while true; do
    poll_jobs
    emit_heartbeat_if_needed
    local active loading
    active="$(count_active_jobs)"
    loading="$(count_loading_window_jobs)"
    if (( active < MAX_ACTIVE_JOBS && loading < MAX_CONCURRENT_LOADS )); then
      return 0
    fi
    sleep 2
  done
}

skip_count=0
run_count=0

for ckpt_path in "${ckpt_paths[@]}"; do
  ckpt_dir="$(cd "$(dirname "$ckpt_path")" && pwd)"
  ckpt_tag="$(ckpt_tag_from_path "$ckpt_path")"
  output_root="${ckpt_dir}/lm_eval_results"
  output_path="${output_root}/${ckpt_tag}-all_tasks.json"
  mkdir -p "$output_root"

  if [[ "$SKIP_EXISTING_RESULTS" == "true" && -f "$output_path" ]]; then
    echo "[SKIP] existing result: $output_path"
    skip_count=$((skip_count + 1))
    continue
  fi

  wait_for_launch_slot
  gpu_id="$(find_free_gpu)"
  if [[ -z "$gpu_id" ]]; then
    sleep 2
    gpu_id="$(find_free_gpu)"
  fi
  if [[ -z "$gpu_id" ]]; then
    echo "[ERROR] No free GPU found unexpectedly."
    exit 1
  fi

  echo "[RUN] gpu=$gpu_id ckpt=$ckpt_path"
  launch_job "$gpu_id" "$ckpt_path" "$output_path" "$LOG_FILE" &
  pid=$!
  JOB_PIDS+=("$pid")
  JOB_GPU+=("$gpu_id")
  JOB_CKPT+=("$ckpt_path")
  JOB_START_TS+=("$(date +%s)")
  run_count=$((run_count + 1))
  sleep "$LAUNCH_STAGGER_SEC"
done

while (( $(count_active_jobs) > 0 )); do
  poll_jobs
  emit_heartbeat_if_needed
  sleep 2
done

echo "[DONE] runs=$run_count skipped=$skip_count"
