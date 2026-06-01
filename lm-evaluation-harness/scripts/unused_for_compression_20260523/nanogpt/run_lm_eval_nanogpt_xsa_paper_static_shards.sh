#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SINGLE_GPU_SCRIPT="$SCRIPT_DIR/run_lm_eval_nanogpt_xsa_paper_single_gpu.sh"

CKPT_ROOT="${CKPT_ROOT:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/nanoGPT/out}"
PATH_FILTER="${PATH_FILTER:-xsa-paper}"
CKPT_NAME="${CKPT_NAME:-}"
CKPT_NAMES="${CKPT_NAMES:-}"
CKPT_PATHS_FILE="${CKPT_PATHS_FILE:-}"
FIND_MAXDEPTH="${FIND_MAXDEPTH:-}"
SORT_RESULTS="${SORT_RESULTS:-true}"
BACKGROUND="${BACKGROUND:-false}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DISPATCH_MODE="${DISPATCH_MODE:-per_ckpt}"

DEFAULT_TASKS="arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
TASKS="${TASKS:-$DEFAULT_TASKS}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
NANOGPT_REPO_ROOT="${NANOGPT_REPO_ROOT:-$HARNESS_DIR/nanoGPT}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
SHARD_FACTOR="${SHARD_FACTOR:-4}"

XSA_FORWARD_ONLY="${XSA_FORWARD_ONLY:-}"
XSA_FORWARD_TARGET="${XSA_FORWARD_TARGET:-}"
XSA_FORWARD_REF="${XSA_FORWARD_REF:-}"
XSA_FORWARD_SPACE="${XSA_FORWARD_SPACE:-}"
XSA_FORWARD_OP="${XSA_FORWARD_OP:-}"
XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-}"

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_logs}"
PID_DIR="${PID_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_pids}"
SHARD_DIR="${SHARD_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_shards}"
mkdir -p "$LOG_DIR" "$PID_DIR" "$SHARD_DIR"

timestamp="$(date +"%Y-%m-%dT%H-%M-%S")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${timestamp}-nanogpt-xsa-paper-static-shards.log}"
PID_FILE="${PID_FILE:-$PID_DIR/${timestamp}-nanogpt-xsa-paper-static-shards.pid}"

trim_spaces() {
  echo "${1:-}" | xargs
}

build_ckpt_names() {
  local names_raw
  local item
  local cleaned
  local names=()
  if [[ -n "$CKPT_NAMES" ]]; then
    names_raw="$CKPT_NAMES"
  elif [[ -n "$CKPT_NAME" ]]; then
    names_raw="$CKPT_NAME"
  else
    names_raw="ckpt_best.pt,ckpt.pt"
  fi
  IFS=',' read -r -a raw_items <<< "$names_raw"
  for item in "${raw_items[@]}"; do
    cleaned="$(trim_spaces "$item")"
    [[ -n "$cleaned" ]] && names+=("$cleaned")
  done
  printf '%s\n' "${names[@]}"
}

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

is_nonneg_int() {
  local v="${1:-}"
  [[ "$v" =~ ^[0-9]+$ ]]
}

if [[ "$BACKGROUND" == "true" && "${_NANOGPT_XSA_STATIC_SHARDS_BG_CHILD:-0}" != "1" ]]; then
  export _NANOGPT_XSA_STATIC_SHARDS_BG_CHILD=1
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
echo "$$" > "$PID_FILE"

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "[ERROR] CKPT_ROOT not found: $CKPT_ROOT"
  exit 1
fi
if [[ ! -f "$SINGLE_GPU_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $SINGLE_GPU_SCRIPT"
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

echo "[INFO] CKPT_ROOT=$CKPT_ROOT"
echo "[INFO] PATH_FILTER=$PATH_FILTER"
mapfile -t CKPT_NAME_LIST < <(build_ckpt_names)
if [[ ${#CKPT_NAME_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoint filenames configured."
  exit 1
fi
echo "[INFO] CKPT_NAMES=$(IFS=,; echo "${CKPT_NAME_LIST[*]}")"
echo "[INFO] CKPT_PATHS_FILE=${CKPT_PATHS_FILE:-<empty>}"
echo "[INFO] FIND_MAXDEPTH=${FIND_MAXDEPTH:-<empty>}"
echo "[INFO] GPU_LIST=$GPU_LIST"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] MAX_BATCH_SIZE=$MAX_BATCH_SIZE"
echo "[INFO] SHARD_FACTOR=$SHARD_FACTOR"
echo "[INFO] DISPATCH_MODE=$DISPATCH_MODE"
echo "[INFO] SKIP_EXISTING_RESULTS=$SKIP_EXISTING_RESULTS"
echo "[INFO] Logs:"
echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
echo "[INFO] PID file:"
echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"

ckpt_paths=()
if [[ -n "$CKPT_PATHS_FILE" ]]; then
  if [[ ! -f "$CKPT_PATHS_FILE" ]]; then
    echo "[ERROR] CKPT_PATHS_FILE not found: $CKPT_PATHS_FILE"
    exit 1
  fi
  echo "[STAGE] ckpt_discovery_start mode=file source=$CKPT_PATHS_FILE"
  while IFS= read -r ckpt; do
    ckpt="$(trim_spaces "$ckpt")"
    [[ -z "$ckpt" ]] && continue
    [[ ! -f "$ckpt" ]] && continue
    if [[ "$ckpt" != *"$PATH_FILTER"* ]]; then
      continue
    fi
    for ckpt_name in "${CKPT_NAME_LIST[@]}"; do
      if [[ "$(basename "$ckpt")" == "$ckpt_name" ]]; then
        ckpt_paths+=("$ckpt")
        break
      fi
    done
  done < "$CKPT_PATHS_FILE"
  echo "[STAGE] ckpt_discovery_done mode=file found=${#ckpt_paths[@]}"
else
  echo "[STAGE] ckpt_discovery_start mode=find root=$CKPT_ROOT filter=$PATH_FILTER ckpt_names=$(IFS=,; echo "${CKPT_NAME_LIST[*]}")"
  find_cmd=(find "$CKPT_ROOT")
  if is_nonneg_int "$FIND_MAXDEPTH"; then
    find_cmd+=(-maxdepth "$FIND_MAXDEPTH")
  fi
  find_cmd+=(-type f '(')
  for idx in "${!CKPT_NAME_LIST[@]}"; do
    (( idx > 0 )) && find_cmd+=(-o)
    find_cmd+=(-name "${CKPT_NAME_LIST[$idx]}")
  done
  find_cmd+=(')' -path "*${PATH_FILTER}*" -print0)
  while IFS= read -r -d '' ckpt; do
    ckpt_paths+=("$ckpt")
  done < <("${find_cmd[@]}")
  echo "[STAGE] ckpt_discovery_done mode=find found=${#ckpt_paths[@]}"
fi

if [[ "$SORT_RESULTS" == "true" && ${#ckpt_paths[@]} -gt 1 ]]; then
  mapfile -t ckpt_paths < <(printf '%s\n' "${ckpt_paths[@]}" | sort)
fi

if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
  echo "[WARN] No checkpoint matched filter under: $CKPT_ROOT"
  exit 0
fi

echo "[INFO] Found ${#ckpt_paths[@]} checkpoint(s) matching filter."
echo "[INFO] Static shard mode: ${#GPU_IDS[@]} worker(s), one long-lived single-GPU script per worker."

total_shards=$(( ${#GPU_IDS[@]} * SHARD_FACTOR ))
if (( total_shards < ${#GPU_IDS[@]} )); then
  total_shards="${#GPU_IDS[@]}"
fi
if (( total_shards > ${#ckpt_paths[@]} )); then
  total_shards="${#ckpt_paths[@]}"
fi
if (( total_shards < 1 )); then
  total_shards=1
fi
echo "[INFO] Creating $total_shards shard(s) for ${#ckpt_paths[@]} checkpoint(s)."

shard_files=()
for ((idx=0; idx<total_shards; idx++)); do
  shard_file="$SHARD_DIR/${timestamp}-shard$(printf '%02d' "$idx").txt"
  : > "$shard_file"
  shard_files+=("$shard_file")
done

for idx in "${!ckpt_paths[@]}"; do
  shard_idx=$(( idx % total_shards ))
  printf '%s\n' "${ckpt_paths[$idx]}" >> "${shard_files[$shard_idx]}"
done

claim_root="$SHARD_DIR/${timestamp}-claims"
mkdir -p "$claim_root"

run_worker() {
  local gpu_id="$1"
  local worker_log_file="$2"
  local worker_pid_file="$3"
  local shard_file
  local shard_count
  local shard_base
  local claim_dir
  local found_work
  local shard_status
  local failure_marker

  while true; do
    found_work=0
    for shard_file in "${shard_files[@]}"; do
      shard_count="$(wc -l < "$shard_file" | tr -d '[:space:]')"
      [[ "$shard_count" == "0" ]] && continue
      shard_base="$(basename "$shard_file")"
      claim_dir="$claim_root/${shard_base}.claimed"
      if mkdir "$claim_dir" 2>/dev/null; then
        found_work=1
        echo "[CLAIM] gpu=$gpu_id shard_file=$shard_file shard_count=$shard_count"
        if GPU_ID="$gpu_id" \
          CKPT_ROOT="$CKPT_ROOT" \
          PATH_FILTER="$PATH_FILTER" \
          CKPT_NAME="$CKPT_NAME" \
          CKPT_NAMES="$CKPT_NAMES" \
          CKPT_PATHS_FILE="$shard_file" \
          FIND_MAXDEPTH="$FIND_MAXDEPTH" \
          SORT_RESULTS="$SORT_RESULTS" \
          CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
          SKIP_EXISTING_RESULTS="$SKIP_EXISTING_RESULTS" \
          BACKGROUND="false" \
          DISPATCH_MODE="$DISPATCH_MODE" \
          TASKS="$TASKS" \
          BATCH_SIZE="$BATCH_SIZE" \
          MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
          DEVICE="$DEVICE" \
          DTYPE="$DTYPE" \
          NUM_FEWSHOT="$NUM_FEWSHOT" \
          LIMIT="$LIMIT" \
          MAX_LENGTH="$MAX_LENGTH" \
          NANOGPT_REPO_ROOT="$NANOGPT_REPO_ROOT" \
          XSA_FORWARD_ONLY="$XSA_FORWARD_ONLY" \
          XSA_FORWARD_TARGET="$XSA_FORWARD_TARGET" \
          XSA_FORWARD_REF="$XSA_FORWARD_REF" \
          XSA_FORWARD_SPACE="$XSA_FORWARD_SPACE" \
          XSA_FORWARD_OP="$XSA_FORWARD_OP" \
          XSA_FORWARD_ALPHA="$XSA_FORWARD_ALPHA" \
          LOG_FILE="$worker_log_file" \
          PID_FILE="$worker_pid_file" \
          bash "$SINGLE_GPU_SCRIPT"; then
          shard_status=0
        else
          shard_status=$?
        fi
        if [[ "$shard_status" -ne 0 ]]; then
          failure_marker="$claim_dir/failed.exitcode.${shard_status}"
          : > "$failure_marker"
          echo "[WARN] gpu=$gpu_id shard_file=$shard_file exited_nonzero=$shard_status; continue to next shard"
        else
          : > "$claim_dir/success"
          echo "[DONE] gpu=$gpu_id shard_file=$shard_file completed successfully"
        fi
        break
      fi
    done
    if [[ "$found_work" == "0" ]]; then
      echo "[DONE] gpu=$gpu_id no more shards to claim"
      return 0
    fi
  done
}

PIDS=()
FAILS=0
WORKER_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
  worker_log_file="${LOG_DIR}/${timestamp}-gpu${gpu_id}-static-shard.log"
  worker_pid_file="${PID_DIR}/${timestamp}-gpu${gpu_id}-static-shard.pid"
  echo "[RUN] gpu=$gpu_id worker_log=$worker_log_file"

  run_worker "$gpu_id" "$worker_log_file" "$worker_pid_file" &
  PIDS+=("$!")
  WORKER_GPUS+=("$gpu_id")
done

for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  gpu_id="${WORKER_GPUS[$idx]}"
  if wait "$pid"; then
    echo "[DONE] gpu=$gpu_id worker exited successfully"
  else
    echo "[FAIL] gpu=$gpu_id worker exited with failure"
    FAILS=$((FAILS + 1))
  fi
done

while IFS= read -r -d '' failure_file; do
  FAILS=$((FAILS + 1))
done < <(find "$claim_root" -type f -name 'failed.exitcode.*' -print0)

echo "[DONE] workers=${#PIDS[@]} failed=$FAILS"
if [[ "$FAILS" -gt 0 ]]; then
  exit 1
fi
