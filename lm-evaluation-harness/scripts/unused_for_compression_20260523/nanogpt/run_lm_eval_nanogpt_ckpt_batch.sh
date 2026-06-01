#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SINGLE_SCRIPT="$SCRIPT_DIR/run_lm_eval_nanogpt_setting.sh"

CKPT_ROOT="${CKPT_ROOT:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/nanoGPT/out}"
CKPT_GLOB="${CKPT_GLOB:-ckpt_best.pt}"
RECURSIVE="${RECURSIVE:-true}"
SORT_RESULTS="${SORT_RESULTS:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-true}"
BACKGROUND="${BACKGROUND:-false}"
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
LOG_FILE="${LOG_FILE:-$LOG_DIR/${timestamp}-nanogpt-ckpt-batch.log}"
PID_FILE="${PID_FILE:-$PID_DIR/${timestamp}-nanogpt-ckpt-batch.pid}"

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

if [[ "$BACKGROUND" == "true" && "${_NANOGPT_CKPT_BATCH_BG_CHILD:-0}" != "1" ]]; then
  export _NANOGPT_CKPT_BATCH_BG_CHILD=1
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

if [[ -z "$CKPT_ROOT" ]]; then
  echo "[ERROR] CKPT_ROOT is required."
  echo "Example:"
  echo "  CKPT_ROOT=/path/to/nanogpt/runs bash $0"
  exit 1
fi

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "[ERROR] CKPT_ROOT not found: $CKPT_ROOT"
  exit 1
fi

if [[ ! -f "$SINGLE_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $SINGLE_SCRIPT"
  exit 1
fi

echo "[INFO] CKPT_ROOT=$CKPT_ROOT"
echo "[INFO] CKPT_GLOB=$CKPT_GLOB"
echo "[INFO] RECURSIVE=$RECURSIVE"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] SKIP_EXISTING_RESULTS=$SKIP_EXISTING_RESULTS"
echo "[INFO] XSA_FORWARD_ONLY=${XSA_FORWARD_ONLY:-<empty>}"
echo "[INFO] XSA_FORWARD_TARGET=${XSA_FORWARD_TARGET:-<empty>}"
echo "[INFO] XSA_FORWARD_REF=${XSA_FORWARD_REF:-<empty>}"
echo "[INFO] XSA_FORWARD_SPACE=${XSA_FORWARD_SPACE:-<empty>}"
echo "[INFO] XSA_FORWARD_OP=${XSA_FORWARD_OP:-<empty>}"
echo "[INFO] XSA_FORWARD_ALPHA=${XSA_FORWARD_ALPHA:-<empty>}"
echo "[INFO] Logs:"
echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
echo "[INFO] PID file:"
echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"

IFS=',' read -r -a TASK_LIST <<< "$TASKS"
if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] TASKS is empty."
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
if [[ "$RECURSIVE" == "true" ]]; then
  while IFS= read -r -d '' ckpt; do
    ckpt_paths+=("$ckpt")
  done < <(find "$CKPT_ROOT" -type f -name "$CKPT_GLOB" -print0)
else
  while IFS= read -r -d '' ckpt; do
    ckpt_paths+=("$ckpt")
  done < <(find "$CKPT_ROOT" -maxdepth 1 -type f -name "$CKPT_GLOB" -print0)
fi

if [[ "$SORT_RESULTS" == "true" && ${#ckpt_paths[@]} -gt 1 ]]; then
  mapfile -t ckpt_paths < <(printf '%s\n' "${ckpt_paths[@]}" | sort)
fi

if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
  echo "[WARN] No ckpt files found under: $CKPT_ROOT (glob: $CKPT_GLOB)"
  exit 0
fi

echo "[INFO] Found ${#ckpt_paths[@]} checkpoint(s)."

fail_count=0
skip_count=0
for ckpt_path in "${ckpt_paths[@]}"; do
  ckpt_tag="$(sanitize_tag "$ckpt_path")"
  model_log_file="${LOG_DIR}/${timestamp}-${ckpt_tag}.log"
  echo "[INFO] Model log:"
  echo "$(cd "$(dirname "$model_log_file")" && pwd)/$(basename "$model_log_file")"

  for raw_task in "${TASK_LIST[@]}"; do
    task="$(echo "$raw_task" | xargs)"
    if [[ -z "$task" ]]; then
      continue
    fi

    fewshot="${NUM_FEWSHOT:-${DEFAULT_FEWSHOT_MAP[$task]:-0}}"
    if [[ -d "$ckpt_path" ]]; then
      output_root="${ckpt_path}/lm_eval_results"
    else
      output_root="$(cd "$(dirname "$ckpt_path")" && pwd)/lm_eval_results"
    fi
    output_path="${output_root}/${ckpt_tag}-$(sanitize_tag "$task").json"

    if [[ "$SKIP_EXISTING_RESULTS" == "true" && -f "$output_path" ]]; then
      echo "--------------------------------------------------"
      echo "[SKIP] existing result: $output_path"
      skip_count=$((skip_count + 1))
      continue
    fi

    echo "--------------------------------------------------"
    echo "[RUN] NANOGPT_CKPT=$ckpt_path"
    echo "[RUN] TASK=$task NUM_FEWSHOT=$fewshot"
    echo "[RUN] OUTPUT_PATH=$output_path"
    if ! NANOGPT_CKPT="$ckpt_path" \
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
      bash "$SINGLE_SCRIPT"; then
      echo "[FAIL] ckpt=$ckpt_path task=$task"
      fail_count=$((fail_count + 1))
      if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
        echo "[ERROR] Stopping due to CONTINUE_ON_ERROR=false"
        exit 1
      fi
    fi
  done
done

echo "--------------------------------------------------"
echo "[DONE] Completed ${#ckpt_paths[@]} checkpoint(s) x ${#TASK_LIST[@]} task(s), skipped=$skip_count failures=$fail_count"
if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
