#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

##############################################################################
# 配置
##############################################################################
NANOGPT_OUT_DIR="${NANOGPT_OUT_DIR:-/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out}"
NANOGPT_REPO_ROOT="${NANOGPT_REPO_ROOT:-}"   # 留空则由 wrapper 自动推断
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
BACKGROUND="${BACKGROUND:-false}"
PROMPT_SET="${PROMPT_SET:-short}" # short|long
if [[ -z "${MAX_LENGTH+x}" ]]; then
  if [[ "$PROMPT_SET" == "long" ]]; then
    MAX_LENGTH="512"
  else
    MAX_LENGTH="256"
  fi
fi

sanitize_tag() {
  local s="$1"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

LOG_DIR="${LOG_DIR:-$ROOT_DIR/representation-analysis/outputs/xsa_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-nanogpt-batch-flip-viz.log"
fi

on_unexpected_error() {
  local exit_code="$1"
  local line_no="$2"
  echo "[ERROR] Unexpected batch-script failure at line $line_no (exit_code=$exit_code)" >&2
  echo "[ERROR] The batch launcher itself hit an error outside per-run handling." >&2
}
trap 'on_unexpected_error $? $LINENO' ERR

# 后台模式：re-exec 自身
if [[ "$BACKGROUND" == "true" && "${_BATCH_BG_CHILD:-0}" != "1" ]]; then
  export LOG_FILE
  _BATCH_BG_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  echo "[INFO] Running in background. PID=$!  Logs: $LOG_FILE"
  exit 0
fi

if [[ "${_BATCH_LOG_INIT:-0}" != "1" ]]; then
  export _BATCH_LOG_INIT=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c "import tiktoken" >/dev/null 2>&1; then
  pip install tiktoken
fi

WRAPPER_SCRIPT="$SCRIPT_DIR/run_nanogpt_xsa_single_layer_flip_viz.sh"
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
  echo "[ERROR] Wrapper script not found: $WRAPPER_SCRIPT" >&2
  exit 1
fi

##############################################################################
# checkpoint 解析（同 wrapper 逻辑）
##############################################################################
resolve_ckpt() {
  local run_dir="$1"
  if [[ -f "$run_dir/ckpt_best.pt" ]]; then
    printf '%s\n' "$run_dir/ckpt_best.pt"
  elif [[ -f "$run_dir/ckpt.pt" ]]; then
    printf '%s\n' "$run_dir/ckpt.pt"
  else
    return 1
  fi
}

##############################################################################
# 主循环
##############################################################################
if [[ ! -d "$NANOGPT_OUT_DIR" ]]; then
  echo "[ERROR] NANOGPT_OUT_DIR not found: $NANOGPT_OUT_DIR" >&2
  exit 1
fi

echo "[INFO] Scanning: $NANOGPT_OUT_DIR"
echo "[INFO] BACKGROUND=$BACKGROUND"
echo "[INFO] OVERWRITE_EXISTING=${OVERWRITE_EXISTING:-false}"
echo "[INFO] CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "[INFO] PROMPT_SET=$PROMPT_SET MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] Forcing wrapper RUN_IN_BACKGROUND=false so batch waits for each run."

FAILED=()
TOTAL_RUNS=0
SKIPPED_RUNS=0
COMPLETED_RUNS=0

while IFS= read -r -d '' run_dir; do
  run_name="$(basename "$run_dir")"

  if ! ckpt_path="$(resolve_ckpt "$run_dir")"; then
    echo "[SKIP] No checkpoint in: $run_dir"
    ((SKIPPED_RUNS+=1)) || true
    continue
  fi

  ckpt_file="$(basename "$ckpt_path" .pt)"   # "ckpt" or "ckpt_best"
  output_dir="$run_dir/$ckpt_file"
  prompt_tag="$(sanitize_tag "$PROMPT_SET")"
  output_prefix="$output_dir/flip-prompt${prompt_tag}"
  mkdir -p "$output_dir"

  # Only treat all_prompts_index as “this run is fully done”.
  # If only sample*-all_layers dirs exist, the run may be partial;
  # fall through to wrapper so it can resume/skip per sample.
  final_index="$output_dir/flip-all_prompts_index.json"
  if [[ -f "$final_index" ]]; then
    echo "[SKIP] Final index exists for: $run_name ($ckpt_file)"
    echo "       $final_index"
    ((SKIPPED_RUNS+=1)) || true
    continue
  fi

  shopt -s nullglob
  partial_outputs=( "$output_dir"/flip-sample*-all_layers/ )
  shopt -u nullglob
  if (( ${#partial_outputs[@]} > 0 )); then
    echo "[INFO] Partial outputs found for: $run_name ($ckpt_file)"
    printf '       - %s\n' "${partial_outputs[@]}"
    echo "[INFO] Continuing so wrapper can resume/skip per sample."
  fi

  echo "=========================================="
  echo "[RUN ] $run_name  ($ckpt_file)"
  echo "=========================================="
  ((TOTAL_RUNS+=1)) || true

  if NANOGPT_CKPT="$ckpt_path" \
       BASE_OUTPUT_DIR="$run_dir" \
       OUTPUT_PREFIX="$output_prefix" \
       RUN_IN_BACKGROUND=false \
       PROMPT_SET="$PROMPT_SET" \
       MAX_LENGTH="$MAX_LENGTH" \
       ${NANOGPT_REPO_ROOT:+NANOGPT_REPO_ROOT="$NANOGPT_REPO_ROOT"} \
       bash "$WRAPPER_SCRIPT"; then
    ((COMPLETED_RUNS+=1)) || true
  else
    run_status=$?
    echo "[FAIL] $run_name" >&2
    echo "       exit_code=$run_status" >&2
    FAILED+=("$run_name")
    if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
      echo "[ERROR] Stopping because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR" >&2
      exit "$run_status"
    fi
    echo "[WARN] Continuing to next run because CONTINUE_ON_ERROR=true" >&2
    continue
  fi

done < <(find "$NANOGPT_OUT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "=========================================="
echo "[INFO] Summary: total_attempted=$TOTAL_RUNS completed=$COMPLETED_RUNS skipped=$SKIPPED_RUNS failed=${#FAILED[@]}"
if (( ${#FAILED[@]} > 0 )); then
  echo "[WARN] ${#FAILED[@]} run(s) failed:" >&2
  printf '  - %s\n' "${FAILED[@]}" >&2
else
  echo "[DONE] All runs completed."
fi
