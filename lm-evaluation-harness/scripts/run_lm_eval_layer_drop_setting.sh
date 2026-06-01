#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HARNESS_DIR"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

timestamp() {
  date +"%Y-%m-%dT%H-%M-%S"
}

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-}"
TASKS="${TASKS:-gsm8k_cot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/layer_drop_lm_eval}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
METADATA="${METADATA:-}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"

LAYER_DROP_ENABLED="${LAYER_DROP_ENABLED:-true}"
LAYER_DROP_ATTN="${LAYER_DROP_ATTN:-}"
LAYER_DROP_MLP="${LAYER_DROP_MLP:-}"
LAYER_DROP_CONFIG="${LAYER_DROP_CONFIG:-}"
LAYER_DROP_COMPONENT="${LAYER_DROP_COMPONENT:-both}"
LAYER_DROP_COUNT="${LAYER_DROP_COUNT:-0}"

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/layer_drop_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  model_log_tag="$(sanitize_tag "${MODEL_NAME:-unknown_model}")"
  task_log_tag="$(sanitize_tag "${TASKS:-unknown_task}")"
  drop_log_tag="$(sanitize_tag "${LAYER_DROP_COMPONENT}-drop${LAYER_DROP_COUNT}")"
  LOG_FILE="$LOG_DIR/$(timestamp)-layer-drop-${drop_log_tag}-${model_log_tag}-${task_log_tag}.log"
fi

if [[ "${LM_EVAL_LOGGING_INITIALIZED:-0}" != "1" ]]; then
  export LM_EVAL_LOGGING_INITIALIZED=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "[ERROR] MODEL_NAME is required." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
model_tag="$(sanitize_tag "$MODEL_NAME")"
task_tag="$(sanitize_tag "$TASKS")"
setting_tag="$(sanitize_tag "${LAYER_DROP_COMPONENT}-drop${LAYER_DROP_COUNT}")"
output_path="${OUTPUT_PATH:-$OUTPUT_ROOT/${model_tag}-${task_tag}-${setting_tag}.json}"

model_args=(
  "pretrained=$MODEL_NAME"
  "backend=causal"
  "dtype=$DTYPE"
  "trust_remote_code=$TRUST_REMOTE_CODE"
  "layer_drop_enabled=$LAYER_DROP_ENABLED"
  "layer_drop_attn=$LAYER_DROP_ATTN"
  "layer_drop_mlp=$LAYER_DROP_MLP"
  "layer_drop_config=$LAYER_DROP_CONFIG"
  "layer_drop_component=$LAYER_DROP_COMPONENT"
  "layer_drop_count=$LAYER_DROP_COUNT"
)

if [[ -n "$MAX_LENGTH" ]]; then
  model_args+=("max_length=$MAX_LENGTH")
fi
model_args_string="$(IFS=,; echo "${model_args[*]}")"

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] LAYER_DROP_ENABLED=$LAYER_DROP_ENABLED"
echo "[INFO] LAYER_DROP_COMPONENT=$LAYER_DROP_COMPONENT"
echo "[INFO] LAYER_DROP_COUNT=$LAYER_DROP_COUNT"
echo "[INFO] LAYER_DROP_CONFIG=$LAYER_DROP_CONFIG"
echo "[INFO] LAYER_DROP_ATTN=$LAYER_DROP_ATTN"
echo "[INFO] LAYER_DROP_MLP=$LAYER_DROP_MLP"
echo "[INFO] OUTPUT_PATH=$output_path"
echo "[INFO] LOG_SAMPLES=$LOG_SAMPLES"

run_args=(
  --model hf-layer-drop
  --model_args "$model_args_string"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --num_fewshot "$NUM_FEWSHOT"
  --device "$DEVICE"
  --output_path "$output_path"
)

if [[ "$APPLY_CHAT_TEMPLATE" == "true" ]]; then
  run_args+=(--apply_chat_template)
fi
if [[ -n "$LIMIT" ]]; then
  run_args+=(--limit "$LIMIT")
fi
if [[ -n "$INCLUDE_PATH" ]]; then
  run_args+=(--include_path "$INCLUDE_PATH")
fi
if [[ "$CONFIRM_RUN_UNSAFE_CODE" == "true" ]]; then
  run_args+=(--confirm_run_unsafe_code)
fi
if [[ -n "$METADATA" ]]; then
  run_args+=(--metadata "$METADATA")
fi
if [[ "$LOG_SAMPLES" == "true" ]]; then
  run_args+=(--log_samples)
fi

status=0
if ! env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m lm_eval run "${run_args[@]}"; then
  status=$?
fi

if [[ "$AUTO_COLLECT_RESULTS" == "true" ]]; then
  repo_root="$(cd "$HARNESS_DIR/.." && pwd)"
  "$PYTHON_BIN" "$repo_root/compression/scripts/collect_layer_drop_lm_eval_results.py" || true
fi

exit "$status"
