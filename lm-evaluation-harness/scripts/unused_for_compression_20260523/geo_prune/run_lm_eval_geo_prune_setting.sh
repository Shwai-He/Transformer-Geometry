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
TASKS="${TASKS:-hellaswag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/geo_prune_lm_eval}"
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

GEO_PRUNE_ENABLED="${GEO_PRUNE_ENABLED:-true}"
GEO_PRUNE_METHOD="${GEO_PRUNE_METHOD:-wanda}"
GEO_PRUNE_STRATEGY="${GEO_PRUNE_STRATEGY:-none}"
GEO_PRUNE_TARGETS="${GEO_PRUNE_TARGETS:-q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj}"
GEO_GEOMETRY_MODE="${GEO_GEOMETRY_MODE:-residual_error}"
GEO_GEOMETRY_ALPHA="${GEO_GEOMETRY_ALPHA:-1.0}"
GEO_GEOMETRY_TARGETS="${GEO_GEOMETRY_TARGETS:-o_proj+down_proj}"
GEO_SPARSITY_RATIO="${GEO_SPARSITY_RATIO:-0.5}"
GEO_SPARSITY_TYPE="${GEO_SPARSITY_TYPE:-2:4}"
GEO_THRESHOLD_SCOPE="${GEO_THRESHOLD_SCOPE:-row}"
GEO_TOKEN_SCOPE="${GEO_TOKEN_SCOPE:-last}"
GEO_CALIB_PROMPTS="${GEO_CALIB_PROMPTS:-}"
GEO_CALIB_FILE="${GEO_CALIB_FILE:-}"
GEO_MAX_PROMPTS="${GEO_MAX_PROMPTS:-2}"
GEO_MAX_LENGTH="${GEO_MAX_LENGTH:-256}"
GEO_SCORE_CACHE_PATH="${GEO_SCORE_CACHE_PATH:-}"

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/geo_prune_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  model_log_tag="$(sanitize_tag "${MODEL_NAME:-unknown_model}")"
  task_log_tag="$(sanitize_tag "${TASKS:-unknown_task}")"
  geo_log_tag="$(sanitize_tag "${GEO_SPARSITY_TYPE}-${GEO_PRUNE_STRATEGY}")"
  LOG_FILE="$LOG_DIR/$(timestamp)-geo-prune-${geo_log_tag}-${model_log_tag}-${task_log_tag}.log"
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
setting_tag="$(sanitize_tag "${GEO_SPARSITY_TYPE}-${GEO_PRUNE_STRATEGY}")"
output_path="${OUTPUT_PATH:-$OUTPUT_ROOT/${model_tag}-${task_tag}-${setting_tag}.json}"

model_args=(
  "pretrained=$MODEL_NAME"
  "backend=causal"
  "dtype=$DTYPE"
  "trust_remote_code=$TRUST_REMOTE_CODE"
  "geo_prune_enabled=$GEO_PRUNE_ENABLED"
  "geo_prune_method=$GEO_PRUNE_METHOD"
  "geo_prune_strategy=$GEO_PRUNE_STRATEGY"
  "geo_prune_targets=$GEO_PRUNE_TARGETS"
  "geo_geometry_mode=$GEO_GEOMETRY_MODE"
  "geo_geometry_alpha=$GEO_GEOMETRY_ALPHA"
  "geo_geometry_targets=$GEO_GEOMETRY_TARGETS"
  "geo_sparsity_ratio=$GEO_SPARSITY_RATIO"
  "geo_sparsity_type=$GEO_SPARSITY_TYPE"
  "geo_threshold_scope=$GEO_THRESHOLD_SCOPE"
  "geo_token_scope=$GEO_TOKEN_SCOPE"
  "geo_max_prompts=$GEO_MAX_PROMPTS"
  "geo_max_length=$GEO_MAX_LENGTH"
)

if [[ -n "$MAX_LENGTH" ]]; then
  model_args+=("max_length=$MAX_LENGTH")
fi
if [[ -n "$GEO_CALIB_PROMPTS" ]]; then
  model_args+=("geo_calib_prompts=$GEO_CALIB_PROMPTS")
fi
if [[ -n "$GEO_CALIB_FILE" ]]; then
  model_args+=("geo_calib_file=$GEO_CALIB_FILE")
fi
if [[ -n "$GEO_SCORE_CACHE_PATH" ]]; then
  model_args+=("geo_score_cache_path=$GEO_SCORE_CACHE_PATH")
fi
model_args_string="$(IFS=,; echo "${model_args[*]}")"

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] GEO_PRUNE_ENABLED=$GEO_PRUNE_ENABLED"
echo "[INFO] GEO_PRUNE_METHOD=$GEO_PRUNE_METHOD"
echo "[INFO] GEO_PRUNE_STRATEGY=$GEO_PRUNE_STRATEGY"
echo "[INFO] GEO_PRUNE_TARGETS=$GEO_PRUNE_TARGETS"
echo "[INFO] GEO_GEOMETRY_MODE=$GEO_GEOMETRY_MODE"
echo "[INFO] GEO_GEOMETRY_ALPHA=$GEO_GEOMETRY_ALPHA"
echo "[INFO] GEO_GEOMETRY_TARGETS=$GEO_GEOMETRY_TARGETS"
echo "[INFO] GEO_SPARSITY_TYPE=$GEO_SPARSITY_TYPE"
echo "[INFO] GEO_SPARSITY_RATIO=$GEO_SPARSITY_RATIO"
echo "[INFO] OUTPUT_PATH=$output_path"

run_args=(
  --model hf-geo-prune
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

env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m lm_eval run "${run_args[@]}"
