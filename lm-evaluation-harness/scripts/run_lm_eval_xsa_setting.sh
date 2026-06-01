#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HARNESS_DIR"
# Ensure the local lm_eval (from this repo) takes priority over any globally
# installed version (e.g. demystifying-transformers-main).
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
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
BACKEND="${BACKEND:-causal}"
TASKS="${TASKS:-hellaswag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/xsa_lm_eval}"
SETTING="${SETTING:-xsa_middle}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"   # single | model_parallel | data_parallel
PARALLELIZE="${PARALLELIZE:-false}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-}"
MAX_PORT_RETRIES="${MAX_PORT_RETRIES:-15}"
PORT_RETRY_SLEEP="${PORT_RETRY_SLEEP:-3}"
BACKGROUND="${BACKGROUND:-false}"
# data_parallel 失败后是否自动降级到 single 模式重跑（避免 NCCL OOM 导致结果缺失）
FALLBACK_TO_SINGLE="${FALLBACK_TO_SINGLE:-true}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
XSA_FORWARD_OP="${XSA_FORWARD_OP:-remove_parallel}"
XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-1.0}"
XSA_LAYER_SCALE_MODE="${XSA_LAYER_SCALE_MODE:-none}"
XSA_LAYER_SCALE_SEED="${XSA_LAYER_SCALE_SEED:-0}"
XSA_LAYER_PARA_SCALE_MIN="${XSA_LAYER_PARA_SCALE_MIN:--20.0}"
XSA_LAYER_PARA_SCALE_MAX="${XSA_LAYER_PARA_SCALE_MAX:-20.0}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
ATTN_DIAG_ENABLED="${ATTN_DIAG_ENABLED:-false}"
ATTN_DIAG_MODE="${ATTN_DIAG_MODE:-none}"
ATTN_DIAG_KEEP_FIRST="${ATTN_DIAG_KEEP_FIRST:-true}"
TOKEN_COMPRESSION_ENABLED="${TOKEN_COMPRESSION_ENABLED:-false}"
TOKEN_COMPRESSION_MODE="${TOKEN_COMPRESSION_MODE:-value_parallel_drop}"
TOKEN_COMPRESSION_KEEP_RATIO="${TOKEN_COMPRESSION_KEEP_RATIO:-1.0}"
TOKEN_COMPRESSION_KEEP_TOKENS="${TOKEN_COMPRESSION_KEEP_TOKENS:-0}"
TOKEN_COMPRESSION_LAYER="${TOKEN_COMPRESSION_LAYER:-0}"
TOKEN_COMPRESSION_LAYERS="${TOKEN_COMPRESSION_LAYERS:-}"
TOKEN_COMPRESSION_REF="${TOKEN_COMPRESSION_REF:-last_value}"
TOKEN_COMPRESSION_PRESERVE_FIRST="${TOKEN_COMPRESSION_PRESERVE_FIRST:-32}"
TOKEN_COMPRESSION_PRESERVE_LAST="${TOKEN_COMPRESSION_PRESERVE_LAST:-256}"
TOKEN_COMPRESSION_MIN_CONTEXT="${TOKEN_COMPRESSION_MIN_CONTEXT:-512}"
TOKEN_COMPRESSION_PARALLEL_WEIGHT="${TOKEN_COMPRESSION_PARALLEL_WEIGHT:-0.25}"
SPARSE_ATTENTION_ENABLED="${SPARSE_ATTENTION_ENABLED:-false}"
SPARSE_ATTENTION_MODE="${SPARSE_ATTENTION_MODE:-contribution_perp}"
SPARSE_ATTENTION_KEEP_RATIO="${SPARSE_ATTENTION_KEEP_RATIO:-1.0}"
SPARSE_ATTENTION_KEEP_TOKENS="${SPARSE_ATTENTION_KEEP_TOKENS:-0}"
SPARSE_ATTENTION_SINK_TOKENS="${SPARSE_ATTENTION_SINK_TOKENS:-32}"
SPARSE_ATTENTION_LOCAL_TOKENS="${SPARSE_ATTENTION_LOCAL_TOKENS:-128}"
SPARSE_ATTENTION_PREFILL_ONLY="${SPARSE_ATTENTION_PREFILL_ONLY:-true}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
METADATA="${METADATA:-}"
GEN_KWARGS="${GEN_KWARGS:-}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-}"
USE_CACHE_PATH="${USE_CACHE_PATH:-}"
SAMPLES_PATH="${SAMPLES_PATH:-}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  model_log_tag="$(sanitize_tag "${MODEL_NAME:-unknown_model}")"
  task_log_tag="$(sanitize_tag "${TASKS:-unknown_task}")"
setting_log_tag="$(sanitize_tag "${SETTING:-unknown_setting}")"
op_log_tag="$(sanitize_tag "${XSA_FORWARD_OP:-remove_parallel}")"
  LOG_FILE="$LOG_DIR/$(timestamp)-setting-${setting_log_tag}-${op_log_tag}-${model_log_tag}-${task_log_tag}.log"
fi
if [[ "$BACKGROUND" == "true" && "${_SETTING_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export LOG_FILE
  _SETTING_BACKGROUND_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  echo "[INFO] Running in background. PID=$!  Logs: $LOG_FILE"
  exit 0
fi

if [[ "${LM_EVAL_LOGGING_INITIALIZED:-0}" != "1" ]]; then
  export LM_EVAL_LOGGING_INITIALIZED=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

safe_log_stderr() {
  local msg="$1"
  if ! printf '%s\n' "$msg" >&2 2>/dev/null; then
    printf '%s\n' "$msg"
  fi
}

# Let the OS pick a guaranteed-free ephemeral port (bind to port 0),
# then immediately close it and use that port number.
# The race window shrinks from seconds to microseconds, and SO_REUSEADDR
# on the TCPStore side ensures it can re-bind even in TIME_WAIT.
reserve_port() {
  local port
  port="$("$PYTHON_BIN" - <<'PY'
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("", 0))
    port = s.getsockname()[1]
    print(port)
finally:
    s.close()
PY
)"
  if [[ -z "$port" ]]; then
    safe_log_stderr "[ERROR] Failed to obtain a free port from OS."
    return 1
  fi
  PORT_LOCK_DIR=""
  echo "$port"
}

release_reserved_port() {
  PORT_LOCK_DIR=""
}

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _cuda_ids <<< "$CUDA_VISIBLE_DEVICES"
    local n=0
    local item
    for item in "${_cuda_ids[@]}"; do
      [[ -n "${item// }" ]] && ((n+=1))
    done
    echo "$n"
    return 0
  fi
  "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

if [[ -z "$MODEL_NAME" ]]; then
  echo "[ERROR] MODEL_NAME is required." >&2
  echo "Example: MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct bash $0" >&2
  exit 1
fi

case "$SETTING" in
  none)
    xsa_target="none"
    xsa_site="xsa_middle_multihead"
    ;;
  xsa_middle)
    xsa_target="attn"
    xsa_site="xsa_middle"
    ;;
  xsa_middle_multihead)
    xsa_target="attn"
    xsa_site="xsa_middle_multihead"
    ;;
  residual_attn)
    xsa_target="attn"
    xsa_site="residual_output"
    ;;
  residual_mlp)
    xsa_target="mlp"
    xsa_site="residual_output"
    ;;
  residual_both)
    xsa_target="both"
    xsa_site="residual_output"
    ;;
  *)
    echo "[ERROR] Unknown SETTING=$SETTING" >&2
    exit 1
    ;;
esac

op_tag=""
if [[ "$xsa_target" != "none" ]]; then
  if [[ "$XSA_FORWARD_OP" != "remove_parallel" || "$XSA_FORWARD_ALPHA" != "1.0" ]]; then
    alpha_tag="$(sanitize_tag "$XSA_FORWARD_ALPHA")"
    op_tag="-${XSA_FORWARD_OP}-a${alpha_tag}"
  fi
fi

case "$LAUNCH_MODE" in
  single)
    PARALLELIZE="false"
    ;;
  model_parallel)
    PARALLELIZE="true"
    ;;
  data_parallel)
    PARALLELIZE="false"
    ;;
  *)
    echo "[ERROR] Unknown LAUNCH_MODE=$LAUNCH_MODE (expected: single, model_parallel, data_parallel)" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_ROOT"
model_tag="$(sanitize_tag "$MODEL_NAME")"
tasks_tag="$(sanitize_tag "$TASKS")"
setting_tag="$(sanitize_tag "$SETTING")"
output_path="${OUTPUT_PATH:-$OUTPUT_ROOT/${model_tag}-${tasks_tag}-${setting_tag}${op_tag}.json}"

model_args=(
  "pretrained=$MODEL_NAME"
  "backend=$BACKEND"
  "dtype=$DTYPE"
  "trust_remote_code=$TRUST_REMOTE_CODE"
  "parallelize=$PARALLELIZE"
  "xsa_target=$xsa_target"
  "xsa_intervention_site=$xsa_site"
  "xsa_start_layer=$XSA_START_LAYER"
  "xsa_end_layer=$XSA_END_LAYER"
  "xsa_skip_first_n=$XSA_SKIP_FIRST_N"
  "xsa_skip_last_n=$XSA_SKIP_LAST_N"
  "xsa_forward_op=$XSA_FORWARD_OP"
  "xsa_forward_alpha=$XSA_FORWARD_ALPHA"
  "xsa_layer_scale_mode=$XSA_LAYER_SCALE_MODE"
  "xsa_layer_scale_seed=$XSA_LAYER_SCALE_SEED"
  "xsa_layer_para_scale_min=$XSA_LAYER_PARA_SCALE_MIN"
  "xsa_layer_para_scale_max=$XSA_LAYER_PARA_SCALE_MAX"
  "xsa_track_stats=$XSA_TRACK_STATS"
  "xsa_layerwise_stats=$XSA_LAYERWISE_STATS"
  "attn_diag_enabled=$ATTN_DIAG_ENABLED"
  "attn_diag_mode=$ATTN_DIAG_MODE"
  "attn_diag_keep_first=$ATTN_DIAG_KEEP_FIRST"
  "token_compression_enabled=$TOKEN_COMPRESSION_ENABLED"
  "token_compression_mode=$TOKEN_COMPRESSION_MODE"
  "token_compression_keep_ratio=$TOKEN_COMPRESSION_KEEP_RATIO"
  "token_compression_keep_tokens=$TOKEN_COMPRESSION_KEEP_TOKENS"
  "token_compression_layer=$TOKEN_COMPRESSION_LAYER"
  "token_compression_layers=$TOKEN_COMPRESSION_LAYERS"
  "token_compression_ref=$TOKEN_COMPRESSION_REF"
  "token_compression_preserve_first=$TOKEN_COMPRESSION_PRESERVE_FIRST"
  "token_compression_preserve_last=$TOKEN_COMPRESSION_PRESERVE_LAST"
  "token_compression_min_context=$TOKEN_COMPRESSION_MIN_CONTEXT"
  "token_compression_parallel_weight=$TOKEN_COMPRESSION_PARALLEL_WEIGHT"
  "sparse_attention_enabled=$SPARSE_ATTENTION_ENABLED"
  "sparse_attention_mode=$SPARSE_ATTENTION_MODE"
  "sparse_attention_keep_ratio=$SPARSE_ATTENTION_KEEP_RATIO"
  "sparse_attention_keep_tokens=$SPARSE_ATTENTION_KEEP_TOKENS"
  "sparse_attention_sink_tokens=$SPARSE_ATTENTION_SINK_TOKENS"
  "sparse_attention_local_tokens=$SPARSE_ATTENTION_LOCAL_TOKENS"
  "sparse_attention_prefill_only=$SPARSE_ATTENTION_PREFILL_ONLY"
)

if [[ -n "$TOKENIZER_NAME" ]]; then
  model_args+=("tokenizer=$TOKENIZER_NAME")
fi
if [[ -n "$MAX_LENGTH" ]]; then
  model_args+=("max_length=$MAX_LENGTH")
fi
if [[ -n "$MAX_MEMORY_PER_GPU" ]]; then
  model_args+=("max_memory_per_gpu=$MAX_MEMORY_PER_GPU")
fi
if [[ -n "$MAX_CPU_MEMORY" ]]; then
  model_args+=("max_cpu_memory=$MAX_CPU_MEMORY")
fi
if [[ -n "$OFFLOAD_FOLDER" ]]; then
  model_args+=("offload_folder=$OFFLOAD_FOLDER")
fi
model_args_string="$(IFS=,; echo "${model_args[*]}")"
if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
  model_args_string="${model_args_string},${EXTRA_MODEL_ARGS}"
fi

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] BACKEND=$BACKEND"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] SETTING=$SETTING"
echo "[INFO] XSA_FORWARD_OP=$XSA_FORWARD_OP"
echo "[INFO] XSA_FORWARD_ALPHA=$XSA_FORWARD_ALPHA"
echo "[INFO] XSA_LAYER_SCALE_MODE=$XSA_LAYER_SCALE_MODE"
echo "[INFO] XSA_LAYER_SCALE_SEED=$XSA_LAYER_SCALE_SEED"
echo "[INFO] XSA_LAYER_PARA_SCALE_MIN=$XSA_LAYER_PARA_SCALE_MIN"
echo "[INFO] XSA_LAYER_PARA_SCALE_MAX=$XSA_LAYER_PARA_SCALE_MAX"
echo "[INFO] ATTN_DIAG_ENABLED=$ATTN_DIAG_ENABLED"
echo "[INFO] ATTN_DIAG_MODE=$ATTN_DIAG_MODE"
echo "[INFO] ATTN_DIAG_KEEP_FIRST=$ATTN_DIAG_KEEP_FIRST"
echo "[INFO] TOKEN_COMPRESSION_ENABLED=$TOKEN_COMPRESSION_ENABLED"
echo "[INFO] TOKEN_COMPRESSION_MODE=$TOKEN_COMPRESSION_MODE"
echo "[INFO] TOKEN_COMPRESSION_KEEP_RATIO=$TOKEN_COMPRESSION_KEEP_RATIO"
echo "[INFO] TOKEN_COMPRESSION_KEEP_TOKENS=$TOKEN_COMPRESSION_KEEP_TOKENS"
echo "[INFO] TOKEN_COMPRESSION_LAYER=$TOKEN_COMPRESSION_LAYER"
echo "[INFO] TOKEN_COMPRESSION_REF=$TOKEN_COMPRESSION_REF"
echo "[INFO] TOKEN_COMPRESSION_PRESERVE_FIRST=$TOKEN_COMPRESSION_PRESERVE_FIRST"
echo "[INFO] TOKEN_COMPRESSION_PRESERVE_LAST=$TOKEN_COMPRESSION_PRESERVE_LAST"
echo "[INFO] TOKEN_COMPRESSION_MIN_CONTEXT=$TOKEN_COMPRESSION_MIN_CONTEXT"
echo "[INFO] SPARSE_ATTENTION_ENABLED=$SPARSE_ATTENTION_ENABLED"
echo "[INFO] SPARSE_ATTENTION_MODE=$SPARSE_ATTENTION_MODE"
echo "[INFO] SPARSE_ATTENTION_KEEP_RATIO=$SPARSE_ATTENTION_KEEP_RATIO"
echo "[INFO] SPARSE_ATTENTION_KEEP_TOKENS=$SPARSE_ATTENTION_KEEP_TOKENS"
echo "[INFO] SPARSE_ATTENTION_SINK_TOKENS=$SPARSE_ATTENTION_SINK_TOKENS"
echo "[INFO] SPARSE_ATTENTION_LOCAL_TOKENS=$SPARSE_ATTENTION_LOCAL_TOKENS"
echo "[INFO] SPARSE_ATTENTION_PREFILL_ONLY=$SPARSE_ATTENTION_PREFILL_ONLY"
echo "[INFO] LAUNCH_MODE=$LAUNCH_MODE"
echo "[INFO] OUTPUT_PATH=$output_path"
if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
  echo "[INFO] EXTRA_MODEL_ARGS=$EXTRA_MODEL_ARGS"
fi
if [[ -n "$USE_CACHE_PATH" ]]; then
  echo "[INFO] USE_CACHE_PATH=$USE_CACHE_PATH"
fi
if [[ -n "$SAMPLES_PATH" ]]; then
  echo "[INFO] SAMPLES_PATH=$SAMPLES_PATH"
fi
echo "[INFO] LOG_SAMPLES=$LOG_SAMPLES"
if [[ -n "$INCLUDE_PATH" ]]; then
  echo "[INFO] INCLUDE_PATH=$INCLUDE_PATH"
fi

if [[ "$xsa_target" != "none" ]]; then
  scale_dir="${output_path%.json}.layer_scales"
  "$PYTHON_BIN" "$HARNESS_DIR/../analysis/forward_geometry/write_xsa_layer_scales.py" \
    --model_name "$MODEL_NAME" \
    --output_dir "$scale_dir" \
    --mode "$XSA_LAYER_SCALE_MODE" \
    --seed "$XSA_LAYER_SCALE_SEED" \
    --para_scale_min "$XSA_LAYER_PARA_SCALE_MIN" \
    --para_scale_max "$XSA_LAYER_PARA_SCALE_MAX" \
    --default_para_scale "$("$PYTHON_BIN" - "$XSA_FORWARD_ALPHA" <<'PY'
import sys
print(f"{1.0 - float(sys.argv[1]):g}")
PY
)" \
    --tag "setting=$SETTING task=$TASKS"
fi

run_args=(
  --model hf-xsa
  --model_args "$model_args_string"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --num_fewshot "$NUM_FEWSHOT"
  --output_path "$output_path"
)

if [[ "$APPLY_CHAT_TEMPLATE" == "true" ]]; then
  run_args+=(--apply_chat_template)
fi
if [[ -n "$FEWSHOT_AS_MULTITURN" ]]; then
  run_args+=(--fewshot_as_multiturn "$FEWSHOT_AS_MULTITURN")
fi

if [[ -n "$LIMIT" ]]; then
  run_args+=(--limit "$LIMIT")
fi

if [[ -n "$USE_CACHE_PATH" ]]; then
  run_args+=(--use_cache "$USE_CACHE_PATH")
fi

if [[ -n "$SAMPLES_PATH" ]]; then
  run_args+=(--samples "$SAMPLES_PATH")
fi
if [[ "$LOG_SAMPLES" == "true" ]]; then
  run_args+=(--log_samples)
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
if [[ -n "$GEN_KWARGS" ]]; then
  run_args+=(--gen_kwargs "$GEN_KWARGS")
fi

run_single_process() {
  echo "[INFO] Running in single-process mode."
  env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m lm_eval run "${run_args[@]}"
}

if [[ "$LAUNCH_MODE" == "data_parallel" ]]; then
  if ! command -v accelerate >/dev/null 2>&1; then
    safe_log_stderr "[ERROR] accelerate is required for LAUNCH_MODE=data_parallel"
    exit 1
  fi

  num_processes="$ACCELERATE_NUM_PROCESSES"
  if [[ -z "$num_processes" ]]; then
    num_processes="$(count_visible_gpus)"
  fi
  if [[ -z "$num_processes" || "$num_processes" -le 1 ]]; then
    safe_log_stderr "[WARN] LAUNCH_MODE=data_parallel but detected num_processes=$num_processes; falling back to single process."
    run_single_process
    exit 0
  fi

  dp_ok=false
  for ((attempt=1; attempt<=MAX_PORT_RETRIES; attempt++)); do
    port="$(reserve_port)"
    echo "[INFO] accelerate attempt ${attempt}/${MAX_PORT_RETRIES} with num_processes=${num_processes} port=${port}"
    cmd=(
      accelerate launch
      --multi_gpu
      --num_processes "$num_processes"
      --main_process_port "$port"
      -m lm_eval run
      "${run_args[@]}"
    )
    if env \
        NCCL_ASYNC_ERROR_HANDLING=1 \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        NCCL_DEBUG=WARN \
        PYTHONUNBUFFERED=1 \
        "${cmd[@]}"; then
      release_reserved_port
      dp_ok=true
      break
    fi
    release_reserved_port
    if (( attempt < MAX_PORT_RETRIES )); then
      safe_log_stderr "[WARN] accelerate launch failed on port=${port} (attempt ${attempt}/${MAX_PORT_RETRIES}), retrying in ${PORT_RETRY_SLEEP}s..."
      sleep "$PORT_RETRY_SLEEP"
    fi
  done

  if [[ "$dp_ok" == true ]]; then
    exit 0
  fi

  safe_log_stderr "[ERROR] accelerate launch failed after ${MAX_PORT_RETRIES} attempts."
  if [[ "$FALLBACK_TO_SINGLE" == "true" ]]; then
    safe_log_stderr "[WARN] Falling back to single-process mode (FALLBACK_TO_SINGLE=true)."
    run_single_process
  else
    exit 1
  fi
else
  run_single_process
fi
