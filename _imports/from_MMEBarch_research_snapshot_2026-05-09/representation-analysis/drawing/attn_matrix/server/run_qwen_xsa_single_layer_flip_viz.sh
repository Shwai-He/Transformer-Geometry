#!/usr/bin/env bash
set -euo pipefail
pip install tiktoken

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/mnt/hdfs/shwai.he/DepthBoost/representation-analysis/models/Qwen/Qwen3-4B}"
NANOGPT_CKPT="${NANOGPT_CKPT:-}"
NANOGPT_REPO_ROOT="${NANOGPT_REPO_ROOT:-}"
JSONL_PATH="${JSONL_PATH:-}"
TEXT_KEY="${TEXT_KEY:-text}"
SAMPLE_IDX="${SAMPLE_IDX:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-6}"
MAX_LENGTH="${MAX_LENGTH:-256}"
PROMPT_SET="${PROMPT_SET:-short}"
FLIP_LAYER="${FLIP_LAYER:--1}"
HEAD="${HEAD:--1}"
NUM_EXTRA_LAYERS="${NUM_EXTRA_LAYERS:-2}"
XSA_INTERVENTION_SITE="${XSA_INTERVENTION_SITE:-xsa_middle_multihead}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
MIN_FREE_MEMORY_GB="${MIN_FREE_MEMORY_GB:-20}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-bf16}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-false}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-false}"
SEED="${SEED:-1337}"

infer_transformers_version() {
  if [[ -n "${TRANSFORMERS_VERSION:-}" ]]; then
    echo "$TRANSFORMERS_VERSION"
    return 0
  fi
  local model_name_lc="${MODEL_NAME,,}"
  if [[ "$model_name_lc" == *qwen3.5* || "$model_name_lc" == *qwen3_5* ]]; then
    echo "5.6.2"
  else
    echo "4.52.4"
  fi
}

TRANSFORMERS_VERSION="$(infer_transformers_version)"

ensure_transformers_version() {
  local want="$1"
  if "$PYTHON_BIN" - "$want" <<'PY'
import sys
try:
    import transformers
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if transformers.__version__ == sys.argv[1] else 1)
PY
  then
    return 0
  fi
  echo "[INFO] Installing transformers==$want for Qwen single-layer flip visualization"
  "$PYTHON_BIN" -m pip install --upgrade "transformers==$want"
}

ensure_transformers_version "$TRANSFORMERS_VERSION"

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

is_true() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off) return 1 ;;
    *)
      echo "[ERROR] Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

log_info() {
  echo "[INFO] $*" | tee -a "$ACTIVE_LOG_PATH"
}

log_error() {
  echo "[ERROR] $*" | tee -a "$ACTIVE_LOG_PATH" >&2
}

finalize_logs() {
  if [[ -n "${ACTIVE_LOG_PATH:-}" && -n "${LOG_PATH:-}" && "$ACTIVE_LOG_PATH" != "$LOG_PATH" ]]; then
    mkdir -p "$(dirname "$LOG_PATH")"
    cp "$ACTIVE_LOG_PATH" "$LOG_PATH" 2>/dev/null || true
  fi
  if [[ -n "${ACTIVE_LOG_PATH:-}" && -n "${PARALLEL_LOG_PATH:-}" && "$ACTIVE_LOG_PATH" != "$PARALLEL_LOG_PATH" ]]; then
    mkdir -p "$(dirname "$PARALLEL_LOG_PATH")"
    cp "$ACTIVE_LOG_PATH" "$PARALLEL_LOG_PATH" 2>/dev/null || true
  fi
}

is_remote_live_log_path() {
  local path="$1"
  case "$path" in
    /mnt/hdfs/*) return 0 ;;
    *) return 1 ;;
  esac
}

MODEL_SOURCE_TAG="$MODEL_NAME"
CKPT_TAG=""
if [[ -n "$NANOGPT_CKPT" ]]; then
  CKPT_RUN_TAG="$(basename "$(dirname "$NANOGPT_CKPT")")"
  CKPT_FILE_TAG="$(basename "$NANOGPT_CKPT")"
  CKPT_FILE_TAG="${CKPT_FILE_TAG%.pt}"
  CKPT_TAG="$(sanitize_tag "${CKPT_RUN_TAG}-${CKPT_FILE_TAG}")"
  MODEL_SOURCE_TAG="$CKPT_TAG"
fi
MODEL_TAG="$(sanitize_tag "$MODEL_SOURCE_TAG")"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/drawing-paper/attn_matrix/outputs/${MODEL_TAG}}"
DATA_TAG="$(sanitize_tag "${JSONL_PATH:-short_builtin}")"
PROMPT_TAG="$(sanitize_tag "${PROMPT_SET}")"
SITE_TAG="$(sanitize_tag "$XSA_INTERVENTION_SITE")"
DEVICE_TAG="$(sanitize_tag "${DEVICE:-auto}")"
DTYPE_TAG="$(sanitize_tag "$DTYPE")"
CHAT_TAG="chat${USE_CHAT_TEMPLATE}"
SAMPLE_TAG="${SAMPLE_IDX}"
if [[ "$SAMPLE_TAG" == "-1" ]]; then
  SAMPLE_TAG="all"
fi
CKPT_SUFFIX=""
if [[ -n "$CKPT_TAG" ]]; then
  CKPT_SUFFIX="-ckpt${CKPT_TAG}"
fi
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${BASE_OUTPUT_DIR}/qwen_xsa_single_layer_flip${CKPT_SUFFIX}-data${DATA_TAG}-prompt${PROMPT_TAG}-sample${SAMPLE_TAG}-flip${FLIP_LAYER}-${CHAT_TAG}-site${SITE_TAG}-dtype${DTYPE_TAG}-dev${DEVICE_TAG}}"
if [[ "$OUTPUT_PREFIX" != /* ]]; then
  OUTPUT_PREFIX="$ROOT_DIR/$OUTPUT_PREFIX"
fi

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] NANOGPT_CKPT=${NANOGPT_CKPT:-<none>}"
if [[ -n "$CKPT_TAG" ]]; then
  echo "[INFO] CKPT_TAG=$CKPT_TAG"
fi
echo "[INFO] SAMPLE_IDX=$SAMPLE_IDX MAX_SAMPLES=$MAX_SAMPLES MAX_LENGTH=$MAX_LENGTH PROMPT_SET=$PROMPT_SET"
echo "[INFO] FLIP_LAYER=$FLIP_LAYER HEAD=$HEAD NUM_EXTRA_LAYERS=$NUM_EXTRA_LAYERS"
echo "[INFO] XSA_INTERVENTION_SITE=$XSA_INTERVENTION_SITE"
echo "[INFO] OVERWRITE_EXISTING=$OVERWRITE_EXISTING"
echo "[INFO] OUTPUT_PREFIX=$OUTPUT_PREFIX"
echo "[INFO] RUN_IN_BACKGROUND=$RUN_IN_BACKGROUND"
echo "[INFO] SEED=$SEED"
echo "[INFO] GPU_LIST=${GPU_LIST:-<serial>}"
if [[ "$FLIP_LAYER" == "-1" ]]; then
  echo "[INFO] Mode=all-layer single-flip sweep"
else
  echo "[INFO] Mode=single flip layer $FLIP_LAYER"
fi

args=(
  /mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/drawing-paper/attn_matrix/server/qwen_xsa_single_layer_flip_viz.py
  --text_key "$TEXT_KEY"
  --sample_idx "$SAMPLE_IDX"
  --max_samples "$MAX_SAMPLES"
  --max_length "$MAX_LENGTH"
  --prompt_set "$PROMPT_SET"
  --seed "$SEED"
  --flip_layer "$FLIP_LAYER"
  --head "$HEAD"
  --num_extra_layers "$NUM_EXTRA_LAYERS"
  --xsa_intervention_site "$XSA_INTERVENTION_SITE"
  --dtype "$DTYPE"
  --system_prompt "$SYSTEM_PROMPT"
  --output_prefix "$OUTPUT_PREFIX"
)

if is_true "$OVERWRITE_EXISTING"; then
  args+=(--overwrite_existing)
else
  args+=(--no-overwrite_existing)
fi

if [[ -n "$NANOGPT_CKPT" ]]; then
  args+=(--nanogpt_ckpt "$NANOGPT_CKPT")
  if [[ -n "$NANOGPT_REPO_ROOT" ]]; then
    args+=(--nanogpt_repo_root "$NANOGPT_REPO_ROOT")
  fi
else
  args+=(--model_name "$MODEL_NAME")
fi

if is_true "$USE_CHAT_TEMPLATE"; then
  args+=(--use_chat_template)
else
  args+=(--no-use_chat_template)
fi

if [[ -n "$DEVICE" ]]; then
  args+=(--device "$DEVICE")
fi

if [[ -n "$JSONL_PATH" ]]; then
  args+=(--jsonl_path "$JSONL_PATH")
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
LOG_PATH="${LOG_PATH:-${OUTPUT_PREFIX}.log}"
PID_PATH="${PID_PATH:-${OUTPUT_PREFIX}.pid}"
if [[ "$LOG_PATH" != /* ]]; then
  LOG_PATH="$ROOT_DIR/$LOG_PATH"
fi
if [[ "$PID_PATH" != /* ]]; then
  PID_PATH="$ROOT_DIR/$PID_PATH"
fi
if [[ -z "${ACTIVE_LOG_PATH:-}" ]]; then
  if is_remote_live_log_path "$LOG_PATH"; then
    ACTIVE_LOG_PATH="/tmp/qwen_xsa_single_layer_flip_viz-$(date +%Y%m%d-%H%M%S).log"
  else
    ACTIVE_LOG_PATH="$LOG_PATH"
  fi
fi

parse_gpu_ids() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  read -r -a GPU_IDS <<< "$raw"
}

filter_available_gpu_ids() {
  local min_free_gb="$1"
  local min_free_mb=$((min_free_gb * 1024))
  local -a filtered=()
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "${GPU_IDS[@]}"
    return 0
  fi
  local smi
  smi="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$smi" ]]; then
    echo "${GPU_IDS[@]}"
    return 0
  fi
  for gpu_id in "${GPU_IDS[@]}"; do
    local free_mb
    free_mb="$(printf '%s\n' "$smi" | awk -F',' -v target="$gpu_id" '$1+0==target {gsub(/ /, "", $2); print $2; exit}')"
    if [[ -n "$free_mb" && "$free_mb" -ge "$min_free_mb" ]]; then
      filtered+=("$gpu_id")
    fi
  done
  if [[ "${#filtered[@]}" -eq 0 ]]; then
    echo "${GPU_IDS[@]}"
  else
    echo "${filtered[@]}"
  fi
}

run_child_job() {
  local sample_idx="$1"
  local gpu_id="${2:-}"
  local fifo_path="$3"
  local -a child_args=("${args[@]}")
  for ((idx=0; idx<${#child_args[@]}; ++idx)); do
    if [[ "${child_args[idx]}" == "--sample_idx" ]]; then
      child_args[$((idx + 1))]="$sample_idx"
    fi
  done
  {
    echo "[JOB_START] sample_idx=${sample_idx} gpu=${gpu_id:-none} time=$(date +%Y-%m-%dT%H:%M:%S)"
    if [[ -n "$gpu_id" ]]; then
      env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u "${child_args[@]}"
    else
      env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${child_args[@]}"
    fi
    status=$?
    echo "[JOB_END] sample_idx=${sample_idx} gpu=${gpu_id:-none} status=${status} time=$(date +%Y-%m-%dT%H:%M:%S)"
    exit "$status"
  } >"$fifo_path" 2>&1
}

parse_gpu_ids "$GPU_LIST"
read -r -a GPU_IDS <<< "$(filter_available_gpu_ids "$MIN_FREE_MEMORY_GB")"
parallel_enabled=false
if [[ "$SAMPLE_IDX" == "-1" && -z "$DEVICE" && "${#GPU_IDS[@]}" -gt 1 ]]; then
  parallel_enabled=true
fi

if [[ "$parallel_enabled" == true ]]; then
  echo "[INFO] Parallel single-layer flip visualization enabled across ${#GPU_IDS[@]} GPUs"
  echo "[INFO] Selected GPUs=${GPU_IDS[*]} (MIN_FREE_MEMORY_GB=$MIN_FREE_MEMORY_GB)"
  mkdir -p /mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/drawing-paper/attn_matrix/outputs/xsa_logs
  manifest_path="$(mktemp /tmp/qwen_xsa_single_layer_flip_manifest.XXXXXX)"
  MAX_SAMPLES="$MAX_SAMPLES" JSONL_PATH="$JSONL_PATH" TEXT_KEY="$TEXT_KEY" MANIFEST_PATH="$manifest_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

max_samples = int(os.environ["MAX_SAMPLES"])
jsonl_path = os.environ["JSONL_PATH"]
text_key = os.environ["TEXT_KEY"]
manifest_path = Path(os.environ["MANIFEST_PATH"])

texts = []
if jsonl_path:
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt") or obj.get(text_key) or obj.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
else:
    from qwen_xsa_single_layer_flip_viz import SHORT_TEXTS
    texts = list(SHORT_TEXTS)

if max_samples > 0:
    texts = texts[:max_samples]

with manifest_path.open("w", encoding="utf-8") as f:
    for idx, text in enumerate(texts):
        preview = text.replace("\t", " ").replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        f.write(f"{idx}\t{preview}\n")
PY
  mapfile -t manifest_rows < "$manifest_path"
  timestamp="$(date +%Y-%m-%dT%H-%M-%S)"
  PARALLEL_LOG_PATH="${PARALLEL_LOG_PATH:-representation-analysis/outputs/xsa_logs/qwen_xsa_single_layer_flip_viz-${timestamp}.log}"
  if [[ "$PARALLEL_LOG_PATH" != /* ]]; then
    PARALLEL_LOG_PATH="$ROOT_DIR/$PARALLEL_LOG_PATH"
  fi
  if [[ -z "${ACTIVE_LOG_PATH:-}" ]]; then
    if is_remote_live_log_path "$PARALLEL_LOG_PATH"; then
      ACTIVE_LOG_PATH="/tmp/qwen_xsa_single_layer_flip_viz-${timestamp}.log"
    else
      ACTIVE_LOG_PATH="$PARALLEL_LOG_PATH"
    fi
  fi
  failed_jobs=()
  completed_samples=()
  mkdir -p "$(dirname "$ACTIVE_LOG_PATH")"
  : > "$ACTIVE_LOG_PATH"
  fifo_path="$(mktemp -u /tmp/qwen_xsa_single_layer_flip_fifo.XXXXXX)"
  mkfifo "$fifo_path"
  tee -a "$ACTIVE_LOG_PATH" < "$fifo_path" >/dev/null &
  log_collector_pid=$!
  cleanup_parallel_logging() {
    exec 3>&-
    wait "$log_collector_pid" 2>/dev/null || true
    rm -f "$fifo_path"
    finalize_logs
  }
  trap cleanup_parallel_logging EXIT
  exec 3>"$fifo_path"
  log_info "PARALLEL_LOG_PATH=$PARALLEL_LOG_PATH"
  log_info "ACTIVE_LOG_PATH=$ACTIVE_LOG_PATH"
  batch_size_jobs="${#GPU_IDS[@]}"
  for ((batch_start=0; batch_start<${#manifest_rows[@]}; batch_start+=batch_size_jobs)); do
    pids=()
    pid_descs=()
    pid_logs=()
    batch_end=$((batch_start + batch_size_jobs))
    if (( batch_end > ${#manifest_rows[@]} )); then
      batch_end="${#manifest_rows[@]}"
    fi
    for ((i=batch_start; i<batch_end; ++i)); do
      IFS=$'\t' read -r sample_idx sample_preview <<< "${manifest_rows[i]}"
      gpu_id="${GPU_IDS[$((i - batch_start))]}"
      log_info "Launching sample_idx=${sample_idx} on gpu=${gpu_id}"
      log_info "sample_idx=${sample_idx} preview=${sample_preview}"
      log_info "sample_idx=${sample_idx} log:"
      echo "  $ACTIVE_LOG_PATH"
      run_child_job "$sample_idx" "$gpu_id" "$fifo_path" &
      pids+=("$!")
      pid_descs+=("$sample_idx")
      pid_logs+=("$ACTIVE_LOG_PATH")
    done
    for ((j=0; j<${#pids[@]}; ++j)); do
      if ! wait "${pids[j]}"; then
        failed_jobs+=("${pid_descs[j]}|${pid_logs[j]}")
        log_error "sample_idx=${pid_descs[j]} failed; last log lines:"
        tail -n 40 "${pid_logs[j]}" | tee -a "$PARALLEL_LOG_PATH" >&2 || true
      else
        completed_samples+=("${pid_descs[j]}")
        log_info "Completed sample_idx=${pid_descs[j]} log:"
        echo "  ${pid_logs[j]}"
      fi
    done
  done
  if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
    log_error "One or more single-layer flip jobs failed:"
    for item in "${failed_jobs[@]}"; do
      IFS='|' read -r sample_idx log_path <<< "$item"
      log_error "  sample_idx=$sample_idx"
      log_error "  Logs:"
      log_error "    $log_path"
    done
    exit 1
  fi
  INDEX_PATH="${OUTPUT_PREFIX}-all_prompts_parallel_index.json"
  COMPLETED_SAMPLES="$(printf '%s\n' "${completed_samples[@]}")" INDEX_PATH="$INDEX_PATH" OUTPUT_PREFIX="$OUTPUT_PREFIX" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

index_path = Path(os.environ["INDEX_PATH"])
completed = [int(x) for x in os.environ["COMPLETED_SAMPLES"].splitlines() if x.strip()]
payload = {
    "sample_idx": -1,
    "mode": "parallel_by_sample",
    "output_prefix": os.environ["OUTPUT_PREFIX"],
    "completed_samples": completed,
}
index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[INFO] Wrote parallel sample index to {index_path}")
PY
  cleanup_parallel_logging
  trap - EXIT
  finalize_logs
elif is_true "$RUN_IN_BACKGROUND"; then
  if [[ "$SAMPLE_IDX" != "-1" ]]; then
    echo "[INFO] Parallel mode disabled because SAMPLE_IDX=$SAMPLE_IDX (set SAMPLE_IDX=-1 to fan out by sample)"
  elif [[ -n "$DEVICE" ]]; then
    echo "[INFO] Parallel mode disabled because DEVICE=$DEVICE is pinned"
  elif [[ "${#GPU_IDS[@]}" -le 1 ]]; then
    echo "[INFO] Parallel mode disabled because GPU_LIST resolved to ${#GPU_IDS[@]} usable GPU(s): ${GPU_IDS[*]:-<empty>} (MIN_FREE_MEMORY_GB=$MIN_FREE_MEMORY_GB)"
  fi
  echo "[INFO] Launching in background with nohup"
  echo "[INFO] Live logs:"
  echo "  $ACTIVE_LOG_PATH"
  if [[ "$ACTIVE_LOG_PATH" != "$LOG_PATH" ]]; then
    echo "[INFO] Final logs:"
    echo "  $LOG_PATH"
  fi
  nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${args[@]}" >"$ACTIVE_LOG_PATH" 2>&1 &
  bg_pid=$!
  echo "$bg_pid" > "$PID_PATH"
  echo "[INFO] PID=$bg_pid"
  echo "[INFO] PID_PATH:"
  echo "  $PID_PATH"
  echo "[INFO] Tail logs with:"
  echo "  tail -f $ACTIVE_LOG_PATH"
else
  if [[ "$SAMPLE_IDX" != "-1" ]]; then
    echo "[INFO] Parallel mode disabled because SAMPLE_IDX=$SAMPLE_IDX (set SAMPLE_IDX=-1 to fan out by sample)"
  elif [[ -n "$DEVICE" ]]; then
    echo "[INFO] Parallel mode disabled because DEVICE=$DEVICE is pinned"
  elif [[ "${#GPU_IDS[@]}" -le 1 ]]; then
    echo "[INFO] Parallel mode disabled because GPU_LIST resolved to ${#GPU_IDS[@]} usable GPU(s): ${GPU_IDS[*]:-<empty>} (MIN_FREE_MEMORY_GB=$MIN_FREE_MEMORY_GB)"
  fi
  echo "[INFO] Running in foreground"
  env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${args[@]}" 2>&1 | tee "$ACTIVE_LOG_PATH"
  finalize_logs
fi
