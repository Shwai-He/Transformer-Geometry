#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/mnt/hdfs/shwai.he/DepthBoost/representation-analysis/models/Qwen/Qwen3-4B}"
JSONL_PATH="${JSONL_PATH:-}"
TEXT_KEY="${TEXT_KEY:-text}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-6}"
MAX_LENGTH="${MAX_LENGTH:-96}"
FLIP_LAYER="${FLIP_LAYER:-19}"
HEADS="${HEADS:-all}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-bf16}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-false}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"

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

parse_gpu_ids() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  read -r -a GPU_IDS <<< "$raw"
}

infer_num_heads() {
  "$PYTHON_BIN" - <<'PY' "$MODEL_NAME" "$TRUST_REMOTE_CODE"
import sys
from transformers import AutoConfig
model_name = sys.argv[1]
trust = sys.argv[2].lower() in {"1", "true", "yes", "y", "on"}
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
value = getattr(cfg, "num_attention_heads", None)
if value is None:
    value = getattr(cfg, "num_heads", None)
if value is None:
    raise SystemExit("Could not infer num_attention_heads from config")
print(int(value))
PY
}

build_head_list() {
  HEAD_IDS=()
  if [[ "$HEADS" == "all" ]]; then
    local n_heads
    n_heads="$(infer_num_heads)"
    for ((h=0; h<n_heads; ++h)); do
      HEAD_IDS+=("$h")
    done
  else
    local raw="${HEADS//,/ }"
    read -r -a HEAD_IDS <<< "$raw"
  fi
}

MODEL_TAG="$(sanitize_tag "$MODEL_NAME")"
DATA_TAG="$(sanitize_tag "${JSONL_PATH:-short_builtin}")"
DEVICE_TAG="$(sanitize_tag "${DEVICE:-auto}")"
DTYPE_TAG="$(sanitize_tag "$DTYPE")"
CHAT_TAG="chat${USE_CHAT_TEMPLATE}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/drawing-paper/attn_matrix/outputs/${MODEL_TAG}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${BASE_OUTPUT_DIR}/qwen_attn_all_heads-data${DATA_TAG}-sample${SAMPLE_IDX}-flip${FLIP_LAYER}-${CHAT_TAG}-dtype${DTYPE_TAG}-dev${DEVICE_TAG}}"
if [[ "$OUTPUT_PREFIX" != /* ]]; then
  OUTPUT_PREFIX="$ROOT_DIR/$OUTPUT_PREFIX"
fi
RUN_DIR="${OUTPUT_PREFIX}-all_heads"
LAYER_DIR="${RUN_DIR}/layer$(printf '%02d' "$FLIP_LAYER")"
mkdir -p "$LAYER_DIR"
ARTIFACT_STEM="$(basename "$OUTPUT_PREFIX")"

head_dir() {
  local head="$1"
  printf '%s/head%02d' "$LAYER_DIR" "$head"
}

head_output_prefix() {
  local head="$1"
  printf '%s/%s' "$(head_dir "$head")" "$ARTIFACT_STEM"
}

head_stem() {
  local head="$1"
  local head_prefix
  head_prefix="$(head_output_prefix "$head")"
  printf '%s-sample%02d-flip%02d-head%02d' "$head_prefix" "$SAMPLE_IDX" "$FLIP_LAYER" "$head"
}

should_skip_head() {
  local head="$1"
  local stem
  stem="$(head_stem "$head")"
  local bundle_path="${stem}/plot_bundle.json"
  if [[ -f "$bundle_path" ]] && ! is_true "$OVERWRITE_EXISTING"; then
    echo "$bundle_path"
    return 0
  fi
  return 1
}

run_one_head() {
  local head="$1"
  local gpu="$2"
  local log_path="$RUN_DIR/head$(printf '%02d' "$head").log"
  local head_prefix
  head_prefix="$(head_output_prefix "$head")"
  mkdir -p "$(dirname "$head_prefix")"
  local cmd=(
    "$PYTHON_BIN" -u
    /mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/drawing-paper/attn_matrix/server/qwen_attn_x_parallel_removal_viz.py
    --model_name "$MODEL_NAME"
    --text_key "$TEXT_KEY"
    --sample_idx "$SAMPLE_IDX"
    --max_samples "$MAX_SAMPLES"
    --max_length "$MAX_LENGTH"
    --flip_layer "$FLIP_LAYER"
    --head "$head"
    --dtype "$DTYPE"
    --system_prompt "$SYSTEM_PROMPT"
    --output_prefix "$head_prefix"
  )
  if is_true "$USE_CHAT_TEMPLATE"; then
    cmd+=(--use_chat_template)
  else
    cmd+=(--no-use_chat_template)
  fi
  if is_true "$TRUST_REMOTE_CODE"; then
    cmd+=(--trust_remote_code)
  else
    cmd+=(--no-trust_remote_code)
  fi
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ -n "$JSONL_PATH" ]]; then
    cmd+=(--jsonl_path "$JSONL_PATH")
  fi

  echo "[INFO] head=$head gpu=${gpu:-none} log=$log_path"
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_path" 2>&1
  else
    "${cmd[@]}" > "$log_path" 2>&1
  fi
}

append_manifest_entry() {
  local head="$1"
  local status="$2"
  local plot_bundle_path="${3:-}"
  local image_path="${4:-}"
  local log_path="${5:-}"
  "$PYTHON_BIN" - <<'PY' "$MANIFEST_JSONL" "$head" "$status" "$plot_bundle_path" "$image_path" "$log_path"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
entry = {
    "head": int(sys.argv[2]),
    "status": sys.argv[3],
}
if sys.argv[4]:
    entry["plot_bundle_path"] = sys.argv[4]
if sys.argv[5]:
    entry["image_path"] = sys.argv[5]
if sys.argv[6]:
    entry["log_path"] = sys.argv[6]
with out.open('a', encoding='utf-8') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

parse_gpu_ids "$GPU_LIST"
build_head_list
MANIFEST_JSONL="$(mktemp /tmp/qwen_attn_all_heads_entries.XXXXXX)"

TIMESTAMP="$(date +%Y-%m-%dT%H:%M:%S)"
echo "[INFO] time=$TIMESTAMP"
echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] SAMPLE_IDX=$SAMPLE_IDX FLIP_LAYER=$FLIP_LAYER"
echo "[INFO] HEADS=$HEADS"
echo "[INFO] HEAD_COUNT=${#HEAD_IDS[@]}"
echo "[INFO] GPU_LIST=${GPU_LIST:-<serial>}"
echo "[INFO] OUTPUT_PREFIX=$OUTPUT_PREFIX"
echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] LAYER_DIR=$LAYER_DIR"

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  GPU_IDS=("")
fi
batch_width=${#GPU_IDS[@]}
if [[ $batch_width -le 0 ]]; then
  batch_width=1
fi

launch_batch() {
  local batch_size="$1"
  local start_idx="$2"
  local -a pids=()
  local -a heads=()
  local -a gpus=()

  for ((i=0; i<batch_size; ++i)); do
    local head_idx=$((start_idx + i))
    local head="${HEAD_IDS[$head_idx]}"
    local gpu="${GPU_IDS[$i]:-}"

    if skip_path="$(should_skip_head "$head")"; then
      echo "[INFO] Skipping head=$head existing bundle found at $skip_path"
      local stem
      stem="$(head_stem "$head")"
      append_manifest_entry "$head" "skipped_existing" "$skip_path" "${stem}/layer_flip.png" ""
      continue
    fi

    run_one_head "$head" "$gpu" &
    pids+=("$!")
    heads+=("$head")
    gpus+=("$gpu")
  done

  for ((j=0; j<${#pids[@]}; ++j)); do
    local pid="${pids[$j]}"
    local head="${heads[$j]}"
    local gpu="${gpus[$j]}"
    local stem
    stem="$(head_stem "$head")"
    local bundle_path="${stem}/plot_bundle.json"
    local image_path="${stem}/layer_flip.png"
    local log_path="$RUN_DIR/head$(printf '%02d' "$head").log"

    if wait "$pid"; then
      echo "[INFO] Completed head=$head gpu=${gpu:-none}"
      append_manifest_entry "$head" "ok" "$bundle_path" "$image_path" "$log_path"
    else
      echo "[ERROR] Failed head=$head gpu=${gpu:-none}" >&2
      append_manifest_entry "$head" "failed" "" "" "$log_path"
    fi
  done
}

for ((offset=0; offset<${#HEAD_IDS[@]}; offset+=batch_width)); do
  remaining=$(( ${#HEAD_IDS[@]} - offset ))
  batch_size=$batch_width
  if [[ $remaining -lt $batch_size ]]; then
    batch_size=$remaining
  fi
  launch_batch "$batch_size" "$offset"
done

INDEX_PATH="$RUN_DIR/all_heads_index.json"
"$PYTHON_BIN" - <<'PY' "$MANIFEST_JSONL" "$INDEX_PATH" "$MODEL_NAME" "$SAMPLE_IDX" "$FLIP_LAYER"
import json, sys
from pathlib import Path
jsonl_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
payload = {
    "model_name": sys.argv[3],
    "sample_idx": int(sys.argv[4]),
    "flip_layer": int(sys.argv[5]),
    "runs": [],
}
if jsonl_path.exists():
    for line in jsonl_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            payload["runs"].append(json.loads(line))
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
PY
rm -f "$MANIFEST_JSONL"
echo "[INFO] Saved all-head index to $INDEX_PATH"
