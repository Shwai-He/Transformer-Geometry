#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
MODEL_TAG="${MODEL_TAG:-}"
OUTPUT_TAG="${OUTPUT_TAG:-}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
PROMPT="${PROMPT:-}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
TOKEN_SCOPE="${TOKEN_SCOPE:-all}"
DROP_COUNTS="${DROP_COUNTS:-4,8}"
RANK_METRIC="${RANK_METRIC:-perp_ratio}"
RANK_ORDER="${RANK_ORDER:-ascending}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/compression/outputs/layer_drop_geometry}"

args=(
  "$REPO_ROOT/compression/code/select_layer_drop_by_geometry.py"
  --model_name "$MODEL_NAME"
  --max_prompts "$MAX_PROMPTS"
  --max_length "$MAX_LENGTH"
  --token_scope "$TOKEN_SCOPE"
  --drop_counts "$DROP_COUNTS"
  --rank_metric "$RANK_METRIC"
  --rank_order "$RANK_ORDER"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MODEL_TAG" ]]; then
  args+=(--model_tag "$MODEL_TAG")
fi
if [[ -n "$OUTPUT_TAG" ]]; then
  args+=(--output_tag "$OUTPUT_TAG")
fi
if [[ -n "$PROMPTS_FILE" ]]; then
  args+=(--prompts_file "$PROMPTS_FILE")
fi
if [[ -n "$PROMPT" ]]; then
  args+=(--prompt "$PROMPT")
fi
if [[ "$LOCAL_FILES_ONLY" == "true" ]]; then
  args+=(--local_files_only)
else
  args+=(--no-local_files_only)
fi

echo "[RUN] layer-drop geometry selector"
echo "[RUN] model=$MODEL_NAME"
echo "[RUN] rank_metric=$RANK_METRIC rank_order=$RANK_ORDER drop_counts=$DROP_COUNTS"
"$PYTHON_BIN" "${args[@]}"
