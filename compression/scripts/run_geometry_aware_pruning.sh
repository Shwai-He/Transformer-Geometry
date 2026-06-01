#!/usr/bin/env bash
set -euo pipefail

# Build geometry-aware WANDA/magnitude scores, and optionally save a pruned
# checkpoint. This launcher is offline by default and does not download models.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
MODEL_PATH="${MODEL_PATH:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
GEOMETRY_STRATEGY="${GEOMETRY_STRATEGY:-residual_perp}"
PRUNE_TARGETS="${PRUNE_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
GEOMETRY_MODE="${GEOMETRY_MODE:-contribution}"
GEOMETRY_TARGETS="${GEOMETRY_TARGETS:-o_proj,down_proj,v_proj}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.0}"
SPARSITY_TYPE="${SPARSITY_TYPE:-unstructured}"
THRESHOLD_SCOPE="${THRESHOLD_SCOPE:-global}"
TOKEN_SCOPE="${TOKEN_SCOPE:-last}"
MAX_PROMPTS="${MAX_PROMPTS:-2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-auto}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
PROMPT="${PROMPT:-}"
SAVE_MODEL="${SAVE_MODEL:-false}"

export HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

model_tag="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/compression/outputs/geo_prune/by_model/${model_tag}/${PRUNE_METHOD}_${GEOMETRY_STRATEGY}}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[ERROR] MODEL_PATH has no config.json: $MODEL_PATH" >&2
  exit 1
fi

args=(
  "$REPO_ROOT/compression/code/geometry_aware_pruning.py"
  --model_name_or_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --prune_method "$PRUNE_METHOD"
  --geometry_strategy "$GEOMETRY_STRATEGY"
  --prune_targets "$PRUNE_TARGETS"
  --geometry_mode "$GEOMETRY_MODE"
  --geometry_targets "$GEOMETRY_TARGETS"
  --sparsity_ratio "$SPARSITY_RATIO"
  --sparsity_type "$SPARSITY_TYPE"
  --threshold_scope "$THRESHOLD_SCOPE"
  --token_scope "$TOKEN_SCOPE"
  --max_prompts "$MAX_PROMPTS"
  --max_length "$MAX_LENGTH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --local_files_only
)

if [[ -n "$PROMPTS_FILE" ]]; then
  args+=(--prompts_file "$PROMPTS_FILE")
fi
if [[ -n "$PROMPT" ]]; then
  args+=(--prompt "$PROMPT")
fi
if [[ "$SAVE_MODEL" == "true" ]]; then
  args+=(--save_model)
fi

echo "[RUN] geometry-aware pruning support"
echo "[RUN] model=$MODEL_PATH"
echo "[RUN] method=$PRUNE_METHOD geometry=$GEOMETRY_STRATEGY mode=$GEOMETRY_MODE prune_targets=$PRUNE_TARGETS geometry_targets=$GEOMETRY_TARGETS sparsity=$SPARSITY_RATIO threshold_scope=$THRESHOLD_SCOPE"
echo "[RUN] output=$OUTPUT_DIR"
"$PYTHON_BIN" "${args[@]}"
