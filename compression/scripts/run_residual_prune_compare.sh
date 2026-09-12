#!/usr/bin/env bash
set -euo pipefail

# Compare residual-level compression error for multiple geometry-aware pruning
# scores. This runs in memory and does not save pruned checkpoints.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
STRATEGIES="${STRATEGIES:-none,residual_perp,residual_para,residual_perp_over_para}"
GEOMETRY_TARGETS="${GEOMETRY_TARGETS:-o_proj,down_proj}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.1}"
SPARSITY_TYPE="${SPARSITY_TYPE:-unstructured}"
THRESHOLD_SCOPE="${THRESHOLD_SCOPE:-row}"
TOKEN_SCOPE="${TOKEN_SCOPE:-last}"
MAX_PROMPTS="${MAX_PROMPTS:-2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-auto}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
PROMPT="${PROMPT:-}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$REPO_ROOT/compression/code${PYTHONPATH:+:$PYTHONPATH}"

model_tag="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/compression/outputs/residual_prune_compare/by_model/${model_tag}/${PRUNE_METHOD}_s${SPARSITY_RATIO}}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[ERROR] MODEL_PATH has no config.json: $MODEL_PATH" >&2
  exit 1
fi

args=(
  "$REPO_ROOT/compression/code/residual_geometry_prune_compare.py"
  --model_name_or_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --prune_method "$PRUNE_METHOD"
  --strategies "$STRATEGIES"
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

echo "[RUN] residual prune compare"
echo "[RUN] model=$MODEL_PATH"
echo "[RUN] method=$PRUNE_METHOD strategies=$STRATEGIES sparsity=$SPARSITY_RATIO"
echo "[RUN] output=$OUTPUT_DIR"
"$PYTHON_BIN" "${args[@]}"
