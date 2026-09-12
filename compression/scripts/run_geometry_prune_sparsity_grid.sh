#!/usr/bin/env bash
set -euo pipefail

# Run the standard sparsity grid for geometry-aware pruning.
# Defaults are offline/cache-only and use the cached Qwen3-0.6B-Base smoke model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"
PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
STRATEGIES_CSV="${STRATEGIES_CSV:-none,residual_perp,value_perp}"
SPARSITY_TYPES_CSV="${SPARSITY_TYPES_CSV:-unstructured}"
GEOMETRY_MODE="${GEOMETRY_MODE:-contribution}"
PRUNE_TARGETS="${PRUNE_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
GEOMETRY_TARGETS="${GEOMETRY_TARGETS:-o_proj,down_proj,v_proj}"
VARIANT_TAG_EXTRA="${VARIANT_TAG_EXTRA:-}"
THRESHOLD_SCOPE="${THRESHOLD_SCOPE:-global}"
TOKEN_SCOPE="${TOKEN_SCOPE:-last}"
MAX_PROMPTS="${MAX_PROMPTS:-2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-auto}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
PROMPT="${PROMPT:-}"
SAVE_MODEL="${SAVE_MODEL:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/geo_prune/by_model/$MODEL_TAG}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUTPUT_ROOT/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[ERROR] MODEL_PATH has no config.json: $MODEL_PATH" >&2
  exit 1
fi

IFS=',' read -r -a STRATEGIES <<< "$STRATEGIES_CSV"
IFS=',' read -r -a SPARSITY_TYPES <<< "$SPARSITY_TYPES_CSV"

manifest="$OUTPUT_ROOT/sparsity_grid_manifest.tsv"
mkdir -p "$OUTPUT_ROOT"
if [[ ! -f "$manifest" ]]; then
  printf 'strategy\tsparsity_type\tsparsity_ratio\tstatus\toutput_dir\n' > "$manifest"
fi

for strategy in "${STRATEGIES[@]}"; do
  strategy="${strategy// /}"
  for sparsity_type in "${SPARSITY_TYPES[@]}"; do
    sparsity_type="${sparsity_type// /}"
    tag="${PRUNE_METHOD}_${strategy}${VARIANT_TAG_EXTRA}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
    out_dir="$OUTPUT_ROOT/$tag"
    if [[ -f "$out_dir/summary.json" && -d "$out_dir/model" ]]; then
      echo "[SKIP] existing saved model: $out_dir"
      printf '%s\t%s\t%s\texists\t%s\n' "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$out_dir" >> "$manifest"
      continue
    fi

    args=(
      "$REPO_ROOT/compression/code/geometry_aware_pruning.py"
      --model_name_or_path "$MODEL_PATH"
      --output_dir "$out_dir"
      --prune_method "$PRUNE_METHOD"
      --geometry_strategy "$strategy"
      --prune_targets "$PRUNE_TARGETS"
      --geometry_mode "$GEOMETRY_MODE"
      --geometry_targets "$GEOMETRY_TARGETS"
      --sparsity_ratio "$SPARSITY_RATIO"
      --sparsity_type "$sparsity_type"
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

    echo "[RUN] strategy=$strategy sparsity_type=$sparsity_type sparsity=$SPARSITY_RATIO prune_targets=$PRUNE_TARGETS geometry_targets=$GEOMETRY_TARGETS"
    echo "[RUN] output=$out_dir"
    if "$PYTHON_BIN" "${args[@]}"; then
      printf '%s\t%s\t%s\tok\t%s\n' "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$out_dir" >> "$manifest"
    else
      printf '%s\t%s\t%s\tfailed\t%s\n' "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$out_dir" >> "$manifest"
      if [[ "${CONTINUE_ON_ERROR:-false}" != "true" ]]; then
        exit 1
      fi
    fi
  done
done

echo "[DONE] manifest: $manifest"
