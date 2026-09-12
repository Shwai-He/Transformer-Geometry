#!/usr/bin/env bash
set -euo pipefail

# Evaluate saved geometry-pruned checkpoints with the local lm-eval launcher.
# Defaults to the offline local_mcq smoke task; override TASKS for broader runs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SINGLE_SETTING_SCRIPT="$REPO_ROOT/lm-evaluation-harness/scripts/run_lm_eval_xsa_setting.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"
PRUNE_ROOT="${PRUNE_ROOT:-$REPO_ROOT/compression/outputs/geo_prune/by_model/$MODEL_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/geo_prune_lm_eval/by_model/$MODEL_TAG}"
TASKS="${TASKS:-local_mcq}"
INCLUDE_PATH="${INCLUDE_PATH:-$REPO_ROOT/compression/lm_eval_tasks/local_mcq}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
LIMIT="${LIMIT:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"
STRATEGIES_CSV="${STRATEGIES_CSV:-none,residual_perp,value_perp}"
SPARSITY_TYPES_CSV="${SPARSITY_TYPES_CSV:-unstructured,4:8,2:4}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$OUTPUT_ROOT/hf_datasets_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUTPUT_ROOT/triton_cache}"
mkdir -p "$OUTPUT_ROOT" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

IFS=',' read -r -a STRATEGIES <<< "$STRATEGIES_CSV"
IFS=',' read -r -a SPARSITY_TYPES <<< "$SPARSITY_TYPES_CSV"

manifest="$OUTPUT_ROOT/run_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'variant\tstrategy\tsparsity_type\tsparsity_ratio\ttask\tstatus\toutput_path\n' > "$manifest"
fi

for strategy in "${STRATEGIES[@]}"; do
  strategy="${strategy// /}"
  for sparsity_type in "${SPARSITY_TYPES[@]}"; do
    sparsity_type="${sparsity_type// /}"
    variant="wanda_${strategy}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
    model_path="$PRUNE_ROOT/$variant/model"
    if [[ ! -f "$model_path/config.json" ]]; then
      echo "[SKIP] missing pruned model: $model_path"
      printf '%s\t%s\t%s\t%s\t%s\tmissing_model\t\n' "$variant" "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$TASKS" >> "$manifest"
      continue
    fi

    out_path="$OUTPUT_ROOT/${variant}_${TASKS}.json"
    if compgen -G "${out_path%.json}"'*.json' >/dev/null || [[ -f "$out_path" ]]; then
      echo "[SKIP] existing output: $out_path"
      printf '%s\t%s\t%s\t%s\t%s\texists\t%s\n' "$variant" "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$TASKS" "$out_path" >> "$manifest"
      continue
    fi

    echo "[RUN] variant=$variant task=$TASKS model=$model_path"
    if PYTHON_BIN="$PYTHON_BIN" \
      MODEL_NAME="$model_path" \
      TASKS="$TASKS" \
      SETTING=none \
      BATCH_SIZE="$BATCH_SIZE" \
      DEVICE="$DEVICE" \
      DTYPE="$DTYPE" \
      LAUNCH_MODE="$LAUNCH_MODE" \
      NUM_FEWSHOT="$NUM_FEWSHOT" \
      LIMIT="$LIMIT" \
      INCLUDE_PATH="$INCLUDE_PATH" \
      OUTPUT_PATH="$out_path" \
      bash "$SINGLE_SETTING_SCRIPT"; then
      printf '%s\t%s\t%s\t%s\t%s\tok\t%s\n' "$variant" "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$TASKS" "$out_path" >> "$manifest"
    else
      printf '%s\t%s\t%s\t%s\t%s\tfailed\t%s\n' "$variant" "$strategy" "$sparsity_type" "$SPARSITY_RATIO" "$TASKS" "$out_path" >> "$manifest"
      if [[ "${CONTINUE_ON_ERROR:-true}" != "true" ]]; then
        exit 1
      fi
    fi
  done
done

"$PYTHON_BIN" "$REPO_ROOT/compression/scripts/summarize_geo_prune_lm_eval.py" \
  --manifest "$manifest" \
  --output "$OUTPUT_ROOT/summary_${TASKS}.csv"

echo "[DONE] manifest: $manifest"
echo "[DONE] summary: $OUTPUT_ROOT/summary_${TASKS}.csv"
