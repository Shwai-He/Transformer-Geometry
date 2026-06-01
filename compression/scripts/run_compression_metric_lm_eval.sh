#!/usr/bin/env bash
set -euo pipefail

# Evaluate dense/compressed checkpoints on the paper task panel.
# This launcher never downloads models by itself: each MODEL_PATH must already
# exist locally and contain config.json.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
HARNESS_DIR="$REPO_ROOT/lm-evaluation-harness"

PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
BASE_MODEL="${BASE_MODEL:-/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/compression_metric_lm_eval}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
INCLUDE_LOCAL_TASKS="${INCLUDE_LOCAL_TASKS:-false}"
LOCAL_TASK_PATH="$REPO_ROOT/compression/lm_eval_tasks/local_mcq"

# Keep caches project-local by default and allow offline runs.
export HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$OUTPUT_ROOT/hf_datasets_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_DATASETS_CACHE"

declare -A FEWSHOT=(
  [openbookqa]=0
  [piqa]=0
  [rte]=0
  [winogrande]=5
  [boolq]=0
  [arc_challenge]=25
  [hellaswag]=10
  [mmlu]=5
  [gsm8k]=5
  [gsm8k_cot]=8
  [humaneval]=5
  [nq_open]=5
  [drop]=0
  [mbpp]=3
  [bbh_cot_zeroshot]=0
  [local_mcq]=0
)

# Format: variant|model_path|compression_tag
VARIANTS=(
  "dense|$BASE_MODEL|dense"
  "wanda_2_4|${WANDA_24_MODEL:-/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/2-4/wanda}|wanda"
  "wanda_4_8|${WANDA_48_MODEL:-/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/4-8/wanda}|wanda"
  "wanda_unstructured|${WANDA_UNSTRUCTURED_MODEL:-/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/unstructured/wanda}|wanda"
  "awq_native|${AWQ_MODEL:-/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B-AWQ}|awq"
)

mkdir -p "$OUTPUT_ROOT"
summary_tsv="$OUTPUT_ROOT/run_manifest.tsv"
if [[ ! -f "$summary_tsv" ]]; then
  printf 'variant\tcompression\tmodel_path\ttask\tstatus\toutput_path\n' > "$summary_tsv"
fi

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

sanitize() {
  local s="$1"
  s="${s//[^A-Za-z0-9._-]/_}"
  printf '%s' "$s"
}

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r variant model_path compression <<< "$spec"
  if [[ ! -f "$model_path/config.json" ]]; then
    echo "[SKIP] $variant missing config.json: $model_path"
    for task in "${TASKS[@]}"; do
      printf '%s\t%s\t%s\t%s\tmissing_model\t\n' "$variant" "$compression" "$model_path" "$task" >> "$summary_tsv"
    done
    continue
  fi

  for task in "${TASKS[@]}"; do
    task="${task// /}"
    fewshot="${FEWSHOT[$task]:-0}"
    out_dir="$OUTPUT_ROOT/$variant"
    mkdir -p "$out_dir"
    out_path="$out_dir/$(sanitize "$task").json"
    if compgen -G "${out_path%.json}"'*.json' >/dev/null || [[ -f "$out_path" ]]; then
      echo "[SKIP] existing output: $variant / $task"
      printf '%s\t%s\t%s\t%s\texists\t%s\n' "$variant" "$compression" "$model_path" "$task" "$out_path" >> "$summary_tsv"
      continue
    fi

    args=(
      -m lm_eval run
      --model hf
      --model_args "pretrained=$model_path,trust_remote_code=$TRUST_REMOTE_CODE,dtype=$DTYPE"
      --tasks "$task"
      --num_fewshot "$fewshot"
      --batch_size "$BATCH_SIZE"
      --device "$DEVICE"
      --output_path "$out_path"
    )
    if [[ -n "$LIMIT" ]]; then
      args+=(--limit "$LIMIT")
    fi
    if [[ "$CONFIRM_RUN_UNSAFE_CODE" == "true" ]]; then
      args+=(--confirm_run_unsafe_code)
    fi
    if [[ "$INCLUDE_LOCAL_TASKS" == "true" ]]; then
      args+=(--include_path "$LOCAL_TASK_PATH")
    fi

    echo "[RUN] variant=$variant task=$task fewshot=$fewshot output=$out_path"
    if "$PYTHON_BIN" "${args[@]}"; then
      printf '%s\t%s\t%s\t%s\tok\t%s\n' "$variant" "$compression" "$model_path" "$task" "$out_path" >> "$summary_tsv"
    else
      printf '%s\t%s\t%s\t%s\tfailed\t%s\n' "$variant" "$compression" "$model_path" "$task" "$out_path" >> "$summary_tsv"
      if [[ "${CONTINUE_ON_TASK_ERROR:-true}" != "true" ]]; then
        exit 1
      fi
    fi
  done
done

echo "[DONE] manifest: $summary_tsv"
