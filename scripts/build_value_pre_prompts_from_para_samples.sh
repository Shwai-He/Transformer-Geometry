#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -z "${MODEL_LIST_FILE}" ]]; then
    if [[ -f models.txt ]]; then
      MODEL_LIST_FILE="models.txt"
    elif [[ -f model.text ]]; then
      MODEL_LIST_FILE="model.text"
    fi
  fi
  if [[ -n "${MODEL_LIST_FILE}" ]]; then
    MODEL_PATH="$(grep -v '^[[:space:]]*#' "${MODEL_LIST_FILE}" | sed '/^[[:space:]]*$/d' | head -n 1)"
  fi
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "No model found. Set MODEL_PATH or create models.txt/model.text." >&2
  exit 1
fi

PARA_PROMPTS_FILE="${PARA_PROMPTS_FILE:-prompts.txt}"
if [[ ! -f "${PARA_PROMPTS_FILE}" && -f "results/prompts.txt" ]]; then
  echo "[WARN] ${PARA_PROMPTS_FILE} not found; falling back to results/prompts.txt" >&2
  PARA_PROMPTS_FILE="results/prompts.txt"
fi
if [[ ! -f "${PARA_PROMPTS_FILE}" ]]; then
  echo "Prompt file not found: ${PARA_PROMPTS_FILE}" >&2
  echo "Set PARA_PROMPTS_FILE to the para ablation prompts.txt path." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/value_pre_prompts}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
NUM_QUANTILE_BUCKETS="${NUM_QUANTILE_BUCKETS:-4}"

python3 scripts/build_value_pre_prompt_buckets.py \
  --prompts_file "${PARA_PROMPTS_FILE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_prompts "${MAX_PROMPTS}" \
  --num_quantile_buckets "${NUM_QUANTILE_BUCKETS}"

echo
echo "[NEXT] Run value_pre probe on quantile buckets:"
echo "PROMPT_FILES=\"${OUTPUT_DIR}/q*.txt\" bash scripts/run_value_pre_ratio_probe.sh"
