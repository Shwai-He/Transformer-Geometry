#!/usr/bin/env bash
set -euo pipefail

# Collect value-space geometry for normal attention:
#   y_pre_t = sum_s A_ts v_s
# and decompose y_pre_t into directions parallel/perpendicular to self value v_t.
#
# Examples:
#   bash scripts/run_value_pre_ratio_probe.sh
#   PROMPT_FILES="results/prompts_len128.txt results/prompts_len512.txt" bash scripts/run_value_pre_ratio_probe.sh
#   MODEL_PATH=/path/to/model GPUS="0 1 2 3 4 5 6 7" bash scripts/run_value_pre_ratio_probe.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

sanitize_tag() {
  local s="$1"
  s="${s%/}"
  s="${s##*/}"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//:/_}"
  echo "${s}"
}

MODEL_PATH="${MODEL_PATH:-}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-}"
MODEL_PATHS=()
if [[ -n "${MODEL_PATH}" ]]; then
  MODEL_PATHS+=("${MODEL_PATH}")
else
  if [[ -z "${MODEL_LIST_FILE}" ]]; then
    if [[ -f models.txt ]]; then
      MODEL_LIST_FILE="models.txt"
    elif [[ -f model.text ]]; then
      MODEL_LIST_FILE="model.text"
    else
      echo "No model list found. Set MODEL_PATH, or create models.txt/model.text." >&2
      exit 1
    fi
  fi
  while IFS= read -r line; do
    MODEL_PATHS+=("${line}")
  done < <(grep -v '^[[:space:]]*#' "${MODEL_LIST_FILE}" | sed '/^[[:space:]]*$/d')
fi
if [[ "${#MODEL_PATHS[@]}" -eq 0 ]]; then
  echo "No model found. Set MODEL_PATH or add model paths to ${MODEL_LIST_FILE:-models.txt}." >&2
  exit 1
fi

PROMPT_FILES="${PROMPT_FILES:-results/prompts.txt}"
EXTRACT_SUMMARY="${EXTRACT_SUMMARY:-1}"

GEOMETRY_SPACE="${GEOMETRY_SPACE:-hidden}"
MEASURE_MODE="${MEASURE_MODE:-prefill}"
if [[ "${MEASURE_MODE}" != "prefill" ]]; then
  echo "[ERROR] value_pre ratio probe is prefill-only. Set MEASURE_MODE=prefill." >&2
  exit 1
fi
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TOP_K="${TOP_K:-5}"
DTYPE="${DTYPE:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PREFILL_TOKEN_STRIDE="${PREFILL_TOKEN_STRIDE:-16}"
DEVICE_MAP="${DEVICE_MAP:-}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "${GPUS_STR}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPUS=\"0 1 2 ...\"." >&2
  exit 1
fi
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${#GPU_LIST[@]}}"
LOG_ROOT="${LOG_ROOT:-outputs/value_pre_logs}"
mkdir -p "${LOG_ROOT}"

python3 - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("transformers")
if spec is None:
    print("[ERROR] transformers is not installed. Install transformers before running this script.", file=sys.stderr)
    sys.exit(1)

import transformers
print(f"[INFO] transformers: {transformers.__version__}")
PY

echo "[INFO] models: ${#MODEL_PATHS[@]}"
if [[ -n "${MODEL_LIST_FILE}" ]]; then
  echo "[INFO] model list: ${MODEL_LIST_FILE}"
fi
echo "[INFO] prompt files: ${PROMPT_FILES}"
echo "[INFO] measure mode: ${MEASURE_MODE}, geometry space: ${GEOMETRY_SPACE}"
echo "[INFO] attention implementation: ${ATTN_IMPLEMENTATION}"
echo "[INFO] batch size: ${BATCH_SIZE}"
echo "[INFO] prefill token stride: ${PREFILL_TOKEN_STRIDE}"
echo "[INFO] gpus: ${GPU_LIST[*]}"
echo "[INFO] max parallel jobs: ${MAX_PARALLEL_JOBS}"

job_pids=()
job_logs=()
job_count=0
model_output_roots=()

wait_one_batch() {
  if [[ "${#job_pids[@]}" -eq 0 ]]; then
    return 0
  fi
  local rc=0
  echo "[INFO] waiting for ${#job_pids[@]} jobs..."
  for idx in "${!job_pids[@]}"; do
    pid="${job_pids[$idx]}"
    log="${job_logs[$idx]}"
    if ! wait "${pid}"; then
      rc=1
      echo "[ERROR] job failed pid=${pid} log=${log}" >&2
    else
      echo "[OK] pid=${pid} log=${log}"
    fi
  done
  job_pids=()
  job_logs=()
  return "${rc}"
}

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_TAG_THIS="$(sanitize_tag "${MODEL_PATH}")"
  if [[ -n "${MODEL_TAG:-}" && "${#MODEL_PATHS[@]}" -eq 1 ]]; then
    MODEL_TAG_THIS="${MODEL_TAG}"
  fi
  OUTPUT_ROOT_THIS="${OUTPUT_ROOT:-outputs/${MODEL_TAG_THIS}/value_pre_ratio}"
  mkdir -p "${OUTPUT_ROOT_THIS}"
  model_output_roots+=("${OUTPUT_ROOT_THIS}")

  echo
  echo "[MODEL] ${MODEL_PATH}"
  echo "[INFO] output root: ${OUTPUT_ROOT_THIS}"

  for PROMPT_FILE in ${PROMPT_FILES}; do
    if [[ ! -f "${PROMPT_FILE}" ]]; then
      echo "[WARN] prompt file not found, skipped: ${PROMPT_FILE}" >&2
      continue
    fi

    PROMPT_TAG="$(sanitize_tag "${PROMPT_FILE}")"
    PROMPT_TAG="${PROMPT_TAG%.txt}"
    OUTPUT_PATH="${OUTPUT_ROOT_THIS}/value_pre-${PROMPT_TAG}-${MEASURE_MODE}-${GEOMETRY_SPACE}-tokens${MAX_NEW_TOKENS}.json"

    echo
    gpu="${GPU_LIST[$((job_count % ${#GPU_LIST[@]}))]}"
    job_count=$((job_count + 1))
    LOG_PATH="${LOG_ROOT}/${MODEL_TAG_THIS}-${PROMPT_TAG}-${MEASURE_MODE}-${GEOMETRY_SPACE}.log"

    echo "[LAUNCH] gpu=${gpu} ${PROMPT_FILE}"
    echo "[OUT] ${OUTPUT_PATH}"
    echo "[LOG] ${LOG_PATH}"

    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONUNBUFFERED=1
      cmd=(
        python3 scripts/run_generation_probe.py
        --model_name_or_path "${MODEL_PATH}"
        --prompt_file "${PROMPT_FILE}"
        --all_prompts
        --measure_mode "${MEASURE_MODE}"
        --geometry_space "${GEOMETRY_SPACE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --top_k "${TOP_K}"
        --dtype "${DTYPE}"
        --attn_implementation "${ATTN_IMPLEMENTATION}"
        --batch_size "${BATCH_SIZE}"
        --prefill_token_stride "${PREFILL_TOKEN_STRIDE}"
        --include_sublayer_metrics
        --output "${OUTPUT_PATH}"
      )
      if [[ -n "${DEVICE_MAP}" ]]; then
        cmd+=(--device_map "${DEVICE_MAP}")
      fi
      "${cmd[@]}"
    ) > "${LOG_PATH}" 2>&1 &
    job_pids+=("$!")
    job_logs+=("${LOG_PATH}")

    if [[ "${#job_pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; then
      wait_one_batch
    fi
  done
done

wait_one_batch

echo
echo "[DONE] all value_pre jobs finished."

if [[ "${EXTRACT_SUMMARY}" == "1" ]]; then
  for OUTPUT_ROOT_THIS in "${model_output_roots[@]}"; do
    echo "[INFO] extracting compact CSV summaries: ${OUTPUT_ROOT_THIS}"
    python3 scripts/extract_value_pre_ratio_summary.py \
      "${OUTPUT_ROOT_THIS}" \
      --phase all \
      --output_dir "${OUTPUT_ROOT_THIS}/summary"
  done
fi
