#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Purpose
# - Run lm-eval for multiple XSA forward variants.
# - Default comparison: remove / keep / add / negate on xsa_middle_multihead.
# - Uses run_lm_eval_xsa_setting.sh as the single-run backend.
# - Designed to be usable on a fresh machine with minimal manual setup.
##############################################################################

##############################################################################
# Base environment
##############################################################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export RULER_CACHE_DIR="${RULER_CACHE_DIR:-/mnt/hdfs/shwai.he/DepthBoost/cache/ruler}"

REVERSE_TASKS="${REVERSE_TASKS:-false}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/hdfs/shwai.he/models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-auto}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bfloat16}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-causal}"
DEFAULT_TRUST_REMOTE_CODE="${DEFAULT_TRUST_REMOTE_CODE:-true}"
DEFAULT_APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE:-false}"
DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-4096}"
LAUNCH_MODE="${LAUNCH_MODE:-data_parallel}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
CONTINUE_ON_TASK_ERROR="${CONTINUE_ON_TASK_ERROR:-true}"
BACKGROUND="${BACKGROUND:-false}"
LIMIT="${LIMIT:-}"
METADATA="${METADATA:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="${NLTK_DATA:-/mnt/hdfs/shwai.he/DepthBoost/nltk_data}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$HARNESS_DIR/offload}"
SINGLE_SETTING_SCRIPT="$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-xsa-variants-batch.log"
fi

if [[ "$BACKGROUND" == "true" && "${_XSA_VARIANTS_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export LOG_FILE
  _XSA_VARIANTS_BACKGROUND_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  echo "[INFO] Running in background. PID=$!  Logs: $LOG_FILE"
  exit 0
fi

if [[ "${LM_EVAL_LOGGING_INITIALIZED:-0}" != "1" ]]; then
  export LM_EVAL_LOGGING_INITIALIZED=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [[ ! -f "$SINGLE_SETTING_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $SINGLE_SETTING_SCRIPT" >&2
  exit 1
fi

##############################################################################
# Environment bootstrap
##############################################################################
cd "$HARNESS_DIR"
mkdir -p "$OFFLOAD_FOLDER"
mkdir -p "$RULER_CACHE_DIR"

ensure_import() {
  local module_name="${1:-}"
  local install_spec="${2:-}"
  if [[ -z "$module_name" || -z "$install_spec" ]]; then
    echo "[ERROR] ensure_import requires <module_name> <install_spec>" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    pip install "$install_spec"
  fi
}

ensure_import_or_editable() {
  local module_name="${1:-}"
  if [[ -z "$module_name" ]]; then
    echo "[ERROR] ensure_import_or_editable requires <module_name>" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    pip install -e ".[hf]"
  fi
}

ensure_exact_version() {
  local module_name="${1:-}"
  local expected_version="${2:-}"
  local install_spec="${3:-}"
  if [[ -z "$module_name" || -z "$expected_version" || -z "$install_spec" ]]; then
    echo "[ERROR] ensure_exact_version requires <module_name> <version> <install_spec>" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib, sys
mod = importlib.import_module("${module_name}")
sys.exit(0 if getattr(mod, "__version__", None) == "${expected_version}" else 1)
PY
  then
    pip install "$install_spec"
  fi
}

ensure_min_version() {
  local module_name="${1:-}"
  local min_version="${2:-}"
  local install_spec="${3:-}"
  if [[ -z "$module_name" || -z "$min_version" || -z "$install_spec" ]]; then
    echo "[ERROR] ensure_min_version requires <module_name> <min_version> <install_spec>" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib, sys
from packaging.version import Version
mod = importlib.import_module("${module_name}")
ver = getattr(mod, "__version__", None)
sys.exit(0 if ver is not None and Version(ver) >= Version("${min_version}") else 1)
PY
  then
    pip install "$install_spec"
  fi
}

ensure_import packaging packaging
ensure_import_or_editable lm_eval
ensure_import accelerate accelerate
ensure_import datasets datasets
ensure_exact_version transformers 4.52.4 "transformers==4.52.4"
ensure_import evaluate evaluate
ensure_import bert_score bert_score
ensure_min_version rouge_score 0.1.2 "rouge_score>=0.1.2"
ensure_min_version nltk 3.9.1 "nltk>=3.9.1"
ensure_import absl absl-py
ensure_import bleurt "git+https://github.com/google-research/bleurt.git"
ensure_exact_version triton 3.0.0 "triton==3.0.0"

ensure_nltk_resource() {
  local resource_path="${1:-}"
  local resource_name="${2:-}"
  if [[ -z "$resource_path" || -z "$resource_name" ]]; then
    echo "[ERROR] ensure_nltk_resource requires <resource_path> <resource_name>" >&2
    exit 1
  fi
  mkdir -p "$NLTK_DATA"
  if ! NLTK_DATA="$NLTK_DATA" "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import nltk
nltk.data.find("${resource_path}")
PY
  then
    echo "[INFO] Downloading NLTK resource: ${resource_name} -> ${NLTK_DATA}"
    NLTK_DATA="$NLTK_DATA" "$PYTHON_BIN" - <<PY
import nltk
nltk.download("${resource_name}", download_dir="${NLTK_DATA}")
PY
  fi
}

ensure_nltk_resource "tokenizers/punkt" "punkt"
ensure_nltk_resource "tokenizers/punkt_tab/english" "punkt_tab"

##############################################################################
# Models
##############################################################################
if [[ -n "${MODEL_POSTFIX_LIST:-}" ]]; then
  read -r -a MODEL_POSTFIX <<< "$MODEL_POSTFIX_LIST"
else
  MODEL_POSTFIX=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-30B-A3B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-14B"
  )
fi

##############################################################################
# Settings and variants
##############################################################################
if [[ -n "${SETTINGS_LIST:-}" ]]; then
  read -r -a SETTINGS <<< "$SETTINGS_LIST"
else
  SETTINGS=(
    "xsa_middle_multihead"
  )
fi

if [[ -n "${VARIANTS_LIST:-}" ]]; then
  read -r -a VARIANTS <<< "$VARIANTS_LIST"
else
  VARIANTS=(
    "remove_parallel:1.0"
    "keep_parallel:1.0"
    "add_parallel:1.0"
    "negate_parallel:1.0"
  )
fi

##############################################################################
# Tasks
##############################################################################
declare -A TASKS_MAP=(
  [openbookqa]=0
  [piqa]=0
  [rte]=0
  [winogrande]=5
  [boolq]=0
  [arc_challenge]=25
  [hellaswag]=10
  [mmlu]=5
  [gsm8k_cot]=8
  [humaneval]=5
  [nq_open]=5
  [drop]=0
  [mbpp]=3
  [bbh_cot_zeroshot]=0
)

TASKS_ORDER=(
  openbookqa
  piqa
  rte
  winogrande
  boolq
  arc_challenge
  hellaswag
  mmlu
  gsm8k_cot
  humaneval
  nq_open
  drop
  mbpp
  bbh_cot_zeroshot
)

FINAL_TASKS=("${TASKS_ORDER[@]}")
if [[ "$REVERSE_TASKS" == "true" ]]; then
  echo "[INFO] Reverse option enabled: tasks will run in reverse order."
  FINAL_TASKS=()
  for (( i=${#TASKS_ORDER[@]}-1; i>=0; i-- )); do
    FINAL_TASKS+=("${TASKS_ORDER[i]}")
  done
fi

##############################################################################
# Helpers
##############################################################################
is_true() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off) return 1 ;;
    *) return 1 ;;
  esac
}

sanitize_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="${s//:/-}"
  echo "$s"
}

collect_existing_outputs() {
  local output_path="$1"
  local prefix_no_ext="${output_path%.json}"
  local -n _results_ref="$2"

  _results_ref=()
  if [[ -f "$output_path" ]]; then
    _results_ref+=("$output_path")
  fi

  shopt -s nullglob
  local timestamped_outputs=( "${prefix_no_ext}_"*.json )
  shopt -u nullglob
  if (( ${#timestamped_outputs[@]} > 0 )); then
    _results_ref+=("${timestamped_outputs[@]}")
  fi
}

use_chat_template_for_model() {
  local model_name="$1"
  local task_name="${2:-}"
  if [[ -n "${DEFAULT_APPLY_CHAT_TEMPLATE_FORCE:-}" ]]; then
    if is_true "$DEFAULT_APPLY_CHAT_TEMPLATE_FORCE"; then
      echo "true"
    else
      echo "false"
    fi
    return 0
  fi

  case "$task_name" in
    humaneval|mbpp)
      echo "false"
      return 0
      ;;
  esac

  case "$model_name" in
    *Instruct*|*Chat*|*chat*|*instruction*)
      echo "true"
      ;;
    *)
      echo "$DEFAULT_APPLY_CHAT_TEMPLATE"
      ;;
  esac
}

##############################################################################
# Main loop
##############################################################################
echo "[INFO] Running XSA variants batch eval"
echo "[INFO] Settings: ${SETTINGS[*]}"
echo "[INFO] Variants: ${VARIANTS[*]}"

FAILED_TASKS=()

for setting in "${SETTINGS[@]}"; do
  echo "=========================================="
  echo "[SETTING] ${setting}"
  echo "=========================================="

  for variant in "${VARIANTS[@]}"; do
    op="${variant%%:*}"
    alpha="${variant#*:}"
    if [[ -z "$op" || -z "$alpha" || "$op" == "$variant" ]]; then
      echo "[ERROR] Invalid variant spec: $variant (expected op:alpha)" >&2
      exit 1
    fi
    variant_label="${setting}-${op}-a$(echo "$alpha" | tr '.' 'p')"

    echo "------------------------------------------"
    echo "[VARIANT] op=${op} alpha=${alpha} label=${variant_label}"
    echo "------------------------------------------"

    for model_postfix in "${MODEL_POSTFIX[@]}"; do
      pretrained="${BASE_MODEL_DIR}/${model_postfix}"

      if [[ ! -d "$pretrained" ]]; then
        echo "[SKIP] Missing model path: $pretrained"
        continue
      fi

      model_output_root="${HARNESS_DIR}/outputs/xsa_lm_eval/$(sanitize_name "$model_postfix")"
      mkdir -p "$model_output_root"

      echo "[START] model=${model_postfix} setting=${setting} op=${op} alpha=${alpha}"

      for task in "${FINAL_TASKS[@]}"; do
        apply_chat_template="$(use_chat_template_for_model "$model_postfix" "$task")"
        fewshot="${TASKS_MAP[$task]}"
        current_bs="$DEFAULT_BATCH_SIZE"
        if [[ "$task" == longbench* ]]; then
          current_bs=1
        fi

        output_path="${model_output_root}/${task}-${variant_label}.json"
        existing_outputs=()
        collect_existing_outputs "$output_path" existing_outputs
        if (( ${#existing_outputs[@]} > 0 )); then
          echo "[SKIP] Existing output(s) found for ${task}/${variant_label}"
          printf '  - %s\n' "${existing_outputs[@]}"
          continue
        fi

        echo "[TASK] task=$task fewshot=$fewshot batch_size=$current_bs apply_chat_template=$apply_chat_template"

        if ! MODEL_NAME="$pretrained" \
          BACKEND="$DEFAULT_BACKEND" \
          TASKS="$task" \
          SETTING="$setting" \
          BATCH_SIZE="$current_bs" \
          DTYPE="$DEFAULT_DTYPE" \
          TRUST_REMOTE_CODE="$DEFAULT_TRUST_REMOTE_CODE" \
          APPLY_CHAT_TEMPLATE="$apply_chat_template" \
          NUM_FEWSHOT="$fewshot" \
          MAX_LENGTH="$DEFAULT_MAX_LENGTH" \
          LAUNCH_MODE="$LAUNCH_MODE" \
          MAX_MEMORY_PER_GPU="$MAX_MEMORY_PER_GPU" \
          MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
          OFFLOAD_FOLDER="$OFFLOAD_FOLDER" \
          OUTPUT_PATH="$output_path" \
          LIMIT="$LIMIT" \
          XSA_START_LAYER="$XSA_START_LAYER" \
          XSA_END_LAYER="$XSA_END_LAYER" \
          XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N" \
          XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N" \
          XSA_FORWARD_OP="$op" \
          XSA_FORWARD_ALPHA="$alpha" \
          XSA_TRACK_STATS="$XSA_TRACK_STATS" \
          XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS" \
          CONFIRM_RUN_UNSAFE_CODE="$CONFIRM_RUN_UNSAFE_CODE" \
          METADATA="$METADATA" \
          bash "$SINGLE_SETTING_SCRIPT"; then
          echo "[FAIL] model=${model_postfix} task=${task} setting=${setting} op=${op} alpha=${alpha}" >&2
          FAILED_TASKS+=("${model_postfix}|${task}|${setting}|${op}|${alpha}")
          if [[ "$CONTINUE_ON_TASK_ERROR" != "true" ]]; then
            exit 1
          fi
          continue
        fi
      done

      echo "[DONE] model=${model_postfix} setting=${setting} op=${op} alpha=${alpha}"
      echo ""
    done
  done
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "[WARN] Some task runs failed:" >&2
  printf '  - %s\n' "${FAILED_TASKS[@]}" >&2
fi
