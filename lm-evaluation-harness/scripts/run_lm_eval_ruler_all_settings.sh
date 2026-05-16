#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# 基础环境配置
##############################################################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_HOME="${HF_HOME:-/mnt/hdfs/shwai.he/models/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export RULER_CACHE_DIR="${RULER_CACHE_DIR:-/mnt/hdfs/shwai.he/DepthBoost/cache/ruler}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/hdfs/shwai.he/models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-1}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bfloat16}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-causal}"
DEFAULT_TRUST_REMOTE_CODE="${DEFAULT_TRUST_REMOTE_CODE:-true}"
DEFAULT_APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE:-false}"
DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-}"
DEFAULT_LIMIT="${DEFAULT_LIMIT:-}"
LAUNCH_MODE="${LAUNCH_MODE:-data_parallel}"
RULER_ALLOW_DISTRIBUTED="${RULER_ALLOW_DISTRIBUTED:-true}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
CONTINUE_ON_TASK_ERROR="${CONTINUE_ON_TASK_ERROR:-true}"
BACKGROUND="${BACKGROUND:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Intentionally ignore inherited NLTK_DATA by default to avoid stale unwritable
# paths like /home/tiger/nltk_data on shared machines. Use NLTK_DATA_OVERRIDE
# if you explicitly want a different directory.
DEFAULT_NLTK_DATA="/mnt/hdfs/shwai.he/DepthBoost/nltk_data"
INHERITED_NLTK_DATA="${NLTK_DATA:-}"
NLTK_DATA="${NLTK_DATA_OVERRIDE:-$DEFAULT_NLTK_DATA}"
export NLTK_DATA
SINGLE_SETTING_SCRIPT="$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-ruler-all-settings.log"
fi

if [[ "$BACKGROUND" == "true" && "${_BATCH_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export LOG_FILE
  _BATCH_BACKGROUND_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
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

if [[ -n "$INHERITED_NLTK_DATA" && "$INHERITED_NLTK_DATA" != "$NLTK_DATA" ]]; then
  echo "[INFO] Ignoring inherited NLTK_DATA=$INHERITED_NLTK_DATA"
fi

##############################################################################
# 环境依赖
##############################################################################
cd "$HARNESS_DIR"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$RULER_CACHE_DIR"

ensure_import() {
  local module_name="${1:-}"
  if [[ -z "$module_name" ]]; then
    echo "[ERROR] ensure_import requires a module name." >&2
    return 1
  fi
  shift
  if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    pip install "$@"
  fi
}

ensure_exact_version() {
  local module_name="${1:-}"
  local target_version="${2:-}"
  if [[ -z "$module_name" || -z "$target_version" ]]; then
    echo "[ERROR] ensure_exact_version requires module name and target version." >&2
    return 1
  fi
  shift 2
  if ! "$PYTHON_BIN" - "$module_name" "$target_version" <<'PY' >/dev/null 2>&1
import importlib, sys
module_name = sys.argv[1]
target = sys.argv[2]
module = importlib.import_module(module_name)
version = getattr(module, "__version__", None)
raise SystemExit(0 if version == target else 1)
PY
  then
    pip install "$@"
  fi
}

ensure_min_version() {
  local module_name="${1:-}"
  local min_version="${2:-}"
  if [[ -z "$module_name" || -z "$min_version" ]]; then
    echo "[ERROR] ensure_min_version requires module name and minimum version." >&2
    return 1
  fi
  shift 2
  if ! "$PYTHON_BIN" - "$module_name" "$min_version" <<'PY' >/dev/null 2>&1
import importlib, sys
from packaging.version import Version
module_name = sys.argv[1]
target = Version(sys.argv[2])
module = importlib.import_module(module_name)
version = Version(getattr(module, "__version__", "0"))
raise SystemExit(0 if version >= target else 1)
PY
  then
    pip install "$@"
  fi
}

if ! "$PYTHON_BIN" -c "import lm_eval" >/dev/null 2>&1; then
  pip install -e ".[hf,ruler]"
fi
ensure_import packaging packaging
ensure_import accelerate accelerate
ensure_import datasets datasets
ensure_exact_version transformers 4.52.4 "transformers==4.52.4"
ensure_import evaluate evaluate
ensure_import bert_score bert_score
ensure_import pytablewriter pytablewriter
ensure_min_version rouge_score 0.1.2 "rouge_score>=0.1.2"
ensure_min_version nltk 3.9.1 "nltk>=3.9.1"
ensure_import wonderwords wonderwords
ensure_import absl absl-py
if ! "$PYTHON_BIN" -c "import bleurt" >/dev/null 2>&1; then
  pip install git+https://github.com/google-research/bleurt.git
fi
ensure_exact_version triton 3.0.0 "triton==3.0"

require_nltk_resource() {
  local resource_path="${1:-}"
  local resource_name="${2:-}"
  if [[ -z "$resource_path" || -z "$resource_name" ]]; then
    echo "[ERROR] require_nltk_resource requires <resource_path> <resource_name>" >&2
    return 1
  fi
  if ! NLTK_DATA="$NLTK_DATA" "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import nltk
nltk.data.find("${resource_path}")
PY
  then
    echo "[ERROR] Missing NLTK resource '${resource_name}' under ${NLTK_DATA}" >&2
    echo "[ERROR] Please run scripts/run_download_nltk_resources.sh first." >&2
    return 1
  fi
}

check_nltk_resources() {
  echo "[INFO] Checking NLTK resources under ${NLTK_DATA}"
  require_nltk_resource "tokenizers/punkt" "punkt"
  require_nltk_resource "tokenizers/punkt_tab/english" "punkt_tab"
  echo "[INFO] NLTK resources ready under ${NLTK_DATA}"
}

check_nltk_resources

##############################################################################
# 模型配置
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
# Settings / RULER 配置
##############################################################################
if [[ -n "${SETTINGS_LIST:-}" ]]; then
  read -r -a SETTINGS <<< "$SETTINGS_LIST"
else
  SETTINGS=(
    "none"
    "xsa_middle"
    "xsa_middle_multihead"
    "residual_attn"
    "residual_mlp"
    "residual_both"
  )
fi

# 默认按表格风格跑整个 RULER group。
# 例子：
#   TASKS=ruler bash run_lm_eval_ruler_all_settings.sh
#   TASKS=ruler_fwe bash run_lm_eval_ruler_all_settings.sh
TASKS="${TASKS:-ruler}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
RULER_LENGTHS="${RULER_LENGTHS:-4096 8192 16384 32768 65536 131072}"
RULER_METADATA="${RULER_METADATA:-}"

##############################################################################
# 辅助函数
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

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

build_lengths_metadata() {
  local lengths="$1"
  "$PYTHON_BIN" - <<'PY' "$lengths"
import json, sys
raw = sys.argv[1].replace(",", " ").split()
vals = [int(x) for x in raw if x.strip()]
print(json.dumps({"max_seq_lengths": vals}, separators=(",", ":")))
PY
}

build_lengths_tag() {
  local lengths="$1"
  "$PYTHON_BIN" - <<'PY' "$lengths"
import sys
raw = sys.argv[1].replace(",", " ").split()
vals = [int(x) for x in raw if x.strip()]
parts = []
for v in vals:
    if v % 1024 == 0:
        parts.append(f"{v // 1024}k")
    else:
        parts.append(str(v))
print("_".join(parts))
PY
}

prewarm_ruler_datasets() {
  local lock_root="${RULER_PREWARM_LOCK_ROOT:-/tmp/ruler_prewarm_locks}"
  local lock_dir="${lock_root}/paul_graham_essays.lock"
  mkdir -p "$lock_root"
  while ! mkdir "$lock_dir" 2>/dev/null; do
    echo "[INFO] Waiting for RULER dataset prewarm lock: $lock_dir"
    sleep 2
  done
  trap 'rmdir "$lock_dir" 2>/dev/null || true' RETURN

  echo "[INFO] Prewarming RULER dataset: baber/paul_graham_essays"
  "$PYTHON_BIN" - <<'PY'
from datasets import load_dataset
ds = load_dataset("baber/paul_graham_essays", split="train")
print(f"[INFO] Prewarmed paul_graham_essays rows={len(ds)}")
PY
}

if [[ -z "${RULER_METADATA:-}" ]]; then
  RULER_METADATA="$(build_lengths_metadata "$RULER_LENGTHS")"
fi
if [[ -z "${DEFAULT_MAX_LENGTH:-}" ]]; then
  DEFAULT_MAX_LENGTH="$("$PYTHON_BIN" - <<'PY' "$RULER_LENGTHS"
import sys
raw = sys.argv[1].replace(",", " ").split()
vals = [int(x) for x in raw if x.strip()]
print(max(vals))
PY
)"
fi
LENGTHS_TAG="$(build_lengths_tag "$RULER_LENGTHS")"
prewarm_ruler_datasets

use_chat_template_for_model() {
  local model_name="$1"
  if [[ -n "${DEFAULT_APPLY_CHAT_TEMPLATE_FORCE:-}" ]]; then
    if is_true "$DEFAULT_APPLY_CHAT_TEMPLATE_FORCE"; then
      echo "true"
    else
      echo "false"
    fi
    return 0
  fi

  case "$model_name" in
    *Instruct*|*Chat*|*chat*|*instruction* )
      echo "true"
      ;;
    *)
      echo "$DEFAULT_APPLY_CHAT_TEMPLATE"
      ;;
  esac
}

##############################################################################
# 主循环
##############################################################################
echo "[INFO] Running RULER all-settings eval"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] Settings: ${SETTINGS[*]}"
echo "[INFO] RULER_LENGTHS=$RULER_LENGTHS"
echo "[INFO] RULER_METADATA=$RULER_METADATA"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[INFO] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

if [[ "$LAUNCH_MODE" != "single" && "$RULER_ALLOW_DISTRIBUTED" != "true" ]]; then
  echo "[WARN] RULER tasks build/download data during task initialization and are fragile under distributed launch."
  echo "[WARN] Overriding LAUNCH_MODE=${LAUNCH_MODE} -> single. Set RULER_ALLOW_DISTRIBUTED=true to use distributed mode."
  LAUNCH_MODE="single"
fi
echo "[INFO] LAUNCH_MODE=$LAUNCH_MODE"

FAILED_RUNS=()

for model_postfix in "${MODEL_POSTFIX[@]}"; do
  pretrained="${BASE_MODEL_DIR}/${model_postfix}"

  if [[ ! -d "$pretrained" ]]; then
    echo "⚠️  [SKIP] 找不到模型路径: $pretrained"
    continue
  fi

  model_output_root="${HARNESS_DIR}/outputs/xsa_lm_eval_ruler/$(sanitize_tag "$model_postfix")"
  mkdir -p "$model_output_root"

  apply_chat_template="$(use_chat_template_for_model "$model_postfix")"

  echo "🚀 [START] 模型: ${model_postfix}"
  echo "📍 [INFO] 路径: ${pretrained}"
  echo "🧩 [INFO] APPLY_CHAT_TEMPLATE=${apply_chat_template}"

  for setting in "${SETTINGS[@]}"; do
    output_path="${model_output_root}/${TASKS}-${setting}-${LENGTHS_TAG}.json"
    existing_outputs=()
    if [[ -f "$output_path" ]]; then
      existing_outputs+=("$output_path")
    fi
    shopt -s nullglob
      timestamped_outputs=( "${model_output_root}/${TASKS}-${setting}-${LENGTHS_TAG}_"*.json )
    shopt -u nullglob
    if (( ${#timestamped_outputs[@]} > 0 )); then
      existing_outputs+=("${timestamped_outputs[@]}")
    fi
    if (( ${#existing_outputs[@]} > 0 )); then
      echo "⏭️  Existing output(s) found for ${TASKS}/${setting}, skip."
      printf '   - %s\n' "${existing_outputs[@]}"
      continue
    fi

    echo "--------------------------------------------------"
    echo "📊 Task: $TASKS | Fewshot: $NUM_FEWSHOT | BS: $DEFAULT_BATCH_SIZE | Setting: $setting"

    if ! MODEL_NAME="$pretrained" \
      BACKEND="$DEFAULT_BACKEND" \
      TASKS="$TASKS" \
      SETTING="$setting" \
      BATCH_SIZE="$DEFAULT_BATCH_SIZE" \
      DTYPE="$DEFAULT_DTYPE" \
      TRUST_REMOTE_CODE="$DEFAULT_TRUST_REMOTE_CODE" \
      APPLY_CHAT_TEMPLATE="$apply_chat_template" \
      NUM_FEWSHOT="$NUM_FEWSHOT" \
      MAX_LENGTH="$DEFAULT_MAX_LENGTH" \
      LAUNCH_MODE="$LAUNCH_MODE" \
      OUTPUT_PATH="$output_path" \
      LIMIT="$DEFAULT_LIMIT" \
      XSA_START_LAYER="$XSA_START_LAYER" \
      XSA_END_LAYER="$XSA_END_LAYER" \
      XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N" \
      XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N" \
      XSA_TRACK_STATS="$XSA_TRACK_STATS" \
      XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS" \
      METADATA="$RULER_METADATA" \
      bash "$SINGLE_SETTING_SCRIPT"; then
      echo "❌ [FAIL] model=${model_postfix} task=${TASKS} setting=${setting}" >&2
      FAILED_RUNS+=("${model_postfix} :: ${TASKS} :: ${setting}")
      if [[ "$CONTINUE_ON_TASK_ERROR" != "true" ]]; then
        exit 1
      fi
    fi
  done

  echo "✅ [DONE] 模型评测完成: ${model_postfix}"
  echo ""
done

if (( ${#FAILED_RUNS[@]} > 0 )); then
  echo "[ERROR] Some RULER runs failed:" >&2
  printf '  - %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

wait
