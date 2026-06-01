#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# 基础环境配置
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
SETTING="residual_attn"
# BACKGROUND=true  → nohup 后台运行，打印 PID 后立即返回
# BACKGROUND=false → 前台阻塞运行（默认）
BACKGROUND="${BACKGROUND:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="${NLTK_DATA:-/mnt/hdfs/shwai.he/DepthBoost/nltk_data}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$HARNESS_DIR/offload}"
SINGLE_SETTING_SCRIPT="$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-attn-removal-batch.log"
fi

# 后台模式：re-exec 自身，父进程打印 PID 后退出
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

##############################################################################
# 环境依赖
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
# 模型配置：直接在这里改
##############################################################################
if [[ -n "${MODEL_POSTFIX_LIST:-}" ]]; then
  read -r -a MODEL_POSTFIX <<< "$MODEL_POSTFIX_LIST"
else
  MODEL_POSTFIX=(
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-30B-A3B"
    "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-4B-Base"
    # "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-14B"
  )
fi

##############################################################################
# 任务配置：直接在这里改
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
  # [mts_dialog]=5
  [drop]=0
  # [truthfulqa_gen]=0
  [mbpp]=3
  # [ifeval]=0
  [bbh_cot_zeroshot]=0
  # [longbench_narrativeqa]=3
  # [longbench_triviaqa]=0
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
  # mts_dialog
  drop
  # truthfulqa_gen
  mbpp
  # ifeval
  bbh_cot_zeroshot
  # longbench_narrativeqa
  # longbench_triviaqa
)

FINAL_TASKS=("${TASKS_ORDER[@]}")
if [[ "$REVERSE_TASKS" == "true" ]]; then
  echo "🔄 Reverse option enabled: Tasks will run in reverse order."
  FINAL_TASKS=()
  for (( i=${#TASKS_ORDER[@]}-1; i>=0; i-- )); do
    FINAL_TASKS+=("${TASKS_ORDER[i]}")
  done
fi

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
echo "[INFO] Running attention-removal batch eval (setting=${SETTING})"

FAILED_TASKS=()

for model_postfix in "${MODEL_POSTFIX[@]}"; do
  pretrained="${BASE_MODEL_DIR}/${model_postfix}"

  if [[ ! -d "$pretrained" ]]; then
    echo "⚠️  [SKIP] 找不到模型路径: $pretrained"
    continue
  fi

  model_output_root="${HARNESS_DIR}/outputs/xsa_lm_eval/$(echo "$model_postfix" | tr '/' '_')"
  mkdir -p "$model_output_root"

  echo "🚀 [START] 开始评测模型: ${model_postfix}"
  echo "📍 [INFO] 路径: ${pretrained}"

  for task in "${FINAL_TASKS[@]}"; do
    apply_chat_template="$(use_chat_template_for_model "$model_postfix" "$task")"
    fewshot="${TASKS_MAP[$task]}"
    current_bs="$DEFAULT_BATCH_SIZE"
    if [[ "$task" == longbench* ]]; then
      current_bs=1
    fi

    output_path="${model_output_root}/${task}-${SETTING}.json"
    existing_outputs=()
    if [[ -f "$output_path" ]]; then
      existing_outputs+=("$output_path")
    fi
    shopt -s nullglob
    timestamped_outputs=( "${model_output_root}/${task}-${SETTING}_"*.json )
    shopt -u nullglob
    if (( ${#timestamped_outputs[@]} > 0 )); then
      existing_outputs+=("${timestamped_outputs[@]}")
    fi
    if (( ${#existing_outputs[@]} > 0 )); then
      echo "⏭️  Existing output(s) found for ${task}/${SETTING}, skip."
      printf '   - %s\n' "${existing_outputs[@]}"
      continue
    fi

    echo "--------------------------------------------------"
    echo "📊 Task: $task | Fewshot: $fewshot | BS: $current_bs | Setting: $SETTING"
    echo "🧩 [INFO] APPLY_CHAT_TEMPLATE=${apply_chat_template}"

    if ! MODEL_NAME="$pretrained" \
      BACKEND="$DEFAULT_BACKEND" \
      TASKS="$task" \
      SETTING="$SETTING" \
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
      LIMIT="${LIMIT:-}" \
      XSA_START_LAYER="$XSA_START_LAYER" \
      XSA_END_LAYER="$XSA_END_LAYER" \
      XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N" \
      XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N" \
      XSA_TRACK_STATS="$XSA_TRACK_STATS" \
      XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS" \
      CONFIRM_RUN_UNSAFE_CODE="$CONFIRM_RUN_UNSAFE_CODE" \
      bash "$SINGLE_SETTING_SCRIPT"; then
      echo "❌ [FAIL] model=${model_postfix} task=${task} setting=${SETTING}" >&2
      FAILED_TASKS+=("${model_postfix}|${task}|${SETTING}")
      if [[ "$CONTINUE_ON_TASK_ERROR" != "true" ]]; then
        exit 1
      fi
      continue
    fi
  done

  echo "✅ [DONE] 模型评测完成: ${model_postfix}"
  echo ""
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "[WARN] Some task runs failed:" >&2
  printf '  - %s\n' "${FAILED_TASKS[@]}" >&2
fi

wait
