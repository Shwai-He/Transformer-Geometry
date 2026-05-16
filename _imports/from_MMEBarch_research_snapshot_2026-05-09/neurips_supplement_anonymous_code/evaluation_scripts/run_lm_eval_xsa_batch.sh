#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# 基础环境配置
##############################################################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export NLTK_DATA="${NLTK_DATA:-/path/to/resource}"

REVERSE_TASKS="${REVERSE_TASKS:-false}"   # 是否反转任务顺序
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/path/to/resource}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-auto}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bfloat16}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-causal}"
DEFAULT_TRUST_REMOTE_CODE="${DEFAULT_TRUST_REMOTE_CODE:-true}"
DEFAULT_APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE:-false}"
DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-4096}"
LAUNCH_MODE="${LAUNCH_MODE:-data_parallel}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
XSA_FORWARD_OP="${XSA_FORWARD_OP:-remove_parallel}"
XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-1.0}"
CONTINUE_ON_TASK_ERROR="${CONTINUE_ON_TASK_ERROR:-true}"
# BACKGROUND=true  → nohup 后台运行，打印 PID 后立即返回
# BACKGROUND=false → 前台阻塞运行（默认）
BACKGROUND="${BACKGROUND:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SINGLE_SETTING_SCRIPT="$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-xsa-batch.log"
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

pip install "transformers==4.52.4"
pip install evaluate
pip install bert_score
pip install "rouge_score>=0.1.2"
pip install nltk
pip install absl-py
pip install git+https://github.com/google-research/bleurt.git
"$PYTHON_BIN" -c "import triton,sys; sys.exit(triton.__version__!='3.0.0')" 2>/dev/null || pip install "triton==3.0"

##############################################################################
# 模型配置
##############################################################################
# 你可以直接改这个数组，也可以从环境变量 MODEL_POSTFIX_LIST 覆盖：
# MODEL_POSTFIX_LIST="Qwen/Qwen3-14B-Base Qwen/Qwen3-14B"
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
# 设置集合
##############################################################################
# 默认跑 6 组：
# 1. none
# 2. xsa_middle
# 3. xsa_middle_multihead
# 4. residual_attn
# 5. residual_mlp
# 6. residual_both
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

##############################################################################
# 任务映射表：[任务名]=fewshot数量
##############################################################################
declare -A TASKS_MAP=(
  [mmlu]=5
  [gsm8k]=5
)

# 显式定义任务顺序
TASKS_ORDER=(
  mmlu
  gsm8k
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
  if [[ -n "${DEFAULT_APPLY_CHAT_TEMPLATE_FORCE:-}" ]]; then
    if is_true "$DEFAULT_APPLY_CHAT_TEMPLATE_FORCE"; then
      echo "true"
    else
      echo "false"
    fi
    return 0
  fi

  case "$model_name" in
    *Instruct*|*Chat*|*chat*|*instruction*|*Qwen3-*|*Qwen/Qwen3-* )
      echo "true"
      ;;
    *)
      echo "$DEFAULT_APPLY_CHAT_TEMPLATE"
      ;;
  esac
}

##############################################################################
# 主循环逻辑
##############################################################################
for model_postfix in "${MODEL_POSTFIX[@]}"; do
  pretrained="${BASE_MODEL_DIR}/${model_postfix}"

  if [[ ! -d "$pretrained" ]]; then
    echo "⚠️  [SKIP] 找不到模型路径: $pretrained"
    continue
  fi

  model_output_root="${HARNESS_DIR}/outputs/xsa_lm_eval/$(echo "$model_postfix" | tr '/' '_')"
  mkdir -p "$model_output_root"

  apply_chat_template="$(use_chat_template_for_model "$model_postfix")"

  echo "🚀 [START] 开始评测模型: ${model_postfix}"
  echo "📍 [INFO] 路径: ${pretrained}"
  echo "🧩 [INFO] APPLY_CHAT_TEMPLATE=${apply_chat_template}"

  for task in "${FINAL_TASKS[@]}"; do
    fewshot="${TASKS_MAP[$task]}"
    current_bs="$DEFAULT_BATCH_SIZE"

    if [[ "$task" == longbench* ]]; then
      current_bs=1
    fi

    echo "--------------------------------------------------"
    echo "📊 Task: $task | Fewshot: $fewshot | BS: $current_bs"

    for setting in "${SETTINGS[@]}"; do
      output_path="${model_output_root}/${task}-${setting}.json"
      existing_outputs=()
      if [[ -f "$output_path" ]]; then
        existing_outputs+=("$output_path")
      fi
      shopt -s nullglob
      timestamped_outputs=( "${model_output_root}/${task}-${setting}_"*.json )
      shopt -u nullglob
      if (( ${#timestamped_outputs[@]} > 0 )); then
        existing_outputs+=("${timestamped_outputs[@]}")
      fi
      if (( ${#existing_outputs[@]} > 0 )); then
        echo "⏭️  Existing output(s) found for ${task}/${setting}, skip."
        printf '   - %s\n' "${existing_outputs[@]}"
        continue
      fi

      echo "   ▶ Setting: $setting"

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
      OUTPUT_PATH="$output_path" \
        LIMIT="${LIMIT:-}" \
        XSA_START_LAYER="$XSA_START_LAYER" \
        XSA_END_LAYER="$XSA_END_LAYER" \
      XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N" \
      XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N" \
      XSA_FORWARD_OP="$XSA_FORWARD_OP" \
      XSA_FORWARD_ALPHA="$XSA_FORWARD_ALPHA" \
      XSA_TRACK_STATS="$XSA_TRACK_STATS" \
      XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS" \
      bash "$SINGLE_SETTING_SCRIPT"; then
        echo "❌ [FAIL] model=${model_postfix} task=${task} setting=${setting}" >&2
        if [[ "$CONTINUE_ON_TASK_ERROR" != "true" ]]; then
          exit 1
        fi
        continue
      fi
    done
  done

  echo "✅ [DONE] 模型评测完成: ${model_postfix}"
  echo ""
done

wait
