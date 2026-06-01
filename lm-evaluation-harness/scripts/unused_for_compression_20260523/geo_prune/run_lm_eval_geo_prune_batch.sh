#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Geometry-aware pruning lm-eval batch.
# This mirrors the task/few-shot/skip structure of the baseline XSA scripts, but
# uses hf-geo-prune so pruned checkpoints do not need to be saved to disk.
##############################################################################

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
MODEL_PATH="${MODEL_PATH:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_PATH")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/geo_prune_lm_eval/$MODEL_TAG}"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/geo_prune_lm_eval_logs}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-1}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-auto}"
DEFAULT_DEVICE="${DEFAULT_DEVICE:-cpu}"
DEFAULT_TRUST_REMOTE_CODE="${DEFAULT_TRUST_REMOTE_CODE:-true}"
DEFAULT_APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE:-false}"
DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-4096}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
CONTINUE_ON_TASK_ERROR="${CONTINUE_ON_TASK_ERROR:-true}"
REVERSE_TASKS="${REVERSE_TASKS:-false}"
LIMIT="${LIMIT:-}"
INCLUDE_PATH="${INCLUDE_PATH:-$REPO_ROOT/compression/lm_eval_tasks}"
LOCAL_TASK_PATH="$REPO_ROOT/compression/lm_eval_tasks/local_mcq"

GEO_PRUNE_METHOD="${GEO_PRUNE_METHOD:-wanda}"
GEO_PRUNE_TARGETS="${GEO_PRUNE_TARGETS:-q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj}"
GEO_GEOMETRY_MODE="${GEO_GEOMETRY_MODE:-residual_error}"
GEO_GEOMETRY_ALPHA="${GEO_GEOMETRY_ALPHA:-1.0}"
GEO_GEOMETRY_TARGETS="${GEO_GEOMETRY_TARGETS:-o_proj+down_proj+v_proj}"
GEO_SPARSITY_RATIO="${GEO_SPARSITY_RATIO:-0.5}"
GEO_THRESHOLD_SCOPE="${GEO_THRESHOLD_SCOPE:-global}"
GEO_TOKEN_SCOPE="${GEO_TOKEN_SCOPE:-last}"
GEO_CALIB_PROMPTS="${GEO_CALIB_PROMPTS:-}"
GEO_CALIB_FILE="${GEO_CALIB_FILE:-}"
GEO_MAX_PROMPTS="${GEO_MAX_PROMPTS:-2}"
GEO_MAX_LENGTH="${GEO_MAX_LENGTH:-256}"

case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT" ;;
esac
case "$LOG_DIR" in
  /*) ;;
  *) LOG_DIR="$REPO_ROOT/$LOG_DIR" ;;
esac

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
SINGLE_SETTING_SCRIPT="$SCRIPT_DIR/run_lm_eval_geo_prune_setting.sh"
if [[ ! -f "$SINGLE_SETTING_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $SINGLE_SETTING_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[ERROR] MODEL_PATH has no config.json: $MODEL_PATH" >&2
  exit 1
fi

# Match the full task surface used by the baseline/XSA variant batch scripts.
declare -A TASKS_MAP=(
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

if [[ -n "${TASKS_CSV:-}" ]]; then
  IFS=',' read -r -a TASKS_ORDER <<< "$TASKS_CSV"
fi

FINAL_TASKS=("${TASKS_ORDER[@]}")
if [[ "$REVERSE_TASKS" == "true" ]]; then
  FINAL_TASKS=()
  for (( i=${#TASKS_ORDER[@]}-1; i>=0; i-- )); do
    FINAL_TASKS+=("${TASKS_ORDER[i]}")
  done
fi

if [[ -n "${GEO_SETTINGS:-}" ]]; then
  IFS=';' read -r -a GEO_SETTING_SPECS <<< "$GEO_SETTINGS"
else
  GEO_SETTING_SPECS=(
    "dense|false|none|unstructured"
    "wanda_unstructured|true|none|unstructured"
    "wanda_unstructured_residual_perp|true|residual_perp|unstructured"
    "wanda_unstructured_residual_para|true|residual_para|unstructured"
    "wanda_unstructured_value_perp|true|value_perp|unstructured"
    "wanda_2_4|true|none|2:4"
    "wanda_4_8|true|none|4:8"
  )
fi

is_true() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
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

manifest="${RUN_MANIFEST:-$OUTPUT_ROOT/run_manifest.tsv}"
if [[ ! -f "$manifest" ]]; then
  mkdir -p "$(dirname "$manifest")"
  printf 'setting\ttask\tstatus\toutput_path\n' > "$manifest"
fi

FAILED_TASKS=()
echo "[INFO] Running geo-prune lm-eval batch"
echo "[INFO] model=$MODEL_PATH"
echo "[INFO] settings=${GEO_SETTING_SPECS[*]}"

for spec in "${GEO_SETTING_SPECS[@]}"; do
  IFS='|' read -r setting_name prune_enabled prune_strategy sparsity_type <<< "$spec"
  setting_root="$OUTPUT_ROOT/$setting_name"
  score_cache="${GEO_SCORE_CACHE_PATH:-$OUTPUT_ROOT/_geometry_scores/geometry_scores.pt}"
  mkdir -p "$setting_root" "$(dirname "$score_cache")"

  for task in "${FINAL_TASKS[@]}"; do
    task="${task// /}"
    fewshot="${TASKS_MAP[$task]:-0}"
    apply_chat_template="$(use_chat_template_for_model "$MODEL_TAG" "$task")"
    current_bs="$DEFAULT_BATCH_SIZE"
    if [[ "$task" == longbench* ]]; then
      current_bs=1
    fi
    include_path_this="$INCLUDE_PATH"
    if [[ "$task" == "local_mcq" && -z "$include_path_this" ]]; then
      include_path_this="$LOCAL_TASK_PATH"
    fi

    output_path="$setting_root/${task}.json"
    existing_outputs=()
    if [[ -f "$output_path" ]]; then
      existing_outputs+=("$output_path")
    fi
    shopt -s nullglob
    timestamped_outputs=( "${output_path%.json}"_*.json )
    shopt -u nullglob
    if (( ${#timestamped_outputs[@]} > 0 )); then
      existing_outputs+=("${timestamped_outputs[@]}")
    fi
    if (( ${#existing_outputs[@]} > 0 )); then
      echo "[SKIP] Existing output(s) found for ${setting_name}/${task}"
      printf '%s\t%s\texists\t%s\n' "$setting_name" "$task" "$output_path" >> "$manifest"
      continue
    fi

    echo "--------------------------------------------------"
    echo "[RUN] setting=$setting_name task=$task fewshot=$fewshot sparsity_type=$sparsity_type strategy=$prune_strategy"

    if ! PYTHON_BIN="$PYTHON_BIN" \
      MODEL_NAME="$MODEL_PATH" \
      TASKS="$task" \
      OUTPUT_PATH="$output_path" \
      BATCH_SIZE="$current_bs" \
      DEVICE="$DEFAULT_DEVICE" \
      DTYPE="$DEFAULT_DTYPE" \
      TRUST_REMOTE_CODE="$DEFAULT_TRUST_REMOTE_CODE" \
      APPLY_CHAT_TEMPLATE="$apply_chat_template" \
      NUM_FEWSHOT="$fewshot" \
      LIMIT="$LIMIT" \
      MAX_LENGTH="$DEFAULT_MAX_LENGTH" \
      INCLUDE_PATH="$include_path_this" \
      CONFIRM_RUN_UNSAFE_CODE="$CONFIRM_RUN_UNSAFE_CODE" \
      LOG_DIR="$LOG_DIR" \
      GEO_PRUNE_ENABLED="$prune_enabled" \
      GEO_PRUNE_METHOD="$GEO_PRUNE_METHOD" \
      GEO_PRUNE_STRATEGY="$prune_strategy" \
      GEO_PRUNE_TARGETS="$GEO_PRUNE_TARGETS" \
      GEO_GEOMETRY_MODE="$GEO_GEOMETRY_MODE" \
      GEO_GEOMETRY_ALPHA="$GEO_GEOMETRY_ALPHA" \
      GEO_GEOMETRY_TARGETS="$GEO_GEOMETRY_TARGETS" \
      GEO_SPARSITY_RATIO="$GEO_SPARSITY_RATIO" \
      GEO_SPARSITY_TYPE="$sparsity_type" \
      GEO_THRESHOLD_SCOPE="$GEO_THRESHOLD_SCOPE" \
      GEO_TOKEN_SCOPE="$GEO_TOKEN_SCOPE" \
      GEO_CALIB_PROMPTS="$GEO_CALIB_PROMPTS" \
      GEO_CALIB_FILE="$GEO_CALIB_FILE" \
      GEO_MAX_PROMPTS="$GEO_MAX_PROMPTS" \
      GEO_MAX_LENGTH="$GEO_MAX_LENGTH" \
      GEO_SCORE_CACHE_PATH="$score_cache" \
      bash "$SINGLE_SETTING_SCRIPT"; then
      echo "[FAIL] setting=${setting_name} task=${task}" >&2
      FAILED_TASKS+=("${setting_name}|${task}")
      printf '%s\t%s\tfailed\t%s\n' "$setting_name" "$task" "$output_path" >> "$manifest"
      if [[ "$CONTINUE_ON_TASK_ERROR" != "true" ]]; then
        exit 1
      fi
      continue
    fi
    printf '%s\t%s\tok\t%s\n' "$setting_name" "$task" "$output_path" >> "$manifest"
  done
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "[WARN] Some task runs failed:" >&2
  printf '  - %s\n' "${FAILED_TASKS[@]}" >&2
fi

echo "[DONE] manifest: $manifest"
