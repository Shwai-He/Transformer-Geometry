#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="${MODELS:-qwenimage,ming}"
TASKS="${TASKS:-mmvp,mmbench,mmmu,mme}"
SPACES="${SPACES:-residual,value}"
TARGETS_RESIDUAL="${TARGETS_RESIDUAL:-block,attn,mlp}"
TARGETS_VALUE="${TARGETS_VALUE:-value}"
PARA_SCALES="${PARA_SCALES:-1.0,0.0}"
PERP_SCALES="${PERP_SCALES:-1.0,0.0}"
AXIS_MODE="${AXIS_MODE:-grid}"  # grid, para, or perp

RESULT_ROOT="${RESULT_ROOT:-runs/vlm_geometry_ppl/scaling_benchmark}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-64}"
PARTITION="${PARTITION:-beacon}"
QOS="${QOS:-medium}"
MEM="${MEM:-64G}"
TIME="${TIME:-04:00:00}"
ANSWER_MODE="${ANSWER_MODE:-letter}"
VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}"
VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}"
ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}"
SCALE_MODE="${SCALE_MODE:-none}"
SCALE_SEED="${SCALE_SEED:-0}"
PARA_SCALE_MIN="${PARA_SCALE_MIN:--1.5}"
PARA_SCALE_MAX="${PARA_SCALE_MAX:-1.5}"
PERP_SCALE_MIN="${PERP_SCALE_MIN:--1.5}"
PERP_SCALE_MAX="${PERP_SCALE_MAX:-1.5}"

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
IFS=',' read -r -a SPACE_LIST <<< "$SPACES"
IFS=',' read -r -a TARGET_LIST_RESIDUAL <<< "$TARGETS_RESIDUAL"
IFS=',' read -r -a TARGET_LIST_VALUE <<< "$TARGETS_VALUE"
IFS=',' read -r -a PARA_LIST <<< "$PARA_SCALES"
IFS=',' read -r -a PERP_LIST <<< "$PERP_SCALES"

submit_one() {
  local model="$1"
  local task="$2"
  local space="$3"
  local target="$4"
  local para="$5"
  local perp="$6"
  local para_tag="${para//./p}"
  local perp_tag="${perp//./p}"
  para_tag="${para_tag//-/m}"
  perp_tag="${perp_tag//-/m}"
  local job_name="vlmppl-${model}-${task}-${space}-${target}-p${para_tag}-r${perp_tag}"

  echo "[SUBMIT] model=$model task=$task space=$space target=$target para=$para perp=$perp"
  MODEL_NAME="$model" \
  TASK="$task" \
  SPACE="$space" \
  TARGET="$target" \
  PARA_SCALE="$para" \
  PERP_SCALE="$perp" \
  RESULT_ROOT="$RESULT_ROOT" \
  TOTAL_SAMPLES="$TOTAL_SAMPLES" \
  ANSWER_MODE="$ANSWER_MODE" \
  VALUE_HEAD_MODE="$VALUE_HEAD_MODE" \
  VALUE_REF_EXPANSION="$VALUE_REF_EXPANSION" \
  ATTN_ATTR_SOURCE="$ATTN_ATTR_SOURCE" \
  SCALE_MODE="$SCALE_MODE" \
  SCALE_SEED="$SCALE_SEED" \
  PARA_SCALE_MIN="$PARA_SCALE_MIN" \
  PARA_SCALE_MAX="$PARA_SCALE_MAX" \
  PERP_SCALE_MIN="$PERP_SCALE_MIN" \
  PERP_SCALE_MAX="$PERP_SCALE_MAX" \
  PARTITION="$PARTITION" \
  QOS="$QOS" \
  MEM="$MEM" \
  TIME="$TIME" \
  JOB_NAME="$job_name" \
    bash "$SCRIPT_DIR/submit_vlm_ppl_slurm.sh"
}

for model in "${MODEL_LIST[@]}"; do
  [[ -z "$model" ]] && continue
  for task in "${TASK_LIST[@]}"; do
    [[ -z "$task" ]] && continue
    for space in "${SPACE_LIST[@]}"; do
      [[ -z "$space" ]] && continue
      if [[ "$space" == "value" ]]; then
        TARGET_LIST=("${TARGET_LIST_VALUE[@]}")
      else
        TARGET_LIST=("${TARGET_LIST_RESIDUAL[@]}")
      fi
      for target in "${TARGET_LIST[@]}"; do
        [[ -z "$target" ]] && continue
        if [[ "$AXIS_MODE" == "para" ]]; then
          for para in "${PARA_LIST[@]}"; do
            submit_one "$model" "$task" "$space" "$target" "$para" "1.0"
          done
        elif [[ "$AXIS_MODE" == "perp" ]]; then
          for perp in "${PERP_LIST[@]}"; do
            submit_one "$model" "$task" "$space" "$target" "1.0" "$perp"
          done
        else
          for para in "${PARA_LIST[@]}"; do
            for perp in "${PERP_LIST[@]}"; do
              submit_one "$model" "$task" "$space" "$target" "$para" "$perp"
            done
          done
        fi
      done
    done
  done
done
