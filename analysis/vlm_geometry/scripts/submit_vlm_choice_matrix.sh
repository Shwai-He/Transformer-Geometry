#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SUBMIT_SCRIPT="$REPO_ROOT/analysis/vlm_geometry/scripts/submit_vlm_choice_eval_slurm.sh"

MODELS="${MODELS:-qwenimage,ming}"
TASKS="${TASKS:-mmvp,mmmu,mme,mmbench}"
SETTINGS="${SETTINGS:-baseline,value_no_para,value_no_perp,residual_no_para,residual_no_perp}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000000}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_choice_eval/understanding_full_$(date +%Y%m%d)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
TIME="${TIME:-08:00:00}"
MEM="${MEM:-96G}"
GRES="${GRES:-gpu:1}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
CANDIDATE_MODE="${CANDIDATE_MODE:-letter}"
VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}"
VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}"
ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}"
SCALE_MODE="${SCALE_MODE:-none}"
SCALE_SEED="${SCALE_SEED:-0}"
PARA_SCALE_MIN="${PARA_SCALE_MIN:--1.5}"
PARA_SCALE_MAX="${PARA_SCALE_MAX:-1.5}"
PERP_SCALE_MIN="${PERP_SCALE_MIN:--1.5}"
PERP_SCALE_MAX="${PERP_SCALE_MAX:-1.5}"

IFS=',' read -r -a model_list <<< "$MODELS"
IFS=',' read -r -a task_list <<< "$TASKS"
IFS=',' read -r -a setting_list <<< "$SETTINGS"

submitted=0
for model in "${model_list[@]}"; do
  model="${model//[[:space:]]/}"
  [[ -n "$model" ]] || continue
  for task in "${task_list[@]}"; do
    task="${task//[[:space:]]/}"
    [[ -n "$task" ]] || continue
    for setting in "${setting_list[@]}"; do
      setting="${setting//[[:space:]]/}"
      [[ -n "$setting" ]] || continue

      case "$setting" in
        baseline)
          space="value"; target="value"; para="1.0"; perp="1.0"; suffix="base"
          ;;
        value_no_para)
          space="value"; target="value"; para="0.0"; perp="1.0"; suffix="v-nopara"
          ;;
        value_no_perp)
          space="value"; target="value"; para="1.0"; perp="0.0"; suffix="v-noperp"
          ;;
        residual_no_para)
          space="residual"; target="block"; para="0.0"; perp="1.0"; suffix="r-nopara"
          ;;
        residual_no_perp)
          space="residual"; target="block"; para="1.0"; perp="0.0"; suffix="r-noperp"
          ;;
        *)
          echo "[ERROR] Unknown setting: $setting" >&2
          exit 2
          ;;
      esac

      job_model="$model"
      if [[ "$job_model" == "qwenimage" ]]; then
        job_model="qwen"
      fi
      export MODEL_NAME="$model"
      export TASK="$task"
      export TOTAL_SAMPLES="$TOTAL_SAMPLES"
      export CANDIDATE_MODE="$CANDIDATE_MODE"
      export SPACE="$space"
      export TARGET="$target"
      export PARA_SCALE="$para"
      export PERP_SCALE="$perp"
      export RESULT_ROOT="$RESULT_ROOT"
      export PARTITION="$PARTITION"
      export QOS="$QOS"
      export TIME="$TIME"
      export MEM="$MEM"
      export GRES="$GRES"
      export DTYPE="$DTYPE"
      export DEVICE_MAP="$DEVICE_MAP"
      export VALUE_HEAD_MODE="$VALUE_HEAD_MODE"
      export VALUE_REF_EXPANSION="$VALUE_REF_EXPANSION"
      export ATTN_ATTR_SOURCE="$ATTN_ATTR_SOURCE"
      export SCALE_MODE="$SCALE_MODE"
      export SCALE_SEED="$SCALE_SEED"
      export PARA_SCALE_MIN="$PARA_SCALE_MIN"
      export PARA_SCALE_MAX="$PARA_SCALE_MAX"
      export PERP_SCALE_MIN="$PERP_SCALE_MIN"
      export PERP_SCALE_MAX="$PERP_SCALE_MAX"
      export JOB_NAME="vlmchoice-${job_model}-${task}-${suffix}"
      bash "$SUBMIT_SCRIPT"
      submitted=$((submitted + 1))
    done
  done
done

echo "[INFO] Submitted $submitted VLM choice-eval jobs."
echo "[INFO] RESULT_ROOT=$RESULT_ROOT"
