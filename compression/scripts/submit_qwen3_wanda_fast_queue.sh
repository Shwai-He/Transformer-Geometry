#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
SAVE_GRES="${SAVE_GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
EVAL_GRES="${EVAL_GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
SAVE_MEM="${SAVE_MEM:-160G}"
EVAL_MEM="${EVAL_MEM:-96G}"
TIME="${TIME:-08:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
CALIB_FILE="${CALIB_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
TARGETS="${TARGETS:-q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
NSAMPLES="${NSAMPLES:-128}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
EVAL_APPLY_CHAT_TEMPLATE="${EVAL_APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"

fewshot_for_task() {
  case "$1" in
    openbookqa|piqa|rte|boolq|drop|bbh_cot_zeroshot) echo 0 ;;
    winogrande|mmlu|humaneval|nq_open) echo 5 ;;
    arc_challenge) echo 25 ;;
    hellaswag) echo 10 ;;
    gsm8k_cot) echo 8 ;;
    mbpp) echo 3 ;;
    *) echo 0 ;;
  esac
}

mkdir -p "$REPO_ROOT/compression/outputs/wanda_layerwise_pruned"

submit_save() {
  local model_id="$1"
  local model_tag="$2"
  local variant="$3"
  local sparsity_ratio="$4"
  local sparsity_type="$5"
  local dependency="${6:-}"
  local local_files_only="${7:-true}"

  local out_dir="$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/$model_tag/$variant"
  local slurm_dir="$out_dir/slurm"
  local job_name="wanda-${model_tag}-${variant}-save"
  mkdir -p "$slurm_dir"

  local output
  output="$(
    REPO_ROOT="$REPO_ROOT" \
    PARTITION="$PARTITION" \
    QOS="$QOS" \
    GRES="$SAVE_GRES" \
    CPUS_PER_TASK="$CPUS_PER_TASK" \
    MEM="$SAVE_MEM" \
    TIME="$TIME" \
    JOB_NAME="$job_name" \
    SLURM_DIR="$slurm_dir" \
    PYTHON_BIN="$PYTHON_BIN" \
    HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE="$([[ "$local_files_only" == "true" ]] && echo 1 || echo 0)" \
    TRANSFORMERS_OFFLINE="$([[ "$local_files_only" == "true" ]] && echo 1 || echo 0)" \
    MODEL_PATH="$model_id" \
    CALIB_FILE="$CALIB_FILE" \
    OUTPUT_DIR="$out_dir" \
    LOCAL_FILES_ONLY="$local_files_only" \
    NSAMPLES="$NSAMPLES" \
    MAX_LENGTH="$MAX_LENGTH" \
    SPARSITY_RATIO="$sparsity_ratio" \
    SPARSITY_TYPE="$sparsity_type" \
    TARGETS="$TARGETS" \
    DTYPE="$DTYPE" \
    DEPENDENCY="$dependency" \
    bash "$REPO_ROOT/compression/scripts/submit_wanda_layerwise_prune_slurm.sh"
  )"
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | awk -F= '/job_id=/{print $2}' | tail -1
}

submit_eval() {
  local model_name="$1"
  local model_tag="$2"
  local variant="$3"
  local task="$4"
  local dependency="$5"
  local fewshot="$6"

  local out_root="$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/$model_tag/$variant/lm_eval/$task"
  local slurm_dir="$out_root/slurm"
  local job_name
  mkdir -p "$slurm_dir"
  if [[ "$variant" == "dense" ]]; then
    job_name="dense-${model_tag}-${task}"
  else
    job_name="eval-${model_tag}-${variant}-${task}"
  fi

  local log_samples_flag=""
  if [[ "$LOG_SAMPLES" == "true" ]]; then
    log_samples_flag=" LOG_SAMPLES=true"
  fi

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --dependency="$dependency" \
    --job-name="$job_name" \
    --gres="$EVAL_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EVAL_MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="cd $REPO_ROOT && export HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 && PYTHON_BIN=$PYTHON_BIN MODEL_NAME=$model_name TASKS=$task NUM_FEWSHOT=$fewshot OUTPUT_ROOT=$out_root BATCH_SIZE=auto DTYPE=$DTYPE APPLY_CHAT_TEMPLATE=$EVAL_APPLY_CHAT_TEMPLATE MAX_LENGTH=$EVAL_MAX_LENGTH LAUNCH_MODE=single TRUST_REMOTE_CODE=true$log_samples_flag bash lm-evaluation-harness/scripts/run_lm_eval_compression_setting.sh"
}

submit_model_grid() {
  local model_id="$1"
  local model_tag="$2"

  local s0p5
  local s0p3
  local two4
  local four8
  s0p5="$(submit_save "$model_id" "$model_tag" all_linear_unstructured_s0p5_c4_ns128_seq2048 0.5 unstructured "" false)"
  s0p3="$(submit_save "$model_id" "$model_tag" all_linear_unstructured_s0p3_c4_ns128_seq2048 0.3 unstructured "afterok:$s0p5" true)"
  two4="$(submit_save "$model_id" "$model_tag" all_linear_2to4_s0p5_c4_ns128_seq2048 0.5 2:4 "afterok:$s0p5" true)"
  four8="$(submit_save "$model_id" "$model_tag" all_linear_4to8_s0p5_c4_ns128_seq2048 0.5 4:8 "afterok:$s0p5" true)"

  local ckpt_root="$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/$model_tag"
  IFS=',' read -r -a tasks_order <<< "$TASKS_ORDER_CSV"
  local eval_jobs=()
  local task fewshot
  for task in "${tasks_order[@]}"; do
    task="${task// /}"
    fewshot="$(fewshot_for_task "$task")"
    eval_jobs+=("s0p5:$task:$(submit_eval "$ckpt_root/all_linear_unstructured_s0p5_c4_ns128_seq2048" "$model_tag" all_linear_unstructured_s0p5_c4_ns128_seq2048 "$task" "afterok:$s0p5" "$fewshot")")
    eval_jobs+=("s0p3:$task:$(submit_eval "$ckpt_root/all_linear_unstructured_s0p3_c4_ns128_seq2048" "$model_tag" all_linear_unstructured_s0p3_c4_ns128_seq2048 "$task" "afterok:$s0p3" "$fewshot")")
    eval_jobs+=("2to4:$task:$(submit_eval "$ckpt_root/all_linear_2to4_s0p5_c4_ns128_seq2048" "$model_tag" all_linear_2to4_s0p5_c4_ns128_seq2048 "$task" "afterok:$two4" "$fewshot")")
    eval_jobs+=("4to8:$task:$(submit_eval "$ckpt_root/all_linear_4to8_s0p5_c4_ns128_seq2048" "$model_tag" all_linear_4to8_s0p5_c4_ns128_seq2048 "$task" "afterok:$four8" "$fewshot")")
    eval_jobs+=("dense:$task:$(submit_eval "$model_id" "$model_tag" dense "$task" "afterok:$s0p5" "$fewshot")")
  done

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model_tag" "$s0p5" "$s0p3" "$two4" "$four8" "$(IFS=','; echo "${eval_jobs[*]}")"
}

manifest="$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/qwen3_fast_queue_jobs.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'model_tag\tsave_s0p5\tsave_s0p3\tsave_2to4\tsave_4to8\teval_jobs\n' > "$manifest"
fi

submit_model_grid "Qwen/Qwen3-4B" "qwen3_4b" | tee -a "$manifest"
submit_model_grid "Qwen/Qwen3-1.7B" "qwen3_1p7b" | tee -a "$manifest"
echo "[DONE] manifest=$manifest"
