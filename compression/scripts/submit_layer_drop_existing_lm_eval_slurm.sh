#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
MEM="${MEM:-96G}"
TIME="${TIME:-08:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_lm_eval}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_geometry}"
RANK_METRICS_CSV="${RANK_METRICS_CSV:-hybrid_perp_compw_alltok}"
COMPONENTS_CSV="${COMPONENTS_CSV:-mlp}"
DROP_COUNTS_CSV="${DROP_COUNTS_CSV:-4,8}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"

MODEL_SPECS="${MODEL_SPECS:-qwen3_0p6b=Qwen/Qwen3-0.6B-Base;qwen3_1p7b=Qwen/Qwen3-1.7B;qwen3_4b=Qwen/Qwen3-4B}"

mkdir -p "$OUTPUT_ROOT"
manifest="$OUTPUT_ROOT/submission_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'job_id\tjob_type\tmodel_tag\trank_metric\tcomponent\tdrop_count\ttask\toutput_root\tselection_json\n' > "$manifest"
fi

fewshot_for_task() {
  case "$1" in
    gsm8k_cot) echo 8 ;;
    gsm8k) echo 5 ;;
    mmlu) echo 5 ;;
    openbookqa|piqa|rte|boolq|drop|bbh_cot_zeroshot) echo 0 ;;
    winogrande|humaneval|nq_open) echo 5 ;;
    arc_challenge) echo 25 ;;
    hellaswag) echo 10 ;;
    mbpp) echo 3 ;;
    *) echo 0 ;;
  esac
}

submit_eval() {
  local model_tag="$1"
  local model_name="$2"
  local rank_metric="$3"
  local component="$4"
  local drop_count="$5"
  local task="$6"
  local selection_json="$SELECTION_ROOT/$model_tag/$rank_metric/drop_selection.json"
  if [[ ! -f "$selection_json" ]]; then
    echo "[SKIP] missing selection_json=$selection_json" >&2
    return 0
  fi

  local fewshot
  fewshot="$(fewshot_for_task "$task")"
  local setting="${rank_metric}_${component}_drop${drop_count}"
  local out_root="$OUTPUT_ROOT/$model_tag/$setting/$task"
  local slurm_dir="$out_root/slurm"
  mkdir -p "$slurm_dir"

  local job_id
  job_id="$(sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="ld-${model_tag}-${component}${drop_count}-${task}" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
PYTHON_BIN="$PYTHON_BIN" \
MODEL_NAME="$model_name" \
TASKS="$task" \
OUTPUT_ROOT="$out_root" \
BATCH_SIZE="$BATCH_SIZE" \
DTYPE="$DTYPE" \
APPLY_CHAT_TEMPLATE="$APPLY_CHAT_TEMPLATE" \
NUM_FEWSHOT="$fewshot" \
MAX_LENGTH="$EVAL_MAX_LENGTH" \
TRUST_REMOTE_CODE=true \
LOG_SAMPLES="$LOG_SAMPLES" \
AUTO_COLLECT_RESULTS="$AUTO_COLLECT_RESULTS" \
LAYER_DROP_CONFIG="$selection_json" \
LAYER_DROP_COMPONENT="$component" \
LAYER_DROP_COUNT="$drop_count" \
bash "$REPO_ROOT/lm-evaluation-harness/scripts/run_lm_eval_layer_drop_setting.sh"
EOF
)"
  )"
  printf '%s\teval\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$model_tag" "$rank_metric" "$component" "$drop_count" "$task" "$out_root" "$selection_json" >> "$manifest"
  echo "$job_id"
}

IFS=';' read -r -a MODEL_ITEMS <<< "$MODEL_SPECS"
IFS=',' read -r -a RANK_METRICS <<< "$RANK_METRICS_CSV"
IFS=',' read -r -a COMPONENTS <<< "$COMPONENTS_CSV"
IFS=',' read -r -a DROP_COUNTS <<< "$DROP_COUNTS_CSV"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

for item in "${MODEL_ITEMS[@]}"; do
  model_tag="${item%%=*}"
  model_name="${item#*=}"
  for rank_metric in "${RANK_METRICS[@]}"; do
    rank_metric="${rank_metric// /}"
    for component in "${COMPONENTS[@]}"; do
      component="${component// /}"
      for drop_count in "${DROP_COUNTS[@]}"; do
        drop_count="${drop_count// /}"
        for task in "${TASKS[@]}"; do
          task="${task// /}"
          submit_eval "$model_tag" "$model_name" "$rank_metric" "$component" "$drop_count" "$task"
        done
      done
    done
  done
done

echo "[DONE] manifest=$manifest"
