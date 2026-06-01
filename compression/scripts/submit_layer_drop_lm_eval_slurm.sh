#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
SELECT_PARTITION="${SELECT_PARTITION:-$PARTITION}"
SELECT_QOS="${SELECT_QOS:-$QOS}"
EVAL_PARTITION="${EVAL_PARTITION:-$PARTITION}"
EVAL_QOS="${EVAL_QOS:-$QOS}"
GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
SELECT_GRES="${SELECT_GRES:-$GRES}"
EVAL_GRES="${EVAL_GRES:-$GRES}"
SELECT_MEM="${SELECT_MEM:-96G}"
EVAL_MEM="${EVAL_MEM:-96G}"
TIME="${TIME:-08:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"
TOKEN_SCOPE="${TOKEN_SCOPE:-all}"
if [[ -z "${METHOD_TAG_SUFFIX+x}" ]]; then
  if [[ "$TOKEN_SCOPE" == "all" ]]; then
    METHOD_TAG_SUFFIX="_alltok"
  else
    METHOD_TAG_SUFFIX=""
  fi
fi
PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
DROP_COUNTS_CSV="${DROP_COUNTS_CSV:-4,8}"
RANK_METRICS_CSV="${RANK_METRICS_CSV:-perp_ratio,perp_over_ref,one_minus_cosine}"
COMPONENTS_CSV="${COMPONENTS_CSV:-attn}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_lm_eval}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_geometry}"

MODEL_SPECS="${MODEL_SPECS:-qwen3_0p6b=/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd;qwen3_1p7b=/beacon-projects/traumallm/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

mkdir -p "$OUTPUT_ROOT" "$SELECTION_ROOT"
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

submit_selector() {
  local model_tag="$1"
  local model_name="$2"
  local rank_metric="$3"
  local selector_tag="${rank_metric}${METHOD_TAG_SUFFIX}"
  local out_dir="$SELECTION_ROOT/$model_tag/$selector_tag"
  local slurm_dir="$out_dir/slurm"
  mkdir -p "$slurm_dir"
  sbatch \
    --parsable \
    --partition="$SELECT_PARTITION" \
    --qos="$SELECT_QOS" \
    --job-name="sel-drop-${model_tag}-${selector_tag}" \
    --gres="$SELECT_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$SELECT_MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=$([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo 1 || echo 0)
export TRANSFORMERS_OFFLINE=$([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo 1 || echo 0)
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" "$REPO_ROOT/compression/code/select_layer_drop_by_geometry.py" \
  --model_name "$model_name" \
  --model_tag "$model_tag" \
  --output_tag "$selector_tag" \
  --prompts_file "$PROMPTS_FILE" \
  --max_prompts "$MAX_PROMPTS" \
  --max_length "$MAX_LENGTH" \
  --token_scope "$TOKEN_SCOPE" \
  --drop_counts "$DROP_COUNTS_CSV" \
  --rank_metric "$rank_metric" \
  --rank_order ascending \
  --device cuda \
  --dtype "$DTYPE" \
  --output_dir "$SELECTION_ROOT/$model_tag" \
  $([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo --local_files_only || echo --no-local_files_only)
EOF
)"
}

submit_eval() {
  local model_tag="$1"
  local model_name="$2"
  local rank_metric="$3"
  local selector_job="$4"
  local component="$5"
  local drop_count="$6"
  local task="$7"
  local selector_tag="${rank_metric}${METHOD_TAG_SUFFIX}"
  local fewshot
  fewshot="$(fewshot_for_task "$task")"

  local setting="${selector_tag}_${component}_drop${drop_count}"
  local out_root="$OUTPUT_ROOT/$model_tag/$setting/$task"
  local slurm_dir="$out_root/slurm"
  local selection_json="$SELECTION_ROOT/$model_tag/$selector_tag/drop_selection.json"
  mkdir -p "$slurm_dir"

  sbatch \
    --parsable \
    --partition="$EVAL_PARTITION" \
    --qos="$EVAL_QOS" \
    --dependency="afterok:$selector_job" \
    --job-name="ld-${model_tag}-${component}${drop_count}-${task}" \
    --gres="$EVAL_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EVAL_MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=$([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo 1 || echo 0)
export TRANSFORMERS_OFFLINE=$([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo 1 || echo 0)
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
    selector_tag="${rank_metric}${METHOD_TAG_SUFFIX}"
    selector_job="$(submit_selector "$model_tag" "$model_name" "$rank_metric")"
    selection_json="$SELECTION_ROOT/$model_tag/$selector_tag/drop_selection.json"
    printf '%s\tselector\t%s\t%s\t\t\t\t%s\t%s\n' "$selector_job" "$model_tag" "$selector_tag" "$SELECTION_ROOT/$model_tag/$selector_tag" "$selection_json" >> "$manifest"
    for component in "${COMPONENTS[@]}"; do
      component="${component// /}"
      for drop_count in "${DROP_COUNTS[@]}"; do
        drop_count="${drop_count// /}"
        for task in "${TASKS[@]}"; do
          task="${task// /}"
          eval_job="$(submit_eval "$model_tag" "$model_name" "$rank_metric" "$selector_job" "$component" "$drop_count" "$task")"
          out_root="$OUTPUT_ROOT/$model_tag/${selector_tag}_${component}_drop${drop_count}/$task"
          printf '%s\teval\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$eval_job" "$model_tag" "$selector_tag" "$component" "$drop_count" "$task" "$out_root" "$selection_json" >> "$manifest"
        done
      done
    done
  done
done

echo "[DONE] manifest=$manifest"
