#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SCORE_MEM="${SCORE_MEM:-96G}"
PRUNE_MEM="${PRUNE_MEM:-128G}"
EVAL_MEM="${EVAL_MEM:-96G}"
TIME="${TIME:-1-00:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CALIB_FILE="${CALIB_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
NSAMPLES="${NSAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
DTYPE="${DTYPE:-bfloat16}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
STRATEGIES_CSV="${STRATEGIES_CSV:-residual_perp,value_perp}"
TARGETS="${TARGETS:-all}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"
MODEL_SPECS="${MODEL_SPECS:-}"

if [[ -z "$MODEL_SPECS" ]]; then
  echo "[ERROR] MODEL_SPECS is required: tag=/path/to/model;sibling=/path" >&2
  exit 1
fi

CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/compression/outputs/wanda_layerwise_pruned}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/compression/outputs/geo_prune_lm_eval_fast/by_model}"
SCORE_ROOT="${SCORE_ROOT:-$REPO_ROOT/compression/outputs/geometry_scores/by_model}"
MANIFEST="${MANIFEST:-$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/unstructured_perp_cross_model_jobs.tsv}"

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'model_tag\tvariant\tscore_job\tprune_job\teval_job\tcheckpoint\n' > "$MANIFEST"
fi

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

geometry_targets_for_strategy() {
  case "$1" in
    residual_*) echo "o_proj,down_proj" ;;
    value_*) echo "v_proj" ;;
    *) echo "o_proj,down_proj,v_proj" ;;
  esac
}

submit_score() {
  local model_tag="$1"
  local model_path="$2"
  local score_dir="$SCORE_ROOT/$model_tag/weight_error_c4_ns${NSAMPLES}_seq${MAX_LENGTH}"
  local score_path="$score_dir/geometry_scores.pt"
  local slurm_dir="$score_dir/slurm"
  mkdir -p "$slurm_dir"
  if [[ -f "$score_path" ]]; then
    echo ""
    return 0
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="werr-score-${model_tag}" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$SCORE_MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" "$REPO_ROOT/compression/code/geometry_aware_pruning.py" \
  --model_name_or_path "$model_path" \
  --output_dir "$score_dir" \
  --prompts_file "$CALIB_FILE" \
  --max_prompts "$NSAMPLES" \
  --max_length "$MAX_LENGTH" \
  --token_scope last \
  --device cuda \
  --dtype "$DTYPE" \
  --prune_method wanda \
  --geometry_strategy residual_perp \
  --geometry_mode weight_error \
  --geometry_targets o_proj,down_proj,v_proj \
  --sparsity_ratio 0.0 \
  --sparsity_type unstructured \
  --threshold_scope global \
  --local_files_only
EOF
)"
}

submit_prune() {
  local model_tag="$1"
  local model_path="$2"
  local variant="$3"
  local strategy="$4"
  local score_job="$5"
  local score_path="$SCORE_ROOT/$model_tag/weight_error_c4_ns${NSAMPLES}_seq${MAX_LENGTH}/geometry_scores.pt"
  local out_dir="$CKPT_ROOT/$model_tag/$variant"
  local slurm_dir="$out_dir/slurm"
  mkdir -p "$slurm_dir"
  if [[ -f "$out_dir/model/config.json" ]]; then
    echo ""
    return 0
  fi
  local dep_args=()
  if [[ -n "$score_job" ]]; then
    dep_args=(--dependency="afterok:$score_job")
  fi
  local geometry_args=""
  if [[ "$strategy" != "none" ]]; then
    local targets
    targets="$(geometry_targets_for_strategy "$strategy")"
    geometry_args="--geometry_scores '$score_path' --geometry_strategy '$strategy' --geometry_mode weight_error --geometry_targets '$targets'"
  else
    geometry_args="--geometry_strategy none --geometry_mode residual_error --geometry_targets o_proj,down_proj,v_proj"
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dep_args[@]}" \
    --job-name="uprune-${model_tag}-${strategy}" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$PRUNE_MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" "$REPO_ROOT/compression/code/save_wanda_layerwise_pruned_model.py" \
  --model_name_or_path "$model_path" \
  --calib_file "$CALIB_FILE" \
  --output_dir "$out_dir/model" \
  --nsamples "$NSAMPLES" \
  --max_length "$MAX_LENGTH" \
  --sparsity_ratio "$SPARSITY_RATIO" \
  --sparsity_type unstructured \
  --threshold_scope global \
  --targets "$TARGETS" \
  $geometry_args \
  --dtype "$DTYPE" \
  --device cuda \
  --local_files_only
cp "$out_dir/model/wanda_layerwise_summary.json" "$out_dir/summary.json"
cp "$out_dir/model/prune_records.csv" "$out_dir/prune_records.csv"
EOF
)"
}

submit_eval() {
  local model_tag="$1"
  local variant="$2"
  local prune_job="$3"
  local model_path="$CKPT_ROOT/$model_tag/$variant/model"
  local out_base="$EVAL_ROOT/$model_tag/$variant"
  local slurm_dir="$out_base/slurm"
  mkdir -p "$slurm_dir"
  local dep_args=()
  if [[ -n "$prune_job" ]]; then
    dep_args=(--dependency="afterok:$prune_job")
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dep_args[@]}" \
    --job-name="ueval-${model_tag}-${variant}" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EVAL_MEM" \
    --time="$TIME" \
    --output="$slurm_dir/%x-%j.out" \
    --error="$slurm_dir/%x-%j.err" \
    --wrap="$(cat <<EOF
set -u
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
collect_results() {
  if [[ "$AUTO_COLLECT_RESULTS" == "true" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/compression/scripts/collect_wanda_lm_eval_results.py" || true
  fi
}
status=0
tasks=(${TASKS_ORDER_CSV//,/ })
for task in "\${tasks[@]}"; do
  case "\$task" in
    openbookqa|piqa|rte|boolq|drop|bbh_cot_zeroshot) fewshot=0 ;;
    winogrande|mmlu|humaneval|nq_open) fewshot=5 ;;
    arc_challenge) fewshot=25 ;;
    hellaswag) fewshot=10 ;;
    gsm8k_cot) fewshot=8 ;;
    mbpp) fewshot=3 ;;
    *) fewshot=0 ;;
  esac
  out_root="$out_base/lm_eval_fast/\$task"
  mkdir -p "\$out_root"
  if find "\$out_root" -maxdepth 1 -name '*.json' -type f | grep -q .; then
    echo "[TASK_SKIP] variant=$variant task=\$task existing_json=\$out_root"
    collect_results
    continue
  fi
  echo "[TASK_START] variant=$variant task=\$task fewshot=\$fewshot out_root=\$out_root"
  if PYTHON_BIN="$PYTHON_BIN" MODEL_NAME="$model_path" TASKS="\$task" NUM_FEWSHOT="\$fewshot" OUTPUT_ROOT="\$out_root" BATCH_SIZE=auto DTYPE="$DTYPE" APPLY_CHAT_TEMPLATE=false MAX_LENGTH="$EVAL_MAX_LENGTH" LAUNCH_MODE=single TRUST_REMOTE_CODE=true bash lm-evaluation-harness/scripts/run_lm_eval_compression_setting.sh; then
    echo "[TASK_DONE] variant=$variant task=\$task"
    collect_results
  else
    rc=\$?
    echo "[TASK_FAILED] variant=$variant task=\$task rc=\$rc" >&2
    status=\$rc
    collect_results
  fi
done
collect_results
exit \$status
EOF
)"
}

IFS=';' read -r -a model_items <<< "$MODEL_SPECS"
IFS=',' read -r -a strategies <<< "$STRATEGIES_CSV"

for item in "${model_items[@]}"; do
  [[ -z "$item" ]] && continue
  model_tag="${item%%=*}"
  model_path="${item#*=}"
  score_job="$(submit_score "$model_tag" "$model_path")"

  baseline_variant="all_linear_unstructured_s0p5_c4_ns128_seq2048"
  baseline_prune="$(submit_prune "$model_tag" "$model_path" "$baseline_variant" none "")"
  baseline_eval="$(submit_eval "$model_tag" "$baseline_variant" "$baseline_prune")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_tag" "$baseline_variant" "$score_job" "$baseline_prune" "$baseline_eval" "$CKPT_ROOT/$model_tag/$baseline_variant" | tee -a "$MANIFEST"

  for strategy in "${strategies[@]}"; do
    strategy="${strategy// /}"
    variant="wanda_${strategy}_weight_error_s${SPARSITY_RATIO}_unstructured_c4_ns${NSAMPLES}_seq${MAX_LENGTH}"
    prune_job="$(submit_prune "$model_tag" "$model_path" "$variant" "$strategy" "$score_job")"
    eval_job="$(submit_eval "$model_tag" "$variant" "$prune_job")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_tag" "$variant" "$score_job" "$prune_job" "$eval_job" "$CKPT_ROOT/$model_tag/$variant" | tee -a "$MANIFEST"
  done
done

echo "[DONE] manifest=$MANIFEST"
