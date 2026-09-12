#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SCORE_MEM="${SCORE_MEM:-64G}"
PRUNE_MEM="${PRUNE_MEM:-96G}"
EVAL_MEM="${EVAL_MEM:-64G}"
TIME="${TIME:-1-00:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
CALIB_FILE="${CALIB_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
NSAMPLES="${NSAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DTYPE="${DTYPE:-bfloat16}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
SPARSITY_TYPE="${SPARSITY_TYPE:-unstructured}"
THRESHOLD_SCOPE="${THRESHOLD_SCOPE:-global}"
STRATEGIES_CSV="${STRATEGIES_CSV:-residual_para,residual_perp,value_para,value_perp}"
TARGETS="${TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"
SCORE_JOB_OVERRIDE="${SCORE_JOB_OVERRIDE:-}"
SCORE_DEPENDENCY="${SCORE_DEPENDENCY:-}"

SCORE_ROOT="${SCORE_ROOT:-$REPO_ROOT/compression/outputs/geometry_scores/by_model/$MODEL_TAG}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/$MODEL_TAG}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/compression/outputs/geo_prune_lm_eval_fast/by_model/$MODEL_TAG}"
SCORE_DIR="$SCORE_ROOT/weight_error_c4_ns${NSAMPLES}_seq${MAX_LENGTH}"
SCORE_PATH="$SCORE_DIR/geometry_scores.pt"
MANIFEST="$CKPT_ROOT/weight_error_wanda_fast_jobs.tsv"

mkdir -p "$SCORE_DIR" "$CKPT_ROOT" "$EVAL_ROOT"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'model\tstrategy\tsparsity_type\tscore_job\tprune_job\teval_job\tcheckpoint\n' > "$MANIFEST"
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
  if [[ -n "$SCORE_JOB_OVERRIDE" ]]; then
    echo "$SCORE_JOB_OVERRIDE"
    return 0
  fi
  local slurm_dir="$SCORE_DIR/slurm"
  mkdir -p "$slurm_dir"
  if [[ -f "$SCORE_PATH" ]]; then
    echo ""
    return 0
  fi
  local dependency_args=()
  if [[ -n "$SCORE_DEPENDENCY" ]]; then
    dependency_args=(--dependency="$SCORE_DEPENDENCY")
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dependency_args[@]}" \
    --job-name="werr-score-${MODEL_TAG}" \
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
  --model_name_or_path "$MODEL_PATH" \
  --output_dir "$SCORE_DIR" \
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
  --threshold_scope "$THRESHOLD_SCOPE" \
  --local_files_only
EOF
)"
}

submit_prune() {
  local strategy="$1"
  local dependency="$2"
  local targets
  targets="$(geometry_targets_for_strategy "$strategy")"
  local variant="wanda_${strategy}_weight_error_s${SPARSITY_RATIO}_${SPARSITY_TYPE//:/to}_c4_ns${NSAMPLES}_seq${MAX_LENGTH}"
  local out_dir="$CKPT_ROOT/$variant"
  local slurm_dir="$out_dir/slurm"
  mkdir -p "$slurm_dir"
  if [[ -f "$out_dir/model/config.json" ]]; then
    echo ""
    return 0
  fi
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:$dependency")
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dependency_args[@]}" \
    --job-name="werr-prune-${MODEL_TAG}-${strategy}" \
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
  --model_name_or_path "$MODEL_PATH" \
  --calib_file "$CALIB_FILE" \
  --output_dir "$out_dir/model" \
  --nsamples "$NSAMPLES" \
  --max_length "$MAX_LENGTH" \
  --sparsity_ratio "$SPARSITY_RATIO" \
  --sparsity_type "$SPARSITY_TYPE" \
  --threshold_scope "$THRESHOLD_SCOPE" \
  --targets "$TARGETS" \
  --geometry_scores "$SCORE_PATH" \
  --geometry_strategy "$strategy" \
  --geometry_mode weight_error \
  --geometry_targets "$targets" \
  --dtype "$DTYPE" \
  --device cuda \
  --local_files_only
cp "$out_dir/model/wanda_layerwise_summary.json" "$out_dir/summary.json"
cp "$out_dir/model/prune_records.csv" "$out_dir/prune_records.csv"
EOF
)"
}

submit_eval() {
  local strategy="$1"
  local dependency="$2"
  local variant="wanda_${strategy}_weight_error_s${SPARSITY_RATIO}_${SPARSITY_TYPE//:/to}_c4_ns${NSAMPLES}_seq${MAX_LENGTH}"
  local model_path="$CKPT_ROOT/$variant/model"
  local out_base="$EVAL_ROOT/$variant"
  local slurm_dir="$out_base/slurm"
  mkdir -p "$slurm_dir"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:$dependency")
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dependency_args[@]}" \
    --job-name="werr-eval-${MODEL_TAG}-${strategy}" \
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
  if PYTHON_BIN="$PYTHON_BIN" MODEL_NAME="$model_path" TASKS="\$task" NUM_FEWSHOT="\$fewshot" OUTPUT_ROOT="\$out_root" BATCH_SIZE="$BATCH_SIZE" DTYPE="$DTYPE" APPLY_CHAT_TEMPLATE=false MAX_LENGTH="$EVAL_MAX_LENGTH" LAUNCH_MODE=single TRUST_REMOTE_CODE=true bash lm-evaluation-harness/scripts/run_lm_eval_compression_setting.sh; then
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

score_job="$(submit_score)"
IFS=',' read -r -a strategies <<< "$STRATEGIES_CSV"
for strategy in "${strategies[@]}"; do
  strategy="${strategy// /}"
  prune_job="$(submit_prune "$strategy" "$score_job")"
  eval_job="$(submit_eval "$strategy" "$prune_job")"
  variant="wanda_${strategy}_weight_error_s${SPARSITY_RATIO}_${SPARSITY_TYPE//:/to}_c4_ns${NSAMPLES}_seq${MAX_LENGTH}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MODEL_TAG" "$strategy" "$SPARSITY_TYPE" "$score_job" "$prune_job" "$eval_job" "$CKPT_ROOT/$variant" >> "$MANIFEST"
done

echo "[DONE] manifest=$MANIFEST"
