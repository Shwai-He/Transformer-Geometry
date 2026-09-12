#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-beacon}"
QOS="${QOS:-medium}"
GRES="${GRES:-gpu:nvidia_l40s:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PRUNE_MEM="${PRUNE_MEM:-64G}"
EVAL_MEM="${EVAL_MEM:-64G}"
TIME="${TIME:-1-00:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CALIB_FILE="${CALIB_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
NSAMPLES="${NSAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
SPARSITY_TYPES_CSV="${SPARSITY_TYPES_CSV:-2:4,4:8}"
SPARSITY_AXIS="${SPARSITY_AXIS:-input}"
STRATEGIES_CSV="${STRATEGIES_CSV:-residual_perp,value_perp}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
MODEL_SPECS="${MODEL_SPECS:-qwen3_1p7b=Qwen/Qwen3-1.7B;qwen3_4b=Qwen/Qwen3-4B}"
START_DEPENDENCY="${START_DEPENDENCY:-}"
VARIANT_TAG_EXTRA="${VARIANT_TAG_EXTRA:-_exact}"
GEOMETRY_EXACT_OBJECTIVE="${GEOMETRY_EXACT_OBJECTIVE:-absolute}"
GEOMETRY_EXACT_MARGIN_LAMBDA="${GEOMETRY_EXACT_MARGIN_LAMBDA:-1.0}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"

PRUNE_ROOT="${PRUNE_ROOT:-$REPO_ROOT/compression/outputs/geo_prune/by_model}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/compression/outputs/geo_prune_lm_eval_fast/by_model}"
MANIFEST="${MANIFEST:-$REPO_ROOT/compression/outputs/geo_prune/structured_exact_projection_jobs.tsv}"

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'model_tag\tstrategy\tsparsity_type\tprune_job\teval_job\tcheckpoint\n' > "$MANIFEST"
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

submit_prune() {
  local model_tag="$1"
  local model_path="$2"
  local strategy="$3"
  local sparsity_type="$4"
  local axis_tag=""
  if [[ "$SPARSITY_AXIS" != "input" ]]; then
    axis_tag="_axis-${SPARSITY_AXIS}"
  fi
  local variant="wanda_${strategy}${VARIANT_TAG_EXTRA}${axis_tag}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
  local out_dir="$PRUNE_ROOT/$model_tag/$variant"
  local model_out="$out_dir/model"
  local slurm_dir="$out_dir/slurm"
  mkdir -p "$slurm_dir"
  if [[ -f "$model_out/config.json" ]]; then
    echo ""
    return 0
  fi
  local dep_args=()
  if [[ -n "$START_DEPENDENCY" ]]; then
    dep_args=(--dependency="$START_DEPENDENCY")
  fi
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dep_args[@]}" \
    --job-name="exact-prune-${model_tag}-${strategy}-${sparsity_type//:/to}" \
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
  --output_dir "$model_out" \
  --nsamples "$NSAMPLES" \
  --max_length "$MAX_LENGTH" \
  --sparsity_ratio "$SPARSITY_RATIO" \
  --sparsity_type "$sparsity_type" \
  --sparsity_axis "$SPARSITY_AXIS" \
  --threshold_scope global \
  --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --geometry_strategy "$strategy" \
  --geometry_mode exact_projection \
  --geometry_exact_objective "$GEOMETRY_EXACT_OBJECTIVE" \
  --geometry_exact_margin_lambda "$GEOMETRY_EXACT_MARGIN_LAMBDA" \
  --geometry_targets o_proj,down_proj,v_proj \
  --dtype "$DTYPE" \
  --device cuda \
  --local_files_only
cp "$model_out/wanda_layerwise_summary.json" "$out_dir/summary.json"
cp "$model_out/prune_records.csv" "$out_dir/prune_records.csv"
EOF
)"
}

submit_eval() {
  local model_tag="$1"
  local strategy="$2"
  local sparsity_type="$3"
  local prune_job="$4"
  local axis_tag=""
  if [[ "$SPARSITY_AXIS" != "input" ]]; then
    axis_tag="_axis-${SPARSITY_AXIS}"
  fi
  local variant="wanda_${strategy}${VARIANT_TAG_EXTRA}${axis_tag}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
  local model_path="$PRUNE_ROOT/$model_tag/$variant/model"
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
    --job-name="exact-eval-${model_tag}-${strategy}-${sparsity_type//:/to}" \
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
  if PYTHON_BIN="$PYTHON_BIN" MODEL_NAME="$model_path" TASKS="\$task" NUM_FEWSHOT="\$fewshot" OUTPUT_ROOT="\$out_root" BATCH_SIZE="$BATCH_SIZE" DTYPE="$DTYPE" APPLY_CHAT_TEMPLATE=false MAX_LENGTH="$EVAL_MAX_LENGTH" LAUNCH_MODE=single TRUST_REMOTE_CODE=true bash evaluation/scripts/run_lm_eval_compression_setting.sh; then
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
IFS=',' read -r -a sparsity_types <<< "$SPARSITY_TYPES_CSV"

for item in "${model_items[@]}"; do
  [[ -z "$item" ]] && continue
  model_tag="${item%%=*}"
  model_path="${item#*=}"
  for strategy in "${strategies[@]}"; do
    strategy="${strategy// /}"
    for sparsity_type in "${sparsity_types[@]}"; do
      sparsity_type="${sparsity_type// /}"
      prune_job="$(submit_prune "$model_tag" "$model_path" "$strategy" "$sparsity_type")"
      eval_job="$(submit_eval "$model_tag" "$strategy" "$sparsity_type" "$prune_job")"
      axis_tag=""
      if [[ "$SPARSITY_AXIS" != "input" ]]; then
        axis_tag="_axis-${SPARSITY_AXIS}"
      fi
      variant="wanda_${strategy}${VARIANT_TAG_EXTRA}${axis_tag}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_tag" "$strategy" "$sparsity_type" "$prune_job" "$eval_job" "$PRUNE_ROOT/$model_tag/$variant" | tee -a "$MANIFEST"
    done
  done
done

echo "[DONE] manifest=$MANIFEST"
