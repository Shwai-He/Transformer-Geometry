#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PRUNE_MEM="${PRUNE_MEM:-96G}"
EVAL_MEM="${EVAL_MEM:-64G}"
TIME="${TIME:-08:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/grad_geo_prune/by_model/$MODEL_TAG}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DTYPE="${DTYPE:-bfloat16}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
SPARSITY_TYPES_CSV="${SPARSITY_TYPES_CSV:-unstructured,2:4}"
GRAD_SETTINGS_CSV="${GRAD_SETTINGS_CSV:-residual:total,residual:perp,value:total,value:perp}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
GRAD_SCORE="${GRAD_SCORE:-snip}"
THRESHOLD_SCOPE="${THRESHOLD_SCOPE:-global}"
PRUNE_TARGETS="${PRUNE_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
VARIANT_TAG_EXTRA="${VARIANT_TAG_EXTRA:-}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"

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
  local grad_space="$1"
  local grad_component="$2"
  local sparsity_type="$3"
  local tag="grad_${grad_space}_${grad_component}_${GRAD_SCORE}${VARIANT_TAG_EXTRA}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
  local out_dir="$OUTPUT_ROOT/$tag"
  local slurm_dir="$out_dir/slurm"
  mkdir -p "$slurm_dir"
  if [[ -f "$out_dir/model/config.json" ]]; then
    echo "[SKIP] existing grad-pruned model: $out_dir" >&2
    echo ""
    return 0
  fi

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="gprune-${MODEL_TAG}-${grad_space}-${grad_component}-${sparsity_type//:/to}" \
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
"$PYTHON_BIN" "$REPO_ROOT/compression/code/gradient_geometry_pruning.py" \
  --model_name_or_path "$MODEL_PATH" \
  --output_dir "$out_dir" \
  --prompts_file "$PROMPTS_FILE" \
  --max_prompts "$MAX_PROMPTS" \
  --max_length "$MAX_LENGTH" \
  --device cuda \
  --dtype "$DTYPE" \
  --grad_space "$grad_space" \
  --grad_component "$grad_component" \
  --grad_score "$GRAD_SCORE" \
  --prune_targets "$PRUNE_TARGETS" \
  --sparsity_ratio "$SPARSITY_RATIO" \
  --sparsity_type "$sparsity_type" \
  --threshold_scope "$THRESHOLD_SCOPE" \
  --local_files_only \
  --save_model
EOF
)"
}

submit_eval() {
  local grad_space="$1"
  local grad_component="$2"
  local sparsity_type="$3"
  local dependency="$4"
  local tag="grad_${grad_space}_${grad_component}_${GRAD_SCORE}${VARIANT_TAG_EXTRA}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
  local model_path="$OUTPUT_ROOT/$tag/model"
  local out_base="$OUTPUT_ROOT/$tag/lm_eval_fast"
  local slurm_dir="$OUTPUT_ROOT/$tag/slurm"
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
    --job-name="geval-${MODEL_TAG}-${grad_space}-${grad_component}-${sparsity_type//:/to}" \
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
  out_root="$out_base/\$task"
  mkdir -p "\$out_root"
  if find "\$out_root" -maxdepth 1 -name '*.json' -type f | grep -q .; then
    echo "[TASK_SKIP] variant=$tag task=\$task existing_json=\$out_root"
    collect_results
    continue
  fi
  echo "[TASK_START] variant=$tag task=\$task fewshot=\$fewshot out_root=\$out_root"
  if PYTHON_BIN="$PYTHON_BIN" MODEL_NAME="$model_path" TASKS="\$task" NUM_FEWSHOT="\$fewshot" OUTPUT_ROOT="\$out_root" BATCH_SIZE=auto DTYPE="$DTYPE" APPLY_CHAT_TEMPLATE=false MAX_LENGTH=4096 LAUNCH_MODE=single TRUST_REMOTE_CODE=true bash lm-evaluation-harness/scripts/run_lm_eval_compression_setting.sh; then
    echo "[TASK_DONE] variant=$tag task=\$task"
    collect_results
  else
    rc=\$?
    echo "[TASK_FAILED] variant=$tag task=\$task rc=\$rc" >&2
    status=\$rc
    collect_results
  fi
done
collect_results
exit \$status
EOF
)"
}

IFS=',' read -r -a grad_settings <<< "$GRAD_SETTINGS_CSV"
IFS=',' read -r -a sparsity_types <<< "$SPARSITY_TYPES_CSV"

mkdir -p "$OUTPUT_ROOT"
manifest="$OUTPUT_ROOT/grad_prune_fast_jobs.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'grad_space\tgrad_component\tsparsity_type\tprune_job\teval_job\n' > "$manifest"
fi

for setting in "${grad_settings[@]}"; do
  setting="${setting// /}"
  grad_space="${setting%%:*}"
  grad_component="${setting#*:}"
  for sparsity_type in "${sparsity_types[@]}"; do
    sparsity_type="${sparsity_type// /}"
    prune_job="$(submit_prune "$grad_space" "$grad_component" "$sparsity_type")"
    eval_job="$(submit_eval "$grad_space" "$grad_component" "$sparsity_type" "$prune_job")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$grad_space" "$grad_component" "$sparsity_type" "$prune_job" "$eval_job" >> "$manifest"
  done
done

echo "[DONE] manifest=$manifest"
