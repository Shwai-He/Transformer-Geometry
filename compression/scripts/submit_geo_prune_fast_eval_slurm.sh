#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-08:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"
PRUNE_ROOT="${PRUNE_ROOT:-$REPO_ROOT/compression/outputs/geo_prune/by_model/$MODEL_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/geo_prune_lm_eval_fast/by_model/$MODEL_TAG}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
STRATEGIES_CSV="${STRATEGIES_CSV:-residual_perp,value_perp}"
SPARSITY_TYPES_CSV="${SPARSITY_TYPES_CSV:-unstructured,2:4}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
VARIANT_TAG_EXTRA="${VARIANT_TAG_EXTRA:-}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"
DEPENDENCY="${DEPENDENCY:-}"

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

submit_variant() {
  local strategy="$1"
  local sparsity_type="$2"
  local variant="wanda_${strategy}${VARIANT_TAG_EXTRA}_s${SPARSITY_RATIO}_${sparsity_type//:/to}"
  local model_path="$PRUNE_ROOT/$variant/model"
  local out_base="$OUTPUT_ROOT/$variant"
  local slurm_dir="$out_base/slurm"
  mkdir -p "$slurm_dir"
  if [[ ! -f "$model_path/config.json" && -z "$DEPENDENCY" ]]; then
    echo "[SKIP] missing model: $model_path" >&2
    return 0
  fi
  local dependency_args=()
  if [[ -n "$DEPENDENCY" ]]; then
    dependency_args=(--dependency="$DEPENDENCY")
  fi

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${dependency_args[@]}" \
    --job-name="geoeval-${MODEL_TAG}-${strategy}-${sparsity_type//:/to}" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
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
  if PYTHON_BIN="$PYTHON_BIN" MODEL_NAME="$model_path" TASKS="\$task" NUM_FEWSHOT="\$fewshot" OUTPUT_ROOT="\$out_root" BATCH_SIZE="$BATCH_SIZE" DTYPE="$DTYPE" APPLY_CHAT_TEMPLATE=false MAX_LENGTH="$MAX_LENGTH" LAUNCH_MODE=single TRUST_REMOTE_CODE=true bash lm-evaluation-harness/scripts/run_lm_eval_compression_setting.sh; then
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

IFS=',' read -r -a strategies <<< "$STRATEGIES_CSV"
IFS=',' read -r -a sparsity_types <<< "$SPARSITY_TYPES_CSV"
for strategy in "${strategies[@]}"; do
  strategy="${strategy// /}"
  for sparsity_type in "${sparsity_types[@]}"; do
    sparsity_type="${sparsity_type// /}"
    submit_variant "$strategy" "$sparsity_type"
  done
done
