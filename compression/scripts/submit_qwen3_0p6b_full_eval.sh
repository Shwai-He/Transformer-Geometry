#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
MEM="${MEM:-64G}"
TIME="${TIME:-1-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DEPENDENCY="${DEPENDENCY:-}"
AUTO_COLLECT_RESULTS="${AUTO_COLLECT_RESULTS:-true}"
SUBMIT_TAGS="${SUBMIT_TAGS:-dense,s0p5,2to4,4to8}"
TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"

MODEL_TAG="${MODEL_TAG:-qwen3_0p6b}"
DENSE_MODEL="${DENSE_MODEL:-Qwen/Qwen3-0.6B-Base}"
ROOT="${ROOT:-$REPO_ROOT/compression/outputs/wanda_layerwise_pruned/$MODEL_TAG}"

IFS=',' read -r -a TASKS_ORDER <<< "$TASKS_ORDER_CSV"

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

submit_eval_job() {
  local tag="$1"
  local model_name="$2"
  local out_base="$3"
  local dependency="${4:-}"
  local slurm_dir="$out_base/lm_eval_full/slurm"
  mkdir -p "$slurm_dir"

  local export_dependency=()
  if [[ -n "$dependency" ]]; then
    export_dependency=(--dependency="$dependency")
  fi

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    "${export_dependency[@]}" \
    --job-name="eval-${MODEL_TAG}-${tag}-full" \
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
tasks=(${TASKS_ORDER[*]})
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
  out_root="$out_base/lm_eval_full/\$task"
  mkdir -p "\$out_root"
  if find "\$out_root" -maxdepth 1 -name '*.json' -type f | grep -q .; then
    echo "[TASK_SKIP] tag=$tag task=\$task existing_json=\$out_root"
    collect_results
    continue
  fi
  echo "[TASK_START] tag=$tag task=\$task fewshot=\$fewshot out_root=\$out_root"
  if PYTHON_BIN="$PYTHON_BIN" MODEL_NAME="$model_name" TASKS="\$task" NUM_FEWSHOT="\$fewshot" OUTPUT_ROOT="\$out_root" BATCH_SIZE="$BATCH_SIZE" DTYPE="$DTYPE" APPLY_CHAT_TEMPLATE=false MAX_LENGTH="$MAX_LENGTH" LAUNCH_MODE=single TRUST_REMOTE_CODE=true bash evaluation/scripts/run_lm_eval_compression_setting.sh; then
    echo "[TASK_DONE] tag=$tag task=\$task"
    collect_results
  else
    rc=\$?
    echo "[TASK_FAILED] tag=$tag task=\$task rc=\$rc" >&2
    status=\$rc
    collect_results
  fi
done
collect_results
exit \$status
EOF
)"
}

should_submit_tag() {
  local tag="$1"
  case ",$SUBMIT_TAGS," in
    *",$tag,"*) return 0 ;;
    *) return 1 ;;
  esac
}

if should_submit_tag dense; then
  submit_eval_job "dense" "$DENSE_MODEL" "$ROOT/dense" "$DEPENDENCY"
fi
if should_submit_tag s0p5; then
  submit_eval_job "s0p5" "$ROOT/all_linear_unstructured_s0p5_c4_ns128_seq2048" "$ROOT/all_linear_unstructured_s0p5_c4_ns128_seq2048" "$DEPENDENCY"
fi
if should_submit_tag s0p3; then
  submit_eval_job "s0p3" "$ROOT/all_linear_unstructured_s0p3_c4_ns128_seq2048" "$ROOT/all_linear_unstructured_s0p3_c4_ns128_seq2048" "${DEPENDENCY_S0P3:-$DEPENDENCY}"
fi
if should_submit_tag s0p7; then
  submit_eval_job "s0p7" "$ROOT/all_linear_unstructured_s0p7_c4_ns128_seq2048" "$ROOT/all_linear_unstructured_s0p7_c4_ns128_seq2048" "${DEPENDENCY_S0P7:-$DEPENDENCY}"
fi
if should_submit_tag 2to4; then
  submit_eval_job "2to4" "$ROOT/all_linear_2to4_s0p5_c4_ns128_seq2048" "$ROOT/all_linear_2to4_s0p5_c4_ns128_seq2048" "${DEPENDENCY_2TO4:-$DEPENDENCY}"
fi
if should_submit_tag 4to8; then
  submit_eval_job "4to8" "$ROOT/all_linear_4to8_s0p5_c4_ns128_seq2048" "$ROOT/all_linear_4to8_s0p5_c4_ns128_seq2048" "${DEPENDENCY_4TO8:-$DEPENDENCY}"
fi
