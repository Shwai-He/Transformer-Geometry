#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_l40s:1}"
MEM="${MEM:-96G}"
TIME="${TIME:-8:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
SELECT_PROMPTS_FILE="${SELECT_PROMPTS_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
SELECT_MAX_PROMPTS="${SELECT_MAX_PROMPTS:-32}"
SELECT_MAX_LENGTH="${SELECT_MAX_LENGTH:-2048}"
TOKEN_SCOPE="${TOKEN_SCOPE:-all}"
if [[ -z "${TOKEN_SCOPE_TAG:-}" ]]; then
  if [[ "$TOKEN_SCOPE" == "all" ]]; then
    TOKEN_SCOPE_TAG="alltok"
  else
    TOKEN_SCOPE_TAG="$TOKEN_SCOPE"
  fi
fi
DROP_COUNTS="${DROP_COUNTS:-4,8}"
EVAL_DROP_COUNTS="${EVAL_DROP_COUNTS:-8}"
RANK_METRICS="${RANK_METRICS:-joint_perp_over_final,joint_perp_ratio_final}"
MIN_GAP="${MIN_GAP:-0}"
COMPONENTS="${COMPONENTS:-attn,mlp}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_lm_eval}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_geometry}"
MODEL_SPECS="${MODEL_SPECS:-qwen3_0p6b=/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/compression/outputs/layer_drop_joint_residual/slurm"
manifest="$OUTPUT_ROOT/submission_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'job_id\tjob_type\tmodel_tag\trank_metric\tcomponent\tdrop_count\ttask\toutput_root\tselection_json\n' > "$manifest"
fi

job_id="$(
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="ld-joint-resid" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$REPO_ROOT/compression/outputs/layer_drop_joint_residual/slurm/%x-%j.out" \
    --error="$REPO_ROOT/compression/outputs/layer_drop_joint_residual/slurm/%x-%j.err" \
    --wrap="$(cat <<EOF
set -u
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

fewshot_for_task() {
  case "\$1" in
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

IFS=';' read -r -a MODEL_ITEMS <<< "$MODEL_SPECS"
IFS=',' read -r -a METRICS <<< "$RANK_METRICS"
IFS=',' read -r -a COMPONENT_ITEMS <<< "$COMPONENTS"
IFS=',' read -r -a DROP_ITEMS <<< "$EVAL_DROP_COUNTS"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

status=0
for item in "\${MODEL_ITEMS[@]}"; do
  model_tag="\${item%%=*}"
  model_name="\${item#*=}"
  for metric in "\${METRICS[@]}"; do
    metric="\${metric// /}"
    output_tag="\${metric}_${TOKEN_SCOPE_TAG}_mp${SELECT_MAX_PROMPTS}"
    if [[ "$MIN_GAP" != "0" ]]; then
      output_tag="\${output_tag}_gap${MIN_GAP}"
    fi
    selection_json="$SELECTION_ROOT/\$model_tag/\$output_tag/drop_selection.json"
    if [[ ! -f "\$selection_json" ]]; then
      echo "[SELECT_START] model=\$model_tag metric=\$metric output_tag=\$output_tag"
      if "$PYTHON_BIN" "$REPO_ROOT/compression/code/select_layer_drop_joint_residual.py" \\
        --model_name "\$model_name" \\
        --model_tag "\$model_tag" \\
        --output_tag "\$output_tag" \\
        --prompts_file "$SELECT_PROMPTS_FILE" \\
        --max_prompts "$SELECT_MAX_PROMPTS" \\
        --max_length "$SELECT_MAX_LENGTH" \\
        --token_scope "$TOKEN_SCOPE" \\
        --drop_counts "$DROP_COUNTS" \\
        --rank_metric "\$metric" \\
        --min_gap "$MIN_GAP" \\
        --device cuda \\
        --metric_device cuda \\
        --dtype "$DTYPE" \\
        --local_files_only; then
        :
      else
        rc=\$?
        echo "[SELECT_FAILED] model=\$model_tag metric=\$metric rc=\$rc" >&2
        status=\$rc
        continue
      fi
    else
      echo "[SELECT_SKIP] existing \$selection_json"
    fi

    for component in "\${COMPONENT_ITEMS[@]}"; do
      component="\${component// /}"
      for drop_count in "\${DROP_ITEMS[@]}"; do
        drop_count="\${drop_count// /}"
        setting="\${output_tag}_\${component}_drop\${drop_count}"
        for task in "\${TASKS[@]}"; do
          task="\${task// /}"
          fewshot="\$(fewshot_for_task "\$task")"
          task_root="$OUTPUT_ROOT/\$model_tag/\$setting/\$task"
          mkdir -p "\$task_root"
          if find "\$task_root" -maxdepth 1 -name '*.json' -type f | grep -q .; then
            echo "[TASK_SKIP] model=\$model_tag setting=\$setting task=\$task"
            continue
          fi
          echo "[TASK_START] model=\$model_tag setting=\$setting task=\$task fewshot=\$fewshot"
          if PYTHON_BIN="$PYTHON_BIN" \\
            MODEL_NAME="\$model_name" \\
            TASKS="\$task" \\
            OUTPUT_ROOT="\$task_root" \\
            BATCH_SIZE="$BATCH_SIZE" \\
            DTYPE="$DTYPE" \\
            APPLY_CHAT_TEMPLATE="$APPLY_CHAT_TEMPLATE" \\
            NUM_FEWSHOT="\$fewshot" \\
            MAX_LENGTH="$EVAL_MAX_LENGTH" \\
            TRUST_REMOTE_CODE=true \\
            LOG_SAMPLES="$LOG_SAMPLES" \\
            AUTO_COLLECT_RESULTS=true \\
            LAYER_DROP_CONFIG="\$selection_json" \\
            LAYER_DROP_COMPONENT="\$component" \\
            LAYER_DROP_COUNT="\$drop_count" \\
            bash "$REPO_ROOT/lm-evaluation-harness/scripts/run_lm_eval_layer_drop_setting.sh"; then
            echo "[TASK_DONE] model=\$model_tag setting=\$setting task=\$task"
          else
            rc=\$?
            echo "[TASK_FAILED] model=\$model_tag setting=\$setting task=\$task rc=\$rc" >&2
            status=\$rc
          fi
        done
        "$PYTHON_BIN" "$REPO_ROOT/compression/scripts/collect_layer_drop_lm_eval_results.py" || true
      done
    done
  done
done

"$PYTHON_BIN" "$REPO_ROOT/compression/scripts/collect_layer_drop_lm_eval_results.py" || true
exit \$status
EOF
)"
)"

printf '%s\tjoint_residual\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$job_id" "$MODEL_SPECS" "$RANK_METRICS" "$COMPONENTS" "$EVAL_DROP_COUNTS" "$TASKS_CSV" "$OUTPUT_ROOT" "$SELECTION_ROOT" >> "$manifest"

echo "[SUBMITTED] job_id=$job_id"
echo "[LOG] $REPO_ROOT/compression/outputs/layer_drop_joint_residual/slurm/ld-joint-resid-$job_id.out"
