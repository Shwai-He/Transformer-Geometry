#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_l40s:1}"
MEM="${MEM:-96G}"
TIME="${TIME:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
NODELIST="${NODELIST:-}"
EXCLUDE="${EXCLUDE:-}"
DEPENDENCY="${DEPENDENCY:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_lm_eval}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_geometry}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
SETTINGS_CSV="${SETTINGS_CSV:-perp_ratio_alltok|both|4,one_minus_cosine_alltok|both|4,hybrid_perp_ratio_alltok_gap1|attn|8,hybrid_perp_ratio_alltok_gap1|mlp|8,hybrid_one_minus_cosine_alltok_gap1|attn|8,hybrid_one_minus_cosine_alltok_gap1|mlp|8}"
MODEL_SPECS="${MODEL_SPECS:-qwen3_0p6b=Qwen/Qwen3-0.6B-Base;qwen3_1p7b=Qwen/Qwen3-1.7B}"

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/compression/outputs/layer_drop_high_compression/slurm"
manifest="$OUTPUT_ROOT/submission_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'job_id\tjob_type\tmodel_tag\trank_metric\tcomponent\tdrop_count\ttask\toutput_root\tselection_json\n' > "$manifest"
fi

job_id="$(
  extra_sbatch_args=()
  if [[ -n "$NODELIST" ]]; then
    extra_sbatch_args+=(--nodelist="$NODELIST")
  fi
  if [[ -n "$EXCLUDE" ]]; then
    extra_sbatch_args+=(--exclude="$EXCLUDE")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    extra_sbatch_args+=(--dependency="$DEPENDENCY")
  fi

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="ld-highcomp" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$REPO_ROOT/compression/outputs/layer_drop_high_compression/slurm/%x-%j.out" \
    --error="$REPO_ROOT/compression/outputs/layer_drop_high_compression/slurm/%x-%j.err" \
    "${extra_sbatch_args[@]}" \
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
IFS=',' read -r -a SETTINGS <<< "$SETTINGS_CSV"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

status=0
for item in "\${MODEL_ITEMS[@]}"; do
  model_tag="\${item%%=*}"
  model_name="\${item#*=}"
  for spec in "\${SETTINGS[@]}"; do
    IFS='|' read -r rank_metric component drop_count <<< "\$spec"
    rank_metric="\${rank_metric// /}"
    component="\${component// /}"
    drop_count="\${drop_count// /}"
    selection_json="$SELECTION_ROOT/\$model_tag/\$rank_metric/drop_selection.json"
    if [[ ! -f "\$selection_json" ]]; then
      echo "[SKIP] missing selection_json=\$selection_json" >&2
      status=1
      continue
    fi
    setting="\${rank_metric}_\${component}_drop\${drop_count}"
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
        bash "$REPO_ROOT/evaluation/scripts/run_lm_eval_layer_drop_setting.sh"; then
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

"$PYTHON_BIN" "$REPO_ROOT/compression/scripts/collect_layer_drop_lm_eval_results.py" || true
exit \$status
EOF
)"
)"

printf '%s\thigh_compression\t%s\t%s\t\t\t%s\t%s\t%s\n' \
  "$job_id" "$MODEL_SPECS" "$SETTINGS_CSV" "$TASKS_CSV" "$OUTPUT_ROOT" "$SELECTION_ROOT" >> "$manifest"

echo "[SUBMITTED] job_id=$job_id"
echo "[LOG] $REPO_ROOT/compression/outputs/layer_drop_high_compression/slurm/ld-highcomp-$job_id.out"
