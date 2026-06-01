#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Submit one Slurm job per geometry-pruning setting/task pair.
# Defaults: full baseline/XSA variant task surface via run_lm_eval_geo_prune_batch.sh.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"

MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_geo_nm_s0p5_full_tasks}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu,gsm8k,humaneval,nq_open,drop,mbpp,bbh_cot_zeroshot}"
JOB_PREFIX="${JOB_PREFIX:-geo-q06}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/geo_prune_lm_eval/$MODEL_TAG}"

if [[ -n "${GEO_SETTINGS:-}" ]]; then
  IFS=';' read -r -a GEO_SETTING_SPECS <<< "$GEO_SETTINGS"
else
  GEO_SETTING_SPECS=(
    "dense|false|none|unstructured"
    "wanda_2_4|true|none|2:4"
    "wanda_2_4_residual_perp|true|residual_perp|2:4"
    "wanda_2_4_residual_para|true|residual_para|2:4"
    "wanda_2_4_residual_perp_over_para|true|residual_perp_over_para|2:4"
    "wanda_4_8|true|none|4:8"
    "wanda_4_8_residual_perp|true|residual_perp|4:8"
    "wanda_4_8_residual_para|true|residual_para|4:8"
    "wanda_4_8_residual_perp_over_para|true|residual_perp_over_para|4:8"
  )
fi

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS_CSV"
mkdir -p "$OUTPUT_ROOT/_geometry_scores" "$OUTPUT_ROOT/_manifests"
submission_manifest="$OUTPUT_ROOT/submission_manifest.tsv"
if [[ ! -f "$submission_manifest" ]]; then
  printf 'job_id\tsetting\ttask\tjob_name\tpartition\tqos\toutput_root\n' > "$submission_manifest"
fi

submitted=()
for spec in "${GEO_SETTING_SPECS[@]}"; do
  IFS='|' read -r setting_name _prune_enabled _prune_strategy _sparsity_type <<< "$spec"
  for task in "${TASK_ARRAY[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] && continue
    safe_job="${JOB_PREFIX}-${setting_name}-${task}"
    safe_job="${safe_job//_/-}"
    if ((${#safe_job} > 40)); then
      safe_job="${safe_job:0:40}"
    fi
    score_cache="${GEO_SCORE_CACHE_PATH:-$OUTPUT_ROOT/_geometry_scores/c4_wanda_scores.pt}"
    manifest="$OUTPUT_ROOT/_manifests/${setting_name}_${task}.tsv"
    echo "[SUBMIT] setting=$setting_name task=$task job=$safe_job"
    output="$(
      MODEL_TAG="$MODEL_TAG" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      TASKS_CSV="$task" \
      GEO_SETTINGS="$spec" \
      GEO_SCORE_CACHE_PATH="$score_cache" \
      RUN_MANIFEST="$manifest" \
      JOB_NAME="$safe_job" \
      bash "$SCRIPT_DIR/submit_lm_eval_geo_prune_slurm.sh"
    )"
    echo "$output"
    job_id="$(echo "$output" | sed -n 's/.*job_id=//p' | tail -1)"
    if [[ -n "$job_id" ]]; then
      submitted+=("$job_id:$setting_name:$task")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" "$setting_name" "$task" "$safe_job" \
        "${PARTITION:-beacon}" "${QOS:-medium}" "$OUTPUT_ROOT" >> "$submission_manifest"
    fi
  done
done

echo "[DONE] submitted=${#submitted[@]}"
echo "[INFO] submission_manifest=$submission_manifest"
printf '%s\n' "${submitted[@]}"
