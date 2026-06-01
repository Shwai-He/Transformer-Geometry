#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_l40s:1}"
MEM="${MEM:-96G}"
TIME="${TIME:-16:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"
TOKEN_SCOPE="${TOKEN_SCOPE:-all}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/compression/calibration/c4_wanda_ns128_seq2048_seed0.txt}"
DROP_COUNTS_CSV="${DROP_COUNTS_CSV:-4,8}"
ENERGY_METRICS_CSV="${ENERGY_METRICS_CSV:-perp_energy,perp_energy_over_ref}"
REFERENCE_TAGS_CSV="${REFERENCE_TAGS_CSV:-one_minus_cosine_alltok,perp_ratio_energy_alltok,hybrid_perp_dualsum_alltok,perp_over_ref_alltok,perp_ratio_alltok}"
COMPONENTS_CSV="${COMPONENTS_CSV:-attn,mlp}"
TASKS_CSV="${TASKS_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"
RUN_EVAL="${RUN_EVAL:-true}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_lm_eval}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/compression/outputs/layer_drop_geometry}"
RECORD_ROOT="${RECORD_ROOT:-$REPO_ROOT/results/quality_eval/by_model/layer_drop_geometry}"

MODEL_SPECS="${MODEL_SPECS:-qwen3_0p6b=/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd;qwen3_1p7b=/beacon-projects/traumallm/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

mkdir -p "$OUTPUT_ROOT" "$SELECTION_ROOT" "$RECORD_ROOT" "$REPO_ROOT/compression/outputs/layer_drop_energy_reliability/slurm"
manifest="$OUTPUT_ROOT/submission_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'job_id\tjob_type\tmodel_tag\trank_metric\tcomponent\tdrop_count\ttask\toutput_root\tselection_json\n' > "$manifest"
fi

job_id="$(
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="ld-energy-reliability" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$REPO_ROOT/compression/outputs/layer_drop_energy_reliability/slurm/%x-%j.out" \
    --error="$REPO_ROOT/compression/outputs/layer_drop_energy_reliability/slurm/%x-%j.err" \
    --wrap="$(cat <<EOF
set -u
cd "$REPO_ROOT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=$([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo 1 || echo 0)
export TRANSFORMERS_OFFLINE=$([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo 1 || echo 0)
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
IFS=',' read -r -a ENERGY_METRICS <<< "$ENERGY_METRICS_CSV"
IFS=',' read -r -a COMPONENTS <<< "$COMPONENTS_CSV"
IFS=',' read -r -a DROP_COUNTS <<< "$DROP_COUNTS_CSV"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

status=0
for item in "\${MODEL_ITEMS[@]}"; do
  model_tag="\${item%%=*}"
  model_name="\${item#*=}"
  for rank_metric in "\${ENERGY_METRICS[@]}"; do
    rank_metric="\${rank_metric// /}"
    selector_tag="\${rank_metric}_alltok"
    echo "[SELECT_START] model=\$model_tag metric=\$rank_metric"
    if "$PYTHON_BIN" "$REPO_ROOT/compression/code/select_layer_drop_by_geometry.py" \\
      --model_name "\$model_name" \\
      --model_tag "\$model_tag" \\
      --output_tag "\$selector_tag" \\
      --prompts_file "$PROMPTS_FILE" \\
      --max_prompts "$MAX_PROMPTS" \\
      --max_length "$MAX_LENGTH" \\
      --token_scope "$TOKEN_SCOPE" \\
      --drop_counts "$DROP_COUNTS_CSV" \\
      --rank_metric "\$rank_metric" \\
      --rank_order ascending \\
      --device cuda \\
      --dtype "$DTYPE" \\
      --output_dir "$SELECTION_ROOT/\$model_tag" \\
      $([[ "$LOCAL_FILES_ONLY" == "true" ]] && echo --local_files_only || echo --no-local_files_only); then
      echo "[SELECT_DONE] model=\$model_tag metric=\$rank_metric"
    else
      rc=\$?
      echo "[SELECT_FAILED] model=\$model_tag metric=\$rank_metric rc=\$rc" >&2
      status=\$rc
      continue
    fi
  done
done

"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

selection_root = Path("$SELECTION_ROOT")
record_root = Path("$RECORD_ROOT")
models = [item.split("=", 1)[0] for item in "$MODEL_SPECS".split(";") if item.strip()]
energy_tags = [f"{m.strip()}_alltok" for m in "$ENERGY_METRICS_CSV".split(",") if m.strip()]
ref_tags = [x.strip() for x in "$REFERENCE_TAGS_CSV".split(",") if x.strip()]
components = [x.strip() for x in "$COMPONENTS_CSV".split(",") if x.strip()]
drop_counts = [x.strip() for x in "$DROP_COUNTS_CSV".split(",") if x.strip()]

rows = []
for model in models:
    selections = {}
    for tag in energy_tags + ref_tags:
        path = selection_root / model / tag / "drop_selection.json"
        if path.is_file():
            selections[tag] = json.loads(path.read_text(encoding="utf-8")).get("recommendations", {})
    for tag in energy_tags:
        rec = selections.get(tag, {})
        for component in components:
            for count in drop_counts:
                key = f"drop_{count}"
                layers = rec.get(component, {}).get(key, [])
                matching_refs = []
                closest_ref = ""
                closest_overlap = -1
                for ref_tag in ref_tags:
                    ref_layers = selections.get(ref_tag, {}).get(component, {}).get(key, [])
                    overlap = len(set(layers) & set(ref_layers))
                    if overlap > closest_overlap:
                        closest_overlap = overlap
                        closest_ref = ref_tag
                    if layers and layers == ref_layers:
                        matching_refs.append(ref_tag)
                rows.append({
                    "model": model,
                    "setting": tag,
                    "component": component,
                    "drop_count": count,
                    "layers": " ".join(map(str, layers)),
                    "matches_reference": "|".join(matching_refs),
                    "closest_reference": closest_ref,
                    "closest_overlap": closest_overlap,
                })

out_csv = record_root / "energy_reliability_selection_compare.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model", "setting", "component", "drop_count", "layers",
        "matches_reference", "closest_reference", "closest_overlap",
    ])
    writer.writeheader()
    writer.writerows(rows)

md = record_root / "energy_reliability_status.md"
lines = [
    "# Layer Drop Energy Reliability",
    "",
    "Last updated: 2026-05-24",
    "",
    "Setup: Qwen small models, all-token C4/Wanda calibration, residual-level layer-drop metrics.",
    "",
    "Compared energy selectors:",
    "- perp_energy_alltok: sum(||e_perp||^2) / N",
    "- perp_energy_over_ref_alltok: sum(||e_perp||^2) / sum(||h||^2)",
    "",
    f"Selection comparison CSV: {out_csv}",
    "",
    "| Model | Setting | Component | Drop | Layers | Exact Ref Match | Closest Ref | Overlap |",
    "|---|---|---|---:|---|---|---|---:|",
]
for row in rows:
    lines.append(
        f"| {row['model']} | {row['setting']} | {row['component']} | {row['drop_count']} | "
        f"{row['layers']} | {row['matches_reference'] or ''} | {row['closest_reference']} | {row['closest_overlap']} |"
    )
md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"[SELECTION_COMPARE] wrote {out_csv}")
print(f"[STATUS] wrote {md}")
PY

if [[ "$RUN_EVAL" == "true" ]]; then
  for item in "\${MODEL_ITEMS[@]}"; do
    model_tag="\${item%%=*}"
    model_name="\${item#*=}"
    for rank_metric in "\${ENERGY_METRICS[@]}"; do
      rank_metric="\${rank_metric// /}"
      selector_tag="\${rank_metric}_alltok"
      selection_json="$SELECTION_ROOT/\$model_tag/\$selector_tag/drop_selection.json"
      if [[ ! -f "\$selection_json" ]]; then
        echo "[EVAL_SKIP] missing selection_json=\$selection_json" >&2
        status=1
        continue
      fi
      for component in "\${COMPONENTS[@]}"; do
        component="\${component// /}"
        for drop_count in "\${DROP_COUNTS[@]}"; do
          drop_count="\${drop_count// /}"
          setting="\${selector_tag}_\${component}_drop\${drop_count}"
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
fi

"$PYTHON_BIN" "$REPO_ROOT/compression/scripts/collect_layer_drop_lm_eval_results.py" || true
exit \$status
EOF
)"
)"

printf '%s\tenergy_reliability\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$job_id" "qwen3_0p6b,qwen3_1p7b" "$ENERGY_METRICS_CSV" "$COMPONENTS_CSV" "$DROP_COUNTS_CSV" "$TASKS_CSV" \
  "$OUTPUT_ROOT" "$SELECTION_ROOT" >> "$manifest"

echo "[SUBMITTED] job_id=$job_id"
echo "[LOG] $REPO_ROOT/compression/outputs/layer_drop_energy_reliability/slurm/ld-energy-reliability-$job_id.out"
