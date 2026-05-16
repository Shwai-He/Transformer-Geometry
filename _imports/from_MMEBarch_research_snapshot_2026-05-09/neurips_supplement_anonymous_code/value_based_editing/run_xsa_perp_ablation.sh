#!/usr/bin/env bash
set -euo pipefail

# Public knobs: para scale and perp scale.
# Internally the implementation still uses alpha, where alpha = 1 - para_scale.
# The effective update is:
#   y' = perp_scale * y_perp + para_scale * y_para

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/path/to/resource}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
MAX_SAMPLES="${MAX_SAMPLES:-2048}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_COMPARE_STAGE="${RUN_COMPARE_STAGE:-true}"
RUN_GENERATION_STAGE="${RUN_GENERATION_STAGE:-true}"
SKIP_EXISTING_COMPARE="${SKIP_EXISTING_COMPARE:-true}"
SKIP_EXISTING_GENERATION="${SKIP_EXISTING_GENERATION:-true}"
COMPARE_MAX_SAMPLES="${COMPARE_MAX_SAMPLES:-$MAX_SAMPLES}"
GENERATION_MAX_SAMPLES="${GENERATION_MAX_SAMPLES:-64}"
DATASET_NAME="${DATASET_NAME:-allenai/c4}"
DATASET_CONFIG="${DATASET_CONFIG:-en}"
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
DATASET_TEXT_KEY="${DATASET_TEXT_KEY:-text}"
DATASET_MIN_CHARS="${DATASET_MIN_CHARS:-160}"
DATASET_MAX_CHARS="${DATASET_MAX_CHARS:-1200}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-1337}"
DEVICE="${DEVICE:-}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-false}"
PARA_SCALE="${PARA_SCALE:-1.0}"
# PERP_SCALES="${PERP_SCALES:--1.5 -1.25 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 1.25 1.5}"
# PERP_SCALES="${PERP_SCALES:--1.5 -1.25 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 1.25 1.5}"
# PERP_SCALES="${PERP_SCALES:--1.5 -1.25 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 1.25 1.5}"
PERP_SCALES="${PERP_SCALES:--1.5 -1.375 -1.25 -1.125 -1.0 -0.875 -0.75 -0.625 -0.5 -0.375 -0.25 -0.125 0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5}"


OUT_DIR="${OUT_DIR:-/path/to/resource}"
PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR:-$OUT_DIR/prompts}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-$OUT_DIR/compare}"
GENERATE_OUT_DIR="${GENERATE_OUT_DIR:-$OUT_DIR/generate}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/tmp/${USER:-xsa}/perp_ablation}"
LOG_DIR="${LOG_DIR:-$LOCAL_RUN_DIR/logs}"
TMP_BASE_DIR="${TMP_BASE_DIR:-$LOCAL_RUN_DIR/tmp}"
mkdir -p "$OUT_DIR" "$PROMPT_CACHE_DIR" "$COMPARE_OUT_DIR" "$GENERATE_OUT_DIR" "$LOG_DIR" "$TMP_BASE_DIR"
PPL_SUMMARY_TSV="${PPL_SUMMARY_TSV:-$OUT_DIR/ppl_summary.tsv}"
GEN_SUMMARY_JSONL="${GEN_SUMMARY_JSONL:-$OUT_DIR/generation_paths.jsonl}"
GEN_COMBINED_JSONL="${GEN_COMBINED_JSONL:-$OUT_DIR/generation_summary.jsonl}"

# Execution / overwrite policy (file-first knobs).
# - MERGE_ONLY=true: do not run compare/generate; only aggregate from existing files.
# - OVERWRITE_EXISTING=false: if per-setting outputs exist, skip rerun for that setting.
# - OVERWRITE_SUMMARY=false: if summary files exist, write timestamped new files instead.
MERGE_ONLY="${MERGE_ONLY:-false}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"
OVERWRITE_SUMMARY="${OVERWRITE_SUMMARY:-false}"
if [[ "${OVERWRITE_EXISTING,,}" == "true" ]]; then
  SKIP_EXISTING_COMPARE="false"
  SKIP_EXISTING_GENERATION="false"
fi
if [[ "${MERGE_ONLY,,}" == "true" ]]; then
  RUN_COMPARE_STAGE="false"
  RUN_GENERATION_STAGE="false"
fi

build_default_gpu_list() {
  local count="$1"
  local ids=()
  local i
  for ((i=0; i<count; ++i)); do
    ids+=("$i")
  done
  local joined
  joined="$(IFS=,; echo "${ids[*]}")"
  echo "$joined"
}

GPU_LIST="${GPU_LIST:-$(build_default_gpu_list "$NUM_GPUS")}"

echo "[INFO] ===== perp ablation setup ====="

if [[ -n "$PROMPTS_PATH" ]]; then
  EFFECTIVE_PROMPTS_PATH="$PROMPTS_PATH"
  echo "[INFO] Using user-provided PROMPTS_PATH"
else
  DATASET_TAG="${DATASET_NAME##*/}"
  DATASET_CONFIG_TAG="${DATASET_CONFIG:-default}"
  DATASET_SPLIT_TAG="${DATASET_SPLIT:-split}"
  EFFECTIVE_PROMPTS_PATH="${PROMPT_CACHE_DIR}/${DATASET_TAG}_${DATASET_CONFIG_TAG}_${DATASET_SPLIT_TAG}_texts_ms${MAX_SAMPLES}_seed${DATASET_SHUFFLE_SEED}.jsonl"
  if [[ ! -s "$EFFECTIVE_PROMPTS_PATH" ]]; then
    echo "[INFO] No cached prompt file found; preparing dataset-backed prompts"
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_text_dataset_jsonl.py" \
      --dataset_name "$DATASET_NAME" \
      --dataset_config "$DATASET_CONFIG" \
      --dataset_split "$DATASET_SPLIT" \
      --text_key "$DATASET_TEXT_KEY" \
      --max_texts "$MAX_SAMPLES" \
      --min_chars "$DATASET_MIN_CHARS" \
      --max_chars "$DATASET_MAX_CHARS" \
      --shuffle_seed "$DATASET_SHUFFLE_SEED" \
      --output_jsonl "$EFFECTIVE_PROMPTS_PATH"
  else
    echo "[INFO] Reusing cached prompt file: $EFFECTIVE_PROMPTS_PATH"
  fi
fi

alpha_fixed="$("$PYTHON_BIN" - "$PARA_SCALE" <<'PY'
import sys
para_scale = float(sys.argv[1])
print(f"{1.0 - para_scale:g}")
PY
)"

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] PARA_SCALE_FIXED=$PARA_SCALE"
echo "[INFO] PERP_SCALES=$PERP_SCALES"
echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES"
echo "[INFO] NUM_GPUS=$NUM_GPUS"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] RUN_COMPARE_STAGE=$RUN_COMPARE_STAGE"
echo "[INFO] RUN_GENERATION_STAGE=$RUN_GENERATION_STAGE"
echo "[INFO] SKIP_EXISTING_COMPARE=$SKIP_EXISTING_COMPARE"
echo "[INFO] SKIP_EXISTING_GENERATION=$SKIP_EXISTING_GENERATION"
echo "[INFO] MERGE_ONLY=$MERGE_ONLY"
echo "[INFO] OVERWRITE_EXISTING=$OVERWRITE_EXISTING"
echo "[INFO] OVERWRITE_SUMMARY=$OVERWRITE_SUMMARY"
echo "[INFO] COMPARE_MAX_SAMPLES=$COMPARE_MAX_SAMPLES"
echo "[INFO] GENERATION_MAX_SAMPLES=$GENERATION_MAX_SAMPLES"
echo "[INFO] GPU_LIST=$GPU_LIST"
echo "[INFO] PROMPTS_PATH=$EFFECTIVE_PROMPTS_PATH"
echo "[INFO] DATASET_NAME=$DATASET_NAME"
echo "[INFO] DATASET_CONFIG=$DATASET_CONFIG"
echo "[INFO] DATASET_SPLIT=$DATASET_SPLIT"
echo "[INFO] USE_CHAT_TEMPLATE=$USE_CHAT_TEMPLATE"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] PROMPT_CACHE_DIR=$PROMPT_CACHE_DIR"
echo "[INFO] COMPARE_OUT_DIR=$COMPARE_OUT_DIR"
echo "[INFO] GENERATE_OUT_DIR=$GENERATE_OUT_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] TMP_BASE_DIR=$TMP_BASE_DIR"
echo "[INFO] PPL_SUMMARY_TSV=$PPL_SUMMARY_TSV"
echo "[INFO] GEN_SUMMARY_JSONL=$GEN_SUMMARY_JSONL"
echo "[INFO] GEN_COMBINED_JSONL=$GEN_COMBINED_JSONL"
echo "[INFO] Formula: y' = perp_scale * y_perp + para_scale * y_para"
echo "[INFO] Prompt file line count: $(wc -l < "$EFFECTIVE_PROMPTS_PATH")"

read -r -a PERP_SCALE_VALUES <<< "$PERP_SCALES"
total_perp_scales="${#PERP_SCALE_VALUES[@]}"
echo "[INFO] Total perp_scale settings: $total_perp_scales"
tmp_rows="$(mktemp /tmp/xsa_perp_rows.XXXXXX)"
tmp_gen="$(mktemp /tmp/xsa_perp_gen.XXXXXX)"
echo "[INFO] Temporary row accumulator: $tmp_rows"
echo "[INFO] Temporary generation manifest: $tmp_gen"

for ((perp_idx=0; perp_idx<total_perp_scales; ++perp_idx)); do
  ps="${PERP_SCALE_VALUES[$perp_idx]}"
  setting_no=$((perp_idx + 1))
  echo "[INFO] ===== [${setting_no}/${total_perp_scales}] perp_scale=$ps para_scale_fixed=$PARA_SCALE ====="

  # PPL compare: baseline(none) vs xsa(attn)
  if [[ "${RUN_COMPARE_STAGE,,}" == "true" ]]; then
    PPL_PREFIX="$COMPARE_OUT_DIR/compare_para_scale_${PARA_SCALE}_perp_scale_${ps}"
    PPL_JSONL="${PPL_PREFIX}.jsonl"
    PPL_TSV="${PPL_PREFIX}.tsv"
    if [[ "${SKIP_EXISTING_COMPARE,,}" == "true" && -s "$PPL_JSONL" && -s "$PPL_TSV" ]]; then
      echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 1/2: compare middle ppl skipped; found existing outputs"
      echo "[INFO] [${setting_no}/${total_perp_scales}] Existing compare jsonl: $PPL_JSONL"
      echo "[INFO] [${setting_no}/${total_perp_scales}] Existing compare tsv: $PPL_TSV"
    else
      echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 1/2: compare middle ppl (MAX_SAMPLES=$COMPARE_MAX_SAMPLES)"
      MODEL_NAME="$MODEL_NAME" \
      NUM_GPUS="$NUM_GPUS" \
      GPU_LIST="$GPU_LIST" \
      DEVICE="$DEVICE" \
      PROMPTS_PATH="$EFFECTIVE_PROMPTS_PATH" \
      MAX_SAMPLES="$COMPARE_MAX_SAMPLES" \
      BATCH_SIZE="$BATCH_SIZE" \
      XSA_ALPHA="$alpha_fixed" \
      XSA_PERP_SCALE="$ps" \
      TMP_BASE_DIR="$TMP_BASE_DIR" \
      XSA_LOG_DIR="$LOG_DIR" \
      OUTPUT_PREFIX="$PPL_PREFIX" \
      USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE" \
      bash representation-analysis/run_xsa_compare_middle_ppl.sh
      echo "[INFO] [${setting_no}/${total_perp_scales}] compare output prefix: $PPL_PREFIX"
      echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 1/2 complete"
    fi
  else
    echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 1/2 skipped"
  fi

  PPL_PREFIX="$COMPARE_OUT_DIR/compare_para_scale_${PARA_SCALE}_perp_scale_${ps}"
  PPL_JSONL="${PPL_PREFIX}.jsonl"
  if [[ -s "$PPL_JSONL" ]]; then
    "$PYTHON_BIN" - "$ps" "$PPL_JSONL" >> "$tmp_rows" <<'PY'
import json, math, statistics, sys
perp_scale = sys.argv[1]
path = sys.argv[2]
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
if not rows:
    print(f"{perp_scale}\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
    raise SystemExit(0)
n = len(rows)
avg_baseline = sum(r["baseline_ppl"] for r in rows) / n
avg_xsa = sum(r["xsa_ppl"] for r in rows) / n
avg_xsa_delta = sum(r["delta_ppl"] for r in rows) / n
avg_res_attn = sum(r["residual_attn_ppl"] for r in rows) / n
avg_res_attn_delta = sum(r["delta_ppl_residual_attn"] for r in rows) / n
avg_res_mlp = sum(r["residual_mlp_ppl"] for r in rows) / n
avg_res_mlp_delta = sum(r["delta_ppl_residual_mlp"] for r in rows) / n
avg_res_both = sum(r["residual_both_ppl"] for r in rows) / n
avg_res_both_delta = sum(r["delta_ppl_residual_both"] for r in rows) / n
baseline_tokens = sum(r["baseline_tokens"] for r in rows)
xsa_tokens = sum(r["xsa_tokens"] for r in rows)
res_attn_tokens = sum(r["residual_attn_tokens"] for r in rows)
res_mlp_tokens = sum(r["residual_mlp_tokens"] for r in rows)
res_both_tokens = sum(r["residual_both_tokens"] for r in rows)
weighted_baseline_loss = sum(r["baseline_loss"] * r["baseline_tokens"] for r in rows) / max(1.0, baseline_tokens)
weighted_xsa_loss = sum(r["xsa_loss"] * r["xsa_tokens"] for r in rows) / max(1.0, xsa_tokens)
weighted_res_attn_loss = sum(r["residual_attn_loss"] * r["residual_attn_tokens"] for r in rows) / max(1.0, res_attn_tokens)
weighted_res_mlp_loss = sum(r["residual_mlp_loss"] * r["residual_mlp_tokens"] for r in rows) / max(1.0, res_mlp_tokens)
weighted_res_both_loss = sum(r["residual_both_loss"] * r["residual_both_tokens"] for r in rows) / max(1.0, res_both_tokens)
median_baseline_loss = statistics.median(r["baseline_loss"] for r in rows)
median_xsa_loss = statistics.median(r["xsa_loss"] for r in rows)
median_delta_loss_xsa = statistics.median(r["delta_loss"] for r in rows)
print(
    f"{perp_scale}\t{n}\t{avg_baseline:.8f}\t{avg_xsa:.8f}\t{avg_xsa_delta:.8f}\t"
    f"{avg_res_attn:.8f}\t{avg_res_attn_delta:.8f}\t"
    f"{avg_res_mlp:.8f}\t{avg_res_mlp_delta:.8f}\t"
    f"{avg_res_both:.8f}\t{avg_res_both_delta:.8f}\t"
    f"{weighted_baseline_loss:.8f}\t{math.exp(weighted_baseline_loss):.8f}\t"
    f"{weighted_xsa_loss:.8f}\t{weighted_xsa_loss - weighted_baseline_loss:.8f}\t{math.exp(weighted_xsa_loss):.8f}\t"
    f"{weighted_res_attn_loss:.8f}\t{weighted_res_attn_loss - weighted_baseline_loss:.8f}\t{math.exp(weighted_res_attn_loss):.8f}\t"
    f"{weighted_res_mlp_loss:.8f}\t{weighted_res_mlp_loss - weighted_baseline_loss:.8f}\t{math.exp(weighted_res_mlp_loss):.8f}\t"
    f"{weighted_res_both_loss:.8f}\t{weighted_res_both_loss - weighted_baseline_loss:.8f}\t{math.exp(weighted_res_both_loss):.8f}\t"
    f"{median_baseline_loss:.8f}\t{median_xsa_loss:.8f}\t{median_delta_loss_xsa:.8f}"
)
PY
  else
    echo "[WARN] Missing compare jsonl for perp_scale=$ps: $PPL_JSONL"
    echo -e "${ps}\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t" >> "$tmp_rows"
  fi

  # Generation
  if [[ "${RUN_GENERATION_STAGE,,}" == "true" ]]; then
    GEN_JSONL="$GENERATE_OUT_DIR/generate_para_scale_${PARA_SCALE}_perp_scale_${ps}.jsonl"
    GEN_STATS_JSON="${GEN_JSONL%.jsonl}.stats.json"
    if [[ "${SKIP_EXISTING_GENERATION,,}" == "true" && -s "$GEN_JSONL" && -s "$GEN_STATS_JSON" ]]; then
      echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 2/2: generation skipped; found existing outputs"
      echo "[INFO] [${setting_no}/${total_perp_scales}] Existing generation jsonl: $GEN_JSONL"
      echo "[INFO] [${setting_no}/${total_perp_scales}] Existing generation stats: $GEN_STATS_JSON"
    else
      echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 2/2: generation (MAX_SAMPLES=$GENERATION_MAX_SAMPLES)"
      MODEL_NAME="$MODEL_NAME" \
      NUM_GPUS="$NUM_GPUS" \
      GPU_LIST="$GPU_LIST" \
      PROMPTS_PATH="$EFFECTIVE_PROMPTS_PATH" \
      MAX_SAMPLES="$GENERATION_MAX_SAMPLES" \
      BATCH_SIZE="$BATCH_SIZE" \
      XSA_ALPHA="$alpha_fixed" \
      XSA_PERP_SCALE="$ps" \
      TMP_BASE_DIR="$TMP_BASE_DIR" \
      XSA_LOG_DIR="$LOG_DIR" \
      OUTPUT_JSONL="$GEN_JSONL" \
      USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE" \
      bash representation-analysis/run_xsa_forward_generate.sh
      echo "[INFO] [${setting_no}/${total_perp_scales}] generation output jsonl: $GEN_JSONL"
      echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 2/2 complete"
    fi
    if [[ -s "$GEN_JSONL" ]]; then
      echo "{\"para_scale\": \"$PARA_SCALE\", \"perp_scale\": \"$ps\", \"generation_jsonl\": \"$GEN_JSONL\", \"generation_stats_json\": \"$GEN_STATS_JSON\"}" >> "$tmp_gen"
    fi
  else
    echo "[INFO] [${setting_no}/${total_perp_scales}] Stage 2/2 skipped"
  fi
  echo "[INFO] [${setting_no}/${total_perp_scales}] Finished perp_scale=$ps"
done

SUMMARY_PATH="$PPL_SUMMARY_TSV"
if [[ -e "$SUMMARY_PATH" && "${OVERWRITE_SUMMARY,,}" != "true" ]]; then
  SUMMARY_PATH="${PPL_SUMMARY_TSV%.tsv}-$(date +%Y%m%d-%H%M%S).tsv"
  echo "[INFO] Summary exists and OVERWRITE_SUMMARY=false; writing new file: $SUMMARY_PATH"
fi
{
  echo -e "perp_scale\tn_prompts\tmean_per_text_baseline_ppl\tmean_per_text_xsa_ppl\tmean_per_text_delta_ppl_xsa\tmean_per_text_residual_attn_ppl\tmean_per_text_delta_ppl_residual_attn\tmean_per_text_residual_mlp_ppl\tmean_per_text_delta_ppl_residual_mlp\tmean_per_text_residual_both_ppl\tmean_per_text_delta_ppl_residual_both\tweighted_baseline_loss\tweighted_baseline_ppl\tweighted_xsa_loss\tweighted_delta_loss_xsa\tweighted_xsa_ppl\tweighted_residual_attn_loss\tweighted_delta_loss_residual_attn\tweighted_residual_attn_ppl\tweighted_residual_mlp_loss\tweighted_delta_loss_residual_mlp\tweighted_residual_mlp_ppl\tweighted_residual_both_loss\tweighted_delta_loss_residual_both\tweighted_residual_both_ppl\tmedian_baseline_loss\tmedian_xsa_loss\tmedian_delta_loss_xsa"
  cat "$tmp_rows"
} > "$SUMMARY_PATH"
echo "[INFO] Wrote aggregate PPL summary to $SUMMARY_PATH"

GEN_MANIFEST_PATH="$GEN_SUMMARY_JSONL"
if [[ -e "$GEN_MANIFEST_PATH" && "${OVERWRITE_SUMMARY,,}" != "true" ]]; then
  GEN_MANIFEST_PATH="${GEN_SUMMARY_JSONL%.jsonl}-$(date +%Y%m%d-%H%M%S).jsonl"
  echo "[INFO] Manifest exists and OVERWRITE_SUMMARY=false; writing new file: $GEN_MANIFEST_PATH"
fi
if [[ "${MERGE_ONLY,,}" == "true" && -s "$GEN_SUMMARY_JSONL" && "${OVERWRITE_SUMMARY,,}" != "true" ]]; then
  echo "[INFO] MERGE_ONLY=true and existing generation manifest kept: $GEN_SUMMARY_JSONL"
else
  cp "$tmp_gen" "$GEN_MANIFEST_PATH"
  echo "[INFO] Wrote generation manifest to $GEN_MANIFEST_PATH"
fi
GEN_COMBINED_PATH="$GEN_COMBINED_JSONL"
if [[ -e "$GEN_COMBINED_PATH" && "${OVERWRITE_SUMMARY,,}" != "true" ]]; then
  GEN_COMBINED_PATH="${GEN_COMBINED_JSONL%.jsonl}-$(date +%Y%m%d-%H%M%S).jsonl"
  echo "[INFO] Combined generation summary exists and OVERWRITE_SUMMARY=false; writing new file: $GEN_COMBINED_PATH"
fi
MANIFEST_PATH="$GEN_MANIFEST_PATH" GEN_COMBINED_JSONL="$GEN_COMBINED_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
combined_jsonl = Path(os.environ["GEN_COMBINED_JSONL"])
combined_stats = combined_jsonl.with_suffix(".stats.json")

rows_by_prompt = {}
stats_payload = {"sweep_kind": "perp", "settings": {}}

manifest_rows = []
if manifest_path.exists():
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_rows = [json.loads(line) for line in f if line.strip()]

for item in manifest_rows:
    gen_path = Path(item["generation_jsonl"])
    if not gen_path.exists():
        continue
    stats_path = Path(item["generation_stats_json"])
    setting_key = f'perp_scale={item["perp_scale"]}'
    setting_stats = None
    if stats_path.exists():
        setting_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_payload["settings"][setting_key] = {
        "para_scale": item.get("para_scale"),
        "perp_scale": item["perp_scale"],
        "generation_jsonl": str(gen_path),
        "generation_stats_json": str(stats_path),
        "stats": setting_stats,
    }
    with gen_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            row = rows_by_prompt.setdefault(
                rec["prompt_idx"],
                {
                    "prompt_idx": rec["prompt_idx"],
                    "prompt": rec["prompt"],
                    "settings": {},
                },
            )
            row["settings"][setting_key] = {
                "para_scale": item.get("para_scale"),
                "perp_scale": item["perp_scale"],
                "generations": rec["generations"],
            }

combined_jsonl.parent.mkdir(parents=True, exist_ok=True)
with combined_jsonl.open("w", encoding="utf-8") as f:
    for prompt_idx in sorted(rows_by_prompt):
        f.write(json.dumps(rows_by_prompt[prompt_idx], ensure_ascii=False) + "\n")
combined_stats.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[INFO] Wrote combined generation summary to {combined_jsonl}")
print(f"[INFO] Wrote combined generation summary stats to {combined_stats}")
PY
rm -f "$tmp_rows" "$tmp_gen"

echo "[INFO] Done."
echo "[INFO] Outputs: $OUT_DIR"
echo "[INFO] PPL Summary:"
echo "[INFO]   $SUMMARY_PATH"
echo "[INFO] Generation Paths:"
if [[ "${MERGE_ONLY,,}" == "true" && -s "$GEN_SUMMARY_JSONL" && "${OVERWRITE_SUMMARY,,}" != "true" ]]; then
  echo "[INFO]   $GEN_SUMMARY_JSONL"
else
  echo "[INFO]   $GEN_MANIFEST_PATH"
fi
echo "[INFO] Generation Summary:"
echo "[INFO]   $GEN_COMBINED_PATH"
