#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/path/to/resource}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
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
# Public sweep knob: para scale. The underlying implementation still consumes
# alpha, where alpha = 1 - para_scale.
PARA_SCALES="${PARA_SCALES:--20 -17.5 -15 -12.5 -10 -9 -8 -7 -6 -5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10 12.5 15 17.5 20}"
XSA_PERP_SCALE="${XSA_PERP_SCALE:-1.0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/representation-analysis/outputs/para_ablation}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-$OUT_DIR/compare}"
GENERATE_OUT_DIR="${GENERATE_OUT_DIR:-$OUT_DIR/generate}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/tmp/${USER:-xsa}/para_ablation}"
LOG_DIR="${LOG_DIR:-$LOCAL_RUN_DIR/logs}"
TMP_BASE_DIR="${TMP_BASE_DIR:-$LOCAL_RUN_DIR/tmp}"
mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR" "$GENERATE_OUT_DIR" "$LOG_DIR" "$TMP_BASE_DIR"

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

echo "[INFO] ===== para/perp scale sweep setup ====="

if [[ -n "$PROMPTS_PATH" ]]; then
  EFFECTIVE_PROMPTS_PATH="$PROMPTS_PATH"
  echo "[INFO] Using user-provided PROMPTS_PATH"
else
  DATASET_TAG="${DATASET_NAME##*/}"
  DATASET_CONFIG_TAG="${DATASET_CONFIG:-default}"
  DATASET_SPLIT_TAG="${DATASET_SPLIT:-split}"
  EFFECTIVE_PROMPTS_PATH="${OUT_DIR}/${DATASET_TAG}_${DATASET_CONFIG_TAG}_${DATASET_SPLIT_TAG}_texts_ms${MAX_SAMPLES}_seed${DATASET_SHUFFLE_SEED}.jsonl"
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

PPL_SUMMARY_TSV="${PPL_SUMMARY_TSV:-$OUT_DIR/ppl_summary.tsv}"
GEN_SUMMARY_JSONL="${GEN_SUMMARY_JSONL:-$OUT_DIR/generation_paths.jsonl}"
GEN_COMBINED_JSONL="${GEN_COMBINED_JSONL:-$OUT_DIR/generation_summary.jsonl}"

echo "[INFO] PARA_SCALES=$PARA_SCALES"
echo "[INFO] XSA_PERP_SCALE=$XSA_PERP_SCALE"
echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES"
echo "[INFO] NUM_GPUS=$NUM_GPUS"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] RUN_COMPARE_STAGE=$RUN_COMPARE_STAGE"
echo "[INFO] RUN_GENERATION_STAGE=$RUN_GENERATION_STAGE"
echo "[INFO] SKIP_EXISTING_COMPARE=$SKIP_EXISTING_COMPARE"
echo "[INFO] SKIP_EXISTING_GENERATION=$SKIP_EXISTING_GENERATION"
echo "[INFO] COMPARE_MAX_SAMPLES=$COMPARE_MAX_SAMPLES"
echo "[INFO] GENERATION_MAX_SAMPLES=$GENERATION_MAX_SAMPLES"
echo "[INFO] PROMPTS_PATH=$EFFECTIVE_PROMPTS_PATH"
echo "[INFO] DATASET_NAME=$DATASET_NAME"
echo "[INFO] DATASET_CONFIG=$DATASET_CONFIG"
echo "[INFO] DATASET_SPLIT=$DATASET_SPLIT"
echo "[INFO] USE_CHAT_TEMPLATE=$USE_CHAT_TEMPLATE"
echo "[INFO] GPU_LIST=$GPU_LIST"
echo "[INFO] DEVICE=${DEVICE:-<empty>}"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] COMPARE_OUT_DIR=$COMPARE_OUT_DIR"
echo "[INFO] GENERATE_OUT_DIR=$GENERATE_OUT_DIR"
echo "[INFO] LOCAL_RUN_DIR=$LOCAL_RUN_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] TMP_BASE_DIR=$TMP_BASE_DIR"
echo "[INFO] PPL_SUMMARY_TSV=$PPL_SUMMARY_TSV"
echo "[INFO] GEN_SUMMARY_JSONL=$GEN_SUMMARY_JSONL"
echo "[INFO] GEN_COMBINED_JSONL=$GEN_COMBINED_JSONL"
echo "[INFO] Prompt file line count: $(wc -l < "$EFFECTIVE_PROMPTS_PATH")"

tmp_rows="$(mktemp /tmp/xsa_para_rows.XXXXXX)"
tmp_gen="$(mktemp /tmp/xsa_para_gen.XXXXXX)"
echo "[INFO] Temporary row accumulator: $tmp_rows"
echo "[INFO] Temporary generation manifest: $tmp_gen"

read -r -a PARA_SCALE_VALUES <<< "$PARA_SCALES"
total_para_scales="${#PARA_SCALE_VALUES[@]}"
echo "[INFO] Total para_scale settings: $total_para_scales"

for ((para_idx=0; para_idx<total_para_scales; ++para_idx)); do
  para_scale="${PARA_SCALE_VALUES[$para_idx]}"
  setting_no=$((para_idx + 1))
  alpha="$("$PYTHON_BIN" - "$para_scale" <<'PY'
import sys
para_scale = float(sys.argv[1])
print(f"{1.0 - para_scale:g}")
PY
)"
  echo "[INFO] ===== [${setting_no}/${total_para_scales}] para_scale=$para_scale perp_scale=$XSA_PERP_SCALE ====="
  PPL_PREFIX="$COMPARE_OUT_DIR/compare_para_scale_${para_scale//./p}"
  PPL_JSONL="${PPL_PREFIX}.jsonl"
  PPL_TSV="${PPL_PREFIX}.tsv"
  if [[ "${RUN_COMPARE_STAGE,,}" == "true" ]]; then
    if [[ "${SKIP_EXISTING_COMPARE,,}" == "true" && -s "$PPL_JSONL" && -s "$PPL_TSV" ]]; then
      echo "[INFO] [${setting_no}/${total_para_scales}] Stage 1/2: compare middle ppl skipped; found existing outputs"
      echo "[INFO] [${setting_no}/${total_para_scales}] Existing compare jsonl: $PPL_JSONL"
      echo "[INFO] [${setting_no}/${total_para_scales}] Existing compare tsv: $PPL_TSV"
    else
      echo "[INFO] [${setting_no}/${total_para_scales}] Stage 1/2: compare middle ppl (MAX_SAMPLES=$COMPARE_MAX_SAMPLES)"
      MODEL_NAME="$MODEL_NAME" NUM_GPUS="$NUM_GPUS" GPU_LIST="$GPU_LIST" DEVICE="$DEVICE" PROMPTS_PATH="$EFFECTIVE_PROMPTS_PATH" MAX_SAMPLES="$COMPARE_MAX_SAMPLES" BATCH_SIZE="$BATCH_SIZE" XSA_ALPHA="$alpha" XSA_PERP_SCALE="$XSA_PERP_SCALE" USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE" TMP_BASE_DIR="$TMP_BASE_DIR" XSA_LOG_DIR="$LOG_DIR" OUTPUT_PREFIX="$PPL_PREFIX" bash "$SCRIPT_DIR/run_xsa_compare_middle_ppl.sh"
      echo "[INFO] [${setting_no}/${total_para_scales}] compare output jsonl: $PPL_JSONL"
      echo "[INFO] [${setting_no}/${total_para_scales}] compare output tsv: $PPL_TSV"
      echo "[INFO] [${setting_no}/${total_para_scales}] Stage 1/2 complete"
    fi
  else
    echo "[INFO] [${setting_no}/${total_para_scales}] Stage 1/2 skipped"
  fi

  "$PYTHON_BIN" - "$para_scale" "$PPL_JSONL" >> "$tmp_rows" <<'PY'
import json, math, statistics, sys
para_scale = sys.argv[1]
path = sys.argv[2]
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
if not rows:
    print(f"{para_scale}\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
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
    f"{para_scale}\t{n}\t{avg_baseline:.8f}\t{avg_xsa:.8f}\t{avg_xsa_delta:.8f}\t"
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

  GEN_PATH="$GENERATE_OUT_DIR/generate_para_scale_${para_scale//./p}.jsonl"
  if [[ "${RUN_GENERATION_STAGE,,}" == "true" ]]; then
    if [[ "${SKIP_EXISTING_GENERATION,,}" == "true" && -s "$GEN_PATH" && -s "${GEN_PATH%.jsonl}.stats.json" ]]; then
      echo "[INFO] [${setting_no}/${total_para_scales}] Stage 2/2: generation skipped; found existing outputs"
      echo "[INFO] [${setting_no}/${total_para_scales}] Existing generation jsonl: $GEN_PATH"
      echo "[INFO] [${setting_no}/${total_para_scales}] Existing generation stats: ${GEN_PATH%.jsonl}.stats.json"
    else
      echo "[INFO] [${setting_no}/${total_para_scales}] Stage 2/2: generation (MAX_SAMPLES=$GENERATION_MAX_SAMPLES)"
      MODEL_NAME="$MODEL_NAME" NUM_GPUS="$NUM_GPUS" GPU_LIST="$GPU_LIST" PROMPTS_PATH="$EFFECTIVE_PROMPTS_PATH" MAX_SAMPLES="$GENERATION_MAX_SAMPLES" BATCH_SIZE="$BATCH_SIZE" XSA_ALPHA="$alpha" XSA_PERP_SCALE="$XSA_PERP_SCALE" USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE" TMP_BASE_DIR="$TMP_BASE_DIR" XSA_LOG_DIR="$LOG_DIR" OUTPUT_JSONL="$GEN_PATH" bash "$SCRIPT_DIR/run_xsa_forward_generate.sh"
      echo "[INFO] [${setting_no}/${total_para_scales}] generation output jsonl: $GEN_PATH"
      echo "[INFO] [${setting_no}/${total_para_scales}] Stage 2/2 complete"
    fi
    echo "{\"para_scale\": \"$para_scale\", \"xsa_perp_scale\": \"$XSA_PERP_SCALE\", \"generation_jsonl\": \"$GEN_PATH\", \"generation_stats_json\": \"${GEN_PATH%.jsonl}.stats.json\"}" >> "$tmp_gen"
  else
    echo "[INFO] [${setting_no}/${total_para_scales}] Stage 2/2 skipped"
  fi
  echo "[INFO] [${setting_no}/${total_para_scales}] Finished para_scale=$para_scale"
done

{
  echo -e "para_scale\tn_prompts\tmean_per_text_baseline_ppl\tmean_per_text_xsa_ppl\tmean_per_text_delta_ppl_xsa\tmean_per_text_residual_attn_ppl\tmean_per_text_delta_ppl_residual_attn\tmean_per_text_residual_mlp_ppl\tmean_per_text_delta_ppl_residual_mlp\tmean_per_text_residual_both_ppl\tmean_per_text_delta_ppl_residual_both\tweighted_baseline_loss\tweighted_baseline_ppl\tweighted_xsa_loss\tweighted_delta_loss_xsa\tweighted_xsa_ppl\tweighted_residual_attn_loss\tweighted_delta_loss_residual_attn\tweighted_residual_attn_ppl\tweighted_residual_mlp_loss\tweighted_delta_loss_residual_mlp\tweighted_residual_mlp_ppl\tweighted_residual_both_loss\tweighted_delta_loss_residual_both\tweighted_residual_both_ppl\tmedian_baseline_loss\tmedian_xsa_loss\tmedian_delta_loss_xsa"
  cat "$tmp_rows"
} > "$PPL_SUMMARY_TSV"
echo "[INFO] Wrote aggregate PPL summary to $PPL_SUMMARY_TSV"

cp "$tmp_gen" "$GEN_SUMMARY_JSONL"
echo "[INFO] Wrote generation manifest to $GEN_SUMMARY_JSONL"
MANIFEST_PATH="$GEN_SUMMARY_JSONL" GEN_COMBINED_JSONL="$GEN_COMBINED_JSONL" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
combined_jsonl = Path(os.environ["GEN_COMBINED_JSONL"])
combined_stats = combined_jsonl.with_suffix(".stats.json")

rows_by_prompt = {}
stats_payload = {"sweep_kind": "para", "settings": {}}

manifest_rows = []
if manifest_path.exists():
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_rows = [json.loads(line) for line in f if line.strip()]

for item in manifest_rows:
    gen_path = Path(item["generation_jsonl"])
    if not gen_path.exists():
        continue
    stats_path = Path(item["generation_stats_json"])
    setting_key = f'para_scale={item["para_scale"]}'
    setting_stats = None
    if stats_path.exists():
        setting_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_payload["settings"][setting_key] = {
        "para_scale": item["para_scale"],
        "xsa_perp_scale": item.get("xsa_perp_scale"),
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
                "para_scale": item["para_scale"],
                "xsa_perp_scale": item.get("xsa_perp_scale"),
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
echo "[INFO] PPL Summary:"
echo "[INFO]   $PPL_SUMMARY_TSV"
echo "[INFO] Generation Paths:"
echo "[INFO]   $GEN_SUMMARY_JSONL"
echo "[INFO] Generation Summary:"
echo "[INFO]   $GEN_COMBINED_JSONL"
