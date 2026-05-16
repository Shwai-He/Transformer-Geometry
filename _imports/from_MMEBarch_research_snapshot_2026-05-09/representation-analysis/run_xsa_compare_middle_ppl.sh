#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/mnt/hdfs/shwai.he/DepthBoost/representation-analysis/models/Qwen/Qwen3-4B}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
TEXT_KEY="${TEXT_KEY:-text}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-8}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
XSA_ALPHA="${XSA_ALPHA:-1.0}"
XSA_PERP_SCALE="${XSA_PERP_SCALE:-1.0}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-}"
# For raw LM evaluation on corpus text such as C4, default to plain text rather
# than rendering each sample as a single-turn chat.
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-false}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
TMP_BASE_DIR="${TMP_BASE_DIR:-/tmp}"
XSA_LOG_DIR="${XSA_LOG_DIR:-representation-analysis/outputs/xsa_logs}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
STREAM_FORWARD_ABLATION_LOGS="${STREAM_FORWARD_ABLATION_LOGS:-true}"

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

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

GPU_LIST="${GPU_LIST:-$(build_default_gpu_list "$NUM_GPUS")}"
MODEL_TAG="$(sanitize_tag "$MODEL_NAME")"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-representation-analysis/outputs/${MODEL_TAG}}"
PROMPT_TAG="$(sanitize_tag "${PROMPTS_PATH:-builtin}")"
DEVICE_TAG="$(sanitize_tag "${DEVICE:-auto}")"
DTYPE_TAG="$(sanitize_tag "$DTYPE")"
CHAT_TAG="chat${USE_CHAT_TEMPLATE}"
LAYERWISE_TAG="lw${XSA_LAYERWISE_STATS}"
STAT_TAG="stats${XSA_TRACK_STATS}"
LAYER_TAG="xl${XSA_START_LAYER}-xe${XSA_END_LAYER}-xsf${XSA_SKIP_FIRST_N}-xsl${XSA_SKIP_LAST_N}"
ALPHA_TAG="$(sanitize_tag "${XSA_ALPHA}")"
PERP_TAG="$(sanitize_tag "${XSA_PERP_SCALE}")"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${BASE_OUTPUT_DIR}/qwen_xsa_compare_middle_ppl-prompt${PROMPT_TAG}-ms${MAX_SAMPLES}-len${MAX_LENGTH}-bs${BATCH_SIZE}-xa${ALPHA_TAG}-xp${PERP_TAG}-${CHAT_TAG}-${STAT_TAG}-${LAYERWISE_TAG}-dtype${DTYPE_TAG}-dev${DEVICE_TAG}-${LAYER_TAG}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_PREFIX}.jsonl}"
OUTPUT_TSV="${OUTPUT_TSV:-${OUTPUT_PREFIX}.tsv}"

echo "[INFO] OUTPUT_JSONL=$OUTPUT_JSONL"
echo "[INFO] OUTPUT_TSV=$OUTPUT_TSV"
echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] NUM_GPUS=$NUM_GPUS"
echo "[INFO] GPU_LIST=$GPU_LIST"
echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] XSA_ALPHA=$XSA_ALPHA"
echo "[INFO] XSA_PERP_SCALE=$XSA_PERP_SCALE"
echo "[INFO] STREAM_FORWARD_ABLATION_LOGS=$STREAM_FORWARD_ABLATION_LOGS"

mkdir -p "$TMP_BASE_DIR" "$XSA_LOG_DIR"
tmp_dir="$(mktemp -d "${TMP_BASE_DIR%/}/qwen_xsa_compare_middle_ppl.XXXXXX")"
prepared_prompts_path="${tmp_dir}/prepared_prompts.jsonl"
per_text_output_jsonl="${tmp_dir}/per_text_results.jsonl"
forward_output_json="${tmp_dir}/forward_payload.json"
forward_log_path="${XSA_LOG_DIR%/}/qwen_xsa_compare_middle_ppl-$(date +%Y-%m-%dT%H-%M-%S).log"
echo "[INFO] Detailed compare log: $forward_log_path"

MAX_SAMPLES="$MAX_SAMPLES" NUM_GPUS="$NUM_GPUS" PROMPTS_PATH="$PROMPTS_PATH" TEXT_KEY="$TEXT_KEY" PREPARED_PROMPTS_PATH="$prepared_prompts_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

max_samples = int(os.environ["MAX_SAMPLES"])
num_gpus = max(1, int(os.environ["NUM_GPUS"]))
prompts_path = os.environ["PROMPTS_PATH"]
text_key = os.environ["TEXT_KEY"]
prepared_path = Path(os.environ["PREPARED_PROMPTS_PATH"])

if prompts_path:
    src = Path(prompts_path)
    prompts = []
    if src.suffix == ".jsonl":
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("prompt") or obj.get(text_key) or obj.get("text")
                if isinstance(text, str) and text.strip():
                    prompts.append(text)
    else:
        with src.open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
else:
    from qwen_xsa_forward_ablation import BUILTIN_TEXTS
    prompts = list(BUILTIN_TEXTS)

if not prompts:
    raise SystemExit("No prompts loaded for compare-middle.")

original_count = len(prompts)
target_prompts = max(max_samples, num_gpus) if max_samples > 0 else max(len(prompts), num_gpus)
if len(prompts) >= target_prompts:
    prompts = prompts[:target_prompts]
else:
    repeats = (target_prompts + len(prompts) - 1) // len(prompts)
    prompts = (prompts * repeats)[:target_prompts]
    print(
        f"[INFO] Expanded compare prompt set from {original_count} to {len(prompts)} to satisfy MAX_SAMPLES/parallel scheduling.",
        flush=True,
    )

with prepared_path.open("w", encoding="utf-8") as f:
    for prompt in prompts:
        f.write(json.dumps({"text": prompt}, ensure_ascii=False) + "\n")
PY

if [[ "${STREAM_FORWARD_ABLATION_LOGS,,}" == "true" ]]; then
  MODEL_NAME="$MODEL_NAME" \
  JSONL_PATH="$prepared_prompts_path" \
  TEXT_KEY="text" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  MAX_LENGTH="$MAX_LENGTH" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_GPUS="$NUM_GPUS" \
  GPU_LIST="$GPU_LIST" \
  XSA_START_LAYER="$XSA_START_LAYER" \
  XSA_END_LAYER="$XSA_END_LAYER" \
  XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N" \
  XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N" \
  XSA_ALPHA="$XSA_ALPHA" \
  XSA_PERP_SCALE="$XSA_PERP_SCALE" \
  XSA_INTERVENTION_SITES="xsa_middle_multihead,residual_output" \
  XSA_MIDDLE_TARGETS="attn" \
  RESIDUAL_OUTPUT_TARGETS="none,attn,mlp,both" \
  DTYPE="$DTYPE" \
  USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE" \
  SYSTEM_PROMPT="$SYSTEM_PROMPT" \
  XSA_TRACK_STATS="$XSA_TRACK_STATS" \
  XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS" \
  STREAM_CHILD_LOGS="true" \
  OUTPUT_JSON="$forward_output_json" \
  PER_TEXT_OUTPUT_JSONL="$per_text_output_jsonl" \
  bash representation-analysis/run_xsa_forward_ablation.sh 2>&1 | tee "$forward_log_path"
else
  MODEL_NAME="$MODEL_NAME" \
  JSONL_PATH="$prepared_prompts_path" \
  TEXT_KEY="text" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  MAX_LENGTH="$MAX_LENGTH" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_GPUS="$NUM_GPUS" \
  GPU_LIST="$GPU_LIST" \
  XSA_START_LAYER="$XSA_START_LAYER" \
  XSA_END_LAYER="$XSA_END_LAYER" \
  XSA_SKIP_FIRST_N="$XSA_SKIP_FIRST_N" \
  XSA_SKIP_LAST_N="$XSA_SKIP_LAST_N" \
  XSA_ALPHA="$XSA_ALPHA" \
  XSA_PERP_SCALE="$XSA_PERP_SCALE" \
  XSA_INTERVENTION_SITES="xsa_middle_multihead,residual_output" \
  XSA_MIDDLE_TARGETS="attn" \
  RESIDUAL_OUTPUT_TARGETS="none,attn,mlp,both" \
  DTYPE="$DTYPE" \
  USE_CHAT_TEMPLATE="$USE_CHAT_TEMPLATE" \
  SYSTEM_PROMPT="$SYSTEM_PROMPT" \
  XSA_TRACK_STATS="$XSA_TRACK_STATS" \
  XSA_LAYERWISE_STATS="$XSA_LAYERWISE_STATS" \
  STREAM_CHILD_LOGS="true" \
  OUTPUT_JSON="$forward_output_json" \
  PER_TEXT_OUTPUT_JSONL="$per_text_output_jsonl" \
  bash representation-analysis/run_xsa_forward_ablation.sh >"$forward_log_path" 2>&1
fi

OUTPUT_JSONL="$OUTPUT_JSONL" OUTPUT_TSV="$OUTPUT_TSV" PER_TEXT_OUTPUT_JSONL="$per_text_output_jsonl" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

output_jsonl = Path(os.environ["OUTPUT_JSONL"])
output_tsv = Path(os.environ["OUTPUT_TSV"])
per_text_path = Path(os.environ["PER_TEXT_OUTPUT_JSONL"])

rows = []
with per_text_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        payload = json.loads(line)
        xsa_site = payload["sites"]["xsa_middle_multihead"]["attn"]
        residual_site = payload["sites"]["residual_output"]
        baseline = residual_site["none"]
        residual_attn = residual_site["attn"]
        residual_mlp = residual_site["mlp"]
        residual_both = residual_site["both"]
        prompt_text = payload["text"]
        prompt_preview = prompt_text.replace("\t", " ").replace("\n", " ")
        if len(prompt_preview) > 120:
            prompt_preview = prompt_preview[:117] + "..."
        rows.append(
            {
                "prompt_idx": payload["text_idx"],
                "prompt": prompt_text,
                "prompt_preview": prompt_preview,
                "baseline_loss": baseline["loss"],
                "baseline_ppl": baseline["ppl"],
                "xsa_loss": xsa_site["loss"],
                "xsa_ppl": xsa_site["ppl"],
                "delta_loss": xsa_site["loss"] - baseline["loss"],
                "delta_ppl": xsa_site["ppl"] - baseline["ppl"],
                "baseline_tokens": baseline["tokens"],
                "xsa_tokens": xsa_site["tokens"],
                "residual_attn_loss": residual_attn["loss"],
                "residual_attn_ppl": residual_attn["ppl"],
                "delta_loss_residual_attn": residual_attn["loss"] - baseline["loss"],
                "delta_ppl_residual_attn": residual_attn["ppl"] - baseline["ppl"],
                "residual_attn_tokens": residual_attn["tokens"],
                "residual_mlp_loss": residual_mlp["loss"],
                "residual_mlp_ppl": residual_mlp["ppl"],
                "delta_loss_residual_mlp": residual_mlp["loss"] - baseline["loss"],
                "delta_ppl_residual_mlp": residual_mlp["ppl"] - baseline["ppl"],
                "residual_mlp_tokens": residual_mlp["tokens"],
                "residual_both_loss": residual_both["loss"],
                "residual_both_ppl": residual_both["ppl"],
                "delta_loss_residual_both": residual_both["loss"] - baseline["loss"],
                "delta_ppl_residual_both": residual_both["ppl"] - baseline["ppl"],
                "residual_both_tokens": residual_both["tokens"],
            }
        )

rows.sort(key=lambda x: x["prompt_idx"])
output_jsonl.parent.mkdir(parents=True, exist_ok=True)
with output_jsonl.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with output_tsv.open("w", encoding="utf-8") as f:
    f.write(
        "prompt_idx\tprompt_preview\tbaseline_loss\tbaseline_ppl\txsa_loss\txsa_ppl\tdelta_loss_xsa\tdelta_ppl_xsa\t"
        "residual_attn_loss\tresidual_attn_ppl\tdelta_loss_residual_attn\tdelta_ppl_residual_attn\t"
        "residual_mlp_loss\tresidual_mlp_ppl\tdelta_loss_residual_mlp\tdelta_ppl_residual_mlp\t"
        "residual_both_loss\tresidual_both_ppl\tdelta_loss_residual_both\tdelta_ppl_residual_both\t"
        "baseline_tokens\txsa_tokens\tresidual_attn_tokens\tresidual_mlp_tokens\tresidual_both_tokens\n"
    )
    for row in rows:
        f.write(
            f"{row['prompt_idx']}\t{row['prompt_preview']}\t"
            f"{row['baseline_loss']:.8f}\t{row['baseline_ppl']:.8f}\t"
            f"{row['xsa_loss']:.8f}\t{row['xsa_ppl']:.8f}\t"
            f"{row['delta_loss']:.8f}\t{row['delta_ppl']:.8f}\t"
            f"{row['residual_attn_loss']:.8f}\t{row['residual_attn_ppl']:.8f}\t"
            f"{row['delta_loss_residual_attn']:.8f}\t{row['delta_ppl_residual_attn']:.8f}\t"
            f"{row['residual_mlp_loss']:.8f}\t{row['residual_mlp_ppl']:.8f}\t"
            f"{row['delta_loss_residual_mlp']:.8f}\t{row['delta_ppl_residual_mlp']:.8f}\t"
            f"{row['residual_both_loss']:.8f}\t{row['residual_both_ppl']:.8f}\t"
            f"{row['delta_loss_residual_both']:.8f}\t{row['delta_ppl_residual_both']:.8f}\t"
            f"{row['baseline_tokens']:.0f}\t{row['xsa_tokens']:.0f}\t"
            f"{row['residual_attn_tokens']:.0f}\t{row['residual_mlp_tokens']:.0f}\t{row['residual_both_tokens']:.0f}\n"
        )

print(f"[INFO] Wrote baseline/xsa/residual JSONL to {output_jsonl}")
print(f"[INFO] Wrote baseline/xsa/residual TSV to {output_tsv}")
PY
