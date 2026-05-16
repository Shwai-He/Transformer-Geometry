#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/mnt/hdfs/shwai.he/DepthBoost/representation-analysis/models/Qwen/Qwen3-4B}"
RESIDUAL_OUTPUT_TARGETS="${RESIDUAL_OUTPUT_TARGETS:-none,attn,mlp,both}"
XSA_MIDDLE_TARGETS="${XSA_MIDDLE_TARGETS:-attn}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-8}"
GEN_USE_CACHE="${GEN_USE_CACHE:-true}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
XSA_ALPHA="${XSA_ALPHA:-1.0}"
XSA_PERP_SCALE="${XSA_PERP_SCALE:-1.0}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-false}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-false}"
XSA_INTERVENTION_SITES="${XSA_INTERVENTION_SITES:-xsa_middle_multihead,residual_output}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-1337}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
TMP_BASE_DIR="${TMP_BASE_DIR:-/tmp}"
XSA_LOG_DIR="${XSA_LOG_DIR:-representation-analysis/outputs/xsa_logs}"

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

infer_transformers_version() {
  if [[ -n "${TRANSFORMERS_VERSION:-}" ]]; then
    echo "$TRANSFORMERS_VERSION"
    return 0
  fi
  local model_name_lc="${MODEL_NAME,,}"
  if [[ "$model_name_lc" == *qwen3.5* || "$model_name_lc" == *qwen3_5* ]]; then
    echo "5.6.2"
  else
    echo "4.52.4"
  fi
}

TRANSFORMERS_VERSION="$(infer_transformers_version)"

ensure_transformers_version() {
  local want="$1"
  if "$PYTHON_BIN" - "$want" <<'PY'
import sys
try:
    import transformers
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if transformers.__version__ == sys.argv[1] else 1)
PY
  then
    return 0
  fi
  echo "[INFO] Installing transformers==$want for Qwen XSA generation"
  "$PYTHON_BIN" -m pip install --upgrade "transformers==$want"
}

ensure_transformers_version "$TRANSFORMERS_VERSION"
sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

MODEL_TAG="$(sanitize_tag "$MODEL_NAME")"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-representation-analysis/outputs/${MODEL_TAG}}"
RESIDUAL_TARGET_TAG="$(sanitize_tag "$RESIDUAL_OUTPUT_TARGETS")"
MIDDLE_TARGET_TAG="$(sanitize_tag "$XSA_MIDDLE_TARGETS")"
PROMPT_TAG="$(sanitize_tag "${PROMPTS_PATH:-builtin}")"
DEVICE_TAG="$(sanitize_tag "${DEVICE:-auto}")"
DTYPE_TAG="$(sanitize_tag "$DTYPE")"
SAMPLE_TAG="sample${DO_SAMPLE}-temp${TEMPERATURE}-topp${TOP_P}-topk${TOP_K}"
CACHE_TAG="cache${GEN_USE_CACHE}"
CHAT_TAG="chat${USE_CHAT_TEMPLATE}"
LAYER_TAG="xl${XSA_START_LAYER}-xe${XSA_END_LAYER}-xsf${XSA_SKIP_FIRST_N}-xsl${XSA_SKIP_LAST_N}"
ALPHA_TAG="$(sanitize_tag "${XSA_ALPHA}")"
PERP_TAG="$(sanitize_tag "${XSA_PERP_SCALE}")"
FINAL_OUTPUT_JSONL="${OUTPUT_JSONL:-${BASE_OUTPUT_DIR}/qwen_xsa_generate-prompt${PROMPT_TAG}-ms${MAX_SAMPLES}-plen${MAX_PROMPT_LENGTH}-new${MAX_NEW_TOKENS}-bs${BATCH_SIZE}-xa${ALPHA_TAG}-xp${PERP_TAG}-${CACHE_TAG}-${CHAT_TAG}-dtype${DTYPE_TAG}-dev${DEVICE_TAG}-seed${SEED}-${LAYER_TAG}-${SAMPLE_TAG}.jsonl}"

echo "[INFO] FINAL_OUTPUT_JSONL=$FINAL_OUTPUT_JSONL"
echo "[INFO] GEN_USE_CACHE=$GEN_USE_CACHE"
echo "[INFO] USE_CHAT_TEMPLATE=$USE_CHAT_TEMPLATE"
echo "[INFO] XSA_INTERVENTION_SITES=$XSA_INTERVENTION_SITES"
echo "[INFO] XSA_MIDDLE_TARGETS=$XSA_MIDDLE_TARGETS"
echo "[INFO] RESIDUAL_OUTPUT_TARGETS=$RESIDUAL_OUTPUT_TARGETS"
echo "[INFO] XSA_ALPHA=$XSA_ALPHA"
echo "[INFO] XSA_PERP_SCALE=$XSA_PERP_SCALE"
echo "[INFO] XSA_TRACK_STATS=$XSA_TRACK_STATS"
echo "[INFO] XSA_LAYERWISE_STATS=$XSA_LAYERWISE_STATS"
echo "[INFO] NUM_GPUS=$NUM_GPUS"
echo "[INFO] GPU_LIST=${GPU_LIST:-<serial>}"
echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"

is_true() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off) return 1 ;;
    *)
      echo "[ERROR] Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a xsa_sites <<< "$XSA_INTERVENTION_SITES"
mkdir -p "$TMP_BASE_DIR"
tmp_dir="$(mktemp -d "${TMP_BASE_DIR%/}/qwen_xsa_generate.XXXXXX")"
mkdir -p "$XSA_LOG_DIR"
effective_prompts_path="$PROMPTS_PATH"
site_payloads=()
job_specs=()
for raw_site in "${xsa_sites[@]}"; do
  xsa_site="$(echo "$raw_site" | xargs)"
  [[ -z "$xsa_site" ]] && continue
  site_tag="$(sanitize_tag "$xsa_site")"
  case "$xsa_site" in
    xsa_middle)
      targets="$XSA_MIDDLE_TARGETS"
      target_tag="$MIDDLE_TARGET_TAG"
      ;;
    xsa_middle_multihead)
      targets="$XSA_MIDDLE_TARGETS"
      target_tag="${MIDDLE_TARGET_TAG}_multihead"
      ;;
    residual_output)
      targets="$RESIDUAL_OUTPUT_TARGETS"
      target_tag="$RESIDUAL_TARGET_TAG"
      ;;
    *)
      echo "[ERROR] Unsupported XSA_INTERVENTION_SITE: $xsa_site" >&2
      exit 1
      ;;
  esac
  output_jsonl="${tmp_dir}/site-${site_tag}-tg${target_tag}.jsonl"
  output_stats_json="${output_jsonl%.jsonl}.stats.json"
  job_specs+=("${xsa_site}|${targets}|${target_tag}|${output_jsonl}|${output_stats_json}")
done

prepared_prompts_path="${tmp_dir}/prepared_prompts.jsonl"
MAX_SAMPLES="$MAX_SAMPLES" NUM_GPUS="$NUM_GPUS" PROMPTS_PATH="$PROMPTS_PATH" PREPARED_PROMPTS_PATH="$prepared_prompts_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from qwen_xsa_forward_generate import DEFAULT_PROMPTS, _read_prompts

max_samples = int(os.environ["MAX_SAMPLES"])
num_gpus = max(1, int(os.environ["NUM_GPUS"]))
prompts_path = os.environ["PROMPTS_PATH"]
prepared_path = Path(os.environ["PREPARED_PROMPTS_PATH"])

prompts = _read_prompts(prompts_path) if prompts_path else list(DEFAULT_PROMPTS)
if not prompts:
    raise SystemExit("No prompts loaded for generation.")

original_count = len(prompts)
target_prompts = len(prompts)
if max_samples > 0:
    target_prompts = max(max_samples, num_gpus)
else:
    target_prompts = max(len(prompts), num_gpus)

if len(prompts) >= target_prompts:
    prompts = prompts[:target_prompts]
else:
    repeats = (target_prompts + len(prompts) - 1) // len(prompts)
    prompts = (prompts * repeats)[:target_prompts]
    print(
        f"[INFO] Expanded generation prompt set from {original_count} to {len(prompts)} to satisfy MAX_SAMPLES/parallel scheduling.",
        flush=True,
    )

with prepared_path.open("w", encoding="utf-8") as f:
    for prompt in prompts:
        f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
PY
effective_prompts_path="$prepared_prompts_path"

worker_manifest_path="${tmp_dir}/worker_manifest.tsv"
NUM_GPUS="$NUM_GPUS" PREPARED_PROMPTS_PATH="$prepared_prompts_path" WORKER_MANIFEST_PATH="$worker_manifest_path" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

num_gpus = max(1, int(os.environ["NUM_GPUS"]))
prepared_path = Path(os.environ["PREPARED_PROMPTS_PATH"])
worker_manifest_path = Path(os.environ["WORKER_MANIFEST_PATH"])
tmp_dir = worker_manifest_path.parent

lines = [line for line in prepared_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit("No prepared prompts found for generation workers.")

num_workers = min(num_gpus, len(lines))
chunk_size = (len(lines) + num_workers - 1) // num_workers
with worker_manifest_path.open("w", encoding="utf-8") as mf:
    for worker_idx, start in enumerate(range(0, len(lines), chunk_size)):
        chunk = lines[start : start + chunk_size]
        worker_path = tmp_dir / f"prompt_worker_{worker_idx:03d}.jsonl"
        worker_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        mf.write(f"{worker_idx}\t{worker_path}\t{len(chunk)}\t{start}\t{start + len(chunk) - 1}\n")
PY

parse_gpu_ids() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  read -r -a GPU_IDS <<< "$raw"
  if [[ "${#GPU_IDS[@]}" -gt "$NUM_GPUS" ]]; then
    GPU_IDS=("${GPU_IDS[@]:0:$NUM_GPUS}")
  fi
}

run_site_job() {
  local xsa_site="$1"
  local targets="$2"
  local output_jsonl="$3"
  local prompts_path="$4"
  local prompt_count="$5"
  local prompt_idx_offset="$6"
  local gpu_id="${7:-}"
  local log_path="$8"
  local progress_label="${9:-}"

  echo "[INFO] Running XSA_INTERVENTION_SITE=$xsa_site"
  echo "[INFO] TARGETS=$targets"
  echo "[INFO] TEMP_OUTPUT_JSONL=$output_jsonl"
  echo "[INFO] PROMPTS_PATH=$prompts_path"
  echo "[INFO] PROMPT_COUNT=$prompt_count"
  [[ -n "$gpu_id" ]] && echo "[INFO] GPU=$gpu_id"

  local -a args=(
    representation-analysis/qwen_xsa_forward_generate.py
    --model_name "$MODEL_NAME"
    --targets "$targets"
    --prompts_path "$prompts_path"
    --max_prompts "$prompt_count"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --batch_size "$BATCH_SIZE"
    --xsa_start_layer "$XSA_START_LAYER"
    --xsa_end_layer "$XSA_END_LAYER"
    --xsa_skip_first_n "$XSA_SKIP_FIRST_N"
    --xsa_skip_last_n "$XSA_SKIP_LAST_N"
    --xsa_alpha "$XSA_ALPHA"
    --xsa_perp_scale "$XSA_PERP_SCALE"
    --xsa_intervention_site "$xsa_site"
    --dtype "$DTYPE"
    --seed "$SEED"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --system_prompt "$SYSTEM_PROMPT"
    --output_jsonl "$output_jsonl"
    --prompt_idx_offset "$prompt_idx_offset"
    --progress_label "$progress_label"
  )

  if is_true "$DO_SAMPLE"; then
    args+=(--do_sample)
  else
    args+=(--no-do_sample)
  fi

  if is_true "$GEN_USE_CACHE"; then
    args+=(--gen_use_cache)
  else
    args+=(--no-gen_use_cache)
  fi

  if is_true "$USE_CHAT_TEMPLATE"; then
    args+=(--use_chat_template)
  else
    args+=(--no-use_chat_template)
  fi

  if is_true "$XSA_LAYERWISE_STATS"; then
    args+=(--xsa_layerwise_stats)
  else
    args+=(--no-xsa_layerwise_stats)
  fi

  if is_true "$XSA_TRACK_STATS"; then
    args+=(--xsa_track_stats)
  else
    args+=(--no-xsa_track_stats)
  fi

  if [[ -n "$DEVICE" ]]; then
    args+=(--device "$DEVICE")
  fi

  if [[ -n "$gpu_id" ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "${args[@]}" >"$log_path" 2>&1
  else
    "$PYTHON_BIN" "${args[@]}" >"$log_path" 2>&1
  fi
}

parse_gpu_ids "$GPU_LIST"
mapfile -t worker_rows < "$worker_manifest_path"
total_workers="${#worker_rows[@]}"
parallel_enabled=false
if [[ -z "$DEVICE" && "${#GPU_IDS[@]}" -gt 1 && "$total_workers" -gt 1 ]]; then
  parallel_enabled=true
fi

failed_jobs=()
timestamp="$(date +%Y-%m-%dT%H-%M-%S)"
echo "[INFO] Total prompts scheduled per site: $(wc -l < "$prepared_prompts_path")"
echo "[INFO] Total worker jobs per site: $total_workers"
echo "[INFO] Available GPUs for generation workers: ${#GPU_IDS[@]}"
echo "[INFO] Derived NUM_ROUNDS for generation per site: $(( (total_workers + ${#GPU_IDS[@]} - 1) / ${#GPU_IDS[@]} ))"

aggregate_site_generation() {
  local site_name="$1"
  local site_output_jsonl="$2"
  local site_output_stats_json="$3"
  local site_tmp_dir="$4"
  local worker_count="$5"
  SITE_NAME="$site_name" SITE_OUTPUT_JSONL="$site_output_jsonl" SITE_OUTPUT_STATS_JSON="$site_output_stats_json" SITE_TMP_DIR="$site_tmp_dir" WORKER_COUNT="$worker_count" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

site_name = os.environ["SITE_NAME"]
site_output_jsonl = Path(os.environ["SITE_OUTPUT_JSONL"])
site_output_stats_json = Path(os.environ["SITE_OUTPUT_STATS_JSON"])
site_tmp_dir = Path(os.environ["SITE_TMP_DIR"])
worker_count = int(os.environ["WORKER_COUNT"])

combined_rows = {}
stats_payloads = []
for worker_idx in range(worker_count):
    worker_jsonl = site_tmp_dir / f"worker_{worker_idx:03d}.jsonl"
    worker_stats = site_tmp_dir / f"worker_{worker_idx:03d}.stats.json"
    with worker_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            combined_rows[rec["prompt_idx"]] = rec
    stats_payloads.append(json.loads(worker_stats.read_text(encoding="utf-8")))

def average_numbers(values):
    return sum(values) / len(values) if values else 0.0

def merge_nested_dicts(dicts):
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    merged = {}
    for key in keys:
        vals = [d[key] for d in dicts if key in d]
        if vals and all(isinstance(v, dict) for v in vals):
            merged[key] = merge_nested_dicts(vals)
        elif vals and all(isinstance(v, (int, float)) for v in vals):
            merged[key] = average_numbers([float(v) for v in vals])
        elif vals:
            merged[key] = vals[0]
    return merged

site_stats = {"targets": {}}
first = stats_payloads[0] if stats_payloads else {}
for key in [
    "model_name",
    "batch_size",
    "gen_use_cache",
    "use_chat_template",
    "system_prompt",
    "xsa_layer_selection",
    "xsa_track_stats",
    "xsa_layerwise_stats",
    "xsa_alpha",
    "xsa_perp_scale",
    "xsa_intervention_site",
    "xsa_intervention_pair",
]:
    if key in first:
        site_stats[key] = first[key]

target_names = set()
for payload in stats_payloads:
    target_names.update(payload.get("targets", {}).keys())
for target in target_names:
    site_stats["targets"][target] = {
        "xsa_stats": merge_nested_dicts([p.get("targets", {}).get(target, {}).get("xsa_stats", {}) for p in stats_payloads if p.get("targets", {}).get(target)]),
        "layer_window": next((p.get("targets", {}).get(target, {}).get("layer_window", {}) for p in stats_payloads if p.get("targets", {}).get(target)), {}),
    }

site_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
with site_output_jsonl.open("w", encoding="utf-8") as f:
    for key in sorted(combined_rows):
        f.write(json.dumps(combined_rows[key], ensure_ascii=False) + "\n")
site_output_stats_json.write_text(json.dumps(site_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[INFO] Aggregated generation payload for {site_name} -> {site_output_jsonl}")
PY
}

for spec in "${job_specs[@]}"; do
  IFS='|' read -r xsa_site targets _target_tag output_jsonl output_stats_json <<< "$spec"
  site_tag="$(sanitize_tag "$xsa_site")"
  site_tmp_dir="${tmp_dir}/${site_tag}"
  mkdir -p "$site_tmp_dir"
  echo "[INFO] Starting generation site: $xsa_site"
  if [[ "$parallel_enabled" == true ]]; then
    pids=()
    pid_descs=()
    pid_logs=()
    for ((i=0; i<${#worker_rows[@]}; ++i)); do
      IFS=$'\t' read -r worker_idx worker_file worker_size first_prompt_idx last_prompt_idx <<< "${worker_rows[i]}"
      worker_output_jsonl="$(printf '%s/worker_%03d.jsonl' "$site_tmp_dir" "$worker_idx")"
      progress_label="worker=${worker_idx} prompts=${first_prompt_idx}-${last_prompt_idx} count=${worker_size}"
      gpu_id="${GPU_IDS[$i]}"
      log_path="${XSA_LOG_DIR%/}/qwen_xsa_generate-${timestamp}-${site_tag}-worker${worker_idx}-gpu${gpu_id}.log"
      echo "[INFO]   launching generation worker $((i + 1))/$total_workers on GPU $gpu_id"
      echo "[INFO]     prompts=${first_prompt_idx}-${last_prompt_idx} (count=${worker_size})"
      echo "[INFO]     worker_file=$worker_file"
      echo "[INFO]     log=$log_path"
      run_site_job "$xsa_site" "$targets" "$worker_output_jsonl" "$worker_file" "$worker_size" "$first_prompt_idx" "$gpu_id" "$log_path" "$progress_label" &
      pids+=("$!")
      pid_descs+=("$worker_idx")
      pid_logs+=("$log_path")
    done
    for ((j=0; j<${#pids[@]}; ++j)); do
      if ! wait "${pids[j]}"; then
        failed_jobs+=("${xsa_site}/worker${pid_descs[j]}|${pid_logs[j]}")
      else
        echo "[INFO] Completed generation worker ${pid_descs[j]} for site=${xsa_site} (log: ${pid_logs[j]})"
      fi
    done
  else
    for row in "${worker_rows[@]}"; do
      IFS=$'\t' read -r worker_idx worker_file worker_size first_prompt_idx last_prompt_idx <<< "$row"
      worker_output_jsonl="$(printf '%s/worker_%03d.jsonl' "$site_tmp_dir" "$worker_idx")"
      progress_label="worker=${worker_idx} prompts=${first_prompt_idx}-${last_prompt_idx} count=${worker_size}"
      log_path="${XSA_LOG_DIR%/}/qwen_xsa_generate-${timestamp}-${site_tag}-worker${worker_idx}.log"
      echo "[INFO]   running generation worker $((worker_idx + 1))/$total_workers serially"
      echo "[INFO]     prompts=${first_prompt_idx}-${last_prompt_idx} (count=${worker_size})"
      echo "[INFO]     worker_file=$worker_file"
      echo "[INFO]     log=$log_path"
      if ! run_site_job "$xsa_site" "$targets" "$worker_output_jsonl" "$worker_file" "$worker_size" "$first_prompt_idx" "" "$log_path" "$progress_label"; then
        failed_jobs+=("${xsa_site}/worker${worker_idx}|${log_path}")
      else
        echo "[INFO] Completed generation worker ${worker_idx} for site=${xsa_site} (log: $log_path)"
      fi
    done
  fi
  aggregate_site_generation "$xsa_site" "$output_jsonl" "$output_stats_json" "$site_tmp_dir" "$total_workers"
  site_payloads+=("${xsa_site}:${output_jsonl}:${output_stats_json}")
done

if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
  echo "[ERROR] One or more generation jobs failed:" >&2
  for item in "${failed_jobs[@]}"; do
    IFS='|' read -r site_name log_path <<< "$item"
    echo "[ERROR]   site=$site_name" >&2
    echo "[ERROR]   Logs:" >&2
    echo "[ERROR]     $log_path" >&2
  done
  exit 1
fi

site_payload_lines="$(printf '%s\n' "${site_payloads[@]}")"

SITE_PAYLOAD_LINES="$site_payload_lines" FINAL_OUTPUT_JSONL="$FINAL_OUTPUT_JSONL" python3 - <<'PY'
import json
import os
from pathlib import Path

site_payloads = []
for line in os.environ["SITE_PAYLOAD_LINES"].splitlines():
    line = line.strip()
    if not line:
        continue
    site, jsonl_path, stats_path = line.split(":", 2)
    site_payloads.append({"site": site, "jsonl_path": jsonl_path, "stats_path": stats_path})
final_jsonl = Path(os.environ["FINAL_OUTPUT_JSONL"])
final_stats = final_jsonl.with_suffix(".stats.json")

combined_rows = {}
combined_stats = {"sites": {}}

for item in site_payloads:
    site = item["site"]
    jsonl_path = Path(item["jsonl_path"])
    stats_path = Path(item["stats_path"])

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = rec["prompt_idx"]
            row = combined_rows.setdefault(
                key,
                {
                    "prompt_idx": rec["prompt_idx"],
                    "prompt": rec["prompt"],
                    "generations": {},
                },
            )
            row["generations"][site] = rec["generations"]

    combined_stats["sites"][site] = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in [
        "model_name",
        "batch_size",
        "gen_use_cache",
        "use_chat_template",
        "system_prompt",
        "xsa_layer_selection",
    ]:
        combined_stats.setdefault(key, combined_stats["sites"][site].get(key))

final_jsonl.parent.mkdir(parents=True, exist_ok=True)
with final_jsonl.open("w", encoding="utf-8") as f:
    for key in sorted(combined_rows):
        f.write(json.dumps(combined_rows[key], ensure_ascii=False) + "\n")

final_stats.write_text(json.dumps(combined_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[INFO] Wrote combined generations to {final_jsonl}")
print(f"[INFO] Wrote combined generation stats to {final_stats}")
PY
