#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-/path/to/resource}"
RESIDUAL_OUTPUT_TARGETS="${RESIDUAL_OUTPUT_TARGETS:-none,attn,mlp,both}"
XSA_MIDDLE_TARGETS="${XSA_MIDDLE_TARGETS:-attn}"
DATASET="${DATASET:-builtin}"
JSONL_PATH="${JSONL_PATH:-}"
TEXT_KEY="${TEXT_KEY:-text}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
XSA_START_LAYER="${XSA_START_LAYER:-0}"
XSA_END_LAYER="${XSA_END_LAYER:--1}"
XSA_SKIP_FIRST_N="${XSA_SKIP_FIRST_N:-0}"
XSA_SKIP_LAST_N="${XSA_SKIP_LAST_N:-0}"
XSA_ALPHA="${XSA_ALPHA:-1.0}"
XSA_PERP_SCALE="${XSA_PERP_SCALE:-1.0}"
XSA_TRACK_STATS="${XSA_TRACK_STATS:-true}"
XSA_LAYERWISE_STATS="${XSA_LAYERWISE_STATS:-true}"
XSA_INTERVENTION_SITES="${XSA_INTERVENTION_SITES:-xsa_middle_multihead,residual_output}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-bf16}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
XSA_LOG_DIR="${XSA_LOG_DIR:-representation-analysis/outputs/xsa_logs}"
TMP_BASE_DIR="${TMP_BASE_DIR:-/tmp}"
STREAM_CHILD_LOGS="${STREAM_CHILD_LOGS:-auto}"
PER_TEXT_OUTPUT_JSONL="${PER_TEXT_OUTPUT_JSONL:-}"

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
PROMPTS_PER_JOB="$BATCH_SIZE"

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
  echo "[INFO] Installing transformers==$want for Qwen XSA analysis"
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
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-representation-analysis/outputs/forward_ablation/${MODEL_TAG}}"
RESIDUAL_TARGET_TAG="$(sanitize_tag "$RESIDUAL_OUTPUT_TARGETS")"
MIDDLE_TARGET_TAG="$(sanitize_tag "$XSA_MIDDLE_TARGETS")"
DATA_TAG="$(sanitize_tag "${JSONL_PATH:-$DATASET}")"
TEXT_TAG="$(sanitize_tag "$TEXT_KEY")"
DEVICE_TAG="$(sanitize_tag "${DEVICE:-auto}")"
DTYPE_TAG="$(sanitize_tag "$DTYPE")"
CHAT_TAG="chat${USE_CHAT_TEMPLATE}"
LAYER_TAG="xl${XSA_START_LAYER}-xe${XSA_END_LAYER}-xsf${XSA_SKIP_FIRST_N}-xsl${XSA_SKIP_LAST_N}"
ALPHA_TAG="$(sanitize_tag "${XSA_ALPHA}")"
PERP_TAG="$(sanitize_tag "${XSA_PERP_SCALE}")"
FINAL_OUTPUT_JSON="${OUTPUT_JSON:-${BASE_OUTPUT_DIR}/qwen_xsa_ablation-data${DATA_TAG}-text${TEXT_TAG}-ms${MAX_SAMPLES}-len${MAX_LENGTH}-bs${BATCH_SIZE}-${CHAT_TAG}-dtype${DTYPE_TAG}-dev${DEVICE_TAG}-${LAYER_TAG}-xa${ALPHA_TAG}-xp${PERP_TAG}.json}"

echo "[INFO] FINAL_OUTPUT_JSON=$FINAL_OUTPUT_JSON"
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
echo "[INFO] XSA_LOG_DIR=$XSA_LOG_DIR"
echo "[INFO] TMP_BASE_DIR=$TMP_BASE_DIR"
echo "[INFO] STREAM_CHILD_LOGS=$STREAM_CHILD_LOGS"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES"
echo "[INFO] PROMPTS_PER_JOB=$PROMPTS_PER_JOB"

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
tmp_dir="$(mktemp -d "${TMP_BASE_DIR%/}/qwen_xsa_ablation.XXXXXX")"
mkdir -p "$XSA_LOG_DIR"
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
  output_json="${tmp_dir}/site-${site_tag}-tg${target_tag}.json"
  job_specs+=("${xsa_site}|${targets}|${target_tag}|${output_json}")
done

manifest_path="${tmp_dir}/prompt_manifest.tsv"
MAX_SAMPLES="$MAX_SAMPLES" JSONL_PATH="$JSONL_PATH" TEXT_KEY="$TEXT_KEY" DATASET="$DATASET" MIN_PROMPTS="$NUM_GPUS" MANIFEST_PATH="$manifest_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from qwen_xsa_forward_ablation import BUILTIN_TEXT_SETS, _read_jsonl

max_samples = int(os.environ["MAX_SAMPLES"])
jsonl_path = os.environ["JSONL_PATH"]
text_key = os.environ["TEXT_KEY"]
dataset = os.environ["DATASET"]
min_prompts = max(1, int(os.environ["MIN_PROMPTS"]))
manifest_path = Path(os.environ["MANIFEST_PATH"])
tmp_dir = manifest_path.parent

if jsonl_path:
    texts = _read_jsonl(jsonl_path, text_key)
else:
    texts = list(BUILTIN_TEXT_SETS[dataset])

if not texts:
    raise SystemExit("No texts loaded for prompt manifest.")

original_count = len(texts)
target_prompts = len(texts)
if max_samples > 0:
    target_prompts = max(max_samples, min_prompts)
else:
    target_prompts = max(len(texts), min_prompts)

if len(texts) >= target_prompts:
    texts = texts[:target_prompts]
else:
    repeats = (target_prompts + len(texts) - 1) // len(texts)
    texts = (texts * repeats)[:target_prompts]
    print(
        f"[INFO] Expanded prompt set from {original_count} to {len(texts)} to satisfy MAX_SAMPLES/parallel scheduling.",
        flush=True,
    )

with manifest_path.open("w", encoding="utf-8") as mf:
    for idx, text in enumerate(texts):
        prompt_path = tmp_dir / f"prompt_{idx:03d}.jsonl"
        prompt_path.write_text(json.dumps({text_key: text}, ensure_ascii=False) + "\n", encoding="utf-8")
        preview = text.replace("\t", " ").replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        mf.write(f"{idx}\t{prompt_path}\t{preview}\n")
PY

parse_gpu_ids() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  read -r -a GPU_IDS <<< "$raw"
  if [[ "${#GPU_IDS[@]}" -gt "$NUM_GPUS" ]]; then
    GPU_IDS=("${GPU_IDS[@]:0:$NUM_GPUS}")
  fi
}

shard_manifest_path="${tmp_dir}/shard_manifest.tsv"
NUM_GPUS="$NUM_GPUS" TEXT_KEY="$TEXT_KEY" MANIFEST_PATH="$manifest_path" SHARD_MANIFEST_PATH="$shard_manifest_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

num_gpus = max(1, int(os.environ["NUM_GPUS"]))
text_key = os.environ["TEXT_KEY"]
manifest_path = Path(os.environ["MANIFEST_PATH"])
shard_manifest_path = Path(os.environ["SHARD_MANIFEST_PATH"])
tmp_dir = shard_manifest_path.parent

rows = []
with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        idx, prompt_path, preview = line.split("\t", 2)
        rows.append((int(idx), Path(prompt_path), preview))

if not rows:
    raise SystemExit("No prompt rows found for shard manifest.")

num_workers = min(num_gpus, len(rows))
chunk_size = (len(rows) + num_workers - 1) // num_workers

with shard_manifest_path.open("w", encoding="utf-8") as mf:
    for shard_idx, start in enumerate(range(0, len(rows), chunk_size)):
        chunk = rows[start : start + chunk_size]
        shard_path = tmp_dir / f"prompt_shard_{shard_idx:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as out:
            for _, prompt_path, _ in chunk:
                record = json.loads(prompt_path.read_text(encoding="utf-8").strip())
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        preview = chunk[0][2]
        shard_size = len(chunk)
        first_prompt_idx = chunk[0][0]
        last_prompt_idx = chunk[-1][0]
        mf.write(
            f"{shard_idx}\t{shard_path}\t{shard_size}\t{first_prompt_idx}\t{last_prompt_idx}\t{preview}\n"
        )
PY

run_site_job() {
  local xsa_site="$1"
  local targets="$2"
  local output_json="$3"
  local gpu_id="${4:-}"
  local log_path="$5"
  local stream_logs="${6:-false}"
  local per_text_output_jsonl="${7:-}"
  local text_idx_offset="${8:-0}"
  local progress_label="${9:-}"

  echo "[INFO] Running XSA_INTERVENTION_SITE=$xsa_site"
  echo "[INFO] TARGETS=$targets"
  echo "[INFO] TEMP_OUTPUT_JSON=$output_json"
  [[ -n "$gpu_id" ]] && echo "[INFO] GPU=$gpu_id"

  local -a args=(
    representation-analysis/qwen_xsa_forward_ablation.py
    --model_name "$MODEL_NAME"
    --targets "$targets"
    --dataset "$DATASET"
    --text_key "$TEXT_KEY"
    --max_samples "$MAX_SAMPLES"
    --max_length "$MAX_LENGTH"
    --batch_size "$BATCH_SIZE"
    --xsa_start_layer "$XSA_START_LAYER"
    --xsa_end_layer "$XSA_END_LAYER"
    --xsa_skip_first_n "$XSA_SKIP_FIRST_N"
    --xsa_skip_last_n "$XSA_SKIP_LAST_N"
    --xsa_alpha "$XSA_ALPHA"
    --xsa_perp_scale "$XSA_PERP_SCALE"
    --xsa_intervention_site "$xsa_site"
    --dtype "$DTYPE"
    --system_prompt "$SYSTEM_PROMPT"
    --output_json "$output_json"
    --text_idx_offset "$text_idx_offset"
    --progress_label "$progress_label"
  )

  if [[ -n "$per_text_output_jsonl" ]]; then
    args+=(--per_text_output_jsonl "$per_text_output_jsonl")
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

  if [[ -n "$JSONL_PATH" ]]; then
    args+=(--jsonl_path "$JSONL_PATH")
  fi

  if [[ "$stream_logs" == "true" ]]; then
    if [[ -n "$gpu_id" ]]; then
      env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$log_path"
    else
      "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$log_path"
    fi
  else
    if [[ -n "$gpu_id" ]]; then
      env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "${args[@]}" >"$log_path" 2>&1
    else
      "$PYTHON_BIN" "${args[@]}" >"$log_path" 2>&1
    fi
  fi
}

parse_gpu_ids "$GPU_LIST"
failed_jobs=()
timestamp="$(date +%Y-%m-%dT%H-%M-%S)"
total_sites="${#job_specs[@]}"
mapfile -t manifest_rows < "$manifest_path"
mapfile -t shard_rows < "$shard_manifest_path"
total_prompts="${#manifest_rows[@]}"
total_shards="${#shard_rows[@]}"
num_rounds=$(( (total_shards + ${#GPU_IDS[@]} - 1) / ${#GPU_IDS[@]} ))
child_stream_logs=false
if [[ "${STREAM_CHILD_LOGS,,}" == "true" ]]; then
  child_stream_logs=true
elif [[ "${STREAM_CHILD_LOGS,,}" == "auto" && "$total_shards" -le 8 ]]; then
  child_stream_logs=true
fi
prompt_parallel_enabled=false
if [[ -z "$DEVICE" && "${#GPU_IDS[@]}" -gt 1 && "$total_prompts" -gt 1 ]]; then
  prompt_parallel_enabled=true
fi

echo "[INFO] Total intervention sites scheduled: $total_sites"
echo "[INFO] Total prompts scheduled per site: $total_prompts"
echo "[INFO] Total worker jobs per site: $total_shards"
echo "[INFO] Available GPUs for shard scheduling: ${#GPU_IDS[@]}"
echo "[INFO] Expected prompts per GPU batch step: $PROMPTS_PER_JOB"
echo "[INFO] Derived NUM_ROUNDS per site: $num_rounds"
echo "[INFO] Child worker log streaming enabled: $child_stream_logs"
echo "[INFO] Temporary working directory: $tmp_dir"
if [[ "$prompt_parallel_enabled" == true ]]; then
  echo "[INFO] GPU-worker parallelism enabled across ${#GPU_IDS[@]} GPUs"
  echo "[INFO] Concurrent worker jobs per round: ${#GPU_IDS[@]}"
else
  echo "[INFO] Prompt-level parallelism disabled; running prompts serially within each site"
fi

aggregate_site_results() {
  local site_name="$1"
  local site_output_json="$2"
  local site_tmp_dir="$3"
  local target_csv="$4"
  local shard_count="$5"
  SITE_NAME="$site_name" SITE_OUTPUT_JSON="$site_output_json" SITE_TMP_DIR="$site_tmp_dir" TARGET_CSV="$target_csv" SHARD_COUNT="$shard_count" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

site_name = os.environ["SITE_NAME"]
site_output_json = Path(os.environ["SITE_OUTPUT_JSON"])
site_tmp_dir = Path(os.environ["SITE_TMP_DIR"])
targets = [x.strip() for x in os.environ["TARGET_CSV"].split(",") if x.strip()]
shard_count = int(os.environ["SHARD_COUNT"])

payloads = []
for shard_idx in range(shard_count):
    path = site_tmp_dir / f"result_shard_{shard_idx:03d}.json"
    payloads.append(json.loads(path.read_text(encoding="utf-8")))

first = payloads[0]
site_payload = dict(first)
site_payload["targets"] = {}

def combine_metrics(target_name):
    total_loss_times_tokens = 0.0
    total_tokens = 0.0
    total_batches = 0.0
    for payload in payloads:
        metrics = payload["targets"][target_name]["metrics"]
        tokens = float(metrics.get("tokens", 0.0))
        total_loss_times_tokens += float(metrics["loss"]) * tokens
        total_tokens += tokens
        total_batches += float(metrics.get("batches", 0.0))
    mean_loss = total_loss_times_tokens / max(1.0, total_tokens)
    return {
        "loss": mean_loss,
        "ppl": float(math.exp(mean_loss)) if mean_loss < 50 else float("inf"),
        "tokens": total_tokens,
        "batches": total_batches,
    }

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

per_text_results = []
for shard_idx in range(shard_count):
    per_text_path = site_tmp_dir / f"per_text_result_shard_{shard_idx:03d}.jsonl"
    if per_text_path.exists():
        with per_text_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    per_text_results.append(json.loads(line))

for target in targets:
    target_payloads = [payload["targets"][target] for payload in payloads]
    xsa_stats_dicts = [tp.get("xsa_stats", {}) for tp in target_payloads if tp.get("xsa_stats")]
    layer_window = next((tp.get("layer_window", {}) for tp in target_payloads if tp.get("layer_window")), {})
    site_payload["targets"][target] = {
        "metrics": combine_metrics(target),
        "xsa_stats": merge_nested_dicts(xsa_stats_dicts) if xsa_stats_dicts else {},
        "layer_window": layer_window,
    }

if per_text_results:
    site_payload["per_text_results"] = sorted(per_text_results, key=lambda x: x["text_idx"])

site_output_json.write_text(json.dumps(site_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[INFO] Aggregated site payload for {site_name} -> {site_output_json}")
PY
}

for ((site_idx=0; site_idx<${#job_specs[@]}; ++site_idx)); do
  spec="${job_specs[site_idx]}"
  IFS='|' read -r xsa_site targets _target_tag output_json <<< "$spec"
  site_tag="$(sanitize_tag "$xsa_site")"
  site_tmp_dir="${tmp_dir}/${site_tag}"
  mkdir -p "$site_tmp_dir"
  echo "[INFO] Starting site $((site_idx + 1))/$total_sites: $xsa_site"
  echo "[INFO]   targets=$targets"
  echo "[INFO]   prompts=$total_prompts"
  echo "[INFO]   worker_jobs=$total_shards"
  echo "[INFO]   prompts_per_job=$PROMPTS_PER_JOB"
  echo "[INFO]   texts_per_worker≈$(( (total_prompts + total_shards - 1) / total_shards ))"
  echo "[INFO]   batch_size=$BATCH_SIZE"

  site_failed=0
  site_completed=0
  if [[ "$prompt_parallel_enabled" == true ]]; then
    batch_size_jobs="${#GPU_IDS[@]}"
    for ((batch_start=0; batch_start<${#shard_rows[@]}; batch_start+=batch_size_jobs)); do
      pids=()
      pid_descs=()
      pid_logs=()
      batch_end=$((batch_start + batch_size_jobs))
      if (( batch_end > ${#shard_rows[@]} )); then
        batch_end="${#shard_rows[@]}"
      fi
      for ((i=batch_start; i<batch_end; ++i)); do
        IFS=$'\t' read -r shard_idx shard_file shard_size first_prompt_idx last_prompt_idx shard_preview <<< "${shard_rows[i]}"
        prompt_output="$(printf '%s/result_shard_%03d.json' "$site_tmp_dir" "$shard_idx")"
        per_text_output=""
        if [[ -n "$PER_TEXT_OUTPUT_JSONL" ]]; then
          per_text_output="$(printf '%s/per_text_result_shard_%03d.jsonl' "$site_tmp_dir" "$shard_idx")"
        fi
        progress_label="worker=${shard_idx} prompts=${first_prompt_idx}-${last_prompt_idx} count=${shard_size}"
        gpu_id="${GPU_IDS[$((i - batch_start))]}"
        log_path="${XSA_LOG_DIR%/}/qwen_xsa_ablation-${timestamp}-${site_tag}-shard${shard_idx}-gpu${gpu_id}.log"
        echo "[INFO]   launching worker $((i + 1))/$total_shards on GPU $gpu_id"
        echo "[INFO]     prompts=${first_prompt_idx}-${last_prompt_idx} (count=${shard_size})"
        echo "[INFO]     worker_file=$shard_file"
        echo "[INFO]     log=$log_path"
        JSONL_PATH="$shard_file" MAX_SAMPLES="$shard_size" run_site_job "$xsa_site" "$targets" "$prompt_output" "$gpu_id" "$log_path" "$child_stream_logs" "$per_text_output" "$first_prompt_idx" "$progress_label" &
        pids+=("$!")
        pid_descs+=("$shard_idx")
        pid_logs+=("$log_path")
      done
      for ((j=0; j<${#pids[@]}; ++j)); do
        if ! wait "${pids[j]}"; then
          failed_jobs+=("${xsa_site}/shard${pid_descs[j]}|${pid_logs[j]}")
          site_failed=1
        else
          site_completed=$((site_completed + 1))
          echo "[INFO]   site=$xsa_site progress: ${site_completed}/${total_shards} shard jobs completed"
        fi
      done
    done
  else
    for row in "${shard_rows[@]}"; do
      IFS=$'\t' read -r shard_idx shard_file shard_size first_prompt_idx last_prompt_idx shard_preview <<< "$row"
      prompt_output="$(printf '%s/result_shard_%03d.json' "$site_tmp_dir" "$shard_idx")"
      per_text_output=""
      if [[ -n "$PER_TEXT_OUTPUT_JSONL" ]]; then
        per_text_output="$(printf '%s/per_text_result_shard_%03d.jsonl' "$site_tmp_dir" "$shard_idx")"
      fi
      progress_label="worker=${shard_idx} prompts=${first_prompt_idx}-${last_prompt_idx} count=${shard_size}"
      log_path="${XSA_LOG_DIR%/}/qwen_xsa_ablation-${timestamp}-${site_tag}-shard${shard_idx}.log"
      echo "[INFO]   running worker $((site_completed + 1))/$total_shards serially"
      echo "[INFO]     prompts=${first_prompt_idx}-${last_prompt_idx} (count=${shard_size})"
      echo "[INFO]     worker_file=$shard_file"
      echo "[INFO]     log=$log_path"
      if ! JSONL_PATH="$shard_file" MAX_SAMPLES="$shard_size" run_site_job "$xsa_site" "$targets" "$prompt_output" "" "$log_path" "$child_stream_logs" "$per_text_output" "$first_prompt_idx" "$progress_label"; then
        failed_jobs+=("${xsa_site}/shard${shard_idx}|${log_path}")
        site_failed=1
      else
        site_completed=$((site_completed + 1))
        echo "[INFO]   site=$xsa_site progress: ${site_completed}/${total_shards} shard jobs completed"
      fi
    done
  fi

  if [[ "$site_failed" -ne 0 ]]; then
    echo "[ERROR] Site failed: $xsa_site" >&2
    continue
  fi

  aggregate_site_results "$xsa_site" "$output_json" "$site_tmp_dir" "$targets" "$total_shards"
  echo "[INFO] Completed site $xsa_site"
  site_payloads+=("${xsa_site}:${output_json}")
done

if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
  echo "[ERROR] One or more ablation jobs failed:" >&2
  for item in "${failed_jobs[@]}"; do
    IFS='|' read -r site_name log_path <<< "$item"
    echo "[ERROR]   site=$site_name" >&2
    echo "[ERROR]   Logs:" >&2
    echo "[ERROR]     $log_path" >&2
  done
  exit 1
fi

site_payload_lines="$(printf '%s\n' "${site_payloads[@]}")"

SITE_PAYLOAD_LINES="$site_payload_lines" FINAL_OUTPUT_JSON="$FINAL_OUTPUT_JSON" PER_TEXT_OUTPUT_JSONL="$PER_TEXT_OUTPUT_JSONL" python3 - <<'PY'
import json
import os
from pathlib import Path

site_payloads = []
for line in os.environ["SITE_PAYLOAD_LINES"].splitlines():
    line = line.strip()
    if not line:
        continue
    site, path = line.split(":", 1)
    site_payloads.append({"site": site, "path": path})
final_path = Path(os.environ["FINAL_OUTPUT_JSON"])
combined = {"sites": {}}
combined_per_text = {}

for item in site_payloads:
    site = item["site"]
    path = Path(item["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    combined["sites"][site] = payload
    for rec in payload.get("per_text_results", []):
        row = combined_per_text.setdefault(
            rec["text_idx"],
            {"text_idx": rec["text_idx"], "text": rec["text"], "sites": {}},
        )
        row["sites"][site] = rec["targets"]
    for key in [
        "model_name",
        "dataset",
        "max_samples",
        "max_length",
        "batch_size",
        "use_chat_template",
        "system_prompt",
        "xsa_layer_selection",
    ]:
        combined.setdefault(key, payload.get(key))

final_path.parent.mkdir(parents=True, exist_ok=True)
final_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[INFO] Wrote combined ablation JSON to {final_path}")
per_text_output_jsonl = os.environ.get("PER_TEXT_OUTPUT_JSONL", "").strip()
if per_text_output_jsonl and combined_per_text:
    per_text_path = Path(per_text_output_jsonl)
    per_text_path.parent.mkdir(parents=True, exist_ok=True)
    with per_text_path.open("w", encoding="utf-8") as f:
        for key in sorted(combined_per_text):
            f.write(json.dumps(combined_per_text[key], ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote combined per-text JSONL to {per_text_path}")
PY
