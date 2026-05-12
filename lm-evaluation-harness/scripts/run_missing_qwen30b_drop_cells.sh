#!/usr/bin/env bash
set -euo pipefail

# Dedicated patch run for the two missing Table-1 DROP cells:
#   Qwen3-30B-A3B / Attn Para-Rem. / DROP
#   Qwen3-30B-A3B / Diag. Rem. / DROP
#
# Example:
#   bash scripts/run_missing_qwen30b_drop_cells.sh
#
# By default this runs one setting as eight independent single-GPU shards.
# This is intentional: DROP uses generate_until and tends to under-utilize one
# multi-GPU data-parallel job. Single-GPU shards also make resume/skip safer.
#
# Default setting: residual_attn, because this is the remaining missing
# Qwen3-30B-A3B DROP cell. Change RUN_SETTING below only if another cell
# must be regenerated intentionally.

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/hdfs/shwai.he/models}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_POSTFIX="${MODEL_POSTFIX:-Qwen/Qwen3-30B-A3B}"
MODEL_TAG="${MODEL_TAG:-Qwen_Qwen3-30B-A3B}"
TASK="${TASK:-drop}"
# Which missing setting to run: residual_attn | attn_diag_zero_renorm | both
RUN_SETTING="${RUN_SETTING:-residual_attn}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-4}"
AUTO_RETRY_BATCH="${AUTO_RETRY_BATCH:-true}"
BATCH_SIZE_CANDIDATES="${BATCH_SIZE_CANDIDATES:-8 4 2 1}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-bfloat16}"
DEFAULT_MAX_LENGTH="${DEFAULT_MAX_LENGTH:-4096}"
DROP_MAX_GEN_TOKS="${DROP_MAX_GEN_TOKS:-32}"
# The patch run defaults to one setting at a time, split into eight independent
# single-GPU shards. Do not use the old 4+4 setting split for this script.
LAUNCH_MODE="single"
SPLIT_GPU_GROUPS="false"
GPU_FARM="true"
ALLOW_LEGACY_SPLIT="false"
NUM_SHARDS="${NUM_SHARDS:-8}"
FARM_GPUS="${FARM_GPUS:-$CUDA_VISIBLE_DEVICES}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-true}"
RUN_COLLECT="${RUN_COLLECT:-true}"
USE_CACHE="${USE_CACHE:-true}"
INSTALL_DEPS="${INSTALL_DEPS:-true}"
NLTK_DATA="${NLTK_DATA:-$HARNESS_DIR/nltk_data}"
BACKGROUND="${BACKGROUND:-false}"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
CACHE_DIR="${CACHE_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_cache/$MODEL_TAG}"
SHARD_DIR="${SHARD_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_shards/$MODEL_TAG}"
SAMPLES_DIR="${SAMPLES_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_samples/$MODEL_TAG/$TASK-$NUM_SHARDS-shards}"
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$SHARD_DIR"
mkdir -p "$SAMPLES_DIR"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-missing-qwen30b-drop.log"
fi

if [[ "$BACKGROUND" == "true" && "${_MISSING_QWEN30B_DROP_BG_CHILD:-0}" != "1" ]]; then
  export LOG_FILE
  _MISSING_QWEN30B_DROP_BG_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  echo "[INFO] Running in background. PID=$!  Logs: $LOG_FILE"
  exit 0
fi

if [[ "${LM_EVAL_LOGGING_INITIALIZED:-0}" != "1" ]]; then
  export LM_EVAL_LOGGING_INITIALIZED=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  if [[ -z "${BASE_MODEL_DIR:-}" ]]; then
    echo "[ERROR] Set BASE_MODEL_DIR or MODEL_NAME." >&2
    echo "Example: BASE_MODEL_DIR=/path/to/resource bash scripts/run_missing_qwen30b_drop_cells.sh" >&2
    exit 1
  fi
  MODEL_NAME="${BASE_MODEL_DIR%/}/${MODEL_POSTFIX}"
fi

if [[ ! -e "$MODEL_NAME" && "$MODEL_NAME" == /* ]]; then
  echo "[ERROR] MODEL_NAME path does not exist: $MODEL_NAME" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval/$MODEL_TAG}"
mkdir -p "$OUT_DIR"
mkdir -p "$NLTK_DATA"

ensure_import() {
  local module_name="$1"
  local install_spec="$2"
  if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "[DEPS] Installing ${install_spec}"
    "$PYTHON_BIN" -m pip install "$install_spec"
  fi
}

ensure_editable_lm_eval() {
  if ! "$PYTHON_BIN" -c "import lm_eval" >/dev/null 2>&1; then
    echo "[DEPS] Installing local lm_eval with HF extras"
    (cd "$HARNESS_DIR" && "$PYTHON_BIN" -m pip install -e ".[hf]")
  fi
}

ensure_exact_version() {
  local module_name="$1"
  local expected_version="$2"
  local install_spec="$3"
  if ! "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib, sys
mod = importlib.import_module("${module_name}")
sys.exit(0 if getattr(mod, "__version__", None) == "${expected_version}" else 1)
PY
  then
    echo "[DEPS] Installing ${install_spec}"
    "$PYTHON_BIN" -m pip install "$install_spec"
  fi
}

ensure_min_version() {
  local module_name="$1"
  local min_version="$2"
  local install_spec="$3"
  if ! "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib, sys
from packaging.version import Version
mod = importlib.import_module("${module_name}")
ver = getattr(mod, "__version__", None)
sys.exit(0 if ver is not None and Version(ver) >= Version("${min_version}") else 1)
PY
  then
    echo "[DEPS] Installing ${install_spec}"
    "$PYTHON_BIN" -m pip install "$install_spec"
  fi
}

ensure_nltk_resource() {
  local resource_path="$1"
  local resource_name="$2"
  if ! NLTK_DATA="$NLTK_DATA" "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import nltk
nltk.data.find("${resource_path}")
PY
  then
    echo "[DEPS] Downloading NLTK resource ${resource_name} to ${NLTK_DATA}"
    NLTK_DATA="$NLTK_DATA" "$PYTHON_BIN" - <<PY
import nltk
nltk.download("${resource_name}", download_dir="${NLTK_DATA}")
PY
  fi
}

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[DEPS] Checking lm-eval runtime dependencies"
  ensure_import packaging packaging
  ensure_editable_lm_eval
  ensure_import datasets datasets
  ensure_import evaluate evaluate
  ensure_import pandas pandas
  ensure_import accelerate accelerate
  ensure_import xlsxwriter xlsxwriter
  ensure_import openpyxl openpyxl
  ensure_import bert_score bert_score
  ensure_min_version rouge_score 0.1.2 "rouge_score>=0.1.2"
  ensure_min_version nltk 3.9.1 "nltk>=3.9.1"
  ensure_import absl absl-py
  ensure_exact_version transformers 4.52.4 "transformers==4.52.4"
  ensure_exact_version triton 3.0.0 "triton==3.0.0"
  ensure_nltk_resource "tokenizers/punkt" "punkt"
  ensure_nltk_resource "tokenizers/punkt_tab/english" "punkt_tab"
fi

COMMON_ENV=(
  "MODEL_NAME=$MODEL_NAME"
  "TASKS=$TASK"
  "NUM_FEWSHOT=$NUM_FEWSHOT"
  "LAUNCH_MODE=$LAUNCH_MODE"
  "MAX_LENGTH=$DEFAULT_MAX_LENGTH"
  "DTYPE=$DEFAULT_DTYPE"
  "TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE"
  "APPLY_CHAT_TEMPLATE=$APPLY_CHAT_TEMPLATE"
  "CONFIRM_RUN_UNSAFE_CODE=$CONFIRM_RUN_UNSAFE_CODE"
  "NLTK_DATA=$NLTK_DATA"
)
if [[ -n "$DROP_MAX_GEN_TOKS" ]]; then
  COMMON_ENV+=("GEN_KWARGS=max_gen_toks=${DROP_MAX_GEN_TOKS},until=.")
fi

run_eval() {
  local label="$1"
  local output_path="$2"
  shift 2

  if [[ -f "$output_path" ]]; then
    echo "[SKIP] $label already exists:"
    echo "       $output_path"
    return 0
  fi

  echo "--------------------------------------------------"
  echo "[RUN] $label"
  echo "[MODEL] $MODEL_NAME"
  echo "[TASK] $TASK"
  echo "[OUT] $output_path"

  local batches=()
  if [[ "$AUTO_RETRY_BATCH" == "true" ]]; then
    read -r -a batches <<< "$BATCH_SIZE_CANDIDATES"
  else
    batches=("$DEFAULT_BATCH_SIZE")
  fi

  local status=1
  local bs
  for bs in "${batches[@]}"; do
    echo "[TRY] $label with BATCH_SIZE=$bs"
    if env "${COMMON_ENV[@]}" "BATCH_SIZE=$bs" "OUTPUT_PATH=$output_path" "$@"; then
      return 0
    fi
    status=$?
    echo "[WARN] $label failed with BATCH_SIZE=$bs (status=$status). Trying next batch size if available." >&2
  done
  return "$status"
}

run_eval_async() {
  local label="$1"
  local output_path="$2"
  local gpu_group="$3"
  local job_tag="$4"
  shift 4

  if [[ -f "$output_path" ]]; then
    echo "[SKIP] $label already exists:"
    echo "       $output_path"
    return 0
  fi

  local job_log="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-${job_tag}.log"
  echo "--------------------------------------------------"
  echo "[RUN-BG] $label"
  echo "[GPUS] $gpu_group"
  echo "[OUT] $output_path"
  echo "[LOG] $job_log"

  (
    batches=()
    if [[ "$AUTO_RETRY_BATCH" == "true" ]]; then
      read -r -a batches <<< "$BATCH_SIZE_CANDIDATES"
    else
      batches=("$DEFAULT_BATCH_SIZE")
    fi

    status=1
    for bs in "${batches[@]}"; do
      echo "[TRY] $label with BATCH_SIZE=$bs on GPUs $gpu_group"
      if env \
          "${COMMON_ENV[@]}" \
          "CUDA_VISIBLE_DEVICES=$gpu_group" \
          "BATCH_SIZE=$bs" \
          "OUTPUT_PATH=$output_path" \
          "$@"; then
        exit 0
      fi
      status=$?
      echo "[WARN] $label failed with BATCH_SIZE=$bs (status=$status). Trying next batch size if available." >&2
    done
    exit "$status"
  ) > "$job_log" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_LABELS+=("$label")
  JOB_LOGS+=("$job_log")
}


make_sample_shards() {
  "$PYTHON_BIN" - "$TASK" "$NUM_SHARDS" "$SAMPLES_DIR" <<'PY'
import json
import sys
from pathlib import Path
from datasets import load_dataset

task = sys.argv[1]
num_shards = int(sys.argv[2])
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
if task != "drop":
    raise SystemExit(f"[ERROR] shard helper currently supports task=drop, got {task!r}")
total = len(load_dataset("EleutherAI/drop", split="validation"))
for shard_idx in range(num_shards):
    indices = list(range(shard_idx, total, num_shards))
    path = out_dir / f"{task}-shard{shard_idx:02d}-of-{num_shards:02d}.json"
    path.write_text(json.dumps({task: indices}, separators=(",", ":")))
print(f"[SHARDS] wrote {num_shards} sample files over {total} {task} examples to {out_dir}")
PY
}

combine_drop_shards() {
  local setting_name="$1"
  local final_output="$2"
  shift 2
  "$PYTHON_BIN" - "$setting_name" "$final_output" "$@" <<'PY'
import copy
import json
import sys
from pathlib import Path

setting_name = sys.argv[1]
final_output = Path(sys.argv[2])
paths = [Path(p) for p in sys.argv[3:]]
missing = [str(p) for p in paths if not p.exists()]
if missing:
    raise SystemExit("[ERROR] Missing shard outputs for %s:\n%s" % (setting_name, "\n".join(missing)))
weighted_sum = 0.0
total_n = 0
payloads = []
for path in paths:
    payload = json.loads(path.read_text())
    payloads.append(payload)
    metric = payload.get("results", {}).get("drop", {}).get("f1,none")
    if metric is None:
        raise SystemExit(f"[ERROR] Could not find results.drop['f1,none'] in {path}")
    n_info = payload.get("n-samples", {}).get("drop", {})
    n = n_info.get("effective") or n_info.get("original")
    if not n:
        n = 1
    weighted_sum += float(metric) * int(n)
    total_n += int(n)
combined = copy.deepcopy(payloads[0])
combined.setdefault("results", {}).setdefault("drop", {})["f1,none"] = weighted_sum / total_n
combined["n-samples"] = {"drop": {"original": total_n, "effective": total_n}}
combined.setdefault("configs", {}).setdefault("drop", {})["metadata"] = {
    "combined_from_shards": len(paths),
    "setting": setting_name,
}
final_output.parent.mkdir(parents=True, exist_ok=True)
final_output.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
print(f"[COMBINE] {setting_name}: DROP f1={weighted_sum / total_n * 100.0:.1f} over n={total_n}")
print(f"[COMBINE] wrote {final_output}")
PY
}

run_farm_setting() {
  local setting_name="$1"
  local label="$2"
  local final_output="$3"
  shift 3

  IFS=',' read -r -a farm_gpus <<< "$FARM_GPUS"
  if (( ${#farm_gpus[@]} < NUM_SHARDS )); then
    echo "[ERROR] NUM_SHARDS=$NUM_SHARDS but FARM_GPUS only has ${#farm_gpus[@]} entries: $FARM_GPUS" >&2
    exit 1
  fi

  JOB_PIDS=()
  JOB_LABELS=()
  JOB_LOGS=()
  SHARD_OUTPUTS=()

  for ((shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++)); do
    gpu="${farm_gpus[$shard_idx]}"
    sample_path="$SAMPLES_DIR/${TASK}-shard$(printf '%02d' "$shard_idx")-of-$(printf '%02d' "$NUM_SHARDS").json"
    shard_output="$SHARD_DIR/${TASK}-${setting_name}-shard$(printf '%02d' "$shard_idx")-of-$(printf '%02d' "$NUM_SHARDS").json"
    shard_cache="$CACHE_DIR/${TASK}-${setting_name}-shard$(printf '%02d' "$shard_idx")"
    if [[ "$USE_CACHE" != "true" ]]; then
      shard_cache=""
    fi
    SHARD_OUTPUTS+=("$shard_output")

    run_eval_async \
      "$label / shard $((shard_idx + 1))/$NUM_SHARDS" \
      "$shard_output" \
      "$gpu" \
      "missing-qwen30b-drop-${setting_name}-shard$(printf '%02d' "$shard_idx")" \
      "LAUNCH_MODE=single" \
      "FALLBACK_TO_SINGLE=false" \
      "SAMPLES_PATH=$sample_path" \
      "USE_CACHE_PATH=$shard_cache" \
      "$@"
  done

  failed=0
  for i in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[$i]}"
    label_i="${JOB_LABELS[$i]}"
    job_log="${JOB_LOGS[$i]}"
    echo "[WAIT] $label_i (PID=$pid)"
    if wait "$pid"; then
      echo "[OK] $label_i"
    else
      echo "[ERROR] $label_i failed. See: $job_log" >&2
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    exit 1
  fi

  combine_drop_shards "$setting_name" "$final_output" "${SHARD_OUTPUTS[@]}"
}

RESIDUAL_OUT="$OUT_DIR/${TASK}-residual_attn.json"
DIAG_OUT="$OUT_DIR/${TASK}-attn_diag_zero_renorm.json"
RESIDUAL_CACHE="$CACHE_DIR/${TASK}-residual_attn"
DIAG_CACHE="$CACHE_DIR/${TASK}-attn_diag_zero_renorm"
if [[ "$USE_CACHE" != "true" ]]; then
  RESIDUAL_CACHE=""
  DIAG_CACHE=""
fi

case "$RUN_SETTING" in
  residual_attn|attn_diag_zero_renorm|both) ;;
  *)
    echo "[ERROR] RUN_SETTING must be one of: residual_attn, attn_diag_zero_renorm, both" >&2
    exit 1
    ;;
esac

echo "[CONFIG] RUN_SETTING=$RUN_SETTING GPU_FARM=$GPU_FARM NUM_SHARDS=$NUM_SHARDS FARM_GPUS=$FARM_GPUS SPLIT_GPU_GROUPS=$SPLIT_GPU_GROUPS"
if [[ "$RUN_SETTING" == "both" ]]; then
  echo "[CONFIG] Execution order: residual_attn first, then attn_diag_zero_renorm. They are not launched together."
fi

if [[ "$GPU_FARM" == "true" ]]; then
  echo "--------------------------------------------------"
  echo "[GPU-FARM] Running $TASK as $NUM_SHARDS single-GPU shards over FARM_GPUS=$FARM_GPUS"
  make_sample_shards

  if [[ "$RUN_SETTING" == "residual_attn" || "$RUN_SETTING" == "both" ]]; then
    run_farm_setting \
      "residual_attn" \
      "Qwen3-30B-A3B DROP / Attn Para-Rem. (residual_attn)" \
      "$RESIDUAL_OUT" \
      "SETTING=residual_attn" \
      bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
  fi

  if [[ "$RUN_SETTING" == "attn_diag_zero_renorm" || "$RUN_SETTING" == "both" ]]; then
    run_farm_setting \
      "attn_diag_zero_renorm" \
      "Qwen3-30B-A3B DROP / Diag. Rem. (attn_diag_zero_renorm)" \
      "$DIAG_OUT" \
      "SETTING=none" \
      "ATTN_DIAG_ENABLED=true" \
      "ATTN_DIAG_MODE=zero_renorm" \
      "ATTN_DIAG_KEEP_FIRST=true" \
      bash "$SCRIPT_DIR/run_lm_eval_attn_diag_setting.sh"
  fi
else
  if [[ "$RUN_SETTING" == "residual_attn" || "$RUN_SETTING" == "both" ]]; then
    run_eval \
      "Qwen3-30B-A3B DROP / Attn Para-Rem. (residual_attn)" \
      "$RESIDUAL_OUT" \
      "SETTING=residual_attn" \
      "USE_CACHE_PATH=$RESIDUAL_CACHE" \
      bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
  fi

  if [[ "$RUN_SETTING" == "attn_diag_zero_renorm" || "$RUN_SETTING" == "both" ]]; then
    run_eval \
      "Qwen3-30B-A3B DROP / Diag. Rem. (attn_diag_zero_renorm)" \
      "$DIAG_OUT" \
      "SETTING=none" \
      "ATTN_DIAG_ENABLED=true" \
      "ATTN_DIAG_MODE=zero_renorm" \
      "ATTN_DIAG_KEEP_FIRST=true" \
      "USE_CACHE_PATH=$DIAG_CACHE" \
      bash "$SCRIPT_DIR/run_lm_eval_attn_diag_setting.sh"
  fi
fi

if [[ "$RUN_COLLECT" == "true" ]]; then
  echo "--------------------------------------------------"
  echo "[COLLECT] Refreshing xsa_lm_eval summary CSVs"
  INPUT_ROOT="$HARNESS_DIR/outputs/xsa_lm_eval" \
    bash "$SCRIPT_DIR/collect_xsa_lm_eval_results.sh"
fi

"$PYTHON_BIN" - "$RUN_SETTING" "$RESIDUAL_OUT" "$DIAG_OUT" <<'PY'
import json
import sys
from pathlib import Path

def extract_drop_f1(path: str):
    p = Path(path)
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    metrics = payload.get("results", {}).get("drop", {})
    value = metrics.get("f1,none")
    if value is None:
        raise SystemExit(f"[ERROR] Could not find DROP f1,none in {path}")
    return round(value * 100.0, 1)

run_setting = sys.argv[1]
residual = extract_drop_f1(sys.argv[2])
diag = extract_drop_f1(sys.argv[3])

print("--------------------------------------------------")
print(f"[DONE] RUN_SETTING={run_setting}")
if residual is not None:
    print(f"Attn Para-Rem. / DROP: {residual:.1f}")
if diag is not None:
    print(f"Diag. Rem.      / DROP: {diag:.1f}")
if residual is not None and diag is not None:
    print()
    print("[LATEX] DROP row")
    print(
        "DROP & 7.3 & 5.2 & 0.4 & 7.3 & "
        f"5.5 & {residual:.1f} & {diag:.1f} & 5.4 \\\\"
    )
else:
    print("[INFO] Run the other missing setting to print the full LaTeX row.")
PY
