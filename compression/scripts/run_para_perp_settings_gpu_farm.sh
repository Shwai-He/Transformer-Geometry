#!/usr/bin/env bash
set -euo pipefail

# One-GPU-per-setting launcher for layerwise para/perp runs.
# Each job writes its own log and pid file.

# ===== User config =====
MODEL_NAME="/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B"
PROMPTS_FILE="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/prompts.txt"
OUTPUT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/focused-compression-analysis/outputs/layerwise_para_perp"
MAX_PROMPTS=32
MAX_LENGTH=512
PYTHON_BIN="python3"
SWEEP_ALL_LAYERS=true
REQUIRED_TRANSFORMERS_VERSION="4.52.4"
REQUIRED_TRITON_VERSION="3.0"
AUTO_INSTALL_DEPS=true
# Throttle launches in batches to avoid oversubscription.
BATCH_SIZE=8
WAIT_BETWEEN_BATCH_SEC=3
CLEAN_EMPTY_DIRS=true
AUTO_SUMMARIZE_LOCAL=true

# GPUs used by this launcher (index in nvidia-smi order).
GPUS=(0 1 2 3 4 5 6 7)

# Define one setting per line:
# key|analysis_mode|compression_type|method_name|effect_scope|focus_layer|pruned_or_dropped_root|target_layer|drop_n|strict_quant_loading
# compare_mode is inferred from analysis_mode:
# - pruned  -> dual_model
# - dropped -> single_model_intervene
# - For dropped mode: set pruned_model_name="-" and provide target_layer/drop_n.
# - effect_scope: global | local
# - focus_layer: layer index when effect_scope=local, else -1
#   Use "*" to indicate layer sweep expansion (0..num_layers-1).
# - strict_quant_loading: true/false (effective for quant mode).
SETTINGS=(
  "wanda_24_local|pruned|prune|wanda_2_4|local|*|/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/2-4/wanda|-|0|false"
  "wanda_48_local|pruned|prune|wanda_4_8|local|*|/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/4-8/wanda|-|0|false"
  "wanda_unstructured_local|pruned|prune|wanda_unstructured|local|*|/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/unstructured/wanda|-|0|false"
  "awq_local|pruned|quant|awq_native|local|*|/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B-AWQ|-|0|true"
  # layer-drop baseline: one global run is usually enough for your baseline reference.
  "layer_drop_baseline_attn8|dropped|prune|drop_attn8_baseline|global|-1|-|attn|8|false"
)

# By default run all settings. To run a single setting:
#   SETTING_KEY=wanda_24_local bash scripts/run_para_perp_settings_gpu_farm.sh
SETTING_KEY="${SETTING_KEY:-ALL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/layerwise_para_perp_compare.py"
SUMMARY_PY="$WORKSPACE_ROOT/code/summarize_local_setting_no_plot.py"
AGGREGATE_PY="$WORKSPACE_ROOT/code/aggregate_all_local_results.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/gpu_farm_settings"
mkdir -p "$LOG_DIR"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

check_deps() {
  "$PYTHON_BIN" - <<'PY' "$REQUIRED_TRANSFORMERS_VERSION" "$REQUIRED_TRITON_VERSION"
import sys
req_t, req_triton = sys.argv[1], sys.argv[2]
ok = True

def ver_ok(cur, req):
    # Accept exact match or prefix match on numeric segments:
    # req=3.0 should accept cur=3.0.0
    if cur == req:
        return True
    req_parts = req.split(".")
    cur_parts = cur.split(".")
    if len(cur_parts) < len(req_parts):
        return False
    return cur_parts[:len(req_parts)] == req_parts

try:
    import transformers
    cur_t = transformers.__version__
    if not ver_ok(cur_t, req_t):
        print(f"[WARN] transformers version mismatch: current={cur_t} required={req_t}")
        ok = False
except Exception as e:
    print(f"[WARN] transformers import failed: {e}")
    ok = False
try:
    import triton
    cur_triton = triton.__version__
    if not ver_ok(cur_triton, req_triton):
        print(f"[WARN] triton version mismatch: current={cur_triton} required={req_triton}")
        ok = False
except Exception as e:
    print(f"[WARN] triton import failed: {e}")
    ok = False
sys.exit(0 if ok else 1)
PY
}

if ! check_deps; then
  if [[ "$AUTO_INSTALL_DEPS" == "true" ]]; then
    echo "[INFO] Installing required deps: transformers==${REQUIRED_TRANSFORMERS_VERSION}, triton==${REQUIRED_TRITON_VERSION}"
    "$PYTHON_BIN" -m pip install -U "transformers==${REQUIRED_TRANSFORMERS_VERSION}" "triton==${REQUIRED_TRITON_VERSION}"
    if ! check_deps; then
      echo "[WARN] dependency check still not fully matched after install; continue anyway."
    fi
  else
    echo "[ERROR] Dependency check failed. Set AUTO_INSTALL_DEPS=true to auto-install." >&2
    exit 1
  fi
fi

if [[ ${#SETTINGS[@]} -eq 0 ]]; then
  echo "[ERROR] SETTINGS is empty." >&2
  exit 1
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] GPUS is empty." >&2
  exit 1
fi

detect_num_layers() {
  "$PYTHON_BIN" - <<'PY' "$MODEL_NAME"
import json, os, sys
from transformers import AutoConfig
name = sys.argv[1]
cfg = None
if os.path.isdir(name):
    p = os.path.join(name, "config.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
if cfg is not None:
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if k in cfg:
            print(int(cfg[k]))
            raise SystemExit(0)
ac = AutoConfig.from_pretrained(name, trust_remote_code=True)
for k in ("num_hidden_layers", "n_layer", "num_layers"):
    if hasattr(ac, k):
        print(int(getattr(ac, k)))
        raise SystemExit(0)
raise RuntimeError("Cannot infer num layers.")
PY
}

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/farm_${STAMP}.log"
MASTER_PID="$LOG_DIR/farm_${STAMP}.pid"
echo $$ > "$MASTER_PID"

echo "Logs:"
echo "  $MASTER_LOG"
echo "  $LOG_DIR/<setting>.log"
echo "PID file:"
echo "  $MASTER_PID"

{
  echo "[INFO] start_ts=$STAMP"
  echo "[INFO] model_name=$MODEL_NAME"
  echo "[INFO] prompts_file=$PROMPTS_FILE"
  echo "[INFO] output_dir=$OUTPUT_DIR"
  echo "[INFO] gpus=${GPUS[*]}"
  echo "[INFO] settings_count=${#SETTINGS[@]}"
  echo "[INFO] selected_setting_key=$SETTING_KEY"
} | tee -a "$MASTER_LOG"

job_i=0
NUM_LAYERS="$(detect_num_layers)"
LAST_LAYER=$((NUM_LAYERS - 1))
selected_count=0
selected_method_prefixes=()
selected_effect_scopes=()
batch_pids=()
launched_jobs=0
wait_batch() {
  if [[ ${#batch_pids[@]} -eq 0 ]]; then
    return 0
  fi
  echo "[INFO] waiting current batch (${#batch_pids[@]} jobs)..." | tee -a "$MASTER_LOG"
  local rc=0
  for p in "${batch_pids[@]}"; do
    if ! wait "$p"; then
      rc=1
      echo "[ERROR] job pid=$p failed." | tee -a "$MASTER_LOG"
    fi
  done
  batch_pids=()
  if [[ "$WAIT_BETWEEN_BATCH_SEC" -gt 0 ]]; then
    echo "[INFO] sleep ${WAIT_BETWEEN_BATCH_SEC}s before next batch..." | tee -a "$MASTER_LOG"
    sleep "$WAIT_BETWEEN_BATCH_SEC"
  fi
  return "$rc"
}

overall_rc=0
for setting in "${SETTINGS[@]}"; do
  IFS='|' read -r key analysis_mode compression_type method_name effect_scope focus_layer pruned_or_dropped_root target_layer drop_n strict_quant_loading <<< "$setting"
  if [[ "$SETTING_KEY" != "ALL" && "$key" != "$SETTING_KEY" ]]; then
    continue
  fi
  selected_count=$((selected_count + 1))
  selected_method_prefixes+=("$method_name")
  selected_effect_scopes+=("$effect_scope")
  layer_list=("$focus_layer")
  if [[ "$effect_scope" == "local" && "$focus_layer" == "*" && "$SWEEP_ALL_LAYERS" == "true" ]]; then
    layer_list=()
    for ((li=0; li<=LAST_LAYER; li++)); do
      layer_list+=("$li")
    done
  fi

  for li in "${layer_list[@]}"; do
    gpu="${GPUS[$((job_i % ${#GPUS[@]}))]}"
    job_i=$((job_i + 1))
    run_key="$key"
    run_method="$method_name"
    if [[ "$effect_scope" == "local" ]]; then
      run_key="${key}_l${li}"
      run_method="${method_name}_l${li}"
    fi

    job_log="$LOG_DIR/${STAMP}_${run_key}.log"
    job_pid="$LOG_DIR/${STAMP}_${run_key}.pid"

    compare_mode="single_model_intervene"
    if [[ "$analysis_mode" == "pruned" ]]; then
      compare_mode="dual_model"
    fi

    cmd=(
      "$PYTHON_BIN" "$PY_SCRIPT"
      --model_name "$MODEL_NAME"
      --analysis_mode "$analysis_mode"
      --compare_mode "$compare_mode"
      --compression_type "$compression_type"
      --effect_scope "$effect_scope"
      --method_name "$run_method"
      --prompts_file "$PROMPTS_FILE"
      --max_prompts "$MAX_PROMPTS"
      --max_length "$MAX_LENGTH"
      --output_dir "$OUTPUT_DIR"
    )
    if [[ "$effect_scope" == "local" ]]; then
      cmd+=(--focus_layer "$li")
    fi

    if [[ "$analysis_mode" == "pruned" ]]; then
      cmd+=(--pruned_model_name "$pruned_or_dropped_root")
    fi
    if [[ "$analysis_mode" == "dropped" ]]; then
      cmd+=(--dropped_root_path "$pruned_or_dropped_root" --target_layer "$target_layer" --drop_n "$drop_n")
    fi
    if [[ "$compression_type" == "quant" ]]; then
      cmd+=(--bnb_4bit_compute_dtype float16)
      if [[ "$strict_quant_loading" == "true" ]]; then
        cmd+=(--strict_quant_loading)
      fi
    fi

    echo "[INFO] launch key=$run_key mode=$compare_mode gpu=$gpu log=$job_log" | tee -a "$MASTER_LOG"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONUNBUFFERED=1
      cd "$WORKSPACE_ROOT"
      "${cmd[@]}"
    ) > "$job_log" 2>&1 &
    pid="$!"
    echo "$pid" > "$job_pid"
    batch_pids+=("$pid")
    launched_jobs=$((launched_jobs + 1))

    if [[ "${#batch_pids[@]}" -ge "$BATCH_SIZE" ]]; then
      if ! wait_batch; then
        overall_rc=1
      fi
    fi
  done
done

if [[ "$selected_count" -eq 0 ]]; then
  echo "[ERROR] SETTING_KEY '$SETTING_KEY' not found in SETTINGS." | tee -a "$MASTER_LOG"
  echo "[ERROR] Available keys:" | tee -a "$MASTER_LOG"
  for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r key _ <<< "$setting"
    echo "  - $key" | tee -a "$MASTER_LOG"
  done
  exit 1
fi

if ! wait_batch; then
  overall_rc=1
fi

echo "[INFO] all jobs completed. launched_jobs=$launched_jobs" | tee -a "$MASTER_LOG"

if [[ "$AUTO_SUMMARIZE_LOCAL" == "true" ]]; then
  if [[ -f "$SUMMARY_PY" ]]; then
    summary_out="$OUTPUT_DIR/_summaries"
    mkdir -p "$summary_out"
    for i in "${!selected_method_prefixes[@]}"; do
      method_prefix="${selected_method_prefixes[$i]}"
      effect_scope="${selected_effect_scopes[$i]}"
      if [[ "$effect_scope" != "local" ]]; then
        continue
      fi
      echo "[INFO] summarizing local setting (flip-only rows): method_prefix=$method_prefix" | tee -a "$MASTER_LOG"
      if ! "$PYTHON_BIN" "$SUMMARY_PY" \
        --runs_root "$OUTPUT_DIR" \
        --method_prefix "$method_prefix" \
        --out_dir "$summary_out" >> "$MASTER_LOG" 2>&1; then
        overall_rc=1
        echo "[ERROR] local summary failed for method_prefix=$method_prefix" | tee -a "$MASTER_LOG"
      fi
    done
  else
    echo "[WARN] summary script not found: $SUMMARY_PY" | tee -a "$MASTER_LOG"
  fi

  if [[ -f "$AGGREGATE_PY" ]]; then
    summary_out="$OUTPUT_DIR/_summaries"
    master_tsv="$summary_out/all_settings_master_v2.tsv"
    mkdir -p "$summary_out"
    echo "[INFO] building master table: $master_tsv" | tee -a "$MASTER_LOG"
    if ! "$PYTHON_BIN" "$AGGREGATE_PY" \
      --runs_root "$OUTPUT_DIR" \
      --out_tsv "$master_tsv" >> "$MASTER_LOG" 2>&1; then
      overall_rc=1
      echo "[ERROR] aggregate master table failed" | tee -a "$MASTER_LOG"
    fi
  else
    echo "[WARN] aggregate script not found: $AGGREGATE_PY" | tee -a "$MASTER_LOG"
  fi
fi

if [[ "$CLEAN_EMPTY_DIRS" == "true" ]]; then
  echo "[INFO] cleaning empty directories under output/log roots..." | tee -a "$MASTER_LOG"
  find "$OUTPUT_DIR" -type d -empty -delete 2>/dev/null || true
  find "$LOG_DIR" -type d -empty -delete 2>/dev/null || true
fi
echo "[INFO] tail -f $MASTER_LOG" | tee -a "$MASTER_LOG"
exit "$overall_rc"
