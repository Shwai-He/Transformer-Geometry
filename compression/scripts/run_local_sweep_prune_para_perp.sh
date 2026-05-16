#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
MODEL_NAME="/mnt/bn/seed-aws-va/shwai.he/models/Qwen/Qwen3-4B"
PRUNED_MODEL_NAME="/mnt/bn/seed-aws-va/shwai.he/wanda/out/Qwen/Qwen3-4B/unstructured/wanda"
PROMPTS_FILE="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/prompts.txt"
OUTPUT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/compression/outputs/layerwise_para_perp"
METHOD_PREFIX="local_wanda50"
MAX_PROMPTS=32
MAX_LENGTH=512
START_LAYER=0
END_LAYER=-1            # -1 means last layer
PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/layerwise_para_perp_compare.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/local_sweep_prune"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="local_sweep_prune_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

detect_num_layers() {
  "$PYTHON_BIN" - <<'PY' "$MODEL_NAME"
import json, os, sys
from transformers import AutoConfig
model_name = sys.argv[1]
cfg = None
if os.path.isdir(model_name):
    p = os.path.join(model_name, "config.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
if cfg is not None:
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if k in cfg:
            print(int(cfg[k]))
            raise SystemExit(0)
ac = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
for k in ("num_hidden_layers", "n_layer", "num_layers"):
    if hasattr(ac, k):
        print(int(getattr(ac, k)))
        raise SystemExit(0)
raise RuntimeError("Cannot infer num layers.")
PY
}

run_sweep() {
  cd "$WORKSPACE_ROOT"
  local n_layers
  n_layers="$(detect_num_layers)"
  local last_layer=$((n_layers - 1))
  local end_layer="$END_LAYER"
  if [[ "$end_layer" -lt 0 || "$end_layer" -gt "$last_layer" ]]; then
    end_layer="$last_layer"
  fi
  if [[ "$START_LAYER" -lt 0 || "$START_LAYER" -gt "$end_layer" ]]; then
    echo "[ERROR] Invalid layer range: START_LAYER=$START_LAYER END_LAYER=$end_layer (num_layers=$n_layers)" >&2
    exit 1
  fi

  echo "[INFO] num_layers=$n_layers start=$START_LAYER end=$end_layer"
  for ((layer=START_LAYER; layer<=end_layer; layer++)); do
    local method_name="${METHOD_PREFIX}_l${layer}"
    local layer_log="$LOG_DIR/${RUN_TAG}_l${layer}.log"
    echo "[INFO] ===== layer $layer / $end_layer : $method_name ====="
    echo "[INFO] layer log: $layer_log"
    "$PYTHON_BIN" "$PY_SCRIPT" \
      --model_name "$MODEL_NAME" \
      --analysis_mode pruned \
      --compression_type prune \
      --effect_scope local \
      --focus_layer "$layer" \
      --method_name "$method_name" \
      --pruned_model_name "$PRUNED_MODEL_NAME" \
      --prompts_file "$PROMPTS_FILE" \
      --max_prompts "$MAX_PROMPTS" \
      --max_length "$MAX_LENGTH" \
      --output_dir "$OUTPUT_DIR" \
      >"$layer_log" 2>&1
  done
}

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"

if [[ "$BACKGROUND" == "true" && "${RUN_FOREGROUND:-0}" != "1" ]]; then
  nohup env RUN_FOREGROUND=1 bash "$0" >"$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
  echo "Started in background."
  echo "tail -f $LOG_PATH"
  exit 0
fi

run_sweep 2>&1 | tee "$LOG_PATH"
