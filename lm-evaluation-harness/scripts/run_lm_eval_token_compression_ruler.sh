#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
MODEL_NAME="${MODEL_NAME:-}"
TASKS="${TASKS:-ruler}"
RULER_LENGTHS="${RULER_LENGTHS:-4096}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
RULER_CACHE_DIR="${RULER_CACHE_DIR:-/beacon-projects/traumallm/.cache/ruler}"
NLTK_DATA="${NLTK_DATA:-/beacon-projects/traumallm/.cache/nltk_data}"

TOKEN_COMPRESSION_MODE="${TOKEN_COMPRESSION_MODE:-value_parallel_drop}"
TOKEN_COMPRESSION_KEEP_RATIO="${TOKEN_COMPRESSION_KEEP_RATIO:-0.5}"
TOKEN_COMPRESSION_KEEP_TOKENS="${TOKEN_COMPRESSION_KEEP_TOKENS:-0}"
TOKEN_COMPRESSION_LAYER="${TOKEN_COMPRESSION_LAYER:-0}"
TOKEN_COMPRESSION_LAYERS="${TOKEN_COMPRESSION_LAYERS:-all}"
TOKEN_COMPRESSION_REF="${TOKEN_COMPRESSION_REF:-last_value}"
TOKEN_COMPRESSION_PRESERVE_FIRST="${TOKEN_COMPRESSION_PRESERVE_FIRST:-32}"
TOKEN_COMPRESSION_PRESERVE_LAST="${TOKEN_COMPRESSION_PRESERVE_LAST:-256}"
TOKEN_COMPRESSION_MIN_CONTEXT="${TOKEN_COMPRESSION_MIN_CONTEXT:-512}"
TOKEN_COMPRESSION_PARALLEL_WEIGHT="${TOKEN_COMPRESSION_PARALLEL_WEIGHT:-0.25}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/token_compression_ruler}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-}"
BACKGROUND="${BACKGROUND:-false}"

if [[ -z "$MODEL_NAME" ]]; then
  echo "[ERROR] MODEL_NAME is required." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$RULER_CACHE_DIR" "$NLTK_DATA"
export HF_HOME HF_DATASETS_CACHE RULER_CACHE_DIR NLTK_DATA HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

build_lengths_metadata() {
  "$PYTHON_BIN" - <<'PY' "$1"
import json, sys
vals = [int(x) for x in sys.argv[1].replace(",", " ").split() if x.strip()]
print(json.dumps({"max_seq_lengths": vals}, separators=(",", ":")))
PY
}

build_lengths_tag() {
  "$PYTHON_BIN" - <<'PY' "$1"
import sys
vals = [int(x) for x in sys.argv[1].replace(",", " ").split() if x.strip()]
print("_".join(f"{v//1024}k" if v % 1024 == 0 else str(v) for v in vals))
PY
}

sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}

MAX_LENGTH="${MAX_LENGTH:-$("$PYTHON_BIN" - <<'PY' "$RULER_LENGTHS"
import sys
vals = [int(x) for x in sys.argv[1].replace(",", " ").split() if x.strip()]
print(max(vals))
PY
)}"
METADATA="${METADATA:-$(build_lengths_metadata "$RULER_LENGTHS")}"
LENGTHS_TAG="$(build_lengths_tag "$RULER_LENGTHS")"
MODEL_TAG="$(sanitize_tag "$MODEL_NAME")"
MODE_TAG="$(sanitize_tag "$TOKEN_COMPRESSION_MODE")"
RATIO_TAG="$(sanitize_tag "$TOKEN_COMPRESSION_KEEP_RATIO")"
TOKENS_TAG="$(sanitize_tag "$TOKEN_COMPRESSION_KEEP_TOKENS")"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/${MODEL_TAG}-${TASKS}-${LENGTHS_TAG}-${MODE_TAG}-kr${RATIO_TAG}-kt${TOKENS_TAG}.json}"

if [[ "$BACKGROUND" == "true" && "${_TOKEN_COMP_BACKGROUND_CHILD:-0}" != "1" ]]; then
  LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/$(date +%Y-%m-%dT%H-%M-%S)-${MODEL_TAG}-${TASKS}-${MODE_TAG}.log}"
  export LOG_FILE
  _TOKEN_COMP_BACKGROUND_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  echo "[INFO] Running in background. PID=$! Logs: $LOG_FILE"
  exit 0
fi

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] TASKS=$TASKS RULER_LENGTHS=$RULER_LENGTHS MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] TOKEN_COMPRESSION_MODE=$TOKEN_COMPRESSION_MODE KEEP_RATIO=$TOKEN_COMPRESSION_KEEP_RATIO KEEP_TOKENS=$TOKEN_COMPRESSION_KEEP_TOKENS"
echo "[INFO] OUTPUT_PATH=$OUTPUT_PATH"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("wonderwords", "nltk") if importlib.util.find_spec(name) is None]
if missing:
    print(f"[ERROR] Missing Python package(s): {', '.join(missing)}", file=sys.stderr)
    print("[ERROR] Install them in PYTHON_BIN env, e.g. `python -m pip install wonderwords nltk`.", file=sys.stderr)
    raise SystemExit(1)
PY

MODEL_NAME="$MODEL_NAME" \
TASKS="$TASKS" \
SETTING="none" \
BATCH_SIZE="$BATCH_SIZE" \
DTYPE="$DTYPE" \
TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
APPLY_CHAT_TEMPLATE="$APPLY_CHAT_TEMPLATE" \
NUM_FEWSHOT="$NUM_FEWSHOT" \
MAX_LENGTH="$MAX_LENGTH" \
LAUNCH_MODE="$LAUNCH_MODE" \
OUTPUT_PATH="$OUTPUT_PATH" \
LIMIT="$LIMIT" \
METADATA="$METADATA" \
GEN_KWARGS="$GEN_KWARGS" \
TOKEN_COMPRESSION_ENABLED=true \
TOKEN_COMPRESSION_MODE="$TOKEN_COMPRESSION_MODE" \
TOKEN_COMPRESSION_KEEP_RATIO="$TOKEN_COMPRESSION_KEEP_RATIO" \
TOKEN_COMPRESSION_KEEP_TOKENS="$TOKEN_COMPRESSION_KEEP_TOKENS" \
TOKEN_COMPRESSION_LAYER="$TOKEN_COMPRESSION_LAYER" \
TOKEN_COMPRESSION_LAYERS="$TOKEN_COMPRESSION_LAYERS" \
TOKEN_COMPRESSION_REF="$TOKEN_COMPRESSION_REF" \
TOKEN_COMPRESSION_PRESERVE_FIRST="$TOKEN_COMPRESSION_PRESERVE_FIRST" \
TOKEN_COMPRESSION_PRESERVE_LAST="$TOKEN_COMPRESSION_PRESERVE_LAST" \
TOKEN_COMPRESSION_MIN_CONTEXT="$TOKEN_COMPRESSION_MIN_CONTEXT" \
TOKEN_COMPRESSION_PARALLEL_WEIGHT="$TOKEN_COMPRESSION_PARALLEL_WEIGHT" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
