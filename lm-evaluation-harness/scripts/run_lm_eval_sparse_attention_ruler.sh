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

SPARSE_ATTENTION_MODE="${SPARSE_ATTENTION_MODE:-contribution_perp}"
SPARSE_ATTENTION_KEEP_RATIO="${SPARSE_ATTENTION_KEEP_RATIO:-0.5}"
SPARSE_ATTENTION_KEEP_TOKENS="${SPARSE_ATTENTION_KEEP_TOKENS:-0}"
SPARSE_ATTENTION_SINK_TOKENS="${SPARSE_ATTENTION_SINK_TOKENS:-32}"
SPARSE_ATTENTION_LOCAL_TOKENS="${SPARSE_ATTENTION_LOCAL_TOKENS:-128}"
SPARSE_ATTENTION_PREFILL_ONLY="${SPARSE_ATTENTION_PREFILL_ONLY:-true}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$HARNESS_DIR/outputs/sparse_attention_ruler}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-}"

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
MODE_TAG="$(sanitize_tag "$SPARSE_ATTENTION_MODE")"
RATIO_TAG="$(sanitize_tag "$SPARSE_ATTENTION_KEEP_RATIO")"
TOKENS_TAG="$(sanitize_tag "$SPARSE_ATTENTION_KEEP_TOKENS")"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/${MODEL_TAG}-${TASKS}-${LENGTHS_TAG}-${MODE_TAG}-kr${RATIO_TAG}-kt${TOKENS_TAG}.json}"

echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] TASKS=$TASKS RULER_LENGTHS=$RULER_LENGTHS MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] SPARSE_ATTENTION_MODE=$SPARSE_ATTENTION_MODE KEEP_RATIO=$SPARSE_ATTENTION_KEEP_RATIO KEEP_TOKENS=$SPARSE_ATTENTION_KEEP_TOKENS"
echo "[INFO] OUTPUT_PATH=$OUTPUT_PATH"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("wonderwords", "nltk") if importlib.util.find_spec(name) is None]
if missing:
    print(f"[ERROR] Missing Python package(s): {', '.join(missing)}", file=sys.stderr)
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
SPARSE_ATTENTION_ENABLED=true \
SPARSE_ATTENTION_MODE="$SPARSE_ATTENTION_MODE" \
SPARSE_ATTENTION_KEEP_RATIO="$SPARSE_ATTENTION_KEEP_RATIO" \
SPARSE_ATTENTION_KEEP_TOKENS="$SPARSE_ATTENTION_KEEP_TOKENS" \
SPARSE_ATTENTION_SINK_TOKENS="$SPARSE_ATTENTION_SINK_TOKENS" \
SPARSE_ATTENTION_LOCAL_TOKENS="$SPARSE_ATTENTION_LOCAL_TOKENS" \
SPARSE_ATTENTION_PREFILL_ONLY="$SPARSE_ATTENTION_PREFILL_ONLY" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
