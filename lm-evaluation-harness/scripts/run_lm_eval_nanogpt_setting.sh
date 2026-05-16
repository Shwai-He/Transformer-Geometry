#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HARNESS_DIR"
export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

timestamp() { date +"%Y-%m-%dT%H-%M-%S"; }
sanitize_tag() {
  local s="$1"
  s="${s##*/}"
  s="${s//[^A-Za-z0-9._,-]/_}"
  echo "$s"
}
is_int() {
  local v="${1:-}"
  [[ "$v" =~ ^[0-9]+$ ]]
}

ensure_python_import() {
  local module_name="$1"
  local package_name="$2"
  if "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    return 0
  fi
  echo "[INFO] Installing missing dependency: ${package_name}"
  pip install "${package_name}"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
NANOGPT_CKPT="${NANOGPT_CKPT:-/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/nanoGPT/out}"
NANOGPT_REPO_ROOT="${NANOGPT_REPO_ROOT:-$HARNESS_DIR/nanoGPT}"
TASKS="${TASKS:-hellaswag}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
BACKGROUND="${BACKGROUND:-false}"

# Optional XSA runtime overrides
XSA_FORWARD_ONLY="${XSA_FORWARD_ONLY:-}"
XSA_FORWARD_TARGET="${XSA_FORWARD_TARGET:-}"
XSA_FORWARD_REF="${XSA_FORWARD_REF:-}"
XSA_FORWARD_SPACE="${XSA_FORWARD_SPACE:-}"
XSA_FORWARD_OP="${XSA_FORWARD_OP:-}"
XSA_FORWARD_ALPHA="${XSA_FORWARD_ALPHA:-}"

if [[ -z "$NANOGPT_CKPT" ]]; then
  echo "[ERROR] NANOGPT_CKPT is required." >&2
  echo "Example:" >&2
  echo "  NANOGPT_CKPT=/path/to/ckpt_best.pt bash $0" >&2
  exit 1
fi

if [[ ! -f "$NANOGPT_REPO_ROOT/model.py" ]]; then
  candidates=(
    "$HARNESS_DIR/nanoGPT"
    "$HARNESS_DIR/../nanoGPT"
    "$HARNESS_DIR/../../nanoGPT"
  )
  for cand in "${candidates[@]}"; do
    if [[ -f "$cand/model.py" ]]; then
      NANOGPT_REPO_ROOT="$cand"
      break
    fi
  done
fi

if [[ "$BATCH_SIZE" == auto* ]]; then
  echo "[INFO] BATCH_SIZE=$BATCH_SIZE will use lm-eval style automatic batch-size probing"
  echo "[INFO] MAX_BATCH_SIZE=$MAX_BATCH_SIZE"
fi

LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_logs}"
PID_DIR="${PID_DIR:-$HARNESS_DIR/outputs/nanogpt_lm_eval_pids}"

ckpt_tag="$(sanitize_tag "$NANOGPT_CKPT")"
tasks_tag="$(sanitize_tag "$TASKS")"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -d "$NANOGPT_CKPT" ]]; then
    OUTPUT_ROOT="$NANOGPT_CKPT/lm_eval_results"
  else
    OUTPUT_ROOT="$(cd "$(dirname "$NANOGPT_CKPT")" && pwd)/lm_eval_results"
  fi
fi
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$PID_DIR"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/${ckpt_tag}-${tasks_tag}.json}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$(timestamp)-${ckpt_tag}-${tasks_tag}.log}"
PID_FILE="${PID_FILE:-$PID_DIR/$(timestamp)-${ckpt_tag}-${tasks_tag}.pid}"

ensure_python_import "pytablewriter" "pytablewriter"
ensure_python_import "datasets" "datasets"
ensure_python_import "tiktoken" "tiktoken"

if [[ "$BACKGROUND" == "true" ]]; then
  if [[ "${_NANOGPT_LMEVAL_BG_CHILD:-0}" != "1" ]]; then
    export LOG_FILE PID_FILE
    _NANOGPT_LMEVAL_BG_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
    bg_pid="$!"
    echo "$bg_pid" > "$PID_FILE"
    echo "[INFO] Running in background. PID=$bg_pid"
    echo "[INFO] Logs:"
    echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
    echo "[INFO] PID file:"
    echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"
    exit 0
  fi
fi

if [[ "${NANOGPT_DISABLE_INNER_TEE:-0}" != "1" ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "$$" > "$PID_FILE"

cmd=(
  "$PYTHON_BIN" "-u" "$SCRIPT_DIR/run_nanogpt_lm_eval.py"
  --ckpt_path "$NANOGPT_CKPT"
  --nanogpt_repo_root "$NANOGPT_REPO_ROOT"
  --tasks "$TASKS"
  --output_path "$OUTPUT_PATH"
  --batch_size "$BATCH_SIZE"
  --max_batch_size "$MAX_BATCH_SIZE"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --num_fewshot "$NUM_FEWSHOT"
)

[[ -n "$LIMIT" ]] && cmd+=(--limit "$LIMIT")
[[ -n "$MAX_LENGTH" ]] && cmd+=(--max_length "$MAX_LENGTH")
[[ -n "$XSA_FORWARD_ONLY" ]] && cmd+=(--xsa_forward_only "$XSA_FORWARD_ONLY")
[[ -n "$XSA_FORWARD_TARGET" ]] && cmd+=(--xsa_forward_target "$XSA_FORWARD_TARGET")
[[ -n "$XSA_FORWARD_REF" ]] && cmd+=(--xsa_forward_ref "$XSA_FORWARD_REF")
[[ -n "$XSA_FORWARD_SPACE" ]] && cmd+=(--xsa_forward_space "$XSA_FORWARD_SPACE")
[[ -n "$XSA_FORWARD_OP" ]] && cmd+=(--xsa_forward_op "$XSA_FORWARD_OP")
[[ -n "$XSA_FORWARD_ALPHA" ]] && cmd+=(--xsa_forward_alpha "$XSA_FORWARD_ALPHA")

echo "[INFO] NANOGPT_CKPT=$NANOGPT_CKPT"
echo "[INFO] NANOGPT_REPO_ROOT=$NANOGPT_REPO_ROOT"
echo "[INFO] TASKS=$TASKS"
echo "[INFO] OUTPUT_PATH=$OUTPUT_PATH"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] MAX_BATCH_SIZE=$MAX_BATCH_SIZE"
echo "[INFO] Logs:"
echo "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
echo "[INFO] PID file:"
echo "$(cd "$(dirname "$PID_FILE")" && pwd)/$(basename "$PID_FILE")"

"${cmd[@]}"
