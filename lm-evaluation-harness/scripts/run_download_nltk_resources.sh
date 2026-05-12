#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HARNESS_DIR"

export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Intentionally ignore any inherited NLTK_DATA from the shell by default.
# This avoids accidentally downloading into stale unwritable paths such as
# /home/tiger/nltk_data on shared machines. Use NLTK_DATA_OVERRIDE if you
# explicitly want another target directory.
DEFAULT_NLTK_DATA="/mnt/hdfs/shwai.he/DepthBoost/nltk_data"
INHERITED_NLTK_DATA="${NLTK_DATA:-}"
NLTK_DATA="${NLTK_DATA_OVERRIDE:-$DEFAULT_NLTK_DATA}"
export NLTK_DATA
BACKGROUND="${BACKGROUND:-false}"
LOG_DIR="${LOG_DIR:-$HARNESS_DIR/outputs/xsa_lm_eval_logs}"
mkdir -p "$LOG_DIR"

if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="$LOG_DIR/$(date +"%Y-%m-%dT%H-%M-%S")-download-nltk-resources.log"
fi

if [[ "$BACKGROUND" == "true" && "${_DOWNLOAD_NLTK_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export LOG_FILE
  _DOWNLOAD_NLTK_BACKGROUND_CHILD=1 nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  echo "[INFO] Running in background. PID=$! Logs: $LOG_FILE"
  exit 0
fi

if [[ "${LM_EVAL_LOGGING_INITIALIZED:-0}" != "1" ]]; then
  export LM_EVAL_LOGGING_INITIALIZED=1
  export LOG_FILE
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

mkdir -p "$NLTK_DATA"

if ! "$PYTHON_BIN" -c "import nltk" >/dev/null 2>&1; then
  pip install "nltk>=3.9.1"
fi

if [[ -n "$INHERITED_NLTK_DATA" && "$INHERITED_NLTK_DATA" != "$NLTK_DATA" ]]; then
  echo "[INFO] Ignoring inherited NLTK_DATA=$INHERITED_NLTK_DATA"
fi
echo "[INFO] Downloading NLTK resources into $NLTK_DATA"
"$PYTHON_BIN" "$SCRIPT_DIR/download_nltk_resources.py" --download-dir "$NLTK_DATA" "$@"
