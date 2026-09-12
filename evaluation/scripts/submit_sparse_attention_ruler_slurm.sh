#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-1-00:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-}"
TASKS="${TASKS:-ruler}"
RULER_LENGTHS="${RULER_LENGTHS:-4096}"
KEEP_RATIOS_CSV="${KEEP_RATIOS_CSV:-0.5}"
MODES_CSV="${MODES_CSV:-dense,local_window,longformer,bigbird,sparse_transformer,vertical_slash,block_sparse,minference_mix,attention_topk,contribution_norm_topk,contribution_perp_topk}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/evaluation/outputs/sparse_attention_ruler}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/slurm}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"
SPARSE_ATTENTION_SINK_TOKENS="${SPARSE_ATTENTION_SINK_TOKENS:-32}"
SPARSE_ATTENTION_LOCAL_TOKENS="${SPARSE_ATTENTION_LOCAL_TOKENS:-128}"
SPARSE_ATTENTION_PREFILL_ONLY="${SPARSE_ATTENTION_PREFILL_ONLY:-true}"

if [[ -z "$MODEL_NAME" ]]; then
  echo "[ERROR] MODEL_NAME is required." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
IFS=',' read -r -a modes <<< "$MODES_CSV"
IFS=',' read -r -a ratios <<< "$KEEP_RATIOS_CSV"

for mode in "${modes[@]}"; do
  mode="${mode// /}"
  for ratio in "${ratios[@]}"; do
    ratio="${ratio// /}"
    if [[ "$mode" == "dense" && "$ratio" != "1.0" ]]; then
      continue
    fi
    if [[ "$mode" != "dense" && "$ratio" == "1.0" ]]; then
      continue
    fi
    if [[ "$mode" == "dense" ]]; then
      wrap_body="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-\$HF_HOME/datasets}"
export RULER_CACHE_DIR="${RULER_CACHE_DIR:-$HOME/.cache/ruler}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export TOKENIZERS_PARALLELISM=false
MODEL_NAME="$MODEL_NAME" \
TASKS="$TASKS" \
RULER_LENGTHS="$RULER_LENGTHS" \
LIMIT="$LIMIT" \
BATCH_SIZE="$BATCH_SIZE" \
DTYPE="$DTYPE" \
LAUNCH_MODE="$LAUNCH_MODE" \
SETTING=none \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
OUTPUT_PATH="$OUTPUT_ROOT/${MODEL_NAME##*/}-${TASKS}-dense.json" \
bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"
EOF
)"
    else
      wrap_body="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-\$HF_HOME/datasets}"
export RULER_CACHE_DIR="${RULER_CACHE_DIR:-$HOME/.cache/ruler}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export TOKENIZERS_PARALLELISM=false
MODEL_NAME="$MODEL_NAME" \
TASKS="$TASKS" \
RULER_LENGTHS="$RULER_LENGTHS" \
LIMIT="$LIMIT" \
BATCH_SIZE="$BATCH_SIZE" \
DTYPE="$DTYPE" \
LAUNCH_MODE="$LAUNCH_MODE" \
SPARSE_ATTENTION_MODE="$mode" \
SPARSE_ATTENTION_KEEP_RATIO="$ratio" \
SPARSE_ATTENTION_SINK_TOKENS="$SPARSE_ATTENTION_SINK_TOKENS" \
SPARSE_ATTENTION_LOCAL_TOKENS="$SPARSE_ATTENTION_LOCAL_TOKENS" \
SPARSE_ATTENTION_PREFILL_ONLY="$SPARSE_ATTENTION_PREFILL_ONLY" \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
bash "$SCRIPT_DIR/run_lm_eval_sparse_attention_ruler.sh"
EOF
)"
    fi
    sbatch --parsable \
      --partition="$PARTITION" \
      --qos="$QOS" \
      --job-name="sa-ruler-${mode}-kr${ratio}" \
      --gres="$GRES" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --output="$LOG_ROOT/%x-%j.out" \
      --error="$LOG_ROOT/%x-%j.err" \
      --wrap="$wrap_body"
  done
done
