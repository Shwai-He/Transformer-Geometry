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
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
MODEL_NAME="${MODEL_NAME:-}"
TASKS="${TASKS:-ruler}"
RULER_LENGTHS="${RULER_LENGTHS:-4096}"
KEEP_RATIOS_CSV="${KEEP_RATIOS_CSV:-0.5}"
MODES_CSV="${MODES_CSV:-value_parallel_drop,uniform}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/lm-evaluation-harness/outputs/token_compression_ruler}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/slurm}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"
TOKEN_COMPRESSION_LAYERS="${TOKEN_COMPRESSION_LAYERS:-all}"
TOKEN_COMPRESSION_PARALLEL_WEIGHT="${TOKEN_COMPRESSION_PARALLEL_WEIGHT:-0.25}"

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
    sbatch --parsable \
      --partition="$PARTITION" \
      --qos="$QOS" \
      --job-name="tc-ruler-${mode}-kr${ratio}" \
      --gres="$GRES" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      --output="$LOG_ROOT/%x-%j.out" \
      --error="$LOG_ROOT/%x-%j.err" \
      --wrap="$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-\$HF_HOME/datasets}"
export RULER_CACHE_DIR="${RULER_CACHE_DIR:-/beacon-projects/traumallm/.cache/ruler}"
export NLTK_DATA="${NLTK_DATA:-/beacon-projects/traumallm/.cache/nltk_data}"
export TOKENIZERS_PARALLELISM=false
MODEL_NAME="$MODEL_NAME" \
TASKS="$TASKS" \
RULER_LENGTHS="$RULER_LENGTHS" \
LIMIT="$LIMIT" \
BATCH_SIZE="$BATCH_SIZE" \
DTYPE="$DTYPE" \
LAUNCH_MODE="$LAUNCH_MODE" \
TOKEN_COMPRESSION_MODE="$mode" \
TOKEN_COMPRESSION_KEEP_RATIO="$ratio" \
TOKEN_COMPRESSION_LAYERS="$TOKEN_COMPRESSION_LAYERS" \
TOKEN_COMPRESSION_PARALLEL_WEIGHT="$TOKEN_COMPRESSION_PARALLEL_WEIGHT" \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
bash "$SCRIPT_DIR/run_lm_eval_token_compression_ruler.sh"
EOF
)"
  done
done
