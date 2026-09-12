#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-$HOME/SparseUnifiedModel}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$SPARSE_UNIFIED_ROOT/eval/vlm/data/paper_understanding_jsonl}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_accuracy/scaling_benchmark}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/vlm_geometry_accuracy/logs/slurm}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME=qwenimage or ming}"
TASK="${TASK:-mmvp}"
MODEL_PATH="${MODEL_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SPACE="${SPACE:-residual}"
TARGET="${TARGET:-block}"
PARA_SCALE="${PARA_SCALE:-1.0}"
PERP_SCALE="${PERP_SCALE:-1.0}"
LAYER_PATHS="${LAYER_PATHS:-}"
LAYER_INDICES="${LAYER_INDICES:-}"
TARGET_MODULE_REGEX="${TARGET_MODULE_REGEX:-}"
FAIL_ON_MISSING_TARGET="${FAIL_ON_MISSING_TARGET:-true}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
TIME="${TIME:-04:00:00}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"

case "$MODEL_NAME" in
  qwenimage|qwen|qwen-image)
    MODEL_NAME="qwenimage"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Qwen-Image}"
    MODEL_PRESET="${MODEL_PRESET:-qwen-image}"
    ;;
  ming|ming-omni)
    MODEL_NAME="ming"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Ming-Lite-Omni-1.5}"
    MODEL_PRESET="${MODEL_PRESET:-ming}"
    ;;
  *)
    echo "Unsupported MODEL_NAME=$MODEL_NAME" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_ROOT"

para_tag="${PARA_SCALE//./p}"
perp_tag="${PERP_SCALE//./p}"
para_tag="${para_tag//-/m}"
perp_tag="${perp_tag//-/m}"
SETTING="side_und__space_${SPACE}__target_${TARGET}__para_${para_tag}__perp_${perp_tag}"
OUT_DIR="$RESULT_ROOT/by_model/$MODEL_NAME/accuracy/$TASK/$SETTING"
DATA_FILE="$DATA_ROOT/$TASK.jsonl"
JOB_NAME="${JOB_NAME:-vlmacc-${MODEL_NAME}-${TASK}-${SPACE}-${TARGET}-p${para_tag}-r${perp_tag}}"

quote_shell() {
  printf "%q" "$1"
}

JOB_SCRIPT="$LOG_ROOT/${JOB_NAME}.$(date +%Y%m%d-%H%M%S).job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export REPO_ROOT=$(quote_shell "$REPO_ROOT")
export SPARSE_UNIFIED_ROOT=$(quote_shell "$SPARSE_UNIFIED_ROOT")
export PYTHONPATH="\$REPO_ROOT:\$SPARSE_UNIFIED_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "\$SPARSE_UNIFIED_ROOT"
"$(quote_shell "$PYTHON_BIN")" eval/vlm/eval_qwen_ming_understanding_jsonl.py \\
  --model-name $(quote_shell "$MODEL_NAME") \\
  --model-path $(quote_shell "$MODEL_PATH") \\
  --processor-path $(quote_shell "$PROCESSOR_PATH") \\
  --data-file $(quote_shell "$DATA_FILE") \\
  --output-dir $(quote_shell "$OUT_DIR") \\
  --keep-ratio 1.0 \\
  --total-samples $(quote_shell "$TOTAL_SAMPLES") \\
  --max-new-tokens $(quote_shell "$MAX_NEW_TOKENS") \\
  --dtype $(quote_shell "$DTYPE") \\
  --seed $(quote_shell "$SEED") \\
  --geometry-space $(quote_shell "$SPACE") \\
  --geometry-preset $(quote_shell "$MODEL_PRESET") \\
  --geometry-target $(quote_shell "$TARGET") \\
  --geometry-para-scale $(quote_shell "$PARA_SCALE") \\
  --geometry-perp-scale $(quote_shell "$PERP_SCALE") \\
  --geometry-layer-paths $(quote_shell "$LAYER_PATHS") \\
  --geometry-layer-indices $(quote_shell "$LAYER_INDICES") \\
  --geometry-target-module-regex $(quote_shell "$TARGET_MODULE_REGEX") \\
  --geometry-fail-on-missing-target $(quote_shell "$FAIL_ON_MISSING_TARGET")
EOF
chmod +x "$JOB_SCRIPT"

sbatch \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --job-name="$JOB_NAME" \
  --gres="$GRES" \
  --mem="$MEM" \
  --time="$TIME" \
  --output="$LOG_ROOT/%x-%j.out" \
  --export=ALL \
  "$JOB_SCRIPT"
