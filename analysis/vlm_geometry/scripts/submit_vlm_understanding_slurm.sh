#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/beacon-projects/traumallm/shwaihe/Transformer-Geometry}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-/beacon-projects/traumallm/shwaihe/SparseUnifiedModel}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
DATA_ROOT="${DATA_ROOT:-$SPARSE_UNIFIED_ROOT/eval/vlm/data/paper_understanding_jsonl}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_understanding/results}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/vlm_geometry_understanding/logs/slurm}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME=qwenimage, ming, or bagel}"
TASK="${TASK:-mmvp}"
MODEL_PATH="${MODEL_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
GEOMETRY_SPACE="${GEOMETRY_SPACE:-none}"
GEOMETRY_TARGET="${GEOMETRY_TARGET:-block}"
GEOMETRY_PARA_SCALE="${GEOMETRY_PARA_SCALE:-1.0}"
GEOMETRY_PERP_SCALE="${GEOMETRY_PERP_SCALE:-1.0}"
GEOMETRY_LAYER_PATHS="${GEOMETRY_LAYER_PATHS:-}"
GEOMETRY_LAYER_INDICES="${GEOMETRY_LAYER_INDICES:-}"
GEOMETRY_SKIP_FIRST_N="${GEOMETRY_SKIP_FIRST_N:-0}"
GEOMETRY_SKIP_LAST_N="${GEOMETRY_SKIP_LAST_N:-0}"
GEOMETRY_TARGET_MODULE_REGEX="${GEOMETRY_TARGET_MODULE_REGEX:-}"
GEOMETRY_ATTN_NAME_REGEX="${GEOMETRY_ATTN_NAME_REGEX:-}"
GEOMETRY_MLP_NAME_REGEX="${GEOMETRY_MLP_NAME_REGEX:-}"
GEOMETRY_VALUE_NAME_REGEX="${GEOMETRY_VALUE_NAME_REGEX:-}"
GEOMETRY_FAIL_ON_MISSING_TARGET="${GEOMETRY_FAIL_ON_MISSING_TARGET:-true}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
TIME="${TIME:-03:00:00}"
MEM="${MEM:-96G}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
JOB_NAME="${JOB_NAME:-vlmund-${MODEL_NAME}-${TASK}-${GEOMETRY_SPACE}-${GEOMETRY_TARGET}}"

case "$MODEL_NAME" in
  qwenimage|qwen|qwen-image)
    MODEL_NAME="qwenimage"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Qwen-Image}"
    GEOMETRY_PRESET="${GEOMETRY_PRESET:-qwen-image}"
    ;;
  ming|ming-omni)
    MODEL_NAME="ming"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Ming-Lite-Omni-1.5}"
    GEOMETRY_PRESET="${GEOMETRY_PRESET:-ming}"
    ;;
  bagel|bagel-vlm)
    MODEL_NAME="bagel"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/hf/BAGEL-7B-MoT}"
    GEOMETRY_PRESET="${GEOMETRY_PRESET:-bagel}"
    ;;
  *)
    echo "Unsupported MODEL_NAME=$MODEL_NAME" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_ROOT"

scale_tag="para_${GEOMETRY_PARA_SCALE//./p}__perp_${GEOMETRY_PERP_SCALE//./p}"
scale_tag="${scale_tag//-/m}"
OUT_DIR="$RESULT_ROOT/by_model/$MODEL_NAME/$TASK/space_${GEOMETRY_SPACE}__target_${GEOMETRY_TARGET}__${scale_tag}__n${TOTAL_SAMPLES}"
DATA_FILE="$DATA_ROOT/$TASK.jsonl"

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
  --total-samples $(quote_shell "$TOTAL_SAMPLES") \\
  --max-new-tokens $(quote_shell "$MAX_NEW_TOKENS") \\
  --dtype $(quote_shell "$DTYPE") \\
  --seed $(quote_shell "$SEED") \\
  --keep-ratio 1.0 \\
  --sparse-mode prune \\
  --geometry-space $(quote_shell "$GEOMETRY_SPACE") \\
  --geometry-preset $(quote_shell "$GEOMETRY_PRESET") \\
  --geometry-target $(quote_shell "$GEOMETRY_TARGET") \\
  --geometry-para-scale $(quote_shell "$GEOMETRY_PARA_SCALE") \\
  --geometry-perp-scale $(quote_shell "$GEOMETRY_PERP_SCALE") \\
  --geometry-layer-paths $(quote_shell "$GEOMETRY_LAYER_PATHS") \\
  --geometry-layer-indices $(quote_shell "$GEOMETRY_LAYER_INDICES") \\
  --geometry-skip-first-n $(quote_shell "$GEOMETRY_SKIP_FIRST_N") \\
  --geometry-skip-last-n $(quote_shell "$GEOMETRY_SKIP_LAST_N") \\
  --geometry-target-module-regex $(quote_shell "$GEOMETRY_TARGET_MODULE_REGEX") \\
  --geometry-attn-name-regex $(quote_shell "$GEOMETRY_ATTN_NAME_REGEX") \\
  --geometry-mlp-name-regex $(quote_shell "$GEOMETRY_MLP_NAME_REGEX") \\
  --geometry-value-name-regex $(quote_shell "$GEOMETRY_VALUE_NAME_REGEX") \\
  --geometry-fail-on-missing-target $(quote_shell "$GEOMETRY_FAIL_ON_MISSING_TARGET")
EOF
chmod +x "$JOB_SCRIPT"

sbatch \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --job-name="$JOB_NAME" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME" \
  --output="$LOG_ROOT/%x-%j.out" \
  --export=ALL \
  "$JOB_SCRIPT"
