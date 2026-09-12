#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-$HOME/SparseUnifiedModel}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$SPARSE_UNIFIED_ROOT/eval/vlm/data/paper_understanding_jsonl}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_ppl/results}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/vlm_geometry_ppl/logs/slurm}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME=qwenimage, ming, or bagel}"
TASK="${TASK:-mmvp}"
MODEL_PATH="${MODEL_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-32}"
ANSWER_MODE="${ANSWER_MODE:-letter}"
SPACE="${SPACE:-residual}"
TARGET="${TARGET:-block}"
PARA_SCALE="${PARA_SCALE:-1.0}"
PERP_SCALE="${PERP_SCALE:-1.0}"
LAYER_PATHS="${LAYER_PATHS:-}"
LAYER_INDICES="${LAYER_INDICES:-}"
SKIP_FIRST_N="${SKIP_FIRST_N:-0}"
SKIP_LAST_N="${SKIP_LAST_N:-0}"
TARGET_MODULE_REGEX="${TARGET_MODULE_REGEX:-}"
ATTN_NAME_REGEX="${ATTN_NAME_REGEX:-}"
MLP_NAME_REGEX="${MLP_NAME_REGEX:-}"
VALUE_NAME_REGEX="${VALUE_NAME_REGEX:-}"
FAIL_ON_MISSING_TARGET="${FAIL_ON_MISSING_TARGET:-true}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-42}"
VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}"
VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}"
ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}"
DISABLE_GEOMETRY="${DISABLE_GEOMETRY:-false}"
SCALE_MODE="${SCALE_MODE:-none}"
SCALE_SEED="${SCALE_SEED:-0}"
PARA_SCALE_MIN="${PARA_SCALE_MIN:--1.5}"
PARA_SCALE_MAX="${PARA_SCALE_MAX:-1.5}"
PERP_SCALE_MIN="${PERP_SCALE_MIN:--1.5}"
PERP_SCALE_MAX="${PERP_SCALE_MAX:-1.5}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
TIME="${TIME:-03:00:00}"
MEM="${MEM:-96G}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-vlm-ppl-${MODEL_NAME}-${TASK}-${TARGET}}"

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
  bagel|bagel-vlm)
    MODEL_NAME="bagel"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/hf/BAGEL-7B-MoT}"
    MODEL_PRESET="${MODEL_PRESET:-bagel}"
    ;;
  *)
    echo "Unsupported MODEL_NAME=$MODEL_NAME" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_ROOT"

scale_tag="para_${PARA_SCALE//./p}__perp_${PERP_SCALE//./p}"
scale_tag="${scale_tag//-/m}"
mode_tag="vhead_${VALUE_HEAD_MODE}"
ref_tag="vref_${VALUE_REF_EXPANSION}__attr_${ATTN_ATTR_SOURCE}"
rand_tag="mode_${SCALE_MODE}__seed_${SCALE_SEED}__pr_${PARA_SCALE_MIN//./p}_${PARA_SCALE_MAX//./p}__pe_${PERP_SCALE_MIN//./p}_${PERP_SCALE_MAX//./p}"
rand_tag="${rand_tag//-/m}"
geom_tag="geom_enabled"
if [[ "${DISABLE_GEOMETRY,,}" == "true" || "$DISABLE_GEOMETRY" == "1" ]]; then
  geom_tag="geom_disabled"
fi
OUT_DIR="$RESULT_ROOT/by_model/$MODEL_NAME/ppl/$TASK/${geom_tag}__side_und__space_${SPACE}__target_${TARGET}__${scale_tag}__${mode_tag}__${ref_tag}__${rand_tag}"
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
cd "\$REPO_ROOT"
"$(quote_shell "$PYTHON_BIN")" analysis/vlm_geometry/run_vlm_ppl.py \\
  --model-name $(quote_shell "$MODEL_NAME") \\
  --model-path $(quote_shell "$MODEL_PATH") \\
  --processor-path $(quote_shell "$PROCESSOR_PATH") \\
  --data-file $(quote_shell "$DATA_FILE") \\
  --output-dir $(quote_shell "$OUT_DIR") \\
  --total-samples $(quote_shell "$TOTAL_SAMPLES") \\
  --answer-mode $(quote_shell "$ANSWER_MODE") \\
  --model-preset $(quote_shell "$MODEL_PRESET") \\
  --side und \\
  --space $(quote_shell "$SPACE") \\
  --target $(quote_shell "$TARGET") \\
  --para-scale $(quote_shell "$PARA_SCALE") \\
  --perp-scale $(quote_shell "$PERP_SCALE") \\
  --layer-paths $(quote_shell "$LAYER_PATHS") \\
  --layer-indices $(quote_shell "$LAYER_INDICES") \\
  --skip-first-n $(quote_shell "$SKIP_FIRST_N") \\
  --skip-last-n $(quote_shell "$SKIP_LAST_N") \\
  --target-module-regex $(quote_shell "$TARGET_MODULE_REGEX") \\
  --attn-name-regex $(quote_shell "$ATTN_NAME_REGEX") \\
  --mlp-name-regex $(quote_shell "$MLP_NAME_REGEX") \\
  --value-name-regex $(quote_shell "$VALUE_NAME_REGEX") \\
  --fail-on-missing-target $(quote_shell "$FAIL_ON_MISSING_TARGET") \\
  --dtype $(quote_shell "$DTYPE") \\
  --device-map $(quote_shell "$DEVICE_MAP") \\
  --value-head-mode $(quote_shell "$VALUE_HEAD_MODE") \\
  --value-ref-expansion $(quote_shell "$VALUE_REF_EXPANSION") \\
  --attn-attr-source $(quote_shell "$ATTN_ATTR_SOURCE") \\
  --disable-geometry $(quote_shell "$DISABLE_GEOMETRY") \\
  --scale-mode $(quote_shell "$SCALE_MODE") \\
  --scale-seed $(quote_shell "$SCALE_SEED") \\
  --para-scale-min $(quote_shell "$PARA_SCALE_MIN") \\
  --para-scale-max $(quote_shell "$PARA_SCALE_MAX") \\
  --perp-scale-min $(quote_shell "$PERP_SCALE_MIN") \\
  --perp-scale-max $(quote_shell "$PERP_SCALE_MAX") \\
  --seed $(quote_shell "$SEED") \\
  --sparse-unified-root "\$SPARSE_UNIFIED_ROOT"
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
