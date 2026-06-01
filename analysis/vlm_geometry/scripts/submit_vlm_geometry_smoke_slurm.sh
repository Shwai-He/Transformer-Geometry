#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/beacon-projects/traumallm/shwaihe/Transformer-Geometry}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-/beacon-projects/traumallm/shwaihe/SparseUnifiedModel}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_scaling/results}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/vlm_geometry_scaling/logs/slurm}"

mkdir -p "$LOG_ROOT"

MODEL_SPECS="${MODEL_SPECS:-qwenimage|qwen-image|sparse_qwenimage|$SPARSE_UNIFIED_ROOT/models/Qwen-Image|und,gen;ming|ming|sparse_ming|$SPARSE_UNIFIED_ROOT/models/Ming-Lite-Omni-1.5|und,gen}"
SPACES="${SPACES:-residual,value}"
TARGETS_RESIDUAL="${TARGETS_RESIDUAL:-${TARGETS:-block,attn,mlp}}"
TARGETS_VALUE="${TARGETS_VALUE:-value}"
PARA_SCALES="${PARA_SCALES:-1.0,0.0}"
PERP_SCALES="${PERP_SCALES:-1.0,0.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
PROMPT="${PROMPT:-A small red cube on a wooden table.}"
METADATA_FILE="${METADATA_FILE:-}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"
BENCH_NAME="${BENCH_NAME:-smoke}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-1}"
ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
REQUIRE_STATS="${REQUIRE_STATS:-true}"
DISABLE_GEOMETRY="${DISABLE_GEOMETRY:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
MODE_DEFAULT="${MODE_DEFAULT:-}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
CFG_SCALE="${CFG_SCALE:-5.0}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-1.5}"
SEED="${SEED:-42}"
VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}"
VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}"
ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}"
FAIL_ON_MISSING_TARGET="${FAIL_ON_MISSING_TARGET:-true}"
MING_LOAD_IMAGE_GEN="${MING_LOAD_IMAGE_GEN:-auto}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
TIME="${TIME:-04:00:00}"
MEM="${MEM:-128G}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-vlm-geo-smoke}"

quote_shell() {
  printf "%q" "$1"
}

JOB_SCRIPT="$LOG_ROOT/${JOB_NAME}.$(date +%Y%m%d-%H%M%S).job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export REPO_ROOT=$(quote_shell "$REPO_ROOT")
export SPARSE_UNIFIED_ROOT=$(quote_shell "$SPARSE_UNIFIED_ROOT")
export PYTHON_BIN=$(quote_shell "$PYTHON_BIN")
export MODEL_SPECS=$(quote_shell "$MODEL_SPECS")
export SPACES=$(quote_shell "$SPACES")
export TARGETS_RESIDUAL=$(quote_shell "$TARGETS_RESIDUAL")
export TARGETS_VALUE=$(quote_shell "$TARGETS_VALUE")
export PARA_SCALES=$(quote_shell "$PARA_SCALES")
export PERP_SCALES=$(quote_shell "$PERP_SCALES")
export RESULT_ROOT=$(quote_shell "$RESULT_ROOT")
export NUM_INFERENCE_STEPS=$(quote_shell "$NUM_INFERENCE_STEPS")
export HEIGHT=$(quote_shell "$HEIGHT")
export WIDTH=$(quote_shell "$WIDTH")
export PROMPT=$(quote_shell "$PROMPT")
export METADATA_FILE=$(quote_shell "$METADATA_FILE")
export MAX_PROMPTS=$(quote_shell "$MAX_PROMPTS")
export PROMPT_KEY=$(quote_shell "$PROMPT_KEY")
export NUM_IMAGES_PER_PROMPT=$(quote_shell "$NUM_IMAGES_PER_PROMPT")
export BENCH_NAME=$(quote_shell "$BENCH_NAME")
export ENABLE_MODEL_CPU_OFFLOAD=$(quote_shell "$ENABLE_MODEL_CPU_OFFLOAD")
export ENABLE_SEQUENTIAL_CPU_OFFLOAD=$(quote_shell "$ENABLE_SEQUENTIAL_CPU_OFFLOAD")
export CONTINUE_ON_ERROR=$(quote_shell "$CONTINUE_ON_ERROR")
export REQUIRE_STATS=$(quote_shell "$REQUIRE_STATS")
export DISABLE_GEOMETRY=$(quote_shell "$DISABLE_GEOMETRY")
export SKIP_EXISTING=$(quote_shell "$SKIP_EXISTING")
export MODE_DEFAULT=$(quote_shell "$MODE_DEFAULT")
export DTYPE=$(quote_shell "$DTYPE")
export DEVICE=$(quote_shell "$DEVICE")
export DEVICE_MAP=$(quote_shell "$DEVICE_MAP")
export CFG_SCALE=$(quote_shell "$CFG_SCALE")
export CFG_IMG_SCALE=$(quote_shell "$CFG_IMG_SCALE")
export SEED=$(quote_shell "$SEED")
export VALUE_HEAD_MODE=$(quote_shell "$VALUE_HEAD_MODE")
export VALUE_REF_EXPANSION=$(quote_shell "$VALUE_REF_EXPANSION")
export ATTN_ATTR_SOURCE=$(quote_shell "$ATTN_ATTR_SOURCE")
export FAIL_ON_MISSING_TARGET=$(quote_shell "$FAIL_ON_MISSING_TARGET")
export MING_LOAD_IMAGE_GEN=$(quote_shell "$MING_LOAD_IMAGE_GEN")
cd "\$REPO_ROOT"
export PYTHONPATH="\$REPO_ROOT:\$SPARSE_UNIFIED_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
bash analysis/vlm_geometry/scripts/run_vlm_geometry_grid.sh
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
