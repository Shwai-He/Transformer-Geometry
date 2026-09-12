#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a local/HF model path.}"
MODEL_PRESET="${MODEL_PRESET:-qwen-image}"
LOADER="${LOADER:-diffusers}"
MODE="${MODE:-pipeline_generate}"
SIDE="${SIDE:-gen}"
SPACE="${SPACE:-residual}"
TARGET="${TARGET:-block}"
PARA_SCALE="${PARA_SCALE:-1.0}"
PERP_SCALE="${PERP_SCALE:-1.0}"
LAYER_PATHS="${LAYER_PATHS:-}"
LAYER_INDICES="${LAYER_INDICES:-}"
TARGET_MODULE_REGEX="${TARGET_MODULE_REGEX:-}"
ATTN_NAME_REGEX="${ATTN_NAME_REGEX:-}"
MLP_NAME_REGEX="${MLP_NAME_REGEX:-}"
VALUE_NAME_REGEX="${VALUE_NAME_REGEX:-}"
VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}"
VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}"
ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}"
SKIP_FIRST_N="${SKIP_FIRST_N:-0}"
SKIP_LAST_N="${SKIP_LAST_N:-0}"
FAIL_ON_MISSING_TARGET="${FAIL_ON_MISSING_TARGET:-true}"
REQUIRE_STATS="${REQUIRE_STATS:-true}"
DISABLE_GEOMETRY="${DISABLE_GEOMETRY:-false}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
PROMPT="${PROMPT:-A small red cube on a wooden table.}"
METADATA_FILE="${METADATA_FILE:-}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-$HOME/SparseUnifiedModel}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
CFG_SCALE="${CFG_SCALE:-5.0}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-1.5}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"
ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-0}"
MING_LOAD_IMAGE_GEN="${MING_LOAD_IMAGE_GEN:-auto}"

RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_scaling/results}"
BENCH_NAME="${BENCH_NAME:-smoke}"
MODEL_TAG="${MODEL_TAG:-$MODEL_PRESET}"
SETTING="side_${SIDE}__space_${SPACE}__target_${TARGET}__para_${PARA_SCALE//./p}__perp_${PERP_SCALE//./p}"
OUT_DIR="$RESULT_ROOT/by_model/$MODEL_TAG/$BENCH_NAME/$SETTING"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_ROOT:$SPARSE_UNIFIED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QWENIMAGE_ENABLE_MODEL_CPU_OFFLOAD="$ENABLE_MODEL_CPU_OFFLOAD"
export QWENIMAGE_ENABLE_SEQUENTIAL_CPU_OFFLOAD="$ENABLE_SEQUENTIAL_CPU_OFFLOAD"

extra_args=()
if [[ -n "$HEIGHT" ]]; then
  extra_args+=(--height "$HEIGHT")
fi
if [[ -n "$WIDTH" ]]; then
  extra_args+=(--width "$WIDTH")
fi
if [[ -n "$METADATA_FILE" ]]; then
  extra_args+=(
    --metadata-file "$METADATA_FILE"
    --max-prompts "$MAX_PROMPTS"
    --prompt-key "$PROMPT_KEY"
    --num-images-per-prompt "$NUM_IMAGES_PER_PROMPT"
  )
fi

"$PYTHON_BIN" "$REPO_ROOT/analysis/vlm_geometry/run_vlm_geometry_smoke.py" \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --model-preset "$MODEL_PRESET" \
  --loader "$LOADER" \
  --mode "$MODE" \
  --side "$SIDE" \
  --space "$SPACE" \
  --target "$TARGET" \
  --para-scale "$PARA_SCALE" \
  --perp-scale "$PERP_SCALE" \
  --layer-paths "$LAYER_PATHS" \
  --layer-indices "$LAYER_INDICES" \
  --skip-first-n "$SKIP_FIRST_N" \
  --skip-last-n "$SKIP_LAST_N" \
  --target-module-regex "$TARGET_MODULE_REGEX" \
  --attn-name-regex "$ATTN_NAME_REGEX" \
  --mlp-name-regex "$MLP_NAME_REGEX" \
  --value-name-regex "$VALUE_NAME_REGEX" \
  --value-head-mode "$VALUE_HEAD_MODE" \
  --value-ref-expansion "$VALUE_REF_EXPANSION" \
  --attn-attr-source "$ATTN_ATTR_SOURCE" \
  --fail-on-missing-target "$FAIL_ON_MISSING_TARGET" \
  --require-stats "$REQUIRE_STATS" \
  --disable-geometry "$DISABLE_GEOMETRY" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --device-map "$DEVICE_MAP" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --cfg-img-scale "$CFG_IMG_SCALE" \
  --seed "$SEED" \
  --sparse-unified-root "$SPARSE_UNIFIED_ROOT" \
  --ming-load-image-gen "$MING_LOAD_IMAGE_GEN" \
  --prompt "$PROMPT" \
  --output-json "$OUT_DIR/result.json" \
  --output-image-dir "$OUT_DIR/images" \
  "${extra_args[@]}"

rm -f "$OUT_DIR/failed.txt"
echo "[OK] wrote $OUT_DIR/result.json"
