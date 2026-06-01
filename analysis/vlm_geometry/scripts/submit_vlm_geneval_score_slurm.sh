#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/beacon-projects/traumallm/shwaihe/Transformer-Geometry}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-/beacon-projects/traumallm/shwaihe/SparseUnifiedModel}"
PYTHON_BIN="${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}"
GENEVAL_PYTHON="${GENEVAL_PYTHON:-/beacon-projects/traumallm/shwaihe/envs/geneval-mmdet/bin/python}"
METADATA_FILE="${METADATA_FILE:-$SPARSE_UNIFIED_ROOT/data/evaluation/geneval_stratified_120/evaluation_metadata.jsonl}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$SPARSE_UNIFIED_ROOT/eval/gen/geneval/model}"
EVAL_DETECTOR="${EVAL_DETECTOR:-mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_geneval/results}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/vlm_geometry_geneval/logs/slurm}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME=qwenimage, ming, bagel, or janus}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_PRESET="${MODEL_PRESET:-}"
LOADER="${LOADER:-}"
MODE="${MODE:-}"
SIDE="${SIDE:-gen}"
SPACE="${SPACE:-value}"
TARGET="${TARGET:-value}"
PARA_SCALE="${PARA_SCALE:-1.0}"
PERP_SCALE="${PERP_SCALE:-1.0}"
VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}"
VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}"
ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}"
SCALE_MODE="${SCALE_MODE:-none}"
SCALE_SEED="${SCALE_SEED:-0}"
PARA_SCALE_MIN="${PARA_SCALE_MIN:--1.5}"
PARA_SCALE_MAX="${PARA_SCALE_MAX:-1.5}"
PERP_SCALE_MIN="${PERP_SCALE_MIN:--1.5}"
PERP_SCALE_MAX="${PERP_SCALE_MAX:-1.5}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
CFG_SCALE="${CFG_SCALE:-5.0}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-1.5}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MING_LOAD_IMAGE_GEN="${MING_LOAD_IMAGE_GEN:-auto}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-1}"
ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
TIME="${TIME:-08:00:00}"
MEM="${MEM:-128G}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-vlmgeneval-${MODEL_NAME}-${SPACE}-${TARGET}}"

case "$MODEL_NAME" in
  qwenimage|qwen|qwen-image)
    MODEL_NAME="qwenimage"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Qwen-Image}"
    MODEL_PRESET="${MODEL_PRESET:-qwen-image}"
    LOADER="${LOADER:-sparse_qwenimage}"
    MODE="${MODE:-pipeline_generate}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
    HEIGHT="${HEIGHT:-1024}"
    WIDTH="${WIDTH:-1024}"
    ;;
  ming|ming-omni)
    MODEL_NAME="ming"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Ming-Lite-Omni-1.5}"
    MODEL_PRESET="${MODEL_PRESET:-ming}"
    LOADER="${LOADER:-sparse_ming}"
    MODE="${MODE:-ming_image_generate}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
    HEIGHT="${HEIGHT:-544}"
    WIDTH="${WIDTH:-480}"
    ;;
  bagel|bagel-vlm)
    MODEL_NAME="bagel"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/hf/BAGEL-7B-MoT}"
    MODEL_PRESET="${MODEL_PRESET:-bagel}"
    LOADER="${LOADER:-sparse_bagel}"
    MODE="${MODE:-bagel_interleave_generate}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
    HEIGHT="${HEIGHT:-512}"
    WIDTH="${WIDTH:-512}"
    ;;
  janus|janus-pro|janus-pro-7b)
    MODEL_NAME="janus"
    MODEL_PATH="${MODEL_PATH:-$SPARSE_UNIFIED_ROOT/models/Janus-Pro-7B}"
    MODEL_PRESET="${MODEL_PRESET:-janus}"
    LOADER="${LOADER:-sparse_adapter}"
    MODE="${MODE:-adapter_generate}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-576}"
    HEIGHT="${HEIGHT:-384}"
    WIDTH="${WIDTH:-384}"
    ;;
  *)
    echo "Unsupported MODEL_NAME=$MODEL_NAME" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_ROOT"

scale_tag="para_${PARA_SCALE//./p}__perp_${PERP_SCALE//./p}"
scale_tag="${scale_tag//-/m}"
rand_tag="mode_${SCALE_MODE}__seed_${SCALE_SEED}__pr_${PARA_SCALE_MIN//./p}_${PARA_SCALE_MAX//./p}__pe_${PERP_SCALE_MIN//./p}_${PERP_SCALE_MAX//./p}"
rand_tag="${rand_tag//-/m}"
OUT_DIR="$RESULT_ROOT/by_model/$MODEL_NAME/geneval/side_${SIDE}__space_${SPACE}__target_${TARGET}__${scale_tag}__vhead_${VALUE_HEAD_MODE}__vref_${VALUE_REF_EXPANSION}__attr_${ATTN_ATTR_SOURCE}__${rand_tag}"

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
export QWENIMAGE_ENABLE_MODEL_CPU_OFFLOAD=$(quote_shell "$ENABLE_MODEL_CPU_OFFLOAD")
export QWENIMAGE_ENABLE_SEQUENTIAL_CPU_OFFLOAD=$(quote_shell "$ENABLE_SEQUENTIAL_CPU_OFFLOAD")
cd "\$REPO_ROOT"

OUT_DIR=$(quote_shell "$OUT_DIR")
if [[ $(quote_shell "$SKIP_EXISTING") == "true" && -s "\$OUT_DIR/results.jsonl" && -s "\$OUT_DIR/score.log" ]]; then
  echo "[SKIP] scored GenEval exists at \$OUT_DIR"
  exit 0
fi
mkdir -p "\$OUT_DIR"
rm -f "\$OUT_DIR/generate_failed.flag" "\$OUT_DIR/eval_failed.flag" "\$OUT_DIR/summary_failed.flag" "\$OUT_DIR/eval_skipped.flag"

$(quote_shell "$PYTHON_BIN") analysis/vlm_geometry/run_vlm_geometry_smoke.py \\
  --model-name-or-path $(quote_shell "$MODEL_PATH") \\
  --model-preset $(quote_shell "$MODEL_PRESET") \\
  --loader $(quote_shell "$LOADER") \\
  --mode $(quote_shell "$MODE") \\
  --side $(quote_shell "$SIDE") \\
  --space $(quote_shell "$SPACE") \\
  --target $(quote_shell "$TARGET") \\
  --para-scale $(quote_shell "$PARA_SCALE") \\
  --perp-scale $(quote_shell "$PERP_SCALE") \\
  --value-head-mode $(quote_shell "$VALUE_HEAD_MODE") \\
  --value-ref-expansion $(quote_shell "$VALUE_REF_EXPANSION") \\
  --attn-attr-source $(quote_shell "$ATTN_ATTR_SOURCE") \\
  --dtype $(quote_shell "$DTYPE") \\
  --device-map $(quote_shell "$DEVICE_MAP") \\
  --num-inference-steps $(quote_shell "$NUM_INFERENCE_STEPS") \\
  --cfg-scale $(quote_shell "$CFG_SCALE") \\
  --cfg-img-scale $(quote_shell "$CFG_IMG_SCALE") \\
  --seed $(quote_shell "$SEED") \\
  --height $(quote_shell "$HEIGHT") \\
  --width $(quote_shell "$WIDTH") \\
  --metadata-file $(quote_shell "$METADATA_FILE") \\
  --max-prompts $(quote_shell "$MAX_PROMPTS") \\
  --prompt-key prompt \\
  --num-images-per-prompt $(quote_shell "$NUM_IMAGES_PER_PROMPT") \\
  --scale-mode $(quote_shell "$SCALE_MODE") \\
  --scale-seed $(quote_shell "$SCALE_SEED") \\
  --para-scale-min $(quote_shell "$PARA_SCALE_MIN") \\
  --para-scale-max $(quote_shell "$PARA_SCALE_MAX") \\
  --perp-scale-min $(quote_shell "$PERP_SCALE_MIN") \\
  --perp-scale-max $(quote_shell "$PERP_SCALE_MAX") \\
  --sparse-unified-root "\$SPARSE_UNIFIED_ROOT" \\
  --ming-load-image-gen $(quote_shell "$MING_LOAD_IMAGE_GEN") \\
  --output-json "\$OUT_DIR/result.json" \\
  --output-image-dir "\$OUT_DIR/images" || {
    echo "Generation failed." | tee "\$OUT_DIR/generate_failed.flag"
    exit 1
  }

$(quote_shell "$PYTHON_BIN") - "\$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

image_root = Path(sys.argv[1]) / "images"
top_metadata = image_root.parent / "metadata.jsonl"
records = []
if top_metadata.exists():
    records = [json.loads(line) for line in top_metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
for prompt_dir in sorted(p for p in image_root.glob("*") if p.is_dir()):
    jsonl_path = prompt_dir / "metadata.jsonl"
    if jsonl_path.exists():
        continue
    json_path = prompt_dir / "metadata.json"
    if json_path.exists():
        record = json.loads(json_path.read_text(encoding="utf-8"))
    elif prompt_dir.name.isdigit() and int(prompt_dir.name) < len(records):
        record = records[int(prompt_dir.name)]
        json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        continue
    jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
PY

if [[ ! -s $(quote_shell "$EVAL_MODEL_PATH/$EVAL_DETECTOR.pth") ]]; then
  echo "Skip evaluation: missing detector checkpoint." | tee "\$OUT_DIR/eval_skipped.flag"
  exit 1
fi

tmp_results="\$OUT_DIR/results.jsonl.tmp"
rm -f "\$tmp_results"
$(quote_shell "$GENEVAL_PYTHON") "$SPARSE_UNIFIED_ROOT/eval/gen/geneval/evaluation/evaluate_images.py" \\
  "\$OUT_DIR/images" \\
  --outfile "\$tmp_results" \\
  --model-path $(quote_shell "$EVAL_MODEL_PATH") \\
  --options model=$(quote_shell "$EVAL_DETECTOR") 2>&1 | tee "\$OUT_DIR/eval.log" || {
    echo "Evaluation failed." | tee "\$OUT_DIR/eval_failed.flag"
    rm -f "\$tmp_results"
    exit 1
  }
mv "\$tmp_results" "\$OUT_DIR/results.jsonl"

$(quote_shell "$GENEVAL_PYTHON") "$SPARSE_UNIFIED_ROOT/eval/gen/geneval/evaluation/summary_scores.py" "\$OUT_DIR/results.jsonl" 2>&1 | tee "\$OUT_DIR/score.log" || {
  echo "Summary failed." | tee "\$OUT_DIR/summary_failed.flag"
  exit 1
}
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
