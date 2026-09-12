#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPARSE_UNIFIED_ROOT="${SPARSE_UNIFIED_ROOT:-$HOME/SparseUnifiedModel}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GENEVAL_PYTHON="${GENEVAL_PYTHON:-python3}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$SPARSE_UNIFIED_ROOT/eval/gen/geneval/model}"
EVAL_DETECTOR="${EVAL_DETECTOR:-mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/vlm_geometry_geneval/logs/slurm}"

OUT_DIR="${OUT_DIR:?Set OUT_DIR to a GenEval output directory containing images/.}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

PARTITION="${PARTITION:-beacon}"
QOS="${QOS:-medium}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-vlmgeneval-rescore}"

mkdir -p "$LOG_ROOT"

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
cd "\$REPO_ROOT"

OUT_DIR=$(quote_shell "$OUT_DIR")
if [[ $(quote_shell "$SKIP_EXISTING") == "true" && -s "\$OUT_DIR/results.jsonl" && -s "\$OUT_DIR/score.log" ]]; then
  echo "[SKIP] scored GenEval exists at \$OUT_DIR"
  exit 0
fi

$(quote_shell "$PYTHON_BIN") - "\$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

image_root = Path(sys.argv[1]) / "images"
top_metadata = image_root.parent / "metadata.jsonl"
records = []
if top_metadata.exists():
    records = [
        json.loads(line)
        for line in top_metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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

rm -f "\$OUT_DIR/eval_failed.flag" "\$OUT_DIR/summary_failed.flag"
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
