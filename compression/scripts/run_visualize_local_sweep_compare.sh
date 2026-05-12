#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
# If both roots are the same directory, script will auto-filter by label prefix.
RUNS_ROOT_A="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/focused-compression-analysis/outputs/layerwise_para_perp"
RUNS_ROOT_B="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/focused-compression-analysis/outputs/layerwise_para_perp"
LABEL_A="local_wanda50"   # prune prefix
LABEL_B="local_bnb4_nf4"  # quant prefix
OUT_DIR="/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/focused-compression-analysis/outputs/viz_local_compare"

PYTHON_BIN="python3"
BACKGROUND=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$WORKSPACE_ROOT/code/visualize_local_sweep_compare.py"
LOG_DIR="$WORKSPACE_ROOT/logs/compression_analysis/viz_local_compare"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="viz_local_compare_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

tmp_dir=""
cleanup() {
  if [[ -n "$tmp_dir" && -d "$tmp_dir" ]]; then
    rm -rf "$tmp_dir"
  fi
}
trap cleanup EXIT

prepare_roots() {
  local root_a="$RUNS_ROOT_A"
  local root_b="$RUNS_ROOT_B"
  if [[ "$root_a" == "$root_b" ]]; then
    tmp_dir="$(mktemp -d /tmp/viz_local_compare.XXXXXX)"
    mkdir -p "$tmp_dir/a" "$tmp_dir/b"
    for d in "$root_a"/*; do
      [[ -d "$d" ]] || continue
      bn="$(basename "$d")"
      if [[ "$bn" == *"__${LABEL_A}_l"* || "$bn" == "${LABEL_A}_l"* || "$bn" == *"${LABEL_A}"* ]]; then
        ln -s "$d" "$tmp_dir/a/$bn"
      fi
      if [[ "$bn" == *"__${LABEL_B}_l"* || "$bn" == "${LABEL_B}_l"* || "$bn" == *"${LABEL_B}"* ]]; then
        ln -s "$d" "$tmp_dir/b/$bn"
      fi
    done
    RUNS_ROOT_A="$tmp_dir/a"
    RUNS_ROOT_B="$tmp_dir/b"
    echo "[INFO] Filtered same-root runs by labels:"
    echo "[INFO]   A: $RUNS_ROOT_A (${LABEL_A})"
    echo "[INFO]   B: $RUNS_ROOT_B (${LABEL_B})"
  fi
}

run_viz() {
  mkdir -p "$OUT_DIR"
  prepare_roots

  for comp in block_out attn_out mlp_out; do
    comp_out="$OUT_DIR/$comp"
    mkdir -p "$comp_out"
    echo "[INFO] ===== component: $comp ====="
    "$PYTHON_BIN" "$PY_SCRIPT" \
      --runs_root_a "$RUNS_ROOT_A" \
      --runs_root_b "$RUNS_ROOT_B" \
      --label_a "$LABEL_A" \
      --label_b "$LABEL_B" \
      --component "$comp" \
      --out_dir "$comp_out"
  done
}

echo "Logs:"
echo "  $LOG_PATH"
echo "PID file:"
echo "  $PID_PATH"

if [[ "$BACKGROUND" == "true" && "${RUN_FOREGROUND:-0}" != "1" ]]; then
  nohup env RUN_FOREGROUND=1 bash "$0" >"$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"
  echo "Started in background."
  echo "tail -f $LOG_PATH"
  exit 0
fi

run_viz 2>&1 | tee "$LOG_PATH"
