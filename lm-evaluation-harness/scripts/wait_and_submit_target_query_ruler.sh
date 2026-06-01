#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/lm-evaluation-harness/outputs/token_compression_ruler/slurm"
MARKER="$LOG_DIR/target_query_submission.done"

mkdir -p "$LOG_DIR"
if [[ -f "$MARKER" ]]; then
  echo "[INFO] target-query submission marker exists: $MARKER"
  exit 0
fi

MAX_TRIES="${MAX_TRIES:-180}"
SLEEP_SEC="${SLEEP_SEC:-60}"

for ((try = 1; try <= MAX_TRIES; try++)); do
  if sinfo -s >/dev/null 2>&1; then
    echo "[INFO] Slurm is reachable on try $try; submitting target-query RULER jobs."
    cd "$REPO_ROOT"

    MODEL_NAME="${MODEL_NAME:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}" \
    TASKS="${TASKS:-niah_single_1}" \
    RULER_LENGTHS="4096" \
    LIMIT="${LIMIT:-20}" \
    KEEP_RATIOS_CSV="${KEEP_RATIOS_CSV:-0.7,0.8,0.9}" \
    MODES_CSV="target_value_norm_keep" \
    TOKEN_COMPRESSION_LAYERS="${TOKEN_COMPRESSION_LAYERS:-all}" \
    TOKEN_COMPRESSION_PARALLEL_WEIGHT="${TOKEN_COMPRESSION_PARALLEL_WEIGHT:-0.25}" \
    PARTITION="${PARTITION:-scavenger}" \
    QOS="${QOS:-scavenger}" \
    GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}" \
    TIME="${TIME:-2:00:00}" \
    bash "$SCRIPT_DIR/submit_token_compression_ruler_slurm.sh"

    MODEL_NAME="${MODEL_NAME:-/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}" \
    TASKS="${TASKS:-niah_single_1}" \
    RULER_LENGTHS="8192" \
    LIMIT="${LIMIT:-20}" \
    KEEP_RATIOS_CSV="${KEEP_RATIOS_CSV:-0.7,0.8,0.9,1.0}" \
    MODES_CSV="uniform,target_value_norm_keep,target_value_perp_only_norm_keep" \
    TOKEN_COMPRESSION_LAYERS="${TOKEN_COMPRESSION_LAYERS:-all}" \
    TOKEN_COMPRESSION_PARALLEL_WEIGHT="${TOKEN_COMPRESSION_PARALLEL_WEIGHT:-0.25}" \
    PARTITION="${PARTITION:-scavenger}" \
    QOS="${QOS:-scavenger}" \
    GRES="${GRES:-gpu:nvidia_rtx_6000_ada_generation:1}" \
    TIME="${TIME:-3:00:00}" \
    bash "$SCRIPT_DIR/submit_token_compression_ruler_slurm.sh"

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
    echo "[INFO] Wrote marker $MARKER"
    exit 0
  fi
  echo "[INFO] Slurm unreachable on try $try/$MAX_TRIES; sleeping ${SLEEP_SEC}s."
  sleep "$SLEEP_SEC"
done

echo "[ERROR] Slurm did not become reachable after $MAX_TRIES tries." >&2
exit 1
