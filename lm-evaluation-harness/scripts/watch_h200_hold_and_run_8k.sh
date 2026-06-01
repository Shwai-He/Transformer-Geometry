#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${JOB_ID:-64444}"
REPO_ROOT="${REPO_ROOT:-/beacon-projects/traumallm/shwaihe/Transformer-Geometry}"
LOG_ROOT="$REPO_ROOT/lm-evaluation-harness/outputs/token_compression_ruler/slurm"
RUN_LOG="$LOG_ROOT/watch-h200-hold-${JOB_ID}.log"
MARKER="$LOG_ROOT/watch-h200-hold-${JOB_ID}.done"
MAX_TRIES="${MAX_TRIES:-4320}"
SLEEP_SEC="${SLEEP_SEC:-60}"

mkdir -p "$LOG_ROOT"

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$RUN_LOG"
}

if [[ -f "$MARKER" ]]; then
  log "Marker exists, nothing to do: $MARKER"
  exit 0
fi

for ((try = 1; try <= MAX_TRIES; try++)); do
  state="$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null | head -1 || true)"
  if [[ -z "$state" ]]; then
    log "Job $JOB_ID not visible in squeue; checking again in ${SLEEP_SEC}s."
  elif [[ "$state" == "RUNNING" ]]; then
    log "Job $JOB_ID is RUNNING. Launching 8k target-query experiments inside the allocation."
    cd "$REPO_ROOT"
    scancel 64429 64430 2>/dev/null || true
    srun --jobid="$JOB_ID" --overlap --ntasks=1 --cpus-per-task=16 --gres=gpu:nvidia_h200:4 bash -lc '
      set -euo pipefail
      cd /beacon-projects/traumallm/shwaihe/Transformer-Geometry
      echo "[INFO] allocation node: $(hostname)"
      nvidia-smi
      for mode in target_value_norm_keep target_value_perp_only_norm_keep; do
        for keep in 0.5 0.7; do
          echo "[INFO] running mode=${mode} keep=${keep}"
          MODEL_NAME=/beacon-projects/traumallm/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
          TASKS=niah_single_1 \
          RULER_LENGTHS=8192 \
          LIMIT=20 \
          TOKEN_COMPRESSION_MODE="$mode" \
          TOKEN_COMPRESSION_KEEP_RATIO="$keep" \
          TOKEN_COMPRESSION_LAYERS=all \
          TOKEN_COMPRESSION_PARALLEL_WEIGHT=0.0 \
          OUTPUT_ROOT=/beacon-projects/traumallm/shwaihe/Transformer-Geometry/lm-evaluation-harness/outputs/token_compression_ruler_h200_hold \
          LAUNCH_MODE=model_parallel \
          PYTHON_BIN=/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python \
          bash lm-evaluation-harness/scripts/run_lm_eval_token_compression_ruler.sh
        done
      done
    ' >> "$RUN_LOG" 2>&1
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$MARKER"
    log "Finished 8k target-query experiments. Marker: $MARKER"
    exit 0
  else
    log "Job $JOB_ID state=$state; waiting ${SLEEP_SEC}s. try=$try/$MAX_TRIES"
  fi
  sleep "$SLEEP_SEC"
done

log "Timed out waiting for job $JOB_ID to start."
exit 1
