#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HARNESS_DIR/.." && pwd)}"

PARTITION="${PARTITION:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:nvidia_l40s:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-1-00:00:00}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_base}"

TASKS_ORDER_CSV="${TASKS_ORDER_CSV:-openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,mmlu,gsm8k_cot,humaneval,nq_open}"
SETTINGS_CSV="${SETTINGS_CSV:-none,residual_attn,xsa_middle_multihead,attn_diag_zero_renorm}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
XSA_LAYER_SCALE_MODE="${XSA_LAYER_SCALE_MODE:-none}"
XSA_LAYER_SCALE_SEED="${XSA_LAYER_SCALE_SEED:-0}"
XSA_LAYER_PARA_SCALE_MIN="${XSA_LAYER_PARA_SCALE_MIN:--20.0}"
XSA_LAYER_PARA_SCALE_MAX="${XSA_LAYER_PARA_SCALE_MAX:-20.0}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-false}"
LIMIT="${LIMIT:-}"

OUT_ROOT="${OUT_ROOT:-$HARNESS_DIR/outputs/xsa_lm_eval_qwen3_0p6b_table1}"
MODEL_OUT_DIR="$OUT_ROOT/$MODEL_TAG"
SLURM_DIR="${SLURM_DIR:-$OUT_ROOT/slurm}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
mkdir -p "$MODEL_OUT_DIR" "$SLURM_DIR" "$LOG_DIR"

fewshot_for_task() {
  case "$1" in
    openbookqa|piqa|rte|boolq) echo 0 ;;
    winogrande|mmlu|humaneval|nq_open) echo 5 ;;
    arc_challenge) echo 25 ;;
    hellaswag) echo 10 ;;
    gsm8k_cot) echo 8 ;;
    *) echo 0 ;;
  esac
}

submit_setting_job() {
  local setting_label="$1"
  local setting="$setting_label"
  local attn_diag_enabled="false"
  local attn_diag_mode="none"

  if [[ "$setting_label" == "attn_diag_zero_renorm" ]]; then
    setting="none"
    attn_diag_enabled="true"
    attn_diag_mode="zero_renorm"
  fi

  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --job-name="xsaeval-${MODEL_TAG}-${setting_label}" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$SLURM_DIR/%x-%j.out" \
    --error="$SLURM_DIR/%x-%j.err" \
    --wrap="$(cat <<EOF
set -u
cd "$HARNESS_DIR"
export PYTHONPATH="$HARNESS_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHON_BIN="$PYTHON_BIN"
export MODEL_NAME="$MODEL_NAME"
export BATCH_SIZE="$BATCH_SIZE"
export DTYPE="$DTYPE"
export MAX_LENGTH="$MAX_LENGTH"
export XSA_LAYER_SCALE_MODE="$XSA_LAYER_SCALE_MODE"
export XSA_LAYER_SCALE_SEED="$XSA_LAYER_SCALE_SEED"
export XSA_LAYER_PARA_SCALE_MIN="$XSA_LAYER_PARA_SCALE_MIN"
export XSA_LAYER_PARA_SCALE_MAX="$XSA_LAYER_PARA_SCALE_MAX"
export LAUNCH_MODE="$LAUNCH_MODE"
export TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE"
export APPLY_CHAT_TEMPLATE="$APPLY_CHAT_TEMPLATE"
export CONFIRM_RUN_UNSAFE_CODE=true
export LOG_DIR="$LOG_DIR"
export LIMIT="$LIMIT"
status=0
IFS=',' read -r -a tasks <<< "$TASKS_ORDER_CSV"
echo "[INFO] setting_label=$setting_label setting=$setting attn_diag=$attn_diag_enabled mode=$attn_diag_mode"
echo "[INFO] job_id=\$SLURM_JOB_ID host=\$(hostname)"
"\$PYTHON_BIN" - <<'PY'
import torch, transformers
print("[INFO] torch", torch.__version__, "cuda", torch.cuda.is_available(), "n", torch.cuda.device_count(), flush=True)
print("[INFO] transformers", transformers.__version__, flush=True)
PY
for task in "\${tasks[@]}"; do
  task="\$(echo "\$task" | xargs)"
  [[ -z "\$task" ]] && continue
  case "\$task" in
    openbookqa|piqa|rte|boolq) fewshot=0 ;;
    winogrande|mmlu|humaneval|nq_open) fewshot=5 ;;
    arc_challenge) fewshot=25 ;;
    hellaswag) fewshot=10 ;;
    gsm8k_cot) fewshot=8 ;;
    *) fewshot=0 ;;
  esac
  out_path="$MODEL_OUT_DIR/\${task}-${setting_label}.json"
  if [[ -s "\$out_path" ]]; then
    echo "[TASK_SKIP] setting=$setting_label task=\$task out=\$out_path"
    continue
  fi
  echo "[TASK_START] setting=$setting_label task=\$task fewshot=\$fewshot out=\$out_path"
  if SETTING="$setting" \
     ATTN_DIAG_ENABLED="$attn_diag_enabled" \
     ATTN_DIAG_MODE="$attn_diag_mode" \
     ATTN_DIAG_KEEP_FIRST=true \
     TASKS="\$task" \
     NUM_FEWSHOT="\$fewshot" \
     OUTPUT_PATH="\$out_path" \
     bash "$SCRIPT_DIR/run_lm_eval_xsa_setting.sh"; then
    echo "[TASK_DONE] setting=$setting_label task=\$task"
  else
    rc=\$?
    echo "[TASK_FAILED] setting=$setting_label task=\$task rc=\$rc" >&2
    status=\$rc
  fi
done
exit \$status
EOF
)"
}

IFS=',' read -r -a settings <<< "$SETTINGS_CSV"
for setting in "${settings[@]}"; do
  setting="$(echo "$setting" | xargs)"
  [[ -z "$setting" ]] && continue
  job_id="$(submit_setting_job "$setting")"
  echo "[SUBMITTED] setting=$setting job_id=$job_id"
done

echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Model output dir: $MODEL_OUT_DIR"
echo "[INFO] Slurm logs: $SLURM_DIR"
