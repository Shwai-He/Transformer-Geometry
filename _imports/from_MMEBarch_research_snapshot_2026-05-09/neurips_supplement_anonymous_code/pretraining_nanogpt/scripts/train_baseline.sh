#!/usr/bin/env bash
set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT_CANDIDATES="${HF_ENDPOINT_CANDIDATES:-https://huggingface.co,https://hf-mirror.com,https://huggingface.cn}"
export HF_ENDPOINT

ROOT_DIR_CANDIDATES=(
  "${ROOT_DIR:-}"
)
ROOT_DIR=""
for _candidate in "${ROOT_DIR_CANDIDATES[@]}"; do
  if [[ -n "$_candidate" && -d "$_candidate" ]]; then
    ROOT_DIR="$_candidate"
    break
  fi
done
if [[ -z "$ROOT_DIR" ]]; then
  echo "[ERROR] Could not find nanoGPT root. Tried: ${ROOT_DIR_CANDIDATES[*]}" >&2
  exit 1
fi
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Usage example:
#   bash scripts/train_baseline.sh config/train_gpt2.py

BASE_CFG="${1:-config/train_gpt2.py}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}" # fixed default for controlled comparison
MAX_PORT_RETRIES="${MAX_PORT_RETRIES:-5}"
NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-}"
if [[ -z "$MASTER_PORT" && -n "${METIS_WORKER_0_PORT:-}" ]]; then
  IFS=',' read -ra _METIS_PORTS <<< "$METIS_WORKER_0_PORT"
  MASTER_PORT="${_METIS_PORTS[0]}"
fi
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-1337}"
INIT_FROM="${INIT_FROM:-auto}"
COMPILE="${COMPILE:-false}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
BATCH_SIZE="${BATCH_SIZE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
MAX_ITERS="${MAX_ITERS:-}"
WARMUP_ITERS="${WARMUP_ITERS:-}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-}"
MIN_LR="${MIN_LR:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
ATTN_DIAG_MODE="${ATTN_DIAG_MODE:-none}" # none | hard | bias
ATTN_DIAG_BIAS="${ATTN_DIAG_BIAS:-0.0}"
ATTN_DIAG_KEEP_FIRST="${ATTN_DIAG_KEEP_FIRST:-true}"
XSA_SELF_LOSS_OBJECTIVE="${XSA_SELF_LOSS_OBJECTIVE:-none}"
MODEL_SIZE_TAG="${MODEL_SIZE_TAG:-}"
RUN_NAME_PREFIX="${MODEL_SIZE_TAG:+${MODEL_SIZE_TAG}-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME_PREFIX}baseline-xobj${XSA_SELF_LOSS_OBJECTIVE}-s${SEED}}"
OUT_DIR="${OUT_DIR:-out/${RUN_NAME_PREFIX}baseline-xobj${XSA_SELF_LOSS_OBJECTIVE}-s${SEED}}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-anonymous-supplement}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/wandb}"
DATASET="${DATASET:-openwebtext}"
PREPARE_DATA="${PREPARE_DATA:-on_error}" # on_error | auto | true | false
HF_HOME="${HF_HOME:-}"
PREPARE_NUM_PROC="${PREPARE_NUM_PROC:-8}"
PREPARE_NUM_PROC_LOAD_DATASET="${PREPARE_NUM_PROC_LOAD_DATASET:-0}"
PREPARE_LOCAL_DIR="${PREPARE_LOCAL_DIR:-/tmp/${USER:-$(id -u)}/nanogpt_data}"
PREPARE_DISABLE_TQDM="${PREPARE_DISABLE_TQDM:-false}"
PREPARE_RETRY_SINGLE_PROC="${PREPARE_RETRY_SINGLE_PROC:-true}"
RETRY_ON_MISSING_DATA="${RETRY_ON_MISSING_DATA:-true}"
COPY_MODE="${COPY_MODE:-cp}" # auto | rsync | cp (hdfs cli disabled)
HDFS_DEST_DIR="${HDFS_DEST_DIR:-$ROOT_DIR/data/$DATASET}"
PORT_SKILL_SH="${PORT_SKILL_SH:-}"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "[ERROR] Base config not found: $BASE_CFG"
  exit 1
fi

validate_attention_head_dims() {
  "$PYTHON_BIN" - "$BASE_CFG" <<'PY'
import runpy
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1]).resolve()
cfg = runpy.run_path(str(cfg_path))

n_embd = cfg.get("n_embd")
n_head = cfg.get("n_head")
n_head_dim = cfg.get("n_head_dim", 0)

if n_embd is None or n_head is None:
    print(f"[WARN] Config missing n_embd/n_head, skip dim preflight: {cfg_path}")
    raise SystemExit(0)

try:
    n_embd = int(n_embd)
    n_head = int(n_head)
    n_head_dim = int(n_head_dim or 0)
except Exception:
    print(f"[ERROR] Invalid dims in {cfg_path}: n_embd={n_embd}, n_head={n_head}, n_head_dim={n_head_dim}")
    raise SystemExit(1)

if n_head <= 0:
    print(f"[ERROR] Invalid n_head in {cfg_path}: {n_head}")
    raise SystemExit(1)

if n_head_dim > 0:
    head_dim = n_head_dim
else:
    if n_embd % n_head != 0:
        print(
            f"[ERROR] Invalid attention dims in {cfg_path}: "
            f"n_embd={n_embd}, n_head={n_head}, n_embd % n_head = {n_embd % n_head}"
        )
        raise SystemExit(1)
    head_dim = n_embd // n_head

if head_dim % 2 != 0:
    print(
        f"[ERROR] Invalid RoPE head_dim in {cfg_path}: "
        f"head_dim={head_dim} (n_embd={n_embd}, n_head={n_head}, n_head_dim={n_head_dim}). Must be even."
    )
    raise SystemExit(1)

attn_dim = n_head * head_dim
print(f"[INFO] Config dims check passed: n_embd={n_embd}, n_head={n_head}, n_head_dim={n_head_dim}, head_dim={head_dim}, attn_dim={attn_dim}")
PY
}

validate_attention_head_dims

echo "[INFO] HF_ENDPOINT=$HF_ENDPOINT"
echo "[INFO] HF_ENDPOINT_CANDIDATES=$HF_ENDPOINT_CANDIDATES"

resolve_init_from() {
  local requested="${1:-auto}"
  local ckpt_path="$OUT_DIR/ckpt.pt"
  case "$requested" in
    auto|resume_if_available)
      if [[ -s "$ckpt_path" ]]; then
        echo "[INFO] Found checkpoint at $ckpt_path, resuming." >&2
        echo "resume"
      else
        echo "[INFO] No checkpoint found at $ckpt_path, starting from scratch." >&2
        echo "scratch"
      fi
      ;;
    scratch|resume)
      echo "$requested"
      ;;
    gpt2*)
      echo "$requested"
      ;;
    *)
      echo "[ERROR] Unsupported INIT_FROM=$requested. Use auto|scratch|resume|gpt2*" >&2
      exit 1
      ;;
  esac
}

log_init_from_decision() {
  local requested="${1:-auto}"
  local resolved="${2:-}"
  local ckpt_path="$OUT_DIR/ckpt.pt"
  local ckpt_exists="false"
  local ckpt_size="0"
  if [[ -e "$ckpt_path" ]]; then
    ckpt_exists="true"
  fi
  if [[ -f "$ckpt_path" ]]; then
    ckpt_size="$(wc -c < "$ckpt_path" | tr -d '[:space:]')"
  fi
  echo "[INFO] init_from_requested=$requested"
  echo "[INFO] init_from_ckpt_path=$ckpt_path exists=$ckpt_exists size=$ckpt_size"
  if [[ "$resolved" == "resume" ]]; then
    echo "[INFO] init_from_decision=resume checkpoint_found=true"
  elif [[ "$resolved" == "scratch" ]]; then
    echo "[INFO] init_from_decision=scratch checkpoint_found=false"
  else
    echo "[INFO] init_from_decision=$resolved checkpoint_found=$ckpt_exists"
  fi
}

to_python_bool() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) echo "True" ;;
    0|false|no|n|off) echo "False" ;;
    *)
      echo "[ERROR] Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

check_dependency_stack() {
  "$PYTHON_BIN" - <<'PY'
import importlib
from packaging.version import Version

mods = {}
for name in ("numpy", "scipy", "pyarrow", "pandas", "datasets", "tiktoken", "tqdm"):
    try:
        mods[name] = importlib.import_module(name)
    except Exception:
        raise SystemExit(1)

numpy_v = Version(mods["numpy"].__version__)
scipy_v = Version(mods["scipy"].__version__)
pyarrow_v = Version(mods["pyarrow"].__version__)
pandas_v = Version(mods["pandas"].__version__)

ok = (
    numpy_v == Version("1.26.4")
    and scipy_v == Version("1.11.4")
    and Version("14.0.0") <= pyarrow_v < Version("18.0.0")
    and pandas_v < Version("2.3.0")
)
raise SystemExit(0 if ok else 1)
PY
}

if ! check_dependency_stack; then
  echo "[INFO] Installing compatible python packages (numpy/scipy/pyarrow/pandas/datasets/tiktoken/tqdm, plus packaging)"
  "$PYTHON_BIN" -m pip uninstall -y numpy scipy pyarrow pandas >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install --no-cache-dir --force-reinstall \
    "packaging" \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    "pyarrow>=14,<18" \
    "pandas<2.3" \
    "datasets" \
    "tiktoken" \
    "tqdm"
fi

export WANDB_DIR
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-0}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"
mkdir -p "$WANDB_DIR" || true

DATA_DIR="$ROOT_DIR/data/$DATASET"
TRAIN_BIN="$DATA_DIR/train.bin"
VAL_BIN="$DATA_DIR/val.bin"
sync_bins_to_target() {
  local src_dir="$1"
  local dst_dir="$2"
  local mode="$COPY_MODE"
  local src_train="$src_dir/train.bin"
  local src_val="$src_dir/val.bin"
  local dst_train="$dst_dir/train.bin"
  local dst_val="$dst_dir/val.bin"

  if [[ ! -f "$src_train" || ! -f "$src_val" ]]; then
    echo "[ERROR] Missing source bins in $src_dir"
    exit 1
  fi

  if [[ "$mode" == "auto" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      mode="rsync"
    else
      mode="cp"
    fi
  fi

  mkdir -p "$dst_dir"
  if [[ "$mode" == "rsync" ]] && command -v rsync >/dev/null 2>&1; then
    echo "[INFO] Copying bins via rsync (parallel) -> $dst_dir"
    rsync -a --whole-file --inplace "$src_train" "$dst_train" &
    pid1=$!
    rsync -a --whole-file --inplace "$src_val" "$dst_val" &
    pid2=$!
    wait "$pid1" "$pid2"
  else
    echo "[INFO] Copying bins via cp (parallel) -> $dst_dir"
    cp -f "$src_train" "$dst_train" &
    pid1=$!
    cp -f "$src_val" "$dst_val" &
    pid2=$!
    wait "$pid1" "$pid2"
  fi
}

prepare_dataset() {
  echo "[INFO] Preparing dataset=$DATASET ..."
  if [[ "$DATASET" == "openwebtext" ]]; then
    STAGE_DIR="$PREPARE_LOCAL_DIR/$DATASET"
    mkdir -p "$STAGE_DIR"
    local disable_tqdm_arg=""
    if [[ "$(to_python_bool "$PREPARE_DISABLE_TQDM")" == "True" ]]; then
      disable_tqdm_arg="--disable_tqdm"
    fi
    local run_prepare_cmd
    run_prepare_cmd() {
      local num_proc="$1"
      local num_proc_load="$2"
      if [[ -n "$HF_HOME" ]]; then
        "$PYTHON_BIN" data/openwebtext/prepare.py \
          --out_dir "$STAGE_DIR" \
          --hf_home "$HF_HOME" \
          --num_proc "$num_proc" \
          --num_proc_load_dataset "$num_proc_load" \
          --skip_if_exists \
          ${disable_tqdm_arg}
      else
        "$PYTHON_BIN" data/openwebtext/prepare.py \
          --out_dir "$STAGE_DIR" \
          --num_proc "$num_proc" \
          --num_proc_load_dataset "$num_proc_load" \
          --skip_if_exists \
          ${disable_tqdm_arg}
      fi
    }
    local prepare_ok="false"
    local endpoint
    IFS=',' read -ra _HF_EP_LIST <<< "$HF_ENDPOINT_CANDIDATES"
    for endpoint in "${_HF_EP_LIST[@]}"; do
      endpoint="$(echo "$endpoint" | xargs)"
      [[ -z "$endpoint" ]] && continue
      export HF_ENDPOINT="$endpoint"
      echo "[INFO] Trying dataset prepare with HF_ENDPOINT=$HF_ENDPOINT"
      if run_prepare_cmd "$PREPARE_NUM_PROC" "$PREPARE_NUM_PROC_LOAD_DATASET"; then
        prepare_ok="true"
        break
      fi
      if [[ "$(to_python_bool "$PREPARE_RETRY_SINGLE_PROC")" == "True" ]]; then
        echo "[WARN] Prepare failed with HF_ENDPOINT=$HF_ENDPOINT using multi-proc; retrying single-proc..."
        if run_prepare_cmd 1 1; then
          prepare_ok="true"
          break
        fi
      fi
      echo "[WARN] Prepare failed with HF_ENDPOINT=$HF_ENDPOINT, trying next candidate..."
    done
    if [[ "$prepare_ok" != "true" ]]; then
      echo "[ERROR] Dataset prepare failed for all HF endpoints: $HF_ENDPOINT_CANDIDATES"
      return 1
    fi
    sync_bins_to_target "$STAGE_DIR" "$DATA_DIR"
    echo "[INFO] Dataset copied from staging: $STAGE_DIR -> $DATA_DIR (mode=$COPY_MODE)"
  elif [[ -f "data/$DATASET/prepare.py" ]]; then
    (cd "data/$DATASET" && "$PYTHON_BIN" prepare.py)
  else
    echo "[ERROR] No prepare script found for dataset=$DATASET and missing bins."
    exit 1
  fi
}

if [[ ! -f "$TRAIN_BIN" || ! -f "$VAL_BIN" ]]; then
  if [[ "$PREPARE_DATA" == "false" ]]; then
    echo "[ERROR] Missing dataset bins and PREPARE_DATA=false: $TRAIN_BIN or $VAL_BIN"
    exit 1
  fi
  prepare_dataset
fi

resolve_nproc_per_node() {
  if [[ "$NPROC_PER_NODE" != "auto" ]]; then
    echo "$NPROC_PER_NODE"
    return 0
  fi
  "$PYTHON_BIN" - <<'PY'
try:
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(max(1, int(n)))
except Exception:
    print(1)
PY
}

NPROC_PER_NODE_RESOLVED="$(resolve_nproc_per_node)"
if ! [[ "$NPROC_PER_NODE_RESOLVED" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE_RESOLVED" -lt 1 ]]; then
  echo "[ERROR] Invalid NPROC_PER_NODE resolved value: $NPROC_PER_NODE_RESOLVED"
  exit 1
fi
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [[ "$NNODES" -lt 1 ]]; then
  echo "[ERROR] Invalid NNODES value: $NNODES"
  exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [[ "$NODE_RANK" -lt 0 ]] || [[ "$NODE_RANK" -ge "$NNODES" ]]; then
  echo "[ERROR] Invalid NODE_RANK value: $NODE_RANK for NNODES=$NNODES"
  exit 1
fi

echo "[INFO] Running baseline... (nnodes=$NNODES, node_rank=$NODE_RANK, nproc_per_node=$NPROC_PER_NODE_RESOLVED, master=$MASTER_ADDR:$MASTER_PORT)"
run_training_once() {
  local -a run_cmd=("$@")
  local -a train_hparam_args=()
  local init_from_resolved
  init_from_resolved="$(resolve_init_from "$INIT_FROM")"
  log_init_from_decision "$INIT_FROM" "$init_from_resolved"
  [[ -n "$BATCH_SIZE" ]] && train_hparam_args+=(--batch_size="$BATCH_SIZE")
  [[ -n "$BLOCK_SIZE" ]] && train_hparam_args+=(--block_size="$BLOCK_SIZE")
  [[ -n "$LEARNING_RATE" ]] && train_hparam_args+=(--learning_rate="$LEARNING_RATE")
  [[ -n "$MAX_ITERS" ]] && train_hparam_args+=(--max_iters="$MAX_ITERS")
  [[ -n "$WARMUP_ITERS" ]] && train_hparam_args+=(--warmup_iters="$WARMUP_ITERS")
  [[ -n "$LR_DECAY_ITERS" ]] && train_hparam_args+=(--lr_decay_iters="$LR_DECAY_ITERS")
  [[ -n "$MIN_LR" ]] && train_hparam_args+=(--min_lr="$MIN_LR")
  [[ -n "$GRADIENT_ACCUMULATION_STEPS" ]] && train_hparam_args+=(--gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS")
  train_hparam_args+=(--attn_diag_mode="$ATTN_DIAG_MODE")
  train_hparam_args+=(--attn_diag_bias="$ATTN_DIAG_BIAS")
  train_hparam_args+=(--attn_diag_keep_first="$(to_python_bool "$ATTN_DIAG_KEEP_FIRST")")
  local wandb_log_py
  wandb_log_py="$(to_python_bool "$ENABLE_WANDB")"
  "${run_cmd[@]}" "$BASE_CFG" \
  --wandb_log="$wandb_log_py" \
  --wandb_project="$WANDB_PROJECT" \
  --wandb_run_name="$WANDB_RUN_NAME" \
  --out_dir="$OUT_DIR" \
  --seed="$SEED" \
  --init_from="$init_from_resolved" \
  --compile="$(to_python_bool "$COMPILE")" \
  --dataset="$DATASET" \
  "${train_hparam_args[@]}" \
  --log_interval="$LOG_INTERVAL" \
  --enable_residual_patch=False \
  --residual_mode=sum
}

run_training() {
  if [[ "$NNODES" -gt 1 ]]; then
    echo "[INFO] Launching multi-node torchrun: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"
    run_training_once "$PYTHON_BIN" -m torch.distributed.run \
      --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" \
      --nproc_per_node="$NPROC_PER_NODE_RESOLVED" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      train.py
    return $?
  fi

  if [[ "$NPROC_PER_NODE_RESOLVED" -le 1 ]]; then
    run_training_once "$PYTHON_BIN" train.py
    return $?
  fi

  if [[ -f "$PORT_SKILL_SH" ]]; then
    # shellcheck source=/dev/null
    source "$PORT_SKILL_SH"
  fi

  local attempt port
  for ((attempt=1; attempt<=MAX_PORT_RETRIES; attempt++)); do
    if declare -F get_free_port >/dev/null 2>&1; then
      port="$(get_free_port)"
    else
      port="$("$PYTHON_BIN" - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
    fi
    echo "[INFO] Launch attempt ${attempt}/${MAX_PORT_RETRIES}, master_port=${port}"
    if run_training_once "$PYTHON_BIN" -m torch.distributed.run --standalone --master_port="$port" --nproc_per_node="$NPROC_PER_NODE_RESOLVED" train.py; then
      return 0
    fi
    echo "[WARN] torchrun failed on attempt ${attempt}, retrying with a new port..."
  done
  return 1
}

if run_training; then
  exit 0
fi

if [[ "${RETRY_ON_MISSING_DATA,,}" == "true" && "$PREPARE_DATA" == "on_error" ]]; then
  if [[ ! -f "$TRAIN_BIN" || ! -f "$VAL_BIN" ]]; then
    echo "[WARN] Training failed and dataset bins are missing. Preparing data then retrying once..."
    prepare_dataset
    run_training
    exit 0
  fi
fi

echo "[ERROR] Training failed."
exit 1
