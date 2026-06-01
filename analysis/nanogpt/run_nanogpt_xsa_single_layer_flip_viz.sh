#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# 默认路径列表：把多个 run 目录或 ckpt 文件放在这里，脚本会依次读取。
DEFAULT_NANOGPT_CKPTS=(
  # 2.7B
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-2p7b-baseline-fineweb100bt-ctx2048-gb8192-lr3e-4-s1337"
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-2p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr3e-4-s1337"
  # 1.4B
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-1p4b-baseline-fineweb100bt-ctx2048-gb256-lr4e-4-s1337"
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-1p4b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb8192-lr4e-4-s1337"
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-1p4b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h1p0-xl0--1-xs1-gb8192-lr4e-4-minlr4e-5-s1337-lrlr10"
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-1p4b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb8192-lr4e-4-minlr4e-5-s1337-lrlr10"
  # 0.7B
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-0p7b-baseline-fineweb100bt-ctx2048-gb4096-lr5e-4-s1337"
  # "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr5e-4-s1337"
  "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h1p0-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10"
)
DEFAULT_NANOGPT_CKPT="${DEFAULT_NANOGPT_CKPT:-}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"
TRAVERSE_ALL_RUNS="${TRAVERSE_ALL_RUNS:-auto}" # auto|true|false
CKPT_VARIANTS="${CKPT_VARIANTS:-both}" # best|latest|both
BACKGROUND="${BACKGROUND:-false}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/xsa_logs}"
PROMPT_SET="${PROMPT_SET:-short}" # short|long
if [[ -z "${MAX_LENGTH+x}" ]]; then
  if [[ "$PROMPT_SET" == "long" ]]; then
    MAX_LENGTH="512"
  else
    MAX_LENGTH="256"
  fi
fi

sanitize_tag() {
  local s="$1"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

is_remote_live_log_path() {
  local path="$1"
  case "$path" in
    /mnt/hdfs/*) return 0 ;;
    *) return 1 ;;
  esac
}

finalize_live_log() {
  if [[ -n "${ACTIVE_LOG_PATH:-}" && -n "${LOG_PATH:-}" && "$ACTIVE_LOG_PATH" != "$LOG_PATH" ]]; then
    mkdir -p "$(dirname "$LOG_PATH")"
    cp "$ACTIVE_LOG_PATH" "$LOG_PATH" 2>/dev/null || true
  fi
}

resolve_nanogpt_repo_root() {
  local ckpt_path="$1"
  local requested_root="${2:-}"
  local -a candidates=()
  local candidate=""

  if [[ -n "$requested_root" ]]; then
    candidates+=("$requested_root")
  fi
  candidates+=(
    "$(cd "$(dirname "$ckpt_path")/../.." && pwd)"
    "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT"
    "$(cd "$ROOT_DIR/.." && pwd)/nanoGPT"
  )

  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" != /* ]]; then
      candidate="$ROOT_DIR/$candidate"
    fi
    if [[ -f "$candidate/model.py" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "[ERROR] Could not find nanoGPT repo root. Tried:" >&2
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" != /* ]]; then
      candidate="$ROOT_DIR/$candidate"
    fi
    echo "  - $candidate" >&2
  done
  return 1
}

resolve_nanogpt_ckpt() {
  local input_path="$1"

  if [[ "$input_path" != /* ]]; then
    input_path="$(cd "$ROOT_DIR" && pwd)/$input_path"
  fi

  # 直接传了 .pt 文件
  if [[ -f "$input_path" ]]; then
    printf '%s\n' "$input_path"
    return 0
  fi

  # 传入目录：优先 ckpt_best.pt，没有则用 ckpt.pt
  if [[ -d "$input_path" ]]; then
    if [[ -f "$input_path/ckpt_best.pt" ]]; then
      printf '%s\n' "$input_path/ckpt_best.pt"
      return 0
    fi
    if [[ -f "$input_path/ckpt.pt" ]]; then
      printf '%s\n' "$input_path/ckpt.pt"
      return 0
    fi
    echo "[ERROR] No ckpt_best.pt or ckpt.pt found in: $input_path" >&2
  fi

  return 1
}

collect_run_ckpts() {
  local run_dir="$1"
  local variant="${2:-both}"
  local best="$run_dir/ckpt_best.pt"
  local latest="$run_dir/ckpt.pt"
  case "$variant" in
    best)
      [[ -f "$best" ]] && echo "$best"
      ;;
    latest)
      [[ -f "$latest" ]] && echo "$latest"
      ;;
    both)
      [[ -f "$best" ]] && echo "$best"
      [[ -f "$latest" ]] && echo "$latest"
      ;;
    *)
      echo "[ERROR] Invalid CKPT_VARIANTS=$variant (use best|latest|both)" >&2
      exit 1
      ;;
  esac
}

is_true() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

should_traverse_all() {
  local requested="$1"
  local input_path="$2"
  local v
  v="$(echo "$requested" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    auto)
      [[ -d "$input_path" ]] && [[ ! -f "$input_path/ckpt.pt" ]] && [[ ! -f "$input_path/ckpt_best.pt" ]]
      ;;
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off) return 1 ;;
    *)
      echo "[ERROR] Invalid TRAVERSE_ALL_RUNS=$requested" >&2
      exit 1
      ;;
  esac
}

collect_ckpts_from_dir() {
  local parent="$1"
  local variant="$2"
  local found=0
  local p run_dir
  # include parent itself if it's already a run dir
  if [[ -d "$parent" ]] && { [[ -f "$parent/ckpt_best.pt" ]] || [[ -f "$parent/ckpt.pt" ]]; }; then
    while IFS= read -r p; do
      [[ -n "$p" ]] || continue
      echo "$p"
      found=1
    done < <(collect_run_ckpts "$parent" "$variant")
  fi
  # include direct subdirectories
  while IFS= read -r -d '' run_dir; do
    if [[ -f "$run_dir/ckpt_best.pt" || -f "$run_dir/ckpt.pt" ]]; then
      while IFS= read -r p; do
        [[ -n "$p" ]] || continue
        echo "$p"
        found=1
      done < <(collect_run_ckpts "$run_dir" "$variant")
    fi
  done < <(find "$parent" -mindepth 1 -maxdepth 1 -type d -print0)
  if [[ "$found" -eq 0 ]]; then
    # fallback: single explicit file/dir path behavior
    if p="$(resolve_nanogpt_ckpt "$parent" 2>/dev/null)"; then
      found=1
      echo "$p"
    fi
  fi
  [[ "$found" -eq 1 ]]
}

build_input_list() {
  local -a inputs=()
  if [[ "$#" -gt 0 ]]; then
    inputs=("$@")
  elif [[ -n "${NANOGPT_CKPT_LIST:-}" ]]; then
    while IFS= read -r line; do
      [[ -n "${line// }" ]] || continue
      inputs+=("$line")
    done <<< "$NANOGPT_CKPT_LIST"
  elif [[ -n "${NANOGPT_CKPT:-}" ]]; then
    inputs=("$NANOGPT_CKPT")
  elif [[ -n "$DEFAULT_NANOGPT_CKPT" ]]; then
    inputs=("$DEFAULT_NANOGPT_CKPT")
  else
    inputs=("${DEFAULT_NANOGPT_CKPTS[@]}")
  fi
  printf '%s\n' "${inputs[@]}"
}

declare -a CKPT_INPUTS=()
while IFS= read -r _input; do
  [[ -n "${_input:-}" ]] || continue
  CKPT_INPUTS+=("$_input")
done < <(build_input_list "$@")

if [[ "${#CKPT_INPUTS[@]}" -eq 0 ]]; then
  echo "[ERROR] Please provide a checkpoint file or run directory." >&2
  echo "Usage: bash $0 /path/to/run_dir [/path/to/another_run ...]" >&2
  echo "   or: bash $0 /path/to/ckpt_best.pt [/path/to/another_ckpt.pt ...]" >&2
  echo "   or: NANOGPT_CKPT=/path/to/run_dir bash $0" >&2
  echo "   or: NANOGPT_CKPT_LIST=$'path1\\npath2' bash $0" >&2
  echo "" >&2
  echo "When a directory is given, ckpt_best.pt is used if present, otherwise ckpt.pt." >&2
  exit 1
fi

if is_true "$BACKGROUND" && [[ "${_XSA_NANOGPT_BG_LAUNCHED:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  LOG_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
  LOG_PATH="${LOG_PATH:-$LOG_DIR/run_nanogpt_xsa_single_layer_flip_viz-${LOG_TIMESTAMP}.log}"
  ACTIVE_LOG_PATH="${ACTIVE_LOG_PATH:-$LOG_PATH}"
  if is_remote_live_log_path "$LOG_PATH"; then
    ACTIVE_LOG_PATH="/tmp/run_nanogpt_xsa_single_layer_flip_viz-${LOG_TIMESTAMP}.log"
  fi
  echo "[INFO] Launching in background"
  echo "[INFO] Live logs: $ACTIVE_LOG_PATH"
  if [[ "$ACTIVE_LOG_PATH" != "$LOG_PATH" ]]; then
    echo "[INFO] Final logs: $LOG_PATH"
  fi
  nohup env _XSA_NANOGPT_BG_LAUNCHED=1 BACKGROUND=false LOG_PATH="$LOG_PATH" ACTIVE_LOG_PATH="$ACTIVE_LOG_PATH" \
    bash "$0" "${CKPT_INPUTS[@]}" >"$ACTIVE_LOG_PATH" 2>&1 &
  BG_PID=$!
  echo "[INFO] PID: $BG_PID"
  exit 0
fi

trap finalize_live_log EXIT

declare -a CKPT_LIST=()
for NANOGPT_CKPT_INPUT in "${CKPT_INPUTS[@]}"; do
  if should_traverse_all "$TRAVERSE_ALL_RUNS" "$NANOGPT_CKPT_INPUT"; then
    while IFS= read -r ckpt; do
      CKPT_LIST+=("$ckpt")
    done < <(collect_ckpts_from_dir "$NANOGPT_CKPT_INPUT" "$CKPT_VARIANTS")
  else
    if [[ -d "$NANOGPT_CKPT_INPUT" ]]; then
      while IFS= read -r ckpt; do
        CKPT_LIST+=("$ckpt")
      done < <(collect_run_ckpts "$NANOGPT_CKPT_INPUT" "$CKPT_VARIANTS")
    else
      if ! NANOGPT_CKPT="$(resolve_nanogpt_ckpt "$NANOGPT_CKPT_INPUT")"; then
        echo "[ERROR] Could not resolve checkpoint from: $NANOGPT_CKPT_INPUT" >&2
        exit 1
      fi
      CKPT_LIST+=("$NANOGPT_CKPT")
    fi
  fi
done

if [[ "${#CKPT_LIST[@]}" -eq 0 ]]; then
  echo "[ERROR] No valid checkpoints found from the provided inputs." >&2
  exit 1
fi

declare -A _seen_ckpts=()
declare -a _deduped_ckpts=()
for _ckpt in "${CKPT_LIST[@]}"; do
  if [[ -n "${_seen_ckpts[$_ckpt]:-}" ]]; then
    continue
  fi
  _seen_ckpts["$_ckpt"]=1
  _deduped_ckpts+=("$_ckpt")
done
CKPT_LIST=("${_deduped_ckpts[@]}")

echo "[INFO] nanoGPT single-layer-flip wrapper"
echo "[INFO] INPUT_COUNT=${#CKPT_INPUTS[@]}"
echo "[INFO] CKPT_COUNT=${#CKPT_LIST[@]}"
echo "[INFO] OVERWRITE_EXISTING=$OVERWRITE_EXISTING"
echo "[INFO] PROMPT_SET=$PROMPT_SET MAX_LENGTH=$MAX_LENGTH"
"${PYTHON_BIN:-python3}" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("tiktoken") is None:
    print("[INFO] Missing dependency: tiktoken. Installing...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
else:
    print("[INFO] Dependency check passed: tiktoken", flush=True)
PY
fail_count=0
for NANOGPT_CKPT in "${CKPT_LIST[@]}"; do
  if ! _repo_root="$(resolve_nanogpt_repo_root "$NANOGPT_CKPT" "${NANOGPT_REPO_ROOT:-}")"; then
    echo "[ERROR] Failed to resolve nanoGPT repo root for checkpoint: $NANOGPT_CKPT" >&2
    fail_count=$((fail_count + 1))
    continue
  fi
  if [[ -z "${BASE_OUTPUT_DIR:-}" ]]; then
    _base_output_dir="$(dirname "$NANOGPT_CKPT")"
  else
    _base_output_dir="$BASE_OUTPUT_DIR"
    if [[ "$_base_output_dir" != /* ]]; then
      _base_output_dir="$ROOT_DIR/$_base_output_dir"
    fi
  fi
  _ckpt_filename="$(basename "$NANOGPT_CKPT" .pt)"
  _prompt_tag="$(sanitize_tag "$PROMPT_SET")"
  _output_prefix="${OUTPUT_PREFIX:-${_base_output_dir}/flip-${_ckpt_filename}-prompt${_prompt_tag}}"

  echo "[INFO] ===== Running ckpt: $NANOGPT_CKPT ====="
  echo "[INFO] NANOGPT_REPO_ROOT=$_repo_root"
  echo "[INFO] BASE_OUTPUT_DIR=$_base_output_dir"
  echo "[INFO] OUTPUT_PREFIX=$_output_prefix"
  "${PYTHON_BIN:-python3}" - "$NANOGPT_CKPT" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
state = ckpt.get("model", {}) if isinstance(ckpt, dict) else {}
gamma_keys = [k for k in state.keys() if "xsa_forward_gamma_raw" in k]

def _cfg(name, default=None):
    return cfg.get(name, default) if isinstance(cfg, dict) else default

print(f"[INFO] ckpt_config_xsa_forward_only={_cfg('xsa_forward_only', None)}", flush=True)
print(f"[INFO] ckpt_config_xsa_forward_ref={_cfg('xsa_forward_ref', None)}", flush=True)
print(f"[INFO] ckpt_config_xsa_forward_space={_cfg('xsa_forward_space', None)}", flush=True)
print(f"[INFO] ckpt_config_xsa_forward_target={_cfg('xsa_forward_target', None)}", flush=True)
print(f"[INFO] ckpt_config_xsa_forward_learnable_gamma={_cfg('xsa_forward_learnable_gamma', None)}", flush=True)
print(f"[INFO] ckpt_config_xsa_forward_gamma_per_head={_cfg('xsa_forward_gamma_per_head', None)}", flush=True)
print(f"[INFO] ckpt_config_xsa_forward_gamma_init={_cfg('xsa_forward_gamma_init', None)}", flush=True)
print(f"[INFO] ckpt_state_gamma_param_count={len(gamma_keys)}", flush=True)
if gamma_keys:
    print(f"[INFO] ckpt_state_gamma_param_example={gamma_keys[0]}", flush=True)
PY

  if ! MODEL_NAME="" NANOGPT_CKPT="$NANOGPT_CKPT" NANOGPT_REPO_ROOT="$_repo_root" BASE_OUTPUT_DIR="$_base_output_dir" OUTPUT_PREFIX="$_output_prefix" OVERWRITE_EXISTING="$OVERWRITE_EXISTING" RUN_IN_BACKGROUND=false PROMPT_SET="$PROMPT_SET" MAX_LENGTH="$MAX_LENGTH" \
    bash "$SCRIPT_DIR/../visualization/run_qwen_xsa_single_layer_flip_viz.sh"; then
    echo "[ERROR] Failed for checkpoint: $NANOGPT_CKPT" >&2 || true
    fail_count=$((fail_count + 1))
  fi
done

if [[ "$fail_count" -gt 0 ]]; then
  echo "[ERROR] Completed with failures: $fail_count/${#CKPT_LIST[@]}" >&2 || true
  exit 1
fi
