#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEFAULT_NANOGPT_CKPT="/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-2p7b-baseline-fineweb100bt-ctx2048-gb8192-lr3e-4-s1337/ckpt_best.pt"

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
  local resolved=""
  local latest=""

  if [[ "$input_path" != /* ]]; then
    input_path="$(cd "$ROOT_DIR" && cd "$(dirname "$input_path")" && pwd)/$(basename "$input_path")"
  fi

  if [[ -f "$input_path" ]]; then
    printf '%s\n' "$input_path"
    return 0
  fi

  if [[ -d "$input_path" ]]; then
    latest="$(find "$input_path" -maxdepth 1 -type f \( -name 'ckpt.pt' -o -name 'ckpt_best.pt' -o -name 'ckpt_iter_*.pt' \) -print0 | xargs -0 ls -1t 2>/dev/null | head -n 1)"
    if [[ -n "$latest" ]]; then
      printf '%s\n' "$latest"
      return 0
    fi
  fi

  return 1
}

NANOGPT_CKPT_INPUT="${NANOGPT_CKPT:-${1:-$DEFAULT_NANOGPT_CKPT}}"
if [[ -z "$NANOGPT_CKPT_INPUT" ]]; then
  echo "[ERROR] NANOGPT_CKPT is required." >&2
  echo "Usage: NANOGPT_CKPT=/path/to/ckpt.pt bash $0" >&2
  echo "   or: NANOGPT_CKPT=/path/to/run_dir bash $0" >&2
  echo "   or: bash $0 /path/to/ckpt.pt" >&2
  echo "   or: bash $0 /path/to/run_dir" >&2
  exit 1
fi

if ! NANOGPT_CKPT="$(resolve_nanogpt_ckpt "$NANOGPT_CKPT_INPUT")"; then
  echo "[ERROR] Could not resolve checkpoint from: $NANOGPT_CKPT_INPUT" >&2
  exit 1
fi

if ! NANOGPT_REPO_ROOT="$(resolve_nanogpt_repo_root "$NANOGPT_CKPT" "${NANOGPT_REPO_ROOT:-}")"; then
  exit 1
fi

if [[ -z "${BASE_OUTPUT_DIR:-}" ]]; then
  BASE_OUTPUT_DIR="$(dirname "$NANOGPT_CKPT")"
fi
if [[ "$BASE_OUTPUT_DIR" != /* ]]; then
  BASE_OUTPUT_DIR="$ROOT_DIR/$BASE_OUTPUT_DIR"
fi

export MODEL_NAME=""
export NANOGPT_CKPT
export NANOGPT_REPO_ROOT
export BASE_OUTPUT_DIR

echo "[INFO] nanoGPT attention-matrix wrapper"
echo "[INFO] NANOGPT_CKPT=$NANOGPT_CKPT"
echo "[INFO] NANOGPT_REPO_ROOT=$NANOGPT_REPO_ROOT"
echo "[INFO] BASE_OUTPUT_DIR=$BASE_OUTPUT_DIR"

exec bash "$SCRIPT_DIR/run_qwen_xsa_attention_matrix_viz.sh"
