#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$REPO_ROOT"

# Semicolon-separated specs:
#   tag|preset|loader|model_path|sides[|layer_paths]
# Example:
#   qwenimage|qwen-image|diffusers|/ckpt/qwen-image|gen;bagel|bagel|transformers|/ckpt/bagel|und,gen|language_model.model.layers
MODEL_SPECS="${MODEL_SPECS:?Set MODEL_SPECS as tag|preset|loader|model_path|sides entries.}"
SPACES="${SPACES:-residual,value}"
TARGETS_RESIDUAL="${TARGETS_RESIDUAL:-${TARGETS:-block,attn,mlp}}"
TARGETS_VALUE="${TARGETS_VALUE:-value}"
PARA_SCALES="${PARA_SCALES:-1.0,0.0}"
PERP_SCALES="${PERP_SCALES:-1.0,0.0}"
MODE_DEFAULT="${MODE_DEFAULT:-}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/runs/vlm_geometry_scaling/results}"
BENCH_NAME="${BENCH_NAME:-smoke}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

IFS=';' read -r -a SPECS <<< "$MODEL_SPECS"
IFS=',' read -r -a SPACE_LIST <<< "$SPACES"
IFS=',' read -r -a TARGET_LIST_RESIDUAL <<< "$TARGETS_RESIDUAL"
IFS=',' read -r -a TARGET_LIST_VALUE <<< "$TARGETS_VALUE"
IFS=',' read -r -a PARA_LIST <<< "$PARA_SCALES"
IFS=',' read -r -a PERP_LIST <<< "$PERP_SCALES"

for spec in "${SPECS[@]}"; do
  [[ -z "$spec" ]] && continue
  IFS='|' read -r tag preset loader model_path sides layer_paths <<< "$spec"
  if [[ -z "${tag:-}" || -z "${preset:-}" || -z "${loader:-}" || -z "${model_path:-}" || -z "${sides:-}" ]]; then
    echo "[ERROR] Bad MODEL_SPECS entry: $spec" >&2
    exit 2
  fi
  mode="$MODE_DEFAULT"
  if [[ -z "$mode" ]]; then
    if [[ "$loader" == "diffusers" || "$loader" == "sparse_qwenimage" ]]; then
      mode="pipeline_generate"
    elif [[ "$loader" == "sparse_ming" ]]; then
      mode="ming_image_generate"
    elif [[ "$loader" == "sparse_bagel" ]]; then
      mode="bagel_interleave_generate"
    else
      mode="text_forward"
    fi
  fi

  IFS=',' read -r -a SIDE_LIST <<< "$sides"
  for side in "${SIDE_LIST[@]}"; do
    for space in "${SPACE_LIST[@]}"; do
      if [[ "$space" == "value" ]]; then
        TARGET_LIST=("${TARGET_LIST_VALUE[@]}")
      else
        TARGET_LIST=("${TARGET_LIST_RESIDUAL[@]}")
      fi
      for target in "${TARGET_LIST[@]}"; do
        for para in "${PARA_LIST[@]}"; do
          for perp in "${PERP_LIST[@]}"; do
            setting="side_${side}__space_${space}__target_${target}__para_${para//./p}__perp_${perp//./p}"
            out_dir="$RESULT_ROOT/by_model/$tag/$BENCH_NAME/$setting"
            if [[ "$SKIP_EXISTING" == "true" && -s "$out_dir/result.json" ]]; then
              echo "[SKIP] $tag $setting"
              continue
            fi
            echo "[RUN] $tag preset=$preset loader=$loader side=$side space=$space target=$target para=$para perp=$perp"
            if ! MODEL_TAG="$tag" \
              MODEL_PRESET="$preset" \
              MODEL_NAME_OR_PATH="$model_path" \
              LAYER_PATHS="${layer_paths:-}" \
              LOADER="$loader" \
              MODE="$mode" \
              SIDE="$side" \
              SPACE="$space" \
              TARGET="$target" \
              PARA_SCALE="$para" \
              PERP_SCALE="$perp" \
              RESULT_ROOT="$RESULT_ROOT" \
              BENCH_NAME="$BENCH_NAME" \
              MODE_DEFAULT="$MODE_DEFAULT" \
              DTYPE="${DTYPE:-bf16}" \
              DEVICE="${DEVICE:-cuda}" \
              DEVICE_MAP="${DEVICE_MAP:-auto}" \
              VALUE_HEAD_MODE="${VALUE_HEAD_MODE:-multihead}" \
              VALUE_REF_EXPANSION="${VALUE_REF_EXPANSION:-model_type}" \
              ATTN_ATTR_SOURCE="${ATTN_ATTR_SOURCE:-model_type}" \
              FAIL_ON_MISSING_TARGET="${FAIL_ON_MISSING_TARGET:-true}" \
              MING_LOAD_IMAGE_GEN="${MING_LOAD_IMAGE_GEN:-auto}" \
                bash "$SCRIPT_DIR/run_vlm_geometry_smoke.sh"; then
              echo "[FAIL] $tag $setting"
              mkdir -p "$out_dir"
              date -u +"%Y-%m-%dT%H:%M:%SZ" > "$out_dir/failed.txt"
              if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
                exit 1
              fi
            fi
          done
        done
      done
    done
  done
done
