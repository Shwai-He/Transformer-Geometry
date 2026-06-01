#!/usr/bin/env bash
set -euo pipefail

# One-command matrix runner.
# Run: bash scripts/run_models.sh

RUN_TYPE="generation"
TOP_K="5"
MAX_NEW_TOKENS="32"
PROMPT_FILE="results/prompts.txt"
MODEL_NAME_OR_PATH="/path/to/your/model"
TRANSFORMERS_VERSION="4.52.4"

# Matrix options
MEASURE_MODES=("prefill" "decode" "both")
GEOMETRY_SPACES=("hidden" "logits")
GPUS="4,5,6,7"

# Random init toggle
USE_RANDOM_INIT="0"
RANDOM_SEED="42"

EXTRA_ARGS=(--top_k "${TOP_K}" --prompt_file "${PROMPT_FILE}" --all_prompts)
if [[ "${RUN_TYPE}" == "generation" ]]; then
  EXTRA_ARGS+=(--max_new_tokens "${MAX_NEW_TOKENS}")
fi
if [[ "${USE_RANDOM_INIT}" == "1" ]]; then
  EXTRA_ARGS+=(--random_init --random_seed "${RANDOM_SEED}")
fi

echo "[INFO] model=${MODEL_NAME_OR_PATH} -> transformers==${TRANSFORMERS_VERSION}"
python3 -c "import importlib.util,sys;spec=importlib.util.find_spec('transformers');sys.exit(0 if (spec is not None and __import__('transformers').__version__=='${TRANSFORMERS_VERSION}') else 1)" \
  || python3 -m pip install "transformers==${TRANSFORMERS_VERSION}"

for geometry_space in "${GEOMETRY_SPACES[@]}"; do
  for mode in "${MEASURE_MODES[@]}"; do
    echo "============================================================"
    echo "[RUN ] run_type=${RUN_TYPE} mode=${mode} geometry_space=${geometry_space} gpus=${GPUS} model=${MODEL_NAME_OR_PATH}"
    python3 scripts/run_models_on_single_gpus.py \
      --run_type "${RUN_TYPE}" \
      --geometry_space "${geometry_space}" \
      --model "${MODEL_NAME_OR_PATH}" \
      --gpus "${GPUS}" \
      -- "${EXTRA_ARGS[@]}" --measure_mode "${mode}"
    echo "[DONE] run_type=${RUN_TYPE} mode=${mode} geometry_space=${geometry_space} gpus=${GPUS} model=${MODEL_NAME_OR_PATH}"
  done
done
