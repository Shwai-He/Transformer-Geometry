#!/usr/bin/env bash
set -euo pipefail

# Simple single-model launcher:
# Just run:
#   bash scripts/run_models_groups.sh 1
#   bash scripts/run_models_groups.sh 2
#
# It picks the first model from model.text.
OPTION="${1:-1}"
case "${OPTION}" in
  1)
    VISIBLE_DEVICES="0,1"
    GEOMETRY_SPACE="logits"
    ;;
  2)
    VISIBLE_DEVICES="2,3"
    GEOMETRY_SPACE="hidden"
    ;;
  *)
    echo "Invalid option: ${OPTION}. Use 1 or 2."
    exit 1
    ;;
esac

MAX_NEW_TOKENS="32"

python3 -c "import importlib.util,sys;spec=importlib.util.find_spec('transformers');sys.exit(0 if (spec is not None and __import__('transformers').__version__=='4.52.4') else 1)" || python3 -m pip install transformers==4.52.4

MODEL_PATH="$(grep -v '^[[:space:]]*#' model.text | sed '/^[[:space:]]*$/d' | head -n 1)"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "No model found in model.text"
  exit 1
fi

CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
python3 scripts/run_generation_probe.py \
  --model_name_or_path "${MODEL_PATH}" \
  --geometry_space "${GEOMETRY_SPACE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --include_sublayer_metrics \
  --device_map auto
