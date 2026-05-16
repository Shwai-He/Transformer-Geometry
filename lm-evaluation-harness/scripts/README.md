# lm-eval Script Layout

This folder is the benchmark-evaluation backend for the NeurIPS transformer-geometry paper. It contains both reusable launchers and paper-specific batch scripts for the inference-time editing experiments.

## How this relates to the paper

Use this workspace for the parts of the paper that evaluate component edits without task-specific fine-tuning:

- general-task evaluation under attention-side edits
- long-context RULER evaluation
- diagonal-removal and residual-space/value-space comparisons
- checkpoint evaluation for training-side experiments

## Reusable launchers

- `run_lm_eval_xsa_setting.sh`: single-setting HF/XSA launcher.
- `run_lm_eval_attn_diag_setting.sh`: wrapper for diagonal-removal settings.
- `run_lm_eval_nanogpt_setting.sh`: single-setting nanoGPT launcher.

These are the lowest-level entrypoints to prefer when building a new batch script.

## Paper or benchmark batches

- `run_lm_eval_ruler_all_settings.sh`: RULER benchmark runs.
- `run_lm_eval_xsa_multihead_batch.sh`: general XSA multihead batch.
- `run_lm_eval_attn_diag_batch.sh`: diagonal-removal batch.
- `run_lm_eval_attn_removal_batch.sh`: residual-space removal batch.

## Collection scripts

- `collect_xsa_lm_eval_results.py` / `.sh`
- `collect_xsa_lm_eval_ruler_results.py` / `.sh`
- `collect_nanogpt_lm_eval_results.py` / `.sh`

These scripts convert raw lm-eval outputs into paper-facing CSV summaries.

## One-off repair scripts

- `run_missing_qwen30b_drop_cells.sh`: one-off script for the missing Qwen3-30B-A3B DROP residual-attention cell. It should not be used as a general benchmark launcher.

## Suggested usage pattern

1. Pick the paper setting you want to reproduce.
2. Identify whether it is a single setting, a batch, or a one-off repair.
3. Run the corresponding batch or launcher.
4. Collect paper-facing summaries with the collection scripts.

Future cleanup can move these groups into `launchers/`, `paper/`, `collect/`, and `oneoff/` once active remote jobs no longer depend on the current flat paths.
