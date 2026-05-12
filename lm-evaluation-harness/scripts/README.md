# lm-eval Script Layout

This folder currently contains both reusable launchers and paper-specific batch scripts. Until files are physically moved, use the categories below to avoid running the wrong entrypoint.

## Reusable launchers

- `run_lm_eval_xsa_setting.sh`: single-setting HF/XSA launcher.
- `run_lm_eval_attn_diag_setting.sh`: wrapper for diagonal-removal settings.
- `run_lm_eval_nanogpt_setting.sh`: single-setting nanoGPT launcher.

## Paper or benchmark batches

- `run_lm_eval_ruler_all_settings.sh`: RULER benchmark runs.
- `run_lm_eval_xsa_multihead_batch.sh`: general XSA multihead batch.
- `run_lm_eval_attn_diag_batch.sh`: diagonal-removal batch.
- `run_lm_eval_attn_removal_batch.sh`: residual-space removal batch.

## Collection scripts

- `collect_xsa_lm_eval_results.py` / `.sh`
- `collect_xsa_lm_eval_ruler_results.py` / `.sh`
- `collect_nanogpt_lm_eval_results.py` / `.sh`

## One-off repair scripts

- `run_missing_qwen30b_drop_cells.sh`: one-off script for the missing Qwen3-30B-A3B DROP residual-attention cell. It should not be used as a general benchmark launcher.

Future cleanup can move these groups into `launchers/`, `paper/`, `collect/`, and `oneoff/` once active remote jobs no longer depend on the current flat paths.
