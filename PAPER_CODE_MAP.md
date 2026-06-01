# Paper-to-Code Map

This document maps the main themes of the NeurIPS 2026 paper to the repository workspaces. It is intended as a practical index rather than a full reproducibility guide.

## Paper root

- `_NeurIPS_2026_/`: manuscript source, paper-facing figures, and paper-facing result CSVs.

Use this directory to understand what is reported in the paper. Use the root-level workspaces below to locate the code that produces or analyzes those results.

## 1. Residual-space and value-space geometry

These parts of the paper ask how transformer updates decompose into parallel and perpendicular components, and how the interpretation changes between residual space and value space.

Primary code locations:

- `src/repgeo/`: reusable geometry utilities.
- `scripts/run_probe.py`, `scripts/run_batch_probe.py`, `scripts/run_generation_probe.py`: basic probing and visualization-oriented entrypoints.
- `analysis/`: standalone intervention and visualization scripts, especially Qwen/XSA forward analysis.
- `drawing/para_dist/` and related plotting folders: editable figure-source code for geometric profiles.

## 2. Inference-time component editing

These parts of the paper evaluate component scaling, residual-space versus value-space interventions, and diagonal editing.

Primary code locations:

- `lm-evaluation-harness/`: benchmark-facing evaluation backend.
- `lm-evaluation-harness/lm_eval/models/hf_xsa.py`: HF model integration for XSA-style interventions.
- `lm-evaluation-harness/lm_eval/models/xsa_hooks.py`: forward hooks for component edits.
- `lm-evaluation-harness/lm_eval/models/attn_diag_hooks.py`: diagonal-edit hooks.
- `lm-evaluation-harness/scripts/run_lm_eval_xsa_setting.sh`: single-setting runner.
- `lm-evaluation-harness/scripts/run_lm_eval_attn_diag_setting.sh`: diagonal-removal runner.
- `analysis/forward_geometry/qwen_xsa_forward_ablation.py`: direct forward ablations and geometry inspection.

## 3. General-task and long-context evaluation

These parts of the paper report data-free, no-finetuning evaluation after inference-time edits.

Primary code locations:

- `lm-evaluation-harness/scripts/run_lm_eval_xsa_multihead_batch.sh`: general benchmark batch.
- `lm-evaluation-harness/scripts/run_lm_eval_attn_removal_batch.sh`: residual-space attention-removal batch.
- `lm-evaluation-harness/scripts/run_lm_eval_attn_diag_batch.sh`: diagonal-removal batch.
- `lm-evaluation-harness/scripts/run_lm_eval_ruler_all_settings.sh`: RULER long-context benchmark.
- `lm-evaluation-harness/scripts/collect_xsa_lm_eval_results.py`: general-task result collection.
- `lm-evaluation-harness/scripts/collect_xsa_lm_eval_ruler_results.py`: RULER result collection.

## 4. Training-time intervention

These parts of the paper study whether suppressing self-value-parallel updates changes training dynamics and downstream behavior.

Primary code locations:

- `training/`: training-side workspace.
- `lm-evaluation-harness/scripts/run_lm_eval_nanogpt_*.sh`: evaluation of training-produced checkpoints.
- `analysis/nanogpt/check_nanogpt_gamma_ckpt.py`, `analysis/nanogpt/cleanup_invalid_nanogpt_checkpoints.py`: checkpoint-side diagnostics.
- `analysis/utils/download_wandb_history.py`, `analysis/utils/plot_wandb_history.py`: training trace inspection.

## 5. Compression diagnostics

These parts of the paper use the same geometry to compare dense and compressed models under quantization or pruning.

Primary code locations:

- `compression/`: canonical compression-analysis workspace.
- `compression/code/layerwise_para_perp_compare.py`: core layerwise geometry comparison.
- `compression/scripts/run_layerwise_para_perp_compare.sh`: main launcher.
- `compression/scripts/run_intra_layer_quant_para_perp.sh`: quantization-side analysis.
- `compression/scripts/run_intra_layer_prune_para_perp.sh`: pruning-side analysis.
- `compression/scripts/run_inter_layer_drop_para_perp.sh`: inter-layer comparisons.
- `compression/code/visualize_local_sweep_compare.py` and related plotting helpers.

## 6. Figure-generation sources

The paper-facing figures in `_NeurIPS_2026_/figs` are typically produced from editable code under `drawing/`.

Representative folders:

- `drawing/overview/`: overview or schematic assets.
- `drawing/para_dist/`: parallel/perpendicular profile plots.
- `drawing/comp_analysis/`: compression-geometry plots.
- `drawing/loss_curves/`: training curves.
- `drawing/attn_matrix/`: attention-matrix or diagonal-edit visualizations.

## Reading order for new contributors

1. Start with `_NeurIPS_2026_/` to understand the paper narrative.
2. Read `README.md` and `PROJECT_STRUCTURE.md` for repository layout.
3. Use this file to jump from a paper claim to the relevant workspace.
4. For benchmark results, go to `lm-evaluation-harness/`.
5. For training results, go to `training/`.
6. For compression analysis, go to `compression/`.
7. For figure reproduction or editing, go to `drawing/`.
