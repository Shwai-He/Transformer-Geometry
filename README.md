# Transformer Geometry

This repository accompanies the NeurIPS 2026 paper in `_NeurIPS_2026_/`. The paper studies transformer computation through a geometric decomposition of module updates into **parallel** and **perpendicular** components, compares **residual-space** and **value-space** views, and uses that geometry for three connected goals:

- inference-time component editing
- compression diagnostics
- training-time intervention

## Paper directory

The manuscript itself lives in:

- `_NeurIPS_2026_/`

This paper directory contains the LaTeX source, paper-facing figures, and paper-facing result CSVs. The rest of the repository contains the code used to run, analyze, and visualize the experiments reported there.

## Repository map

```text
_NeurIPS_2026_/            paper source, final figures, final CSV tables
lm-evaluation-harness/     benchmark evaluation for general-task and RULER experiments
training/                  training and optimization runs related to parallel suppression
compression/               pruning and quantization geometry analysis
drawing/                   editable figure-generation code and plotting assets
analysis/                  standalone intervention, visualization, and diagnostic scripts
src/repgeo/                reusable geometry and intervention utilities
notebooks/                 exploratory analysis notebooks
```

## Paper-related code by topic

### 1. Geometry of transformer updates

These parts of the code support the paper's core decomposition into parallel and perpendicular components.

- `src/repgeo/`: reusable geometry utilities.
- `scripts/run_probe.py`: single-prompt probing.
- `scripts/run_batch_probe.py`: batch probing across prompts.
- `scripts/run_generation_probe.py`: generation-step geometry analysis.
- `analysis/`: standalone diagnostic and visualization scripts for deeper inspection.

### 2. Inference-time component editing

These parts support the paper's editing experiments, including residual-space versus value-space interventions and diagonal edits.

- `lm-evaluation-harness/lm_eval/models/hf_xsa.py`
- `lm-evaluation-harness/lm_eval/models/xsa_hooks.py`
- `lm-evaluation-harness/lm_eval/models/attn_diag_hooks.py`
- `analysis/qwen_xsa_forward_ablation.py`
- `analysis/qwen_xsa_forward_generate.py`

Key runners:

- `lm-evaluation-harness/scripts/run_lm_eval_xsa_setting.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_attn_diag_setting.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_xsa_multihead_batch.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_attn_removal_batch.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_attn_diag_batch.sh`

### 3. General-task and long-context evaluation

These parts correspond to the paper's evaluation without task-specific fine-tuning, including both standard downstream tasks and RULER.

- `lm-evaluation-harness/scripts/run_lm_eval_xsa_multihead_batch.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_attn_removal_batch.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_attn_diag_batch.sh`
- `lm-evaluation-harness/scripts/run_lm_eval_ruler_all_settings.sh`

Result collection:

- `lm-evaluation-harness/scripts/collect_xsa_lm_eval_results.py`
- `lm-evaluation-harness/scripts/collect_xsa_lm_eval_ruler_results.py`

### 4. Training-time intervention

These parts correspond to the paper's experiments on suppressing self-value-parallel updates during training and tracking optimization effects.

- `training/`: training-side workspace.
- `analysis/check_nanogpt_gamma_ckpt.py`
- `analysis/cleanup_invalid_nanogpt_checkpoints.py`
- `analysis/download_wandb_history.py`
- `analysis/plot_wandb_history.py`
- `lm-evaluation-harness/scripts/run_lm_eval_nanogpt_setting.sh`
- `lm-evaluation-harness/scripts/collect_nanogpt_lm_eval_results.py`

### 5. Compression diagnostics

These parts correspond to the paper's pruning and quantization analysis.

- `compression/code/layerwise_para_perp_compare.py`
- `compression/code/visualize_local_sweep_compare.py`
- `compression/code/visualize_local_sweep_summary.py`
- `compression/scripts/run_layerwise_para_perp_compare.sh`
- `compression/scripts/run_intra_layer_quant_para_perp.sh`
- `compression/scripts/run_intra_layer_prune_para_perp.sh`
- `compression/scripts/run_inter_layer_drop_para_perp.sh`

## Figure-generation code

Editable plotting and figure-source code lives under `drawing/`.

Representative subfolders:

- `drawing/overview/`: schematics and overview assets
- `drawing/para_dist/`: parallel/perpendicular profile plots
- `drawing/comp_analysis/`: compression-geometry plots
- `drawing/loss_curves/`: training curves
- `drawing/attn_matrix/`: attention-matrix and diagonal-edit visualizations

Paper-facing figures used by LaTeX should be copied into `_NeurIPS_2026_/figs`.

## Recommended reading order

1. Read `_NeurIPS_2026_/` for the paper narrative.
2. Read `PAPER_CODE_MAP.md` for a direct paper-to-code index.
3. Use `lm-evaluation-harness/` for benchmark evaluation.
4. Use `training/` for training-side experiments.
5. Use `compression/` for pruning/quantization analysis.
6. Use `drawing/` for figure reproduction or editing.

## Related docs

- `PAPER_CODE_MAP.md`: paper claims mapped to code locations.
- `PROJECT_STRUCTURE.md`: repository layout policy.
- `_NeurIPS_2026_/README.md`: notes for the paper directory.
- `lm-evaluation-harness/scripts/README.md`: launcher guide for benchmark evaluation.
- `analysis/README.md`: grouped guide to standalone research scripts.

## Compatibility note

`representation-analysis/` is retained only as a compatibility layer for older commands. New work should use the root-level paths directly.
