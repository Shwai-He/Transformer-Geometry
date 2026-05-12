<h1 align="center">Transformer Geometry</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch-red">
  <img alt="Models" src="https://img.shields.io/badge/Models-Transformers-orange">
  <img alt="Evaluation" src="https://img.shields.io/badge/Evaluation-lm--eval-success">
</p>

<p align="center">
  <a href="#what-you-can-run-here">What You Can Run</a> |
  <a href="#selected-figures-and-code">Figures and Code</a> |
  <a href="#installation">Installation</a> |
  <a href="#repository-layout">Layout</a> |
  <a href="#related-documentation">Docs</a>
</p>

<p align="center">
  Public codebase for studying transformer computation through parallel and perpendicular update geometry.
</p>

<p align="center">
  <img src="docs/assets/transformer_geometry_overview.svg" alt="Overview of parallel and perpendicular transformer update geometry" width="860">
</p>
<p align="center">
  <em>Overview. Transformer updates are decomposed into direction-preserving and direction-changing components, then used for inference-time editing, compression diagnostics, and training-time analysis.</em>
</p>

This repository supports experiments for a geometric view of transformer computation. The central decomposition separates each module update into a **parallel** component, which mostly rescales the current representation, and a **perpendicular** component, which changes direction. We compare residual-space and value-space versions of this decomposition and use them to study editing, compression, and optimization behavior.

This public release focuses on code, scripts, lightweight plotting data, and exported figure assets.

## What You Can Run Here

- **Geometry probing**: measure parallel and perpendicular update structure across layers, branches, prompts, and generation steps.
- **Inference-time component editing**: run residual-space and value-space interventions, including attention-parallel removal and diagonal edits.
- **Benchmark evaluation**: evaluate edited models on general tasks and long-context RULER through the included lm-evaluation-harness fork.
- **Training-time intervention**: inspect scratch pretraining runs that suppress or rescale parallel updates.
- **Compression diagnostics**: decompose pruning and quantization error into parallel and perpendicular components.
- **Figure reproduction**: regenerate selected paper plots from lightweight saved summaries.

## Installation

Create an environment, install the light root dependencies, then install the evaluation harness when running benchmark experiments.

```bash
conda create -n transformer-geometry python=3.10 -y
conda activate transformer-geometry
pip install -r requirements.txt

cd lm-evaluation-harness
pip install -e .
cd ..
```

For model-scale benchmark runs, install the backend packages required by your local setup, such as `transformers`, `accelerate`, `datasets`, and CUDA-compatible PyTorch builds.

## Selected figures and code

The figures below are exported from the paper experiments as SVG assets for quick browsing. Each block gives the local entrypoints that reproduce the figure or the corresponding experiment.

### Component structure across depth

These profiles measure how much transformer updates preserve the current direction versus change it across depth. The plotting scripts use saved probe summaries, while the probe runners regenerate the underlying activations.

<table>
  <tr>
    <td width="50%" align="center"><img src="docs/assets/component_profiles_qwen3_4b.svg" alt="Parallel and perpendicular component profiles for Qwen3-4B" width="390"></td>
    <td width="50%" align="center"><img src="docs/assets/component_profiles_qwen3_30b_a3b.svg" alt="Parallel and perpendicular component profiles for Qwen3-30B-A3B" width="390"></td>
  </tr>
</table>

```text
# Plot saved component profiles
drawing/para_dist/
drawing/embedded_data/

# Regenerate geometry probes
scripts/run_probe.py
scripts/run_batch_probe.py
```

### Manual component scaling at inference time

These ablations manually scale parallel or perpendicular components and measure the resulting perplexity change. They are the lightweight diagnostic counterpart to the benchmark evaluations in `lm-evaluation-harness/`.

<table>
  <tr>
    <td width="50%" align="center"><img src="docs/assets/component_scaling_parallel.svg" alt="Perplexity change under parallel component scaling" width="360"></td>
    <td width="50%" align="center"><img src="docs/assets/component_scaling_perpendicular.svg" alt="Perplexity change under perpendicular component scaling" width="360"></td>
  </tr>
</table>

```text
# Plot scaling summaries
drawing/para_ablation/plot_para_ppl_summary.py
drawing/para_ablation/data/

# Run intervention evaluations
lm-evaluation-harness/scripts/run_lm_eval_xsa_setting.sh
lm-evaluation-harness/scripts/run_lm_eval_attn_removal_batch.sh
```

### Attention diagonal editing

This view compares attention maps after value-space and residual-space diagonal edits. The hook code implements the edit, and the drawing code replots saved attention-map bundles.

<p align="center">
  <img src="docs/assets/diagonal_edit_attention_maps.svg" alt="Attention maps under diagonal editing" width="660">
</p>

```text
# Implement diagonal edits
lm-evaluation-harness/lm_eval/models/attn_diag_hooks.py
lm-evaluation-harness/scripts/run_lm_eval_attn_diag_setting.sh

# Replot saved attention maps
drawing/attn_matrix/replot_from_saved_data.py
drawing/attn_matrix/
```

### Compression error geometry

The compression experiments decompose pruning and quantization error into parallel and perpendicular parts. The example below shows that the direction-changing error is a useful descriptor of compression behavior.

<p align="center">
  <img src="docs/assets/compression_attention_perp.svg" alt="Attention compression error decomposed by perpendicular component" width="620">
</p>

```text
# Run compression geometry analysis
compression/code/layerwise_para_perp_compare.py
compression/code/visualize_local_sweep_compare.py

# Recreate paper plot
drawing/comp_analysis/plot_local_flip_compare_v2.py
drawing/comp_analysis/data/all_settings_master_v2.tsv
```

### Training-time parallel removal

The training experiments test whether suppressing parallel updates changes optimization. The plot summarizes scratch pretraining runs across model sizes, with downstream evaluation handled through the same lm-eval workspace.

<p align="center">
  <img src="docs/assets/pretraining_parallel_removal.svg" alt="Pretraining loss curves under parallel removal" width="700">
</p>

```text
# Training workspace and loss plots
training/
drawing/loss_curves/plot_loss_csv_sizes_overview.py
drawing/loss_curves/data/

# Post-training evaluation
lm-evaluation-harness/scripts/run_lm_eval_nanogpt_setting.sh
lm-evaluation-harness/scripts/collect_nanogpt_lm_eval_results.py
```

## Repository layout

```text
lm-evaluation-harness/     benchmark evaluation and intervention runners
training/                  training-side experiments and optimization studies
compression/               pruning and quantization geometry analysis
drawing/                   editable plotting code and figure assets
analysis/                  standalone diagnostic and visualization scripts
scripts/                   small repo-level runners and probes
src/repgeo/                reusable geometry and intervention utilities
notebooks/                 exploratory notebooks
results/                   non-paper intermediate outputs and local artifacts
```

## Quick Usage

Start from the figure or experiment family you care about, then use the nearby scripts listed above. The most common entrypoints are:

```bash
# Geometry probes
python scripts/run_probe.py
python scripts/run_batch_probe.py

# Inference-time component editing and diagonal edits
bash lm-evaluation-harness/scripts/run_lm_eval_xsa_setting.sh
bash lm-evaluation-harness/scripts/run_lm_eval_attn_diag_setting.sh

# Long-context RULER evaluation
bash lm-evaluation-harness/scripts/run_lm_eval_ruler_all_settings.sh

# Compression geometry
bash compression/scripts/run_layerwise_para_perp_compare.sh

# Figure regeneration examples
python drawing/para_ablation/plot_para_ppl_summary.py
python drawing/comp_analysis/plot_local_flip_compare_v2.py
python drawing/loss_curves/plot_loss_csv_sizes_overview.py
```

For a paper-to-code index, use [`PAPER_CODE_MAP.md`](PAPER_CODE_MAP.md). For workspace-specific details, use the README files under `lm-evaluation-harness/scripts/`, `training/`, `compression/`, and `analysis/`.

## Related documentation

- `PAPER_CODE_MAP.md`: maps paper claims to code locations
- `PROJECT_STRUCTURE.md`: repository structure policy
- `lm-evaluation-harness/scripts/README.md`: evaluation launcher guide
- `training/README.md`: training workspace notes
- `compression/README.md`: compression workspace notes
- `analysis/README.md`: standalone analysis script guide

## Compatibility note

`representation-analysis/` is retained only as a compatibility layer for older commands. New work should use the root-level paths directly.
