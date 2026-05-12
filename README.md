# Transformer Geometry

This repository contains the public codebase for our work on the geometry of transformer computation. The central idea is to decompose a module update into a **parallel** component that mainly rescales the current representation and a **perpendicular** component that changes direction. We use this view to study three connected problems:

- **inference-time component editing**
- **compression diagnostics**
- **training-time intervention**

The public repository is organized around the experiments and analysis code. Paper source files, private drafting assets, and Overleaf-specific materials are not included here.

<p align="center">
  <img src="docs/assets/transformer_geometry_overview.svg" alt="Overview of parallel and perpendicular transformer update geometry" width="860">
</p>

## What this repository reproduces

This codebase supports the main experimental threads of the paper:

1. **Geometry probing of pretrained transformers**
   Measure parallel and perpendicular update structure across layers, branches, prompts, and generation steps.
2. **Inference-time editing without additional training**
   Compare residual-space and value-space interventions, including attention-parallel removal and diagonal edits, on general-task benchmarks and long-context RULER evaluation.
3. **Training-time suppression of parallel updates**
   Study how parallel suppression affects optimization and downstream behavior in scratch pretraining runs.
4. **Compression-induced error geometry**
   Analyze pruning and quantization through the same parallel/perpendicular decomposition.
5. **Figure generation and diagnostic visualization**
   Rebuild analysis plots, schematic figures, and inspection notebooks used to interpret the experiments.

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

## Workspace index

Most experiment-specific guidance now lives next to the figures above. This section is only a quick directory map.

```text
scripts/                    geometry probing and small runners
lm-evaluation-harness/       benchmark interventions and collectors
training/                    scratch pretraining experiments
compression/                 pruning and quantization analysis
drawing/                     plotting code and lightweight figure data
analysis/                    standalone diagnostics and exploratory checks
```

## How to use this repository

### Read the code in this order

1. Start with `PAPER_CODE_MAP.md` for a paper-to-code index.
2. Use `lm-evaluation-harness/` if you want to reproduce benchmark results.
3. Use `training/` for training-side experiments.
4. Use `compression/` for pruning and quantization analysis.
5. Use `drawing/` and `notebooks/` for figure reproduction and exploratory analysis.

### Install and run

This repository is a research workspace rather than a single packaged library. In practice, workflows are launched from the subdirectories above. Typical setup is:

1. Create a Python environment and install the root requirements if needed.
2. Follow workspace-specific setup inside `lm-evaluation-harness/`, `training/`, or `compression/`.
3. Use the shell runners in each workspace as the canonical entrypoints for paper experiments.

## Related documentation

- `PAPER_CODE_MAP.md`: maps paper claims to code locations
- `PROJECT_STRUCTURE.md`: repository structure policy
- `lm-evaluation-harness/scripts/README.md`: evaluation launcher guide
- `training/README.md`: training workspace notes
- `compression/README.md`: compression workspace notes
- `analysis/README.md`: standalone analysis script guide

## Compatibility note

`representation-analysis/` is retained only as a compatibility layer for older commands. New work should use the root-level paths directly.
