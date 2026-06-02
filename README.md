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
  Private research codebase for studying transformer computation through parallel and perpendicular update geometry.
</p>

This repository supports experiments for a ***geometric view of transformer computation***. The central decomposition separates each module update into a ***parallel*** component, which mostly rescales the current representation, and a ***perpendicular*** component, which changes direction. We compare ***residual-space*** and ***value-space*** versions of this decomposition and use them to study editing, compression, and optimization behavior.

This private research workspace focuses on ***runnable code and experiment scripts***. Large raw outputs, plotting workspaces, figure assets, drafting workspaces, local model paths, and Overleaf-specific files are intentionally not included in git.

## What You Can Run Here

- **Geometry probing**: measure parallel and perpendicular update structure across layers, branches, prompts, and generation steps.
- **Inference-time component editing**: run residual-space and value-space interventions, including attention-parallel removal and diagonal edits.
- **Benchmark evaluation**: evaluate edited models on general tasks and long-context RULER through the included lm-evaluation-harness fork.
- **Training-time intervention**: inspect scratch pretraining runs that suppress or rescale parallel updates.
- **Compression diagnostics**: decompose pruning and quantization error into parallel and perpendicular components.

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

For model-scale benchmark runs, install the backend packages required by your local setup, such as `transformers`, `accelerate`, `datasets`, and CUDA-compatible PyTorch builds. Most launch scripts are ***file-first***: edit model paths, output roots, and GPU settings near the top of the script before running.

## Selected experiment entrypoints

The sections below list the main runnable entrypoints without including private plotting workspaces or paper figure assets.

### Component structure across depth

These profiles measure how much transformer updates ***preserve the current direction versus change it*** across depth. The probe runners regenerate the underlying activations and summaries.

> **Key result:** transformer updates contain both rescaling-aligned and direction-changing structure across depth, and the profile remains visible across model scales.

```text
# Regenerate geometry probes
scripts/run_probe.py
scripts/run_batch_probe.py
```

### Manual component scaling at inference time

These ablations manually scale ***parallel or perpendicular components*** and measure the resulting perplexity change. They are the lightweight diagnostic counterpart to the benchmark evaluations in `lm-evaluation-harness/`.

> **Key result:** value-parallel scaling is comparatively robust, while changing the perpendicular component more directly disrupts model behavior.

```text
# Run intervention evaluations
lm-evaluation-harness/scripts/run_lm_eval_xsa_setting.sh
lm-evaluation-harness/scripts/run_lm_eval_attn_removal_batch.sh
```

### Attention diagonal editing

This view compares attention maps after ***value-space and residual-space diagonal edits***. The hook code implements the edit used to produce these diagnostics.

> **Key result:** edits that look similar as scalar diagonal controls can behave differently depending on whether the constraint is solved in value space or residual space.

```text
# Implement diagonal edits
lm-evaluation-harness/lm_eval/models/attn_diag_hooks.py
lm-evaluation-harness/scripts/run_lm_eval_attn_diag_setting.sh

```

### Compression error geometry

The compression experiments decompose ***pruning and quantization error*** into parallel and perpendicular parts. The paired attention-side views show that the direction-changing component separates pruning severity more clearly, while the parallel component gives the complementary rescaling view.

> **Key result:** stronger pruning is most visible in the direction-changing error, whereas quantization stays closer to the dense update geometry in both components.

```text
# Run compression geometry analysis
compression/code/layerwise_para_perp_compare.py
```

### Training-time parallel removal

The training experiments test whether ***suppressing parallel updates changes optimization***. The plot summarizes scratch pretraining runs across model sizes, with downstream evaluation handled through the same lm-eval workspace.

> **Key result:** parallel-update control affects scratch-training loss curves, so the geometry is relevant to optimization as well as inference-time editing.

```text
# Training workspace
training/

# Post-training evaluation
lm-evaluation-harness/scripts/run_lm_eval_nanogpt_setting.sh
lm-evaluation-harness/scripts/collect_nanogpt_lm_eval_results.py
```

## Repository layout

```text
lm-evaluation-harness/     benchmark evaluation and intervention runners
training/                  training-side experiments and optimization studies
compression/               pruning and quantization geometry analysis
analysis/                  standalone diagnostic and visualization scripts
scripts/                   small repo-level runners and probes
src/repgeo/                reusable geometry and intervention utilities
```

Full benchmark logs, plotting workspaces, model checkpoints, raw activations, and private paper exports are excluded from git.

## Quick Usage

Start from the ***figure or experiment family*** you care about, then use the nearby scripts listed above. The most common entrypoints are shown below; edit paths and resource settings inside each script before launching long jobs.

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
```

For workspace-specific details, use the README files under `lm-evaluation-harness/scripts/`, `training/`, `compression/`, and `analysis/`.

## Technical reproduction extras

The repository also keeps a small set of technical-reproduction utilities that are separate from the paper mainline:

- `scripts/reproduce_technical.py`
- `scripts/reproduce_technical_batch.py`
- `src/repgeo/technical_reproduction.py`

Generated sample prompts and outputs are treated as local artifacts and are not tracked.

## Related documentation

- `lm-evaluation-harness/scripts/README.md`: evaluation launcher guide
- `training/README.md`: training workspace notes
- `compression/README.md`: compression workspace notes
- `analysis/README.md`: standalone analysis script guide

## Citation

Citation information will be added after the paper metadata is public.

## Compatibility note

Older local experiments may have used a `representation-analysis/` prefix. Public entrypoints in this repository use the root-level workspaces shown above.
