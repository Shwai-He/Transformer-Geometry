# Anonymous NeurIPS Supplementary Code

This directory contains the code used for the supplementary experiments in the submission. It is prepared for anonymous review and includes only code needed to reproduce the main experiment pipelines.

## Structure

- `analysis_core/`: residual-stream geometry probes and supporting utilities.
- `compression/`: compression and local counterfactual analysis scripts.
- `value_based_editing/`: forward-intervention and ablation scripts for pretrained models.
- `evaluation_scripts/`: evaluation wrappers and result collection helpers.
- `pretraining_nanogpt/`: nanoGPT-style training code and intervention launchers.

## What Is Omitted

This package intentionally excludes checkpoints, datasets, cached outputs, generated figures, notebooks, and local experiment results.

## Usage Notes

- Some shell launchers expect the user to set model paths, dataset locations, or output directories in the environment before running.
- Logging to Weights & Biases is disabled by default in this anonymous package. It can be re-enabled by setting `ENABLE_WANDB=true`.
- The pretraining launchers assume `ROOT_DIR` points to `pretraining_nanogpt/`; the bundled entry scripts set this automatically.

## Minimal Examples

```bash
cd analysis_core
PYTHONPATH=src python scripts/run_probe.py \
  --model_name_or_path gpt2 \
  --prompt "The model writes an update into the residual stream." \
  --output probe.json
```

```bash
cd value_based_editing
bash run_xsa_forward_ablation.sh
```

```bash
cd pretraining_nanogpt
bash scripts/run_0p7b_xsa.sh
```
