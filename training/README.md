# Training and Optimization Experiments

Use this directory for training-side experiments in the transformer-geometry paper, especially the question of whether suppressing self-value-parallel updates changes optimization and downstream behavior.

## How this relates to the paper

This workspace corresponds to the training-time intervention part of the paper:

- small-model pretraining sweeps
- retained larger-model runs
- optimization diagnostics connected to parallel suppression
- downstream evaluation of trained checkpoints

## Suggested substructure

```text
scripts/      # launch and collection scripts
configs/      # model/training configs
outputs/      # logs and generated metrics, gitignored when large
notebooks/    # exploratory analysis notebooks
```

The directory name is intentionally `training` rather than `nanogpt` so that implementation details do not define the conceptual structure.
