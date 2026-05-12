# Training and Optimization Experiments

Use this directory for training-side experiments, including nanoGPT-style pretraining runs, optimization diagnostics, and training-time parallel/perpendicular geometry measurements.

Suggested substructure:

```text
scripts/      # launch and collection scripts
configs/      # model/training configs
outputs/      # logs and generated metrics, gitignored when large
notebooks/    # exploratory analysis notebooks
```

The directory name is intentionally `training` rather than `nanogpt` so that implementation details do not define the conceptual structure.
