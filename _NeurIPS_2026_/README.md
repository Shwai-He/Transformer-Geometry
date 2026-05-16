# NeurIPS 2026 Paper Directory

This directory is the stable paper root for the transformer-geometry project. It contains the manuscript, paper-facing figures, paper-facing result CSVs, and lightweight utilities used to prepare the submission.

## What belongs here

```text
sections/     # paper text
figs/         # final figures used by LaTeX
results/      # paper-facing tables/CSVs
outputs/      # build outputs or derived paper artifacts
tools/        # lightweight paper utilities
```

## Relationship to the rest of the repository

This directory is only one part of the full codebase. The experiment and analysis workspaces live at the repository root and feed into the paper:

- `../lm-evaluation-harness/`: general-task and RULER evaluation.
- `../training/`: training-time experiments and optimization diagnostics.
- `../compression/`: pruning and quantization geometry analysis.
- `../drawing/`: editable figure-generation code.
- `../analysis/`: standalone intervention and visualization scripts.

`drawing/` and `representation-analysis/` inside this directory are compatibility symlinks into the repository root. Treat them as pointers to experiment workspaces, not as native paper-source folders.

## Practical workflow

- Run experiments and produce raw outputs in the root-level workspaces.
- Copy or collect paper-ready artifacts into `_NeurIPS_2026_/figs` and `_NeurIPS_2026_/results`.
- Keep LaTeX-facing filenames and table schemas stable here even if the underlying experiment code evolves.
