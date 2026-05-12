# Transformer Geometry

This repository is organized around the NeurIPS 2026 paper in `_NeurIPS_2026_/`. The paper studies transformer updates through a geometric decomposition into parallel and perpendicular components, compares residual-space and value-space views, and uses that geometry for three connected goals: inference-time editing, compression diagnostics, and training-time intervention.

## Paper-first entrypoint

If you want the paper itself, start here:

- `_NeurIPS_2026_/`: LaTeX source, paper-facing figures, and paper-facing result tables.

The rest of the repository is arranged to support that paper with experiment code, reusable analysis, and benchmark infrastructure.

## How the codebase maps to the paper

```text
_NeurIPS_2026_/            paper source, final figures, final CSV tables
lm-evaluation-harness/     benchmark evaluation for general-task and RULER experiments
training/                  training and optimization runs related to parallel suppression
compression/               compression-analysis workspace for pruning and quantization studies
drawing/                   editable figure-generation code and plotting assets
analysis/                  standalone intervention, visualization, and diagnostic scripts
src/repgeo/                reusable geometry and intervention utilities
```

A good mental model is:

- `_NeurIPS_2026_/` contains what is cited by the paper.
- `lm-evaluation-harness/`, `training/`, and `compression/` contain the main experimental backends.
- `drawing/` contains editable figure-source code.
- `analysis/` contains smaller research scripts that do not belong in a benchmark harness.
- `src/repgeo/` is where reusable logic should accumulate over time.

## Main research themes represented in the code

- Parallel vs perpendicular decomposition of transformer updates.
- Residual-space vs value-space measurement and intervention.
- Attention-side component scaling and diagonal editing.
- Compression-induced geometry distortion under pruning and quantization.
- Training-time suppression of self-value-parallel updates.

## What to use for new work

- Put paper text, final figures, and final tables in `_NeurIPS_2026_/`.
- Put benchmark runners and result collection under `lm-evaluation-harness/`.
- Put pretraining or optimization experiments under `training/`.
- Put compression-specific studies under `compression/`.
- Put editable figure-source code under `drawing/`.
- Put standalone research scripts under `analysis/`.
- Put reusable Python logic under `src/repgeo/`.

## Compatibility paths

`representation-analysis/` is now a compatibility directory. Older commands still resolve through it, but new work should use the root-level paths directly.

Examples:

- `representation-analysis/lm-evaluation-harness/...` -> `lm-evaluation-harness/...`
- `representation-analysis/drawing/...` -> `drawing/...`
- `representation-analysis/training/...` -> `training/...`
- `representation-analysis/compression/...` -> `compression/...`

## Related navigation docs

- `PROJECT_STRUCTURE.md`: higher-level structure policy.
- `_NeurIPS_2026_/README.md`: paper-directory notes and links back to the experiment workspaces.
- `lm-evaluation-harness/scripts/README.md`: benchmark launcher guide.
- `analysis/README.md`: grouped guide to standalone analysis scripts.
