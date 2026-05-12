# Transformer Geometry

This repository now separates the NeurIPS paper, active experiment workspaces, reusable code, and compatibility paths. The goal is to keep paper-facing assets stable while making it obvious where new code and new outputs should live.

## Primary workspaces

```text
_NeurIPS_2026_/            paper source, paper figures, paper-facing CSVs
lm-evaluation-harness/     forked lm-eval workspace for general and RULER evaluation
training/                  training and optimization experiments
compression/               canonical compression-analysis workspace
drawing/                   editable plotting and figure-generation sources
analysis/                  standalone research scripts and one-off analyses
src/repgeo/                reusable geometry and intervention code
```

## What to use for new work

- Put paper text, final figures, and final tables in `_NeurIPS_2026_/`.
- Put benchmark runners and result collection under `lm-evaluation-harness/`.
- Put pretraining or optimization experiments under `training/`.
- Put editable figure-source code under `drawing/`.
- Put standalone research scripts that do not belong to a specific benchmark harness under `analysis/`.
- Put reusable Python logic under `src/repgeo/`.

## Compatibility paths

`representation-analysis/` is now a compatibility directory. Older commands still resolve through it, but new work should use the root-level paths directly.

Examples:

- `representation-analysis/lm-evaluation-harness/...` -> `lm-evaluation-harness/...`
- `representation-analysis/drawing/...` -> `drawing/...`
- `representation-analysis/training/...` -> `training/...`

## Suggested mental model

```text
paper assets      -> _NeurIPS_2026_/
active experiments -> lm-evaluation-harness/ , training/ , compression/ , analysis/
editable figures   -> drawing/
reusable code      -> src/repgeo/
legacy aliases     -> representation-analysis/
archive/imports    -> archive/ , _imports/
```

## Files that look miscellaneous

A few paths are intentionally still preserved in place because they may still be referenced by older commands or remote jobs:

- `representation-analysis/`
- `_imports/`
- `focused-compression-analysis/` (legacy alias to `compression/`)
- one-off PDFs, zips, and local outputs in the repository root

These can be further cleaned once active jobs and scripts no longer depend on them.

## Related navigation docs

- `PROJECT_STRUCTURE.md`: higher-level structure policy.
- `_NeurIPS_2026_/README.md`: paper-specific notes.
- `lm-evaluation-harness/scripts/README.md`: benchmark launcher guide.
- `analysis/README.md`: grouped guide to standalone analysis scripts.
