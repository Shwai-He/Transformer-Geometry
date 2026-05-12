# Project Structure

This repository separates the paper, reusable analysis code, experiment runners, and generated artifacts. The goal is to keep the NeurIPS paper directory stable while making it clear which paths are active experiment code, which paths are compatibility aliases, and which paths are archival snapshots.

## Top-level layout

```text
_NeurIPS_2026_/               # Paper source and paper-facing artifacts
lm-evaluation-harness/        # forked lm-eval for general/RULER/XSA evaluations
training/                     # training and optimization experiments
compression/                  # canonical compression-analysis workspace
drawing/                      # editable figure-generation sources
analysis/                     # standalone research scripts and one-off analyses
src/repgeo/                   # reusable geometry/intervention Python code
scripts/                      # small repo-level utilities
notebooks/                    # exploratory notebooks
results/                      # non-paper raw or intermediate results
representation-analysis/      # compatibility directory with symlinks/aliases only
focused-compression-analysis/ # legacy alias kept for compatibility
archive/                      # retired assets once no longer active
_imports/                     # imported snapshots kept for provenance
```

## Paper directory

`_NeurIPS_2026_` stays as the paper root. It should contain only paper source, paper figures, paper-facing CSVs, and lightweight tools.

```text
_NeurIPS_2026_/
  neurips_2026.tex
  sections/
  figs/          # final figures referenced by LaTeX
  results/       # paper-facing CSVs/tables
  outputs/       # paper build outputs or derived text outputs
  tools/         # small paper utilities
  drawing/       # compatibility link to editable figure sources
```

## Active experiment workspaces

```text
lm-evaluation-harness/  # forked lm-eval used for XSA, general-task, and RULER evaluation
training/               # pretraining and optimization experiments
compression/            # canonical compression-analysis workspace
drawing/                # editable plotting and figure-source scripts
analysis/               # standalone scripts that do not belong in a benchmark harness
```

RULER belongs under `lm-evaluation-harness` because it is a benchmark/task family inside the evaluation workflow rather than a separate top-level project.

## Compatibility policy

`representation-analysis/` is transitional. Keep it available for old commands, but treat it as an alias layer rather than the main home for new code. When adding or revising scripts, prefer the root-level workspaces above.

## Artifact policy

- Paper-ready artifacts go to `_NeurIPS_2026_/figs` and `_NeurIPS_2026_/results`.
- Raw experiment outputs, caches, and logs stay under the relevant experiment workspace and should be gitignored where possible.
- One-off repair scripts should be clearly marked and not used as general-purpose launchers.
- Imported snapshots and retired assets should move into `archive/` or remain in `_imports/` only when provenance matters.
