# Project Structure

This repository separates reusable analysis code, experiment runners, plotting sources, and generated artifacts. The goal is to make the public codebase easy to navigate while keeping private paper drafts and raw outputs outside version control.

## Top-level layout

```text
lm-evaluation-harness/        # forked lm-eval for general/RULER/XSA evaluations
training/                     # training and optimization experiments
compression/                  # canonical compression-analysis workspace
drawing/                      # editable figure-generation sources
analysis/                     # standalone research scripts and one-off analyses
src/repgeo/                   # reusable geometry/intervention Python code
scripts/                      # small repo-level utilities
notebooks/                    # exploratory notebooks
docs/assets/                  # lightweight public SVG assets for README/docs
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

## Artifact policy

- Raw experiment outputs, caches, private paper drafts, and logs stay under ignored local paths.
- Public figure assets used by documentation go under `docs/assets/`.
- One-off repair scripts should be clearly marked and not used as general-purpose launchers.
- Imported snapshots and retired assets should remain outside the public repository unless they are required for reproduction.
