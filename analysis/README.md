# Analysis Workspace

This directory collects standalone research scripts that do not naturally belong inside the lm-eval harness, the training workspace, or the paper directory. In the paper workflow, this is where one-off diagnostics, forward-ablation checks, and visualization helpers usually live before they are promoted into a more stable workspace.

For the current non-breaking script map, see `INDEX.md`.

Top-level category directories such as `gsm_math/`, `forward_geometry/`, and
`model_compare/` now own the scripts directly.  The root of `analysis/` is kept
for the index and package marker only.

## Layout

- `forward_geometry/`: residual/value-space interventions, XSA generation, and para/perp sweeps.
- `vlm_geometry/`: model-agnostic para/perp scaling hooks for VLM understanding and generation modules.
- `visualization/`: Qwen attention and layer/head visualization entrypoints.
- `gsm_math/`: GSM8K and targeted math sanity checks.
- `model_compare/`: dense, dropped, pruned, and masked-teacher comparisons.
- `nanogpt/`: nanoGPT checkpoint diagnostics and wrappers.
- `utils/`: shared helpers and data-prep utilities.

Prefer keeping reusable logic small here, or promoting it into `src/repgeo/`
when it becomes shared infrastructure.

## Boundaries with other directories

- If a script is a benchmark launcher, it should usually live under `lm-evaluation-harness/scripts/`.
- If a script is specifically for pretraining or optimization runs, it should usually live under `training/`.
- If a script is specifically for compression geometry, it should usually live under `compression/`.
- If a script directly generates a paper figure, consider placing the editable source under `drawing/` and only keeping lightweight analysis helpers here.
- If logic becomes reusable across multiple scripts, move it into `src/repgeo/`
  or `analysis/utils/` depending on whether it is project infrastructure or a
  local analysis helper.

## Compatibility note

Older private commands may still reference a historical `representation-analysis/...` prefix. Public scripts should use `analysis/<category>/...` paths in this repository.
