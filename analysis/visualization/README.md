# Visualization Workspace

This directory is the single owner for editable plotting code and plot-specific
notebooks used by the paper and supporting analyses.

Each active plotting directory keeps its small local inputs in a `data/` subdirectory, so the code and the table/CSV it reads are visible together. Large server-side JSON traces are not duplicated here; the plotted geometry arrays are preserved in `embedded_data/*.py`.

## Geometry Figure

- `plot_paper_para_perp_sampled_layers.py`
  - Reads hidden-space generation geometry JSONs from the configured server `ROOT`.
  - Writes one PNG and one embedded-data Python script per model to
    `results/figures/para_dist/`.
  - Uses six uniformly sampled layers in a single row.

- `embedded_data/*.py`
  - Self-contained reproductions of already generated geometry figures.
  - These scripts do not require the original large JSON files.

- `plot_prompt_*.py`
  - Full-grid and probe-step diagnostics consolidated from the former
    `scripts/probe_steps_viz/` workspace.
  - Use these for appendix/debug plots; the cleaner paper entry point is
    `plot_paper_para_perp_sampled_layers.py`.

## Ablation Sweeps

- `visualize_xsa_ablation_sweeps.py`
  - Set `SWEEP_NAME = "para_ablation"` or `"perp_ablation"`.
  - Reads local summary tables from `para_ablation/data/`.
  - Writes figures and aggregate files to
    `results/figures/{SWEEP_NAME}`.

## Output ownership

`analysis/visualization/` owns editable plotting sources and their small local inputs.
Generated figures are durable evidence and therefore belong under
`results/figures/`; plotting code must not recreate a top-level `figs/`
directory. The older numbered copy is preserved only under
`archive/figures/figs_2_legacy/`.

## Current Overleaf figure sources

The tracked Overleaf package is `_overleaf_/ARR_May_Revision_Overleaf`.
Paper-facing plotting entry points and their lightweight inputs are:

- geometry depth panels: `embedded_data/*.py`;
- component-scaling panels: `para_ablation/`;
- attention-diagonal panel: `recompute_arr_attention_diag_figure.py
  --replot-only` with `attn_matrix/source/effective_diagonal/`;
- compression panels: `comp_analysis/plot_local_flip_compare_v2.py`;
- 1.4B loss panel: `loss_curves/plot_loss_1p4_gate_vs_xsa.py`;
- pretraining trajectories: `train/scripts/plot_arr_figure6_retained_curves.py`;
- per-head causal sensitivity:
  `analysis/forward_geometry/analyze_per_head_causal_sensitivity.py`.

The method overview is a composed publication asset. Its source raster/vector
assets and conversion helper are under `overview/`, but there is no single
deterministic composition script for the final PDF.

## Archived legacy branches

Older draft branches that are not part of the active figure pipeline now live under `archive/drawing_legacy/`:

- `archive/drawing_legacy/para_ablation_old/`
- `archive/drawing_legacy/para_dist_v2/`
- `archive/drawing_legacy/attn_matrix_server/`
- `archive/drawing_legacy/probe_steps_viz_copies/`

Focused-compression and exploratory notebook plots are intentionally not copied here unless they become paper figures.
