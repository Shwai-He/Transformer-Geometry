# Drawing Scripts

This directory collects plotting code used for the NeurIPS 2026 draft.

Each active plotting directory keeps its small local inputs in a `data/` subdirectory, so the code and the table/CSV it reads are visible together. Large server-side JSON traces are not duplicated here; the plotted geometry arrays are preserved in `embedded_data/*.py`.

## Geometry Figure

- `plot_paper_para_perp_sampled_layers.py`
  - Reads hidden-space generation geometry JSONs from the configured server `ROOT`.
  - Writes one PNG and one embedded-data Python script per model to `_NeurIPS_2026_/figs/para_dist/`.
  - Uses six uniformly sampled layers in a single row.

- `embedded_data/*.py`
  - Self-contained reproductions of already generated geometry figures.
  - These scripts do not require the original large JSON files.

- `probe_steps_viz/*.py`
  - Full-grid and diagnostic versions copied from `scripts/probe_steps_viz/`.
  - Use these for appendix/debug plots; the cleaner paper entry point is the top-level `plot_paper_para_perp_sampled_layers.py`.

## Ablation Sweeps

- `visualize_xsa_ablation_sweeps.py`
  - Set `SWEEP_NAME = "para_ablation"` or `"perp_ablation"`.
  - Reads local summary tables from `para_ablation/data/`.
  - Writes figures and aggregate files to `_NeurIPS_2026_/figs/{SWEEP_NAME}`.

Focused-compression and exploratory notebook plots are intentionally not copied here unless they become paper figures.
