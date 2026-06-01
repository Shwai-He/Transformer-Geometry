# Drawing Catalog

This note separates paper-facing plotting code from supporting utilities and legacy branches.

## Verified paper-facing outputs

These are the figure artifacts that `_NeurIPS_2026_` currently references.

| Figure family | Referenced output | Current status |
| --- | --- | --- |
| Method overview | `figs/overview/parallel_updates_main_figure_analysis_v11_midline_arrows.pdf` | present |
| Geometry depth plot | `figs/para_dist/paper_Qwen3-4B_hidden_both_block_para_perp_sampled_layers.pdf` | present |
| Geometry depth plot | `figs/para_dist/paper_Qwen3-30B-A3B_hidden_both_block_para_perp_sampled_layers.pdf` | present |
| Para ablation | `figs/para_ablation/para_scale_weighted_delta_ppl.pdf` | present |
| Para ablation | `figs/para_ablation/perp_scale_weighted_delta_ppl.pdf` | present |
| Attention-matrix edit view | `figs/attn_matrix/outputs/qwen3-4b/layer19/h6_three_map_compare.pdf` | present |
| Compression geometry | `figs/comp_analysis/local_flip_compare-attn_out-perp_over_base_update.pdf` | present |
| Pretraining loss | `figs/loss_curves/pretraining_loss_by_size.pdf` | present |
| Appendix loss ablation | `figs/loss_curves/loss_1p4_gate_vs_xsa_absolute.pdf` | present |

## Paper-facing plotting entry points

These are the scripts/notebooks we should treat as active figure sources.

| Path | Role | Status |
| --- | --- | --- |
| `drawing/plot_paper_para_perp_sampled_layers.py` | Main geometry figure generator | works after falling back to the populated generation trace |
| `drawing/embedded_data/*.py` | Self-contained geometry figure reproductions | active |
| `drawing/para_ablation/plot_para_ppl_summary.py` | Para/perp ablation paper figure | works |
| `drawing/loss_curves/plot_loss_from_cleaned_csv.py` | Cleaned loss-curve figure | works |
| `drawing/loss_curves/plot_loss_1p4_gate_vs_xsa.py` | 1.4B gate vs XSA loss figure | works |
| `drawing/loss_curves/plot_loss_curves_296m.py` | 296M loss plot family | notebook/script pair present; data-dependent |
| `drawing/loss_curves/plot_loss_csv_sizes_overview.py` | Cross-size overview plot family | notebook/script pair present; data-dependent |
| `drawing/comp_analysis/plot_local_flip_compare_v2.py` | Compression / local-flip paper figure | works |
| `drawing/attn_matrix/replot_from_saved_data.py` | Rebuilds saved attention/XSA panels | runnable with `--input`; no bundled sample input is currently checked in |
| `drawing/visualize_xsa_ablation_sweeps.py` | Sweep summary figure generator | active |
| `drawing/probe_steps_viz/*.py` | Appendix/debug-depth figures reused in paper flow | active, but not all are main-text figures |

## Supporting utilities

These are useful, but they are not the primary paper-figure sources.

| Path | Why it exists |
| --- | --- |
| `drawing/common.py` | Shared result loading and reduction helpers |
| `drawing/overview/vectorize_pngs_to_svg.py` | Converts raster artwork to SVG for publication assets |
| `drawing/attn_matrix/server/*` | Long-form runner and server-side generation helpers |
| `drawing/attn_matrix/data/README.md` | Data notes for attention-matrix inputs |
| `drawing/para_ablation/old/*` | Legacy predecessor for the current para-ablation pipeline |
| `drawing/para_dist/v2/*` | Alternate geometry draft branch; keep for reference only |

## Non-drawing experiment code

These directories are still useful, but they should be read as analysis or training code rather than figure sources.

| Path | Role |
| --- | --- |
| `analysis/*` | Evaluation, diagnostics, and sweep helpers |
| `scripts/*` | Repo-level launchers and data-prep utilities |
| `training/*` | Pretraining / optimization workspaces |
| `compression/*` | Compression-analysis workspace |

## Notes from local verification

- `drawing/para_ablation/plot_para_ppl_summary.py` runs successfully against the local summary tables.
- `drawing/loss_curves/plot_loss_from_cleaned_csv.py` and `drawing/loss_curves/plot_loss_1p4_gate_vs_xsa.py` both run successfully.
- `drawing/comp_analysis/plot_local_flip_compare_v2.py` runs successfully and writes the expected PDF/PNG/SVG outputs.
- `drawing/overview/vectorize_pngs_to_svg.py` runs successfully.
- `drawing/plot_paper_para_perp_sampled_layers.py` now works locally after falling back to the populated traces in
  `results/generation_probe.json` and `results/generation_probe/generation_probe.json`.
- `drawing/attn_matrix/replot_from_saved_data.py` is fine as a helper, but it needs `--input` on the command line.
