# Camera-ready figure reproduction audit

Audit baseline: `main@3626beabfe60407b9fd493d84edbdb6399180b84` on 2026-08-24.
The plotting environment used Python 3.10 with the exact packages in
`requirements-visualization.txt`. Generated files were written below
`results/figures/`; no paper-facing PDF was overwritten.

## Status

| Paper asset | Reproduction entry point | Status | Evidence |
| --- | --- | --- | --- |
| Qwen3-4B and Qwen3-30B depth profiles | `analysis/visualization/para_dist/*.py` | Pass | The dedicated paper scripts select indices 1/3/5 from their six-layer embedded bundles, yielding the displayed three layers. Both outputs have identical 150-DPI rasters to the Overleaf PDFs (RMSE 0). The similarly named `embedded_data/*.py` files are six-layer source bundles, not the paper entry points. |
| Matched component scaling | `analysis/visualization/plot_arr_matched_component_scaling.py --source-manifest ...` | Pass | The lightweight Overleaf source manifest contains every plotted value, retained scale, no-op delta, and original result path. Replot-only mode validates all expected series/scales and no-op points; its 150-DPI raster is identical to the paper PDF (RMSE 0). The script retains its stricter raw-JSON mode for source-level experiment validation when the large result trees are available. |
| Attention diagonal | `analysis/visualization/recompute_arr_attention_diag_figure.py --replot-only` | Pass | Replot from the checked-in effective-diagonal bundle has identical dimensions; 150-DPI raster RMSE is 0.000261 (rendering-level only). |
| Compression panels | `analysis/visualization/comp_analysis/plot_local_flip_compare_v2.py` | Pass | All six paper-facing parallel/perpendicular panels and the baseline-update panel reproduce pixel-for-pixel at 150 DPI (RMSE 0). |
| 1.4B training loss | `analysis/visualization/loss_curves/plot_loss_1p4_gate_vs_xsa.py` | Pass | Paper-facing absolute-loss panel reproduces pixel-for-pixel at 150 DPI (RMSE 0). |
| Pretraining trajectories | `training/scripts/plot_arr_figure6_retained_curves.py` | Pass | Paper-facing restored curve panel reproduces pixel-for-pixel at 150 DPI (RMSE 0). |
| Per-head causal sensitivity | `analysis/forward_geometry/analyze_per_head_causal_sensitivity.py` | Gap | Script executes, but produces early/middle/late head-index panels, while the paper uses four all-layer model panels with a single-head min--max envelope. The committed CSVs contain only three layers per model, so they cannot reconstruct the all-layer paper asset. Their manifests point to `/beacon-projects/traumallm`; read-only recovery attempts from ECEC to both `ihccs050v.ihc.umd.edu` and `10.7.103.35` timed out on port 22 on 2026-08-24. Do not infer missing values from the rendered figure. |
| Method overview | composed asset under `analysis/visualization/overview/` | Gap | The checked-in arrow SVG/PNG assets are only component inputs. PDF metadata identifies the paper-facing figure as created by Microsoft PowerPoint 2021 (author metadata `Lenovo`, creation time `2026-05-25T14:33:54-07:00`), but no `.pptx`, `.ppt`, `.key`, `.ai`, `.drawio`, or `.fig` source exists in the repository. Recover the original PowerPoint deck/source slide; do not approximate the final composition from the rendered PDF. |

## Commands exercised

```bash
python analysis/visualization/para_dist/paper_Qwen3-4B-Instruct-2507_hidden_both_block_para_perp_sampled_layers.py
python analysis/visualization/para_dist/paper_Qwen3-30B-A3B_hidden_both_block_para_perp_sampled_layers.py
python analysis/visualization/plot_arr_matched_component_scaling.py \
  --source-manifest _overleaf_/ARR_May_Revision_Overleaf/figs/para_ablation_corrected/ppl_component_scaling_corrected_sources.csv \
  --output results/figures/camera_ready_audit/component_scaling/ppl_component_scaling_corrected.pdf
python analysis/visualization/comp_analysis/plot_local_flip_compare_v2.py
python analysis/visualization/loss_curves/plot_loss_1p4_gate_vs_xsa.py
python training/scripts/plot_arr_figure6_retained_curves.py --output-dir results/figures/camera_ready_audit/pretraining
python analysis/forward_geometry/analyze_per_head_causal_sensitivity.py \
  --bootstrap 5000 --sign-flips 10000 --seed 2026 \
  --figure-dir results/figures/camera_ready_audit/per_head
python analysis/visualization/recompute_arr_attention_diag_figure.py \
  --replot-only \
  --result-dir analysis/visualization/attn_matrix/source/effective_diagonal/qwen3-4b/layer19/h6 \
  --figure-output results/figures/camera_ready_audit/attention/h6_diag_delta_corrected_titles.pdf
```

## Verified output hashes

- Attention replot PDF: `aa8d8992743dbd989a8ba151147fd62a1dac78011ead88e92cf34c9eeeb379c3`
- Qwen3-4B depth-profile PDF: `331a0cdfb52187b18d4754c11a3587b02cb4fa0f0a811d7a7938935e36c2e382` (150-DPI raster RMSE 0 against the paper asset)
- Qwen3-30B-A3B depth-profile PDF: `86fe36369e4eaedba9e9a2559f2ce1fec533675a4ca9710c627d924f50549afe` (150-DPI raster RMSE 0 against the paper asset)
- Matched component-scaling PDF: `36d71551419474b583d342a5b905770106e073b7bb0a3643db93e9cf4492bbd1` (150-DPI raster RMSE 0 against the paper asset)
- Matched component-scaling replot manifest: `70a5c295cfb276d51ff7e7c5183035e2229a76b399e27c0cd1cc5acb3cd50a23`
- Compression baseline-update PDF: `78cae45cf48d1666e74d60bb4ee0f615200d9a1422a62d69a15bb4313f007835`
- 1.4B absolute-loss PDF: `301507ac511cae382770f1eef29cdf16f27ce83e29065c47e276699a7e98d833`
- Pretraining trajectories PDF: `05beaadf57cb6be69cddc32739c66d939cb354b81c4d75c3b75e69913cf61899`

## Camera-ready action items

1. Recover the exact four-model, all-layer head-localization plotting entry point.
2. Recover the original PowerPoint deck/source slide for the method overview, then check it in or export a deterministic composition recipe. The two committed arrow assets alone are insufficient to reproduce the final layout.
3. Keep current Overleaf assets unchanged until replacements pass a raster/content comparison and caption audit.
