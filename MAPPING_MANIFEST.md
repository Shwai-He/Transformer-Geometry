# Mapping Manifest

Source tree:
- Current working tree: `drawing/`, `analysis/`, `compression/`, `training/`, `lm-evaluation-harness/`, `src/`, plus the symlink wrapper `representation-analysis/`
- Imported mirror tree: `_imports/from_MMEBarch_research_snapshot_2026-05-09/`

This file records the directory-level correspondence between the active tree and the imported mirror, plus the current migration state.

For the file-level manifest, see [MAPPING_MANIFEST.tsv](./MAPPING_MANIFEST.tsv).

## Status Legend

- `active`: this is the tree we are treating as the working copy.
- `mirror`: imported historical copy kept for reference.
- `verified`: we confirmed both sides exist and correspond.
- `partially migrated`: active tree has been cleaned or patched, but the mirror is still preserved untouched.

## Directory Correspondence

| Current path | Imported path | Status | Migration state | Notes |
| --- | --- | --- | --- | --- |
| `representation-analysis` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis` | verified | wrapper | Current `representation-analysis/` is a symlink bundle pointing back to active top-level directories; imported tree is the archived snapshot. |
| `representation-analysis/drawing` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing` | verified | active + mirror | Current side is a wrapper symlink to `drawing/`; imported side is the historical drawing snapshot. |
| `drawing/attn_matrix` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/attn_matrix` | verified | partially migrated | Active notebook/scripts were updated for local paths; mirror kept as archive. |
| `drawing/comp_analysis` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/comp_analysis` | verified | partially migrated | Current plotting notebooks were repaired and executed successfully. |
| `drawing/loss_curves` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/loss_curves` | verified | partially migrated | Active notebooks need data filtering fixes; mirror remains unchanged. |
| `drawing/para_ablation` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/para_ablation` | verified | partially migrated | Current notebooks were adjusted and run successfully. |
| `drawing/para_dist` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/para_dist` | verified | mirror | Current tree exists and is the working copy; imported tree is the historical source snapshot. |
| `drawing/embedded_data` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/embedded_data` | verified | mirror | Both trees exist; treat current tree as the active copy. |
| `drawing/overview` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/overview` | verified | mirror | Current tree is the active copy for overview assets. |
| `drawing/probe_steps_viz` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing/probe_steps_viz` | verified | partial | Current tree exists; imported tree is the historical counterpart. |
| `analysis` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/analysis_core` | verified | active + mirror | Current `analysis/` is the working tree; imported `analysis_core/` is the archived snapshot. |
| `compression` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/compression` | verified | active + mirror | Current `compression/` is the working tree; imported `compression/` is the archived snapshot. |
| `training` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/pretraining_nanogpt` | verified | active + mirror | Current `training/` is the working tree; imported `pretraining_nanogpt/` is the archived snapshot. |
| `lm-evaluation-harness` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/evaluation_scripts` | verified | active + mirror | Current `lm-evaluation-harness/` is the working tree; imported `evaluation_scripts/` is the archived snapshot. |
| `src` | `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/value_based_editing` | verified | active + mirror | Current `src/` is the working tree; imported `value_based_editing/` is the archived snapshot. |

## Full Imported Tree Coverage

### `representation-analysis`

- `representation-analysis` → current symlink wrapper
- `representation-analysis/drawing` ↔ `_imports/from_MMEBarch_research_snapshot_2026-05-09/representation-analysis/drawing`
- `representation-analysis/analysis` → current `analysis/`
- `representation-analysis/compression` → current `compression/`
- `representation-analysis/training` → current `training/`
- `representation-analysis/lm-evaluation-harness` → current `lm-evaluation-harness/`

### `neurips_supplement_anonymous_code`

- `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/analysis_core` ↔ current `analysis/`
- `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/compression` ↔ current `compression/`
- `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/evaluation_scripts` ↔ current `lm-evaluation-harness/`
- `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/pretraining_nanogpt` ↔ current `training/`
- `_imports/from_MMEBarch_research_snapshot_2026-05-09/neurips_supplement_anonymous_code/value_based_editing` ↔ current `src/`

## Current Migration Notes

- The active `drawing/` tree has already been patched for local paths in several notebooks and scripts.
- The imported mirror under `_imports/` is intentionally left untouched for provenance.
- The file-level TSV is the canonical per-file mapping record; this markdown file keeps the higher-level structure readable.
- Notebook verification has already succeeded for:
  - `drawing/attn_matrix/compare_three_attention_maps.ipynb`
  - `drawing/comp_analysis/plot_local_flip_compare_v1.ipynb`
  - `drawing/comp_analysis/plot_local_flip_compare_v2.ipynb`
  - `drawing/loss_curves/plot_loss_1p4_gate_vs_xsa.ipynb`
  - `drawing/loss_curves/plot_loss_from_cleaned_csv.ipynb`
  - `drawing/para_ablation/plot_para_ppl_summary.ipynb`
  - `drawing/para_ablation/old/plot_xsa_ablation_summary.ipynb`
  - `_NeurIPS_2026_/results/notebooks/plot_diagonal_change_schematic.ipynb`
- Known unresolved notebook issue in the active tree:
  - `drawing/attn_matrix/replot_from_saved_data.ipynb` still expects a missing `attn/layer_19_plot_bundle.json`
  - several `drawing/loss_curves/*.ipynb` notebooks still need CSV filtering so they do not ingest `loss_csv_by_size_raw3_cleaned.csv`
- Current tree contains a thin `representation-analysis/` symlink wrapper that points at the live top-level directories. This wrapper is not an independent source of truth; it is a convenience entrypoint for the active tree.
- The active `analysis/` tree is a merge point for both `_imports/.../representation-analysis` root scripts and `_imports/.../neurips_supplement_anonymous_code/value_based_editing`.
- The active `scripts/` tree corresponds to `_imports/.../neurips_supplement_anonymous_code/analysis_core/scripts`.
- The active `src/repgeo/` tree corresponds mostly to `_imports/.../neurips_supplement_anonymous_code/analysis_core/src/repgeo`, with `gated_attention.py` currently existing only on the active side.

## Update Rule

When we migrate or consolidate another pair, add a row here with:
- the canonical working location,
- the archival location,
- the migration state,
- the latest verification note.
