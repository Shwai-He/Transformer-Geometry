# Focused Compression Analysis Workspace

This workspace is a focused entrypoint for compression-related para/perp analysis.
It uses symlinks to original files, so existing paths and scripts stay unchanged.

## Structure

- `docs/`: method notes and usage docs
- `scripts/`: runnable shell launchers for experiments
- `code/`: core Python implementation
- `paper/`: paper sections discussing compression geometry
- `outputs/`: recommended place for focused outputs
- `notes/`: ad-hoc notes and TODOs

## Linked Key Files

- `docs/README_layerwise_para_perp.md`
- `scripts/run_layerwise_para_perp_compare.sh`
- `scripts/run_inter_layer_drop_para_perp.sh`
- `scripts/run_intra_layer_prune_para_perp.sh`
- `scripts/run_intra_layer_quant_para_perp.sh`
- `code/layerwise_para_perp_compare.py`
- `paper/experiments.tex`
- `paper/analysis.tex`

## Suggested Workflow

1. Read `docs/README_layerwise_para_perp.md`.
2. Configure one launcher under `scripts/`.
3. Run and write new artifacts into `outputs/`.
4. Track conclusions under `notes/`.

## Visualization Helpers

- `code/visualize_layerwise_runs.py`
  - Single run:
    - `python3 code/visualize_layerwise_runs.py --mode single --csv <run_dir>/layerwise_metrics.csv --out_dir outputs/viz_single`
  - Compare runs (overlay by method):
    - `python3 code/visualize_layerwise_runs.py --mode compare --csvs <run1.csv>,<run2.csv> --component block_out --out_dir outputs/viz_compare`

- `code/visualize_local_sweep_summary.py`
  - Summarize local sweep folders into focus-layer curves:
    - `python3 code/visualize_local_sweep_summary.py --runs_root outputs/layerwise_para_perp --component block_out --out_dir outputs/viz_local_summary`
