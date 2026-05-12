# Compression Analysis Workspace

This workspace supports the compression-diagnostics part of the transformer-geometry paper. It studies how pruning or quantization changes parallel and perpendicular update geometry, and how those distortions relate to performance degradation.

## How this relates to the paper

Use this directory for:

- layerwise para/perp comparisons between dense and compressed models
- quantization-versus-pruning comparisons
- local flip or sweep analyses of compression-induced error geometry
- compression-focused plots and summaries that later feed the paper figures

## Structure

- `docs/`: method notes and usage docs
- `scripts/`: runnable shell launchers for experiments
- `code/`: core Python implementation
- `outputs/`: recommended place for focused outputs
- `notes/`: ad-hoc notes and TODOs

## Linked key files

- `docs/README_layerwise_para_perp.md`
- `scripts/run_layerwise_para_perp_compare.sh`
- `scripts/run_inter_layer_drop_para_perp.sh`
- `scripts/run_intra_layer_prune_para_perp.sh`
- `scripts/run_intra_layer_quant_para_perp.sh`
- `code/layerwise_para_perp_compare.py`

## Suggested workflow

1. Read `docs/README_layerwise_para_perp.md`.
2. Configure one launcher under `scripts/`.
3. Run and write new artifacts into `outputs/`.
4. Use the visualization helpers in `code/` to summarize trends.
5. Move paper-ready figures or tables into the private paper workspace when preparing a manuscript.
