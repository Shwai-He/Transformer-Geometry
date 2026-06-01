# Compression Analysis Workspace

This workspace supports the compression-diagnostics part of the transformer-geometry paper. It studies how pruning or quantization changes parallel and perpendicular update geometry, and how those distortions relate to performance degradation.

## How this relates to the paper

Use this directory for:

- layerwise para/perp comparisons between dense and compressed models
- geometry-aware pruning scores that inject residual/value-space decomposition into WANDA-style pruning
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
- `scripts/run_geometry_aware_pruning.sh`
- `scripts/run_residual_prune_compare.sh`
- `code/layerwise_para_perp_compare.py`
- `code/geometry_aware_pruning.py`
- `code/residual_geometry_prune_compare.py`

## Geometry-aware pruning support

`code/geometry_aware_pruning.py` collects dense-model geometry scores and uses
them as optional multipliers for WANDA or magnitude pruning. The supported
decomposition signals are:

- attention value-space: `value_para`, `value_perp`, `value_perp_over_para`
- attention residual-space: `residual_para`, `residual_perp`, `residual_perp_over_para`
- MLP residual-space: `residual_para`, `residual_perp`, `residual_perp_over_para`

The first implementation applies geometry directly to the projection exits:

- `self_attn.o_proj`: head-level value/residual scores are repeated over each head dimension
- `mlp.down_proj`: neuron-level residual scores are applied over intermediate channels

Score-only smoke run, using an existing local cache:

```bash
bash compression/scripts/run_geometry_aware_pruning.sh
```

Save a geometry-aware WANDA checkpoint for evaluation:

```bash
MODEL_PATH=/path/to/dense_model \
SPARSITY_RATIO=0.5 \
GEOMETRY_STRATEGY=value_perp \
GEOMETRY_TARGETS=o_proj \
SAVE_MODEL=true \
DEVICE=cuda DTYPE=bfloat16 \
bash compression/scripts/run_geometry_aware_pruning.sh
```

Use residual-space geometry for MLP-side pruning:

```bash
MODEL_PATH=/path/to/dense_model \
SPARSITY_RATIO=0.5 \
GEOMETRY_STRATEGY=residual_perp \
GEOMETRY_TARGETS=down_proj \
SAVE_MODEL=true \
DEVICE=cuda DTYPE=bfloat16 \
bash compression/scripts/run_geometry_aware_pruning.sh
```

Compare residual-level error across pruning scores without saving checkpoints:

```bash
SPARSITY_RATIO=0.5 \
STRATEGIES=none,residual_perp,residual_para,residual_perp_over_para \
DEVICE=cuda DTYPE=bfloat16 \
bash compression/scripts/run_residual_prune_compare.sh
```

## Suggested workflow

1. Read `docs/README_layerwise_para_perp.md`.
2. Configure one launcher under `scripts/`.
3. Run and write new artifacts into `outputs/`.
4. Use the visualization helpers in `code/` to summarize trends.
5. Move paper-ready figures or tables into the private paper workspace when preparing a manuscript.
