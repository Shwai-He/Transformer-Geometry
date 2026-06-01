# Layer-wise Parallel/Perpendicular Decomposition

Script: `representation-analysis/layerwise_para_perp_compare.py`

`Delta` definition in this script:
- `global`: `Delta_l = h_l^comp - h_l^dense`
- `local`: only replace one layer (`focus_layer`) with compressed layer, then `Delta_l = h_l^cf - h_l^dense`

## Example: dropped (layer drop)

```bash
cd third_party/Pruning-on-Representations/representation-analysis
python layerwise_para_perp_compare.py \
  --analysis_mode dropped \
  --effect_scope global \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --model_tag Qwen__Qwen2.5-7B-Instruct \
  --method_name attn_drop_8 \
  --dropped_root_path /path/to/dropped_results \
  --target_layer attn \
  --drop_n 8 \
  --prompts_file /path/to/prompts.txt
```

## Example: pruned (Wanda/SparseGPT)

```bash
cd third_party/Pruning-on-Representations/representation-analysis
python layerwise_para_perp_compare.py \
  --analysis_mode pruned \
  --compression_type prune \
  --effect_scope global \
  --model_name /path/to/dense_model \
  --pruned_model_name /path/to/pruned_model \
  --method_name wanda_50 \
  --prompts_file /path/to/prompts.txt
```

## Example: quantized (4-bit/8-bit)

```bash
cd third_party/Pruning-on-Representations/representation-analysis
python layerwise_para_perp_compare.py \
  --analysis_mode pruned \
  --compression_type quant \
  --effect_scope global \
  --model_name /path/to/dense_model \
  --pruned_model_name /path/to/quantized_or_base_model \
  --method_name bnb4_nf4 \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype float16 \
  --prompts_file /path/to/prompts.txt
```

## Example: local single-layer effect

```bash
cd third_party/Pruning-on-Representations/representation-analysis
python layerwise_para_perp_compare.py \
  --analysis_mode pruned \
  --compression_type prune \
  --effect_scope local \
  --focus_layer 12 \
  --model_name /path/to/dense_model \
  --pruned_model_name /path/to/pruned_model \
  --method_name wanda_l12_local \
  --prompts_file /path/to/prompts.txt
```

## Outputs

Saved under:
`representation-analysis/outputs/layerwise_para_perp/<model_tag>__<method_name>/`

- `layerwise_metrics.csv`: columns `method, layer, alpha, para_abs, perp_abs, perp_ratio`
- `layerwise_metrics.csv`: columns `method, component, layer, alpha, para_abs, perp_abs, perp_ratio`
  - `component in {block_out, attn_out, mlp_out}`
- `summary.json`: includes per-component `mean_perp_ratio`, `std_perp_ratio`
- `fig_a_perp_ratio.png`
- `fig_b_alpha.png`
- `fig_c_heatmap.png`

## Launcher Script

Use:
`scripts/compression_analysis/run_layerwise_para_perp_compare.sh`

Edit variables at the top of the script, then run:

```bash
cd /Users/bytedance/Documents/GitHub/MMEBarch/representation-analysis/demystifying-transformers
bash scripts/compression_analysis/run_layerwise_para_perp_compare.sh
```

The script prints:
- `Logs:` absolute path
- `PID file:` absolute path

Follow progress with:

```bash
tail -f <printed-log-path>
```

## Split Launchers

Inter-layer (drop):
- `scripts/compression_analysis/run_inter_layer_drop_para_perp.sh`

Intra-layer (prune):
- `scripts/compression_analysis/run_intra_layer_prune_para_perp.sh`

Intra-layer (quant):
- `scripts/compression_analysis/run_intra_layer_quant_para_perp.sh`

Each launcher prints absolute `Logs:` and `PID file:` paths.
