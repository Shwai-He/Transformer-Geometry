# Demystifying Transformers through Representation Geometries (Starter)

This is an initial coding scaffold based on the paper notes.

## What is implemented

A minimal representation-geometry probe for causal LMs:
- `z_post_median`: median hidden-state norm across layers (last token).
- `z_post[i+1] / z_post[i]`: layer-wise scale gain.
- `dz_para_median`: median norm of update component parallel to input representation.
- `dz_perp_median`: median norm of update component orthogonal to input representation.
- `log(dz_para / dz_perp)`: dominance of parallel over orthogonal updates.
- top-k index overlap in logits space across adjacent layers.

## Structure

- `src/repgeo/analyzer.py`: core analysis logic.
- `scripts/run_probe.py`: CLI entry.
- `results/`: output JSON.

## Quick start

```bash
cd /Users/heshuai/Documents/Code/demystifying-transformers-representation-geometries
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 scripts/run_probe.py \
  --model_name_or_path gpt2 \
  --prompt "John has twice as many books as Mary. Together they have 18 books. How many books does John have?" \
  --top_k 5 \
  --output results/gpt2_probe.json
```

## Next steps

- Add generation-step analysis (not only single forward on last token).
- Add token-wise trajectories and plotting.
- Add batched prompts and statistics across datasets.
