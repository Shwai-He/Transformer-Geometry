# Demystifying Transformers through Representation Geometries (Starter)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/search/?query=Demystifying+Transformers+through+Representation+Geometries&searchtype=all)
[![Conference](https://img.shields.io/badge/Conference-TBD-lightgrey)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#)

This local project is an initial coding scaffold based on the paper notes.

## What is implemented

A representation-geometry probe for causal LMs:
- `z_post_median`: median hidden-state norm across layers (last token).
- `z_post[i+1] / z_post[i]`: layer-wise scale gain.
- `dz_para_median`: median norm of update component parallel to input representation.
- `dz_perp_median`: median norm of update component orthogonal to input representation.
- `log(dz_para / dz_perp)`: dominance of parallel over orthogonal updates.
- top-k index overlap in logits space across adjacent layers.

And two practical extensions:
- batch analysis over multiple prompts.
- step-wise generation analysis (recompute geometry at each decoding step).

And one architecture-oriented toy study:
- gated attention vs vanilla attention on synthetic inputs, measured by the same parallel/orthogonal geometry decomposition.

And a technical reproduction path:
- layer-wise `z_in -> z_out` decomposition into parallel and orthogonal components.
- logits decomposition (`x_in`, `x_out`, `x_para`, `x_perp`) on top-k tokens.
- top-k rank shifts across layers.
- softmax translation invariance check.
- mHC-style `alpha * z_in + delta_perp` decomposition diagnostics.

## Structure

- `src/repgeo/analyzer.py`: core analysis logic.
- `scripts/run_probe.py`: single-prompt probe.
- `scripts/run_batch_probe.py`: batch prompt probe.
- `scripts/run_generation_probe.py`: generation-step probe.
- `scripts/plot_layer_metrics.py`: basic plotting utility.
- `scripts/run_gated_attention_demo.py`: synthetic gated-attention geometry comparison.
- `scripts/reproduce_technical.py`: technical reproduction script.
- `scripts/plot_technical.py`: plot reproduction metrics.
- `results/`: output JSON/figures.

## Install

```bash
cd /Users/heshuai/Documents/GitHub/demystifying-transformers
python3 -m pip install -r requirements.txt
```

## Quick start

Single prompt:

```bash
PYTHONPATH=src python3 scripts/run_probe.py \
  --model_name_or_path gpt2 \
  --prompt "John has twice as many books as Mary. Together they have 18 books. How many books does John have?" \
  --top_k 5 \
  --granularity both \
  --output results/gpt2_probe.json
```

Batch prompts:

```bash
cat > results/prompts.txt << 'EOF'
John has twice as many books as Mary. Together they have 18 books. How many books does John have?
If all bloops are razzies and some razzies are lazzies, can some bloops be lazzies?
EOF

PYTHONPATH=src python3 scripts/run_batch_probe.py \
  --model_name_or_path gpt2 \
  --prompts_file results/prompts.txt \
  --output results/gpt2_batch_probe.json
```

Generation-step analysis:

```bash
PYTHONPATH=src python3 scripts/run_generation_probe.py \
  --model_name_or_path gpt2 \
  --prompt "Solve: 2x + 3 = 11, x =" \
  --max_new_tokens 8 \
  --output results/gpt2_generation_probe.json
```

Plot layer metrics:

```bash
PYTHONPATH=src python3 scripts/plot_layer_metrics.py \
  --input results/gpt2_probe.json \
  --output results/gpt2_layer_metrics.png
```

Gated attention demo:

```bash
PYTHONPATH=src python3 scripts/run_gated_attention_demo.py \
  --d_model 512 \
  --n_heads 8 \
  --seq_len 64 \
  --batch_size 8 \
  --device cpu \
  --output results/gated_attention_demo.json
```

Technical reproduction:

```bash
PYTHONPATH=src python3 scripts/reproduce_technical.py \
  --model_name_or_path gpt2 \
  --prompt "John has twice as many books as Mary. Together they have 18 books. How many books does John have?" \
  --top_k 5 \
  --translation_shift 100 \
  --output results/gpt2_technical_repro.json
```

Plot technical reproduction:

```bash
PYTHONPATH=src python3 scripts/plot_technical.py \
  --input results/gpt2_technical_repro.json \
  --output results/gpt2_technical_repro.png
```

Batch technical reproduction:

```bash
cat > results/technical_prompts.txt << 'EOF'
John has twice as many books as Mary. Together they have 18 books. How many books does John have?
If all bloops are razzies and some razzies are lazzies, can some bloops be lazzies?
EOF

PYTHONPATH=src python3 scripts/reproduce_technical_batch.py \
  --model_name_or_path gpt2 \
  --prompts_file results/technical_prompts.txt \
  --top_k 5 \
  --translation_shift 100 \
  --output results/gpt2_technical_repro_batch.json
```

Plot batch technical reproduction:

```bash
PYTHONPATH=src python3 scripts/plot_technical_batch.py \
  --input results/gpt2_technical_repro_batch.json \
  --output results/gpt2_technical_repro_batch.png
```

## Next steps

- Add token-level trajectory analysis for full sequence positions (not only final token).
- Add model-to-model comparison (dense vs pruned) in the same interface.
- Add dataset-level benchmark scripts and statistical tests.
