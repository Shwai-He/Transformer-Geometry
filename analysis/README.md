# Analysis Workspace

This directory collects standalone research scripts that do not naturally belong inside the lm-eval harness, the training workspace, or the paper directory. In the paper workflow, this is where one-off diagnostics, forward-ablation checks, and visualization helpers usually live before they are promoted into a more stable workspace.

## Recommended grouping

Use the following mental categories when adding or locating scripts here.

### Forward editing and intervention analysis

- `qwen_xsa_forward_ablation.py`
- `qwen_xsa_forward_generate.py`
- `qwen_attn_x_parallel_removal_viz.py`
- `summarize_para_perp_ppl_ablation.py`
- `visualize_xsa_ablation_sweeps.py`

These scripts analyze test-time edits, value-space or residual-space interventions, and their effect on perplexity or generation.

### Attention and layerwise visualization

- `qwen_xsa_attention_matrix_viz.py`
- `qwen_xsa_single_layer_flip_viz.py`
- `run_qwen_xsa_attention_matrix_viz.sh`
- `run_qwen_xsa_single_layer_flip_viz.sh`
- `run_nanogpt_xsa_batch_flip_viz.sh`

These scripts build qualitative or layerwise views for attention-side geometry.

### Task-specific evaluation scripts

- `gsm8k_*`
- `eval_single_token_math*.py`
- `masked_teacher_*`
- `double_check_drop_eval.py`

These scripts are for targeted experiments that are narrower than the general lm-eval batches.

### Training diagnostics and checkpoint utilities

- `check_nanogpt_gamma_ckpt.py`
- `cleanup_invalid_nanogpt_checkpoints.py`
- `download_wandb_history.py`
- `plot_wandb_history.py`

These scripts support training-side diagnosis, logging inspection, and checkpoint cleanup.

### Shared helpers

- `forward_utils.py`
- `generation_forward_utils.py`
- `nanogpt_attention_utils.py`
- `prepare_text_dataset_jsonl.py`

Prefer keeping reusable logic small here, or promoting it into `src/repgeo/` when it becomes shared infrastructure.

## Boundaries with other directories

- If a script is a benchmark launcher, it should usually live under `lm-evaluation-harness/scripts/`.
- If a script is specifically for pretraining or optimization runs, it should usually live under `training/`.
- If a script is specifically for compression geometry, it should usually live under `compression/`.
- If a script directly generates a paper figure, consider placing the editable source under `drawing/` and only keeping lightweight analysis helpers here.
- If logic becomes reusable across multiple scripts, move it into `src/repgeo/`.

## Compatibility note

Older private commands may still reference a historical `representation-analysis/...` prefix. Public scripts should use the root-level paths in this repository.
