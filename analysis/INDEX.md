# Analysis Script Index

This file is the working map for `analysis/`. The top-level category
directories own the scripts directly, while shared helpers live under
`analysis/utils/`.

## 1. Forward Geometry And XSA

Core scripts for residual/value-space interventions and para/perp sweeps.

| Script | Role |
|---|---|
| `forward_geometry/qwen_xsa_forward_ablation.py` | Main forward-only XSA ablation runner for perplexity/metrics. |
| `forward_geometry/qwen_xsa_forward_generate.py` | Generation runner for XSA targets such as `none`, `attn`, `mlp`, `both`. |
| `forward_geometry/qwen_attn_x_parallel_removal_viz.py` | Visualizes attention-side x-parallel removal. |
| `forward_geometry/summarize_para_perp_ppl_ablation.py` | Aggregates para/perp PPL ablation outputs. |
| `forward_geometry/summarize_alpha_gamma_ablation.py` | Aggregates alpha/gamma sweep outputs. |
| `forward_geometry/visualize_xsa_ablation_sweeps.py` | Plots sweep summaries. |
| `forward_geometry/run_xsa_forward_ablation.sh` | Batch launcher for XSA forward ablation. |
| `forward_geometry/run_xsa_forward_generate.sh` | Batch launcher for generation examples. |
| `forward_geometry/run_xsa_compare_middle_ppl.sh` | PPL comparison launcher for middle-space interventions. |
| `forward_geometry/run_xsa_para_ablation.sh` | Para-scale sweep launcher. |
| `forward_geometry/run_xsa_perp_ablation.sh` | Perp-scale sweep launcher. |
| `forward_geometry/run_summarize_para_perp_ppl_ablation.sh` | Background summary launcher. |

## 2. VLM Geometry Scaling

Hooks and smoke/grid launchers for applying residual-level para/perp scaling to
multimodal model understanding and generation modules.

| Script | Role |
|---|---|
| `vlm_geometry/hooks.py` | Model-agnostic para/perp scaling hooks for block, attention, and MLP updates. |
| `vlm_geometry/presets.py` | Module-path presets for Qwen-Image, Bagel, and Ming. |
| `vlm_geometry/run_vlm_geometry_smoke.py` | Minimal loader/forward runner for checking one VLM geometry setting. |
| `vlm_geometry/scripts/run_vlm_geometry_smoke.sh` | Shell wrapper that writes one result JSON under `runs/vlm_geometry_scaling`. |
| `vlm_geometry/scripts/run_vlm_geometry_grid.sh` | Sequential grid wrapper over model specs, sides, targets, and scale values. |

## 3. Attention And Layerwise Visualization

Scripts that produce qualitative layer/head figures.

| Script | Role |
|---|---|
| `visualization/qwen_xsa_attention_matrix_viz.py` | Qwen attention matrix visualization. |
| `visualization/qwen_xsa_single_layer_flip_viz.py` | Single-layer/head flip visualization. |
| `visualization/run_qwen_xsa_attention_matrix_viz.sh` | Multi-GPU Qwen attention visualization launcher. |
| `visualization/run_qwen_xsa_attention_matrix_viz_single_gpu.sh` | Single-GPU wrapper. |
| `visualization/run_qwen_xsa_single_layer_flip_viz.sh` | Multi-GPU single-layer visualization launcher. |
| `visualization/run_qwen_xsa_single_layer_flip_viz_single_gpu.sh` | Single-GPU wrapper. |
| `visualization/run_qwen_attn_x_parallel_removal_viz.sh` | Multi-GPU x-parallel visualization launcher. |
| `visualization/run_qwen_attn_x_parallel_removal_viz_single_gpu.sh` | Single-GPU wrapper. |

## 4. GSM8K And Math Diagnostics

Targeted sanity checks for math behavior.  These are narrower than lm-eval and
are useful when benchmark numbers look suspicious.

| Script | Role |
|---|---|
| `gsm_math/gsm8k_free_budget_eval.py` | GSM8K evaluation with free generation budget. |
| `gsm_math/gsm8k_free_budget_fewshot_eval.py` | Few-shot GSM8K free-budget variant. |
| `gsm_math/gsm8k_dualpath_eval.py` | Dense/drop dual-path GSM8K check. |
| `gsm_math/gsm8k_wanda_dualpath_eval.py` | WANDA-specific GSM8K dual-path check. |
| `gsm_math/gsm8k_masked_prefix_eval.py` | Masked-prefix GSM8K check. |
| `gsm_math/compare_dense_pruned_gsm_generations.py` | Dense-vs-pruned GSM8K generation comparison for sanity checks. |
| `gsm_math/eval_single_token_math.py` | Single-token math probe for dropped models. |
| `gsm_math/eval_single_token_math_wanda.py` | Single-token math probe for WANDA models. |

## 5. Dense, Dropped, And Pruned Comparisons

Scripts that compare generated outputs, MCQ subspaces, or transition metrics
between dense and modified models.

| Script | Role |
|---|---|
| `model_compare/transition_layerwise_compare.py` | Layerwise transition comparison; supports `analysis_mode=pruned`. |
| `model_compare/compare_generation_metrics.py` | Generation metric comparison for dropped/modified runs. |
| `model_compare/compare_mcq_subspace_metrics.py` | MCQ subspace metric comparison. |
| `model_compare/double_check_drop_eval.py` | Drop-eval sanity checker. |
| `model_compare/masked_teacher_drop_eval.py` | Masked-teacher check for dropped models. |
| `model_compare/masked_teacher_wanda_eval.py` | Masked-teacher check for WANDA models. |

## 6. nanoGPT Diagnostics

Utilities and launchers for nanoGPT checkpoints.

| Script | Role |
|---|---|
| `nanogpt/nanogpt_attention_utils.py` | Shared nanoGPT checkpoint/tokenizer helpers. |
| `nanogpt/check_nanogpt_gamma_ckpt.py` | Validates gamma checkpoint contents. |
| `nanogpt/cleanup_invalid_nanogpt_checkpoints.py` | Removes invalid checkpoint artifacts after inspection. |
| `nanogpt/layerwise_sim_lm_random_init.py` | Random-init layerwise similarity diagnostic. |
| `nanogpt/run_nanogpt_xsa_attention_matrix_viz.sh` | nanoGPT attention visualization launcher. |
| `nanogpt/run_nanogpt_xsa_single_layer_flip_viz.sh` | nanoGPT single-layer visualization launcher. |
| `nanogpt/run_nanogpt_xsa_batch_flip_viz.sh` | Batch single-layer visualization launcher. |
| `nanogpt/run_nanogpt_attn_x_parallel_removal_viz.sh` | nanoGPT x-parallel visualization launcher. |

## 7. Shared Utilities And Data Prep

Keep reusable code small here.  If a helper becomes shared infrastructure,
promote it into `src/repgeo/` and leave a thin compatibility wrapper.

| Script | Role |
|---|---|
| `utils/generation_forward_utils.py` | Shared generation/forward tracing helpers. |
| `utils/forward_utils.py` | Older forward helper module kept for compatibility. |
| `utils/prepare_text_dataset_jsonl.py` | Dataset-to-jsonl prompt preparation. |
| `utils/download_wandb_history.py` | W&B history downloader. |
| `utils/plot_wandb_history.py` | W&B history plotting. |
| `utils/remove_empty_dirs.py` | Local cleanup helper. |

## Refactor Rule

Use package imports such as `analysis.utils.generation_forward_utils` for
cross-category helpers.
