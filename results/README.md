# Results Workspace

This directory stores non-paper result artifacts produced by probes, generation analysis, and standalone measurements.

Current layout:

- `generation_probe/`: generation-step probe JSONs, hidden-state dumps, and per-prompt visualizations
- `para_profiles/`: standalone sampled-layer profile figures and helper scripts
- `value_pre_ratio/`: structured summaries grouped by model
- `prompts.txt`: shared prompt file kept at the top level because several scripts still reference `results/prompts.txt`

Some legacy top-level filenames are kept as symlinks for compatibility with older notebooks and scripts.

Paper-facing tables and figures should still be copied into `_NeurIPS_2026_/results` and `_NeurIPS_2026_/figs`.
