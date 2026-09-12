# lm-eval Scripts

This directory keeps only scripts directly used by the current WANDA
compression and layer-drop run at the top level. Other normal lm-eval/XSA
helpers were not deleted; they were collected under the archive directory below.

## Active Entrypoints

- `run_lm_eval_xsa_setting.sh`: single-setting HF/XSA launcher. The WANDA
  compression launcher wraps this with `SETTING=none`.
- `run_lm_eval_compression_setting.sh`: neutral compression-eval launcher for
  dense/pruned checkpoints. It applies no XSA intervention.
- `run_lm_eval_layer_drop_setting.sh`: single-setting layer-drop launcher.
- `collect_xsa_lm_eval_results.py` / `.sh`: standard XSA/general result
  collector retained for reference and manual collection.
- `collect_xsa_lm_eval_ruler_results.py` / `.sh`: RULER result collector
  retained for manual collection.
- `collect_wanda_lm_eval_results.py`: WANDA compression result collector.
- `collect_layer_drop_lm_eval_results.py`: layer-drop result collector.
- `download_lm_eval_tasks.py`: local task cache helper.
- `download_mmlu_local_snapshot.sh`: local MMLU snapshot helper.

## Archived Scripts

Scripts that were not directly used by the current compression/layer-drop run
were moved to:

```text
unused_for_compression_20260523/
```

The archive is organized by experiment family:

- `xsa_paper/`: normal XSA/paper lm-eval batch launchers and wrappers.
- `collectors/`: reserved for older collectors if new unused collectors are
  archived later.
- `nanogpt/`: nanoGPT checkpoint evaluation scripts.
- `geo_prune/`: older geo-prune launchers and collectors.
- `utilities/`: older helper scripts and table builders.
- `data_prep/`: training-data cleanup utilities.
- `oneoff/`: one-off repair scripts.
- `cache/`: Python cache files moved out of the active script root.

Note: archived scripts may contain relative references that assumed the old flat
layout. Move a script back or update its `SCRIPT_DIR` references before reusing
it as an active launcher.
