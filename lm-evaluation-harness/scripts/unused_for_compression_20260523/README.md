# Scripts Archived From Active Compression Run

These scripts were moved out of the top-level `lm-evaluation-harness/scripts`
directory on 2026-05-23 because they are older helper families not directly
used by the current WANDA compression and layer-drop experiments.

Nothing was deleted. The current active Slurm jobs still use the top-level
launchers:

- `../run_lm_eval_xsa_setting.sh`
- `../run_lm_eval_layer_drop_setting.sh`

Archive layout:

- `xsa_paper/`: normal XSA/paper lm-eval batch launchers and wrappers.
- `collectors/`: reserved for older collectors if new unused collectors are
  archived later. Current XSA/RULER collectors were restored to the top level.
- `nanogpt/`: nanoGPT evaluation scripts.
- `geo_prune/`: older geo-prune scripts.
- `utilities/`: general helper scripts.
- `data_prep/`: data-cleaning utilities.
- `oneoff/`: one-off repair scripts.
- `cache/`: moved `__pycache__` files.

Before running an archived script again, check its relative paths. Some scripts
assume they live next to `run_lm_eval_xsa_setting.sh` in the old flat layout.
