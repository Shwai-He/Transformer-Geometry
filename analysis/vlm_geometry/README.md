# VLM Geometry Scaling

This folder contains model-agnostic hooks for parallel/perpendicular scaling in
multimodal models.  The intervention decomposes a module signal `y` with
respect to the module input `x`:

```text
y_parallel = proj_x(y)
y_perp     = y - y_parallel
y'         = para_scale * y_parallel + perp_scale * y_perp
```

Use `SPACE=residual` for residual-stream interventions.  For `target=block`,
`y` is the residual block update `block(x) - x`; for `target=attn` or
`target=mlp`, `y` is the corresponding branch output.

Use `SPACE=value` with `TARGET=value` for a value-space intervention on the
attention value-projection branch.  This is separated from residual results in
both the JSON config and the output path.

## Supported Presets

The code has presets for:

- `qwen-image`
- `bagel`
- `ming`

Presets only provide likely module paths and branch-name regexes.  If a local
checkpoint uses different names, pass `--layer-paths` or set `LAYER_PATHS`.
This is expected for rapidly moving VLM repos.

## Smoke Run

```bash
MODEL_PRESET=qwen-image \
MODEL_NAME_OR_PATH=/path/to/qwen-image \
LOADER=diffusers \
MODE=pipeline_generate \
SIDE=gen \
SPACE=residual \
TARGET=block \
PARA_SCALE=1.0 \
PERP_SCALE=0.0 \
bash analysis/vlm_geometry/scripts/run_vlm_geometry_smoke.sh
```

For transformer-style understanding models:

```bash
MODEL_PRESET=bagel \
MODEL_NAME_OR_PATH=/path/to/bagel \
LOADER=transformers \
MODE=text_forward \
SIDE=und \
SPACE=residual \
TARGET=block \
bash analysis/vlm_geometry/scripts/run_vlm_geometry_smoke.sh
```

If preset discovery fails, inspect the model and set an explicit layer path:

```bash
LAYER_PATHS=language_model.model.layers \
bash analysis/vlm_geometry/scripts/run_vlm_geometry_smoke.sh
```

## Unified Grid

`scripts/run_vlm_geometry_grid.sh` runs the same scaling grid over multiple
model specs.  Each spec is:

```text
tag|preset|loader|model_path|sides[|layer_paths]
```

Example:

```bash
MODEL_SPECS="qwenimage|qwen-image|diffusers|/ckpt/qwen-image|gen;bagel|bagel|transformers|/ckpt/bagel|und,gen|language_model.model.layers" \
SPACES=residual,value \
TARGETS_RESIDUAL=block,attn,mlp \
TARGETS_VALUE=value \
PARA_SCALES=1.0,0.0 \
PERP_SCALES=1.0,0.0 \
bash analysis/vlm_geometry/scripts/run_vlm_geometry_grid.sh
```

Outputs are written under
`runs/vlm_geometry_scaling/results/by_model/<model>/<task>/<setting>/`.
The setting name includes `side`, `space`, `target`, `para`, and `perp`, and
generation runs also save images under the same result leaf.
