#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    REPO_ROOT,
    build_mean_min_max_series,
    build_result_path,
    collect_step_layer_values,
    load_runs,
)


# ===== config =====
ROOT = REPO_ROOT / "results"
RUN_TYPE = "generation"  # "probe" or "generation"
SPACE = "logits"  # "hidden" or "logits"
MODEL_NAME = "Qwen3-4B-Instruct-2507"
RANDOM_INIT = False
MODE = "both"  # only used for generation: "prefill", "decode", or "both"
MAX_NEW_TOKENS = 128
RESULT_PATH = build_result_path(ROOT, RUN_TYPE, SPACE, MODEL_NAME, RANDOM_INIT, MODE, MAX_NEW_TOKENS)
PHASE = "both"  # "prefill", "decode", or "both"
PART = "block"  # "block", "attn", or "mlp"
OUT_DIR = REPO_ROOT / "figs" / "probe_steps_viz"
OUT_NAME = f"prompt_agg_{PHASE}_{PART}_para_perp_layergrid_step.png"
# ==================


def main() -> None:
    runs = load_runs(RESULT_PATH)
    ratio = collect_step_layer_values(runs, part=PART, metric="para_perp_ratio", phase=PHASE)

    layers = sorted(ratio.keys())
    if not layers:
        print(f"{PART}: no data")
        return

    ncols = 8
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, layer in enumerate(layers):
        ax = axes[i]

        if layer in ratio and ratio[layer]:
            x, mean, lo, hi = build_mean_min_max_series(ratio[layer])
            ax.plot(x, mean, color="tab:blue", lw=1.2, label="para/perp mean")
            ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.18, label="para/perp min-max")

        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("para/perp", fontsize=8, color="tab:blue")
        ax.grid(alpha=0.2)

    for j in range(len(layers), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{PART.upper()} per-layer (x=step), prompt-aggregated para/perp, phase={PHASE}")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
