#!/usr/bin/env python3
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from common import REPO_ROOT, build_mean_min_max_series, build_result_path, collect_step_layer_values, load_runs


# ===== config =====
ROOT = Path("/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/results")
RUN_TYPE = "generation"  # "probe" or "generation"
SPACE = "logits"  # "hidden" or "logits"
MODEL_NAME = "Qwen3-4B-Instruct-2507"
RANDOM_INIT = False
MODE = "both"  # only used for generation: "prefill", "decode", or "both"
MAX_NEW_TOKENS = 128
RESULT_PATH = build_result_path(ROOT, RUN_TYPE, SPACE, MODEL_NAME, RANDOM_INIT, MODE, MAX_NEW_TOKENS)
PHASE = "both"  # "prefill", "decode", or "both"
PART = "block"  # "block", "attn", or "mlp"
LEFT_METRIC = "dz_perp_over_z_plus_dz_para"  # or "para_perp_ratio"
LEFT_LABEL = "dz_perp/(z+dz_para)" if LEFT_METRIC == "dz_perp_over_z_plus_dz_para" else "para_perp_ratio"
TRIM_EDGE_LAYERS = False
OUT_DIR = REPO_ROOT / "results"
OUT_NAME = f"prompt_agg_{PHASE}_{PART}_{LEFT_METRIC}_layergrid_step.png"
# ==================


def main() -> None:
    runs = load_runs(RESULT_PATH)
    left = collect_step_layer_values(runs, part=PART, metric=LEFT_METRIC, phase=PHASE)
    cos = collect_step_layer_values(runs, part=PART, metric="io_cos_sim", phase=PHASE)

    layers = sorted(set(left.keys()) | set(cos.keys()))
    if TRIM_EDGE_LAYERS and len(layers) > 2:
        layers = layers[1:-1]
    if not layers:
        print(f"{PART}: no data")
        return

    ncols = 8
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, layer in enumerate(layers):
        ax = axes[i]
        ax2 = ax.twinx()

        if layer in left and left[layer]:
            x, mean, lo, hi = build_mean_min_max_series(left[layer])
            ax.plot(x, mean, color="tab:blue", lw=1.2, label=f"{LEFT_LABEL} mean")
            ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.18, label=f"{LEFT_LABEL} min-max")

        if layer in cos and cos[layer]:
            x2, mean2, lo2, hi2 = build_mean_min_max_series(cos[layer])
            ax2.plot(x2, mean2, color="tab:red", lw=1.1, linestyle="--", label="cos mean")
            ax2.fill_between(x2, lo2, hi2, color="tab:red", alpha=0.12, label="cos min-max")

        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel(LEFT_LABEL, fontsize=8, color="tab:blue")
        ax2.set_ylabel("cos", fontsize=8, color="tab:red")
        ax2.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

    for j in range(len(layers), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{PART.upper()} per-layer (x=step), prompt-aggregated {LEFT_LABEL} + io_cos_sim, phase={PHASE}")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
