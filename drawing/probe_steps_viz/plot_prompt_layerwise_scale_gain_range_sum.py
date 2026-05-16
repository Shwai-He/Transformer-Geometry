#!/usr/bin/env python3
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from common import (
    REPO_ROOT,
    build_result_path,
    collect_layerwise_from_layer_metrics,
    collect_layerwise_from_sublayer,
    load_runs,
    pick_steps,
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
PROMPT_IDX = 0
LEFT_METRIC = "scale_gain_to_next"  # or "para_perp_ratio"
LEFT_LABEL = "scale_gain_to_next" if LEFT_METRIC == "scale_gain_to_next" else "para_perp_ratio"
TRIM_EDGE_LAYERS = True
START_LAYER = 2
OUT_DIR = REPO_ROOT / "figs" / "probe_steps_viz"
OUT_NAME = f"prompt{PROMPT_IDX}_{PHASE}_layerwise_range_sum.png"
# ==================


def main() -> None:
    runs = load_runs(RESULT_PATH)
    run = runs[PROMPT_IDX]
    steps = pick_steps(run, phase=PHASE)
    if not steps:
        print(f"prompt {PROMPT_IDX}: no steps ({PHASE}), skipped")
        return

    has_sublayer = any(step.get("sublayer_metrics") for step in steps)
    has_layer = any(step.get("layer_metrics") for step in steps)
    print(f"prompt {PROMPT_IDX}: steps={len(steps)}, phase={PHASE}, has_sublayer={has_sublayer}, has_layer={has_layer}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=False)
    series = [("block", "BLOCK", "block_update"), ("attn", "ATTN", "attn_update"), ("mlp", "MLP", "mlp_update")]

    for ax, (name, title, metric_key) in zip(axes, series):
        used_layer_left_fallback = False
        used_layer_cos_fallback = False

        if has_sublayer:
            if name == "block":
                x_l, mean_l, min_l, max_l = collect_layerwise_from_sublayer(
                    steps, "block_update", LEFT_METRIC, TRIM_EDGE_LAYERS, START_LAYER
                )
                x_c, mean_c, min_c, max_c = collect_layerwise_from_sublayer(
                    steps, "block_update", "io_cos_sim", TRIM_EDGE_LAYERS, START_LAYER
                )
                if has_layer and not x_l:
                    x_l, mean_l, min_l, max_l = collect_layerwise_from_layer_metrics(
                        steps, LEFT_METRIC, TRIM_EDGE_LAYERS, START_LAYER
                    )
                    used_layer_left_fallback = True
                if has_layer and not x_c:
                    x_c, mean_c, min_c, max_c = collect_layerwise_from_layer_metrics(
                        steps, "io_cos_sim", TRIM_EDGE_LAYERS, START_LAYER
                    )
                    used_layer_cos_fallback = True
            else:
                x_l, mean_l, min_l, max_l = collect_layerwise_from_sublayer(
                    steps, metric_key, LEFT_METRIC, TRIM_EDGE_LAYERS, START_LAYER
                )
                x_c, mean_c, min_c, max_c = collect_layerwise_from_sublayer(
                    steps, metric_key, "io_cos_sim", TRIM_EDGE_LAYERS, START_LAYER
                )
        else:
            x_l, mean_l, min_l, max_l = collect_layerwise_from_layer_metrics(
                steps, LEFT_METRIC, TRIM_EDGE_LAYERS, START_LAYER
            )
            x_c, mean_c, min_c, max_c = collect_layerwise_from_layer_metrics(
                steps, "io_cos_sim", TRIM_EDGE_LAYERS, START_LAYER
            )

        if x_l:
            ax.plot(x_l, mean_l, color="tab:blue", linestyle="-.", label=LEFT_LABEL)
            ax.fill_between(x_l, min_l, max_l, color="tab:blue", alpha=0.15, label="_nolegend_")

        ax2 = ax.twinx()
        if x_c:
            ax2.plot(x_c, mean_c, color="tab:red", linestyle="--", label="io_cos_sim")
            ax2.fill_between(x_c, min_c, max_c, color="tab:red", alpha=0.12, label="_nolegend_")

        sum_left = float(np.nansum(mean_l)) if len(mean_l) > 0 else None
        ax.set_title(f"{title} (sum: {sum_left})")
        ax.set_xlabel("Layer")
        ax.set_ylabel(LEFT_LABEL, color="tab:blue")
        ax.grid(alpha=0.25)
        ax2.set_ylabel("io_cos_sim", color="tab:red")
        ax2.set_ylim(0, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                ncol=2,
                loc="center",
                frameon=True,
                facecolor="white",
                edgecolor="#cfcfcf",
            )

        if used_layer_left_fallback:
            ax.text(0.01, 0.02, f"{LEFT_LABEL} from layer_metrics", transform=ax.transAxes, fontsize=8, color="tab:blue")
        if used_layer_cos_fallback:
            ax.text(0.01, 0.08, "cos from layer_metrics", transform=ax.transAxes, fontsize=8, color="tab:red")

    fig.suptitle(f"Prompt {PROMPT_IDX}: layer-wise {LEFT_LABEL} + cos (range over steps, phase={PHASE})")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
