#!/usr/bin/env python3
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from common import (
    REPO_ROOT,
    build_result_path,
    collect_layerwise_ratio_from_layer_metrics,
    collect_layerwise_ratio_from_sublayer,
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
TRIM_EDGE_LAYERS = True
START_LAYER = 2
OUT_DIR = REPO_ROOT / "figs" / "probe_steps_viz"
OUT_NAME = f"prompt{PROMPT_IDX}_{PHASE}_layerwise_para_residual_ratios_range.png"
# ==================


def _collect_ratios(steps, metric_key, has_sublayer, has_layer, is_block):
    if has_sublayer:
        x_pp, mean_pp, min_pp, max_pp = collect_layerwise_ratio_from_sublayer(
            steps, metric_key, "dz_para_norm", "dz_perp_norm", TRIM_EDGE_LAYERS, START_LAYER
        )
        x_px, mean_px, min_px, max_px = collect_layerwise_ratio_from_sublayer(
            steps, metric_key, "dz_para_norm", "z_post_norm", TRIM_EDGE_LAYERS, START_LAYER
        )
        if is_block and has_layer and not x_pp:
            x_pp, mean_pp, min_pp, max_pp = collect_layerwise_ratio_from_layer_metrics(
                steps, "dz_para_norm", "dz_perp_norm", TRIM_EDGE_LAYERS, START_LAYER
            )
        if is_block and has_layer and not x_px:
            x_px, mean_px, min_px, max_px = collect_layerwise_ratio_from_layer_metrics(
                steps, "dz_para_norm", "z_post_norm", TRIM_EDGE_LAYERS, START_LAYER
            )
    else:
        x_pp, mean_pp, min_pp, max_pp = collect_layerwise_ratio_from_layer_metrics(
            steps, "dz_para_norm", "dz_perp_norm", TRIM_EDGE_LAYERS, START_LAYER
        )
        x_px, mean_px, min_px, max_px = collect_layerwise_ratio_from_layer_metrics(
            steps, "dz_para_norm", "z_post_norm", TRIM_EDGE_LAYERS, START_LAYER
        )
    return (x_pp, mean_pp, min_pp, max_pp), (x_px, mean_px, min_px, max_px)


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=False, sharey=False)
    series = [("block", "BLOCK", "block_update"), ("attn", "ATTN", "attn_update"), ("mlp", "MLP", "mlp_update")]

    for ax, (name, title, metric_key) in zip(axes, series):
        para_perp, para_x = _collect_ratios(
            steps,
            metric_key=metric_key,
            has_sublayer=has_sublayer,
            has_layer=has_layer,
            is_block=name == "block",
        )
        x_pp, mean_pp, min_pp, max_pp = para_perp
        x_px, mean_px, min_px, max_px = para_x

        if x_pp:
            ax.plot(x_pp, mean_pp, color="tab:blue", lw=1.8, label="para/perp")
            ax.fill_between(x_pp, min_pp, max_pp, color="tab:blue", alpha=0.16, label="para/perp range")

        if x_px:
            ax.plot(x_px, mean_px, color="tab:orange", lw=1.8, linestyle="--", label="para/residual")
            ax.fill_between(x_px, min_px, max_px, color="tab:orange", alpha=0.16, label="para/residual range")

        ax.set_title(f"{title} ratios")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Ratio")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cfcfcf")

    fig.suptitle(f"Prompt {PROMPT_IDX}: dz_para/dz_perp and dz_para/residual (mean + range over steps, phase={PHASE})")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
