#!/usr/bin/env python3
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
ROOT = Path("/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/results")
RUN_TYPE = "generation"  # "probe" or "generation"
SPACE = "logits"  # "hidden" or "logits"
MODEL_NAME = "Qwen3-4B-Instruct-2507"
RANDOM_INIT = False
MODE = "both"  # only used for generation: "prefill", "decode", or "both"
MAX_NEW_TOKENS = 128
RESULT_PATH = build_result_path(ROOT, RUN_TYPE, SPACE, MODEL_NAME, RANDOM_INIT, MODE, MAX_NEW_TOKENS)
PHASE = "both"  # "prefill", "decode", or "both"
PROMPT_IDX = 0
TRIM_EDGE_LAYERS = False
OUT_DIR = REPO_ROOT / "results"
OUT_NAME = f"prompt{PROMPT_IDX}_{PHASE}_layerwise_znorm_dznorm_range.png"
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
        used_layer_z_fallback = False
        used_layer_dz_fallback = False

        if has_sublayer:
            if name == "block":
                x_z, mean_z, min_z, max_z = collect_layerwise_from_sublayer(
                    steps, "block_update", "z_post_norm", TRIM_EDGE_LAYERS
                )
                x_dz, mean_dz, min_dz, max_dz = collect_layerwise_from_sublayer(
                    steps, "block_update", "dz_norm", TRIM_EDGE_LAYERS
                )
                if has_layer and not x_z:
                    x_z, mean_z, min_z, max_z = collect_layerwise_from_layer_metrics(
                        steps, "z_post_norm", TRIM_EDGE_LAYERS
                    )
                    used_layer_z_fallback = True
                if has_layer and not x_dz:
                    x_dz, mean_dz, min_dz, max_dz = collect_layerwise_from_layer_metrics(
                        steps, "dz_norm", TRIM_EDGE_LAYERS
                    )
                    used_layer_dz_fallback = True
            else:
                x_z, mean_z, min_z, max_z = collect_layerwise_from_sublayer(
                    steps, metric_key, "z_post_norm", TRIM_EDGE_LAYERS
                )
                x_dz, mean_dz, min_dz, max_dz = collect_layerwise_from_sublayer(
                    steps, metric_key, "dz_norm", TRIM_EDGE_LAYERS
                )
        else:
            x_z, mean_z, min_z, max_z = collect_layerwise_from_layer_metrics(
                steps, "z_post_norm", TRIM_EDGE_LAYERS
            )
            x_dz, mean_dz, min_dz, max_dz = collect_layerwise_from_layer_metrics(
                steps, "dz_norm", TRIM_EDGE_LAYERS
            )

        if x_z:
            ax.plot(x_z, mean_z, color="tab:blue", linestyle="-.", label="z_post_norm")
            ax.fill_between(x_z, min_z, max_z, color="tab:blue", alpha=0.15, label="_nolegend_")

        ax2 = ax.twinx()
        if x_dz:
            ax2.plot(x_dz, mean_dz, color="tab:orange", linestyle="--", label="dz_norm")
            ax2.fill_between(x_dz, min_dz, max_dz, color="tab:orange", alpha=0.12, label="_nolegend_")

        ax.set_title(f"{title} (range over steps)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("z_post_norm", color="tab:blue")
        ax.grid(alpha=0.25)
        ax2.set_ylabel("dz_norm", color="tab:orange")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, labels1 + labels2, ncol=2, loc="center")

        if used_layer_z_fallback:
            ax.text(0.01, 0.02, "z norm from layer_metrics", transform=ax.transAxes, fontsize=8, color="tab:blue")
        if used_layer_dz_fallback:
            ax.text(0.01, 0.08, "dz norm from layer_metrics", transform=ax.transAxes, fontsize=8, color="tab:orange")

    fig.suptitle(f"Prompt {PROMPT_IDX}: z norm + dz norm (range over steps, phase={PHASE})")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
