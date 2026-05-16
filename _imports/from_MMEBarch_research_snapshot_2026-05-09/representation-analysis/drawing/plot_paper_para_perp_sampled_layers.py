#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import build_mean_min_max_series, build_result_path, collect_step_layer_values, load_runs


# ===== config =====
PAPER_ROOT = Path(__file__).absolute().parents[1]
ROOT = Path("/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/results")
RUN_TYPE = "generation"  # "probe" or "generation"
SPACE = "hidden"  # "hidden" or "logits"
MODEL_NAMES = [
    "Qwen3-4B-Instruct-2507",
    "Qwen3-30B-A3B",
]
MODEL_COLORS = {
    "Qwen3-4B-Instruct-2507": "tab:blue",
    "Qwen3-30B-A3B": "tab:orange",
}
RANDOM_INIT = False
MODE = "both"  # only used for generation: "prefill", "decode", or "both"
MAX_NEW_TOKENS = 128

PHASE = "both"  # "prefill", "decode", or "both"
PART = "block"  # "block", "attn", or "mlp"
N_SAMPLED_LAYERS = 6
OUT_DIR = PAPER_ROOT / "figs" / "para_dist"
# ==================


def sample_layers(layers: list[int], n: int) -> list[int]:
    if len(layers) <= n:
        return layers
    idx = np.linspace(0, len(layers) - 1, n, dtype=int)
    return [layers[i] for i in idx]


def make_out_stem(model_name: str) -> str:
    model_tag = model_name.replace("/", "_")
    return f"paper_{model_tag}_{SPACE}_{PHASE}_{PART}_para_perp_sampled_layers"


def write_embedded_plot_script(model_name: str, color: str, plot_data: list[dict[str, object]], out_path: Path) -> None:
    script = f'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = {model_name!r}
SPACE = {SPACE!r}
PHASE = {PHASE!r}
PART = {PART!r}
COLOR = {color!r}
PLOT_DATA = {pformat(plot_data, width=120)}
OUT_PATH = Path(__file__).with_suffix(".png")


def main() -> None:
    fig, axes = plt.subplots(1, 6, figsize=(18.2, 3.7), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    for i, (ax, item) in enumerate(zip(axes, PLOT_DATA)):
        ax.plot(item["steps"], item["mean"], color=COLOR, lw=2.2)
        ax.fill_between(item["steps"], item["min"], item["max"], color=COLOR, alpha=0.20, linewidth=0)
        ax.text(0.5, -0.20, f"L{{item['layer']}}", transform=ax.transAxes, fontsize=11.5, ha="center", va="top")
        if i == 0:
            ax.set_ylabel(r"$\\|\\Delta_{\\parallel}\\| / \\|\\Delta_{\\perp}\\|$", fontsize=12.5)
        ax.grid(alpha=0.24, linewidth=0.85)
        ax.tick_params(axis="both", which="major", labelsize=10.5, length=4.0, width=0.85)

    for ax in axes[len(PLOT_DATA):]:
        ax.axis("off")

    fig.supxlabel("Generation Step", fontsize=12.5, y=0.015)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
'''
    out_path.write_text(script, encoding="utf-8")


def plot_one_model(model_name: str) -> None:
    color = MODEL_COLORS.get(model_name, "tab:blue")
    result_path = build_result_path(ROOT, RUN_TYPE, SPACE, model_name, RANDOM_INIT, MODE, MAX_NEW_TOKENS)
    runs = load_runs(result_path)
    ratio = collect_step_layer_values(runs, part=PART, metric="para_perp_ratio", phase=PHASE)

    layers = sample_layers(sorted(ratio.keys()), N_SAMPLED_LAYERS)
    if not layers:
        print(f"{model_name} {PART}: no para/perp data")
        return

    fig, axes = plt.subplots(1, 6, figsize=(18.2, 3.7), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    plot_data = []
    for i, (ax, layer) in enumerate(zip(axes, layers)):
        x, mean, lo, hi = build_mean_min_max_series(ratio[layer])
        plot_data.append({"layer": layer, "steps": x, "mean": mean, "min": lo, "max": hi})
        ax.plot(x, mean, color=color, lw=2.2)
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)

        ax.text(0.5, -0.20, f"L{layer}", transform=ax.transAxes, fontsize=11.5, ha="center", va="top")
        if i == 0:
            ax.set_ylabel(r"$\|\Delta_{\parallel}\| / \|\Delta_{\perp}\|$", fontsize=12.5)
        ax.grid(alpha=0.24, linewidth=0.85)
        ax.tick_params(axis="both", which="major", labelsize=10.5, length=4.0, width=0.85)

    for ax in axes[len(layers):]:
        ax.axis("off")

    fig.supxlabel("Generation Step", fontsize=12.5, y=0.015)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = make_out_stem(model_name)
    out = OUT_DIR / f"{out_stem}.png"
    embedded_script = OUT_DIR / f"{out_stem}.py"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_embedded_plot_script(model_name, color, plot_data, embedded_script)
    print("saved:", out)
    print("saved embedded script:", embedded_script)
    print("sampled layers:", layers)


def main() -> None:
    for model_name in MODEL_NAMES:
        plot_one_model(model_name)


if __name__ == "__main__":
    main()
