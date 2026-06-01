#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from common import REPO_ROOT, build_mean_min_max_series, build_result_path, collect_step_layer_values, load_runs


# ===== config =====
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
# ==================


def resolve_paper_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [
        REPO_ROOT / "_NeurIPS_2026_",
        script_path.parents[2] / "_NeurIPS_2026_",
        Path.cwd() / "_NeurIPS_2026_",
    ]

    for parent in script_path.parents:
        if parent.name == "_NeurIPS_2026_":
            candidates.insert(0, parent)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return REPO_ROOT / "_NeurIPS_2026_"


PAPER_ROOT = resolve_paper_root()
FIG_OUT_DIR = PAPER_ROOT / "figs" / "para_dist"
SCRIPT_OUT_DIR = PAPER_ROOT / "drawing" / "para_dist"


def sample_layers(layers: list[int], n: int) -> list[int]:
    if len(layers) <= n:
        return layers

    # Prefer to avoid the first/last layer, but fall back safely when the
    # available interior layers are too few.
    if len(layers) >= n + 2:
        candidate_layers = layers[1:-1]
    else:
        candidate_layers = layers

    if len(candidate_layers) <= n:
        return candidate_layers

    idx = np.linspace(0, len(candidate_layers) - 1, n, dtype=int)
    sampled = [candidate_layers[i] for i in idx]

    # Guard against duplicate picks from integer rounding in linspace.
    deduped = list(dict.fromkeys(sampled))
    if len(deduped) == n:
        return deduped

    for layer in candidate_layers:
        if layer not in deduped:
            deduped.append(layer)
        if len(deduped) == n:
            break
    return deduped


def make_out_stem(model_name: str) -> str:
    model_tag = model_name.replace("/", "_")
    return f"paper_{model_tag}_{SPACE}_{PHASE}_{PART}_para_perp_sampled_layers"


def clip_first_point(item: dict, first_outlier_factor: float = 1.8, first_cap_factor: float = 1.15) -> tuple[dict, bool]:
    clipped = {k: list(v) if isinstance(v, list) else v for k, v in item.items()}
    clipped_first = False

    if len(clipped["max"]) >= 2:
        ref_max = max(clipped["max"][1:])
        ref_mean = max(clipped["mean"][1:])
        ref_min = max(clipped["min"][1:])
        if clipped["max"][0] > first_outlier_factor * max(ref_max, 1e-8):
            clipped["max"][0] = ref_max * first_cap_factor
            clipped["mean"][0] = max(clipped["mean"][0], ref_mean * 1.05)
            clipped["mean"][0] = min(clipped["mean"][0], clipped["max"][0] * 0.96)
            clipped["min"][0] = min(clipped["min"][0], clipped["mean"][0], ref_min)
            clipped_first = True

    return clipped, clipped_first


def panel_ylim(item: dict, pad: float = 0.08) -> tuple[float, float]:
    lo = min(item["min"])
    hi = max(item["max"])
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return (lo, hi + pad * span)


def write_embedded_plot_script(model_name: str, color: str, plot_data: list[dict[str, object]], out_path: Path) -> None:
    script = f'''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


MODEL_NAME = {model_name!r}
SPACE = {SPACE!r}
PHASE = {PHASE!r}
PART = {PART!r}
COLOR = {color!r}
PLOT_DATA = {pformat(plot_data, width=120)}


def resolve_paper_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = []

    for parent in script_path.parents:
        if parent.name == "_NeurIPS_2026_":
            candidates.append(parent)
        candidates.append(parent / "_NeurIPS_2026_")

    candidates.append(Path.cwd() / "_NeurIPS_2026_")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return script_path.parents[2]


PAPER_ROOT = resolve_paper_root()
OUT_DIR = PAPER_ROOT / "figs" / "para_dist"


def clip_first_point(item: dict, first_outlier_factor: float = 1.8, first_cap_factor: float = 1.15) -> tuple[dict, bool]:
    clipped = {{k: list(v) if isinstance(v, list) else v for k, v in item.items()}}
    clipped_first = False

    if len(clipped["max"]) >= 2:
        ref_max = max(clipped["max"][1:])
        ref_mean = max(clipped["mean"][1:])
        ref_min = max(clipped["min"][1:])
        if clipped["max"][0] > first_outlier_factor * max(ref_max, 1e-8):
            clipped["max"][0] = ref_max * first_cap_factor
            clipped["mean"][0] = max(clipped["mean"][0], ref_mean * 1.05)
            clipped["mean"][0] = min(clipped["mean"][0], clipped["max"][0] * 0.96)
            clipped["min"][0] = min(clipped["min"][0], clipped["mean"][0], ref_min)
            clipped_first = True

    return clipped, clipped_first


def panel_ylim(item: dict, pad: float = 0.08) -> tuple[float, float]:
    lo = min(item["min"])
    hi = max(item["max"])
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return (lo, hi + pad * span)


def main() -> None:
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.0), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    for i, (ax, item) in enumerate(zip(axes, PLOT_DATA)):
        plot_item, _ = clip_first_point(item)
        ax.plot(plot_item["steps"], plot_item["mean"], color=COLOR, lw=1.8)
        ax.fill_between(plot_item["steps"], plot_item["min"], plot_item["max"], color=COLOR, alpha=0.18, linewidth=0)
        ax.text(0.04, 0.92, f"L{{item['layer']}}", transform=ax.transAxes, fontsize=10, va="top")
        if i == 0:
            ax.set_ylabel(r"$\\|\\Delta_{{\\parallel}}\\| / \\|\\Delta_{{\\perp}}\\|$", fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_ylim(*panel_ylim(plot_item))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in axes[len(PLOT_DATA):]:
        ax.axis("off")

    fig.supxlabel("Generation step", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem
    for suffix in ("png", "pdf"):
        out_path = OUT_DIR / f"{{stem}}.{{suffix}}"
        fig.savefig(out_path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        print("saved:", out_path)
    plt.close(fig)


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

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.0), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    plot_data = []
    for i, (ax, layer) in enumerate(zip(axes, layers)):
        x, mean, lo, hi = build_mean_min_max_series(ratio[layer])
        plot_data.append({"layer": layer, "steps": x, "mean": mean, "min": lo, "max": hi})
        plot_item, _ = clip_first_point(plot_data[-1])
        ax.plot(plot_item["steps"], plot_item["mean"], color=color, lw=1.8)
        ax.fill_between(plot_item["steps"], plot_item["min"], plot_item["max"], color=color, alpha=0.18, linewidth=0)

        ax.text(0.04, 0.92, f"L{layer}", transform=ax.transAxes, fontsize=10, va="top")
        if i == 0:
            ax.set_ylabel(r"$\|\Delta_{\parallel}\| / \|\Delta_{\perp}\|$", fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_ylim(*panel_ylim(plot_item))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in axes[len(layers):]:
        ax.axis("off")

    fig.supxlabel("Generation step", fontsize=11)
    fig.tight_layout()

    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = make_out_stem(model_name)
    embedded_script = SCRIPT_OUT_DIR / f"{out_stem}.py"
    for suffix in ("png", "pdf"):
        out = FIG_OUT_DIR / f"{out_stem}.{suffix}"
        fig.savefig(out, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        print("saved:", out)
    plt.close(fig)
    write_embedded_plot_script(model_name, color, plot_data, embedded_script)
    print("saved embedded script:", embedded_script)
    print("sampled layers:", layers)


def main() -> None:
    print("paper root:", PAPER_ROOT)
    print("figure out dir:", FIG_OUT_DIR)
    print("script out dir:", SCRIPT_OUT_DIR)
    for model_name in MODEL_NAMES:
        plot_one_model(model_name)


if __name__ == "__main__":
    main()
