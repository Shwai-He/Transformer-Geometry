#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DRAWING_DIR = Path(__file__).absolute().parent
PAPER_ROOT = DRAWING_DIR.parents[2]
INPUT_CSV = DRAWING_DIR / "data" / "loss_csv_by_size_raw3_cleaned.csv"
OUTPUT_DIR = PAPER_ROOT / "results" / "figures" / "loss_curves"
OUTPUT_NAME = "pretraining_loss_by_size_2x2.pdf"
PREVIEW_NAME = "pretraining_loss_by_size_2x2.png"

SIZE_ORDER = ["194m", "296m", "436m", "528m"]
SIZE_TITLES = {
    "194m": "194M",
    "296m": "296M",
    "436m": "436M",
    "528m": "528M",
}
CURVE_ORDER = ["Baseline", "Residual Rotation", "Value-Space Rotation"]
DISPLAY_LABELS = {
    "Baseline": "Baseline",
    "Residual Rotation": "Attn Para-Rem.",
    "Value-Space Rotation": "V-Para Rem.",
}
LINESTYLES = {
    "Baseline": "-",
    "Residual Rotation": "--",
    "Value-Space Rotation": ":",
}
SIZE_COLORS = {
    "194m": "#1f77b4",
    "296m": "#ff7f0e",
    "436m": "#2ca02c",
    "528m": "#d62728",
}

# Match the paper-facing v2 figure geometry and panel layout.
FIGSIZE = (7.0, 5.2)
DPI = 240
X_TICK_STEP = 1000

TITLE_FONTSIZE = 10.5
LABEL_FONTSIZE = 10.5
TICK_FONTSIZE = 9.0
LEGEND_FONTSIZE = 8.2
LINE_WIDTH = 2.0


def load_rows() -> dict[tuple[str, str], list[tuple[float, float]]]:
    rows: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = row["model_size"].strip()
            curve = row["curve_name"].strip()
            try:
                step = float(row["step"])
                loss = float(row["val_loss"])
            except (KeyError, ValueError):
                continue
            rows[(size, curve)].append((step, loss))
    for key in list(rows):
        rows[key].sort()
    return rows


def main() -> None:
    rows = load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI, sharey=False)
    axes_flat = axes.flatten()

    for panel_idx, (ax, size) in enumerate(zip(axes_flat, SIZE_ORDER)):
        for curve in CURVE_ORDER:
            pts = rows.get((size, curve), [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                color=SIZE_COLORS.get(size, "#4C78A8"),
                linestyle=LINESTYLES.get(curve, "-"),
                linewidth=LINE_WIDTH,
                label=DISPLAY_LABELS[curve],
            )

        panel_label = chr(ord("a") + panel_idx)
        ax.set_xlabel(f"({panel_label}) {SIZE_TITLES.get(size, size)}", fontsize=LABEL_FONTSIZE)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, length=3.0, width=0.8)
        ax.grid(True, color="#d0d0d0", linestyle=":", linewidth=0.65, alpha=0.45)
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#cfcfcf",
            fontsize=LEGEND_FONTSIZE,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("val loss", fontsize=LABEL_FONTSIZE)

    fig.tight_layout(w_pad=1.2, h_pad=1.1)

    out_path = OUTPUT_DIR / OUTPUT_NAME
    preview_path = OUTPUT_DIR / PREVIEW_NAME
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(preview_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")
    print(f"[OK] wrote: {preview_path}")


if __name__ == "__main__":
    main()
