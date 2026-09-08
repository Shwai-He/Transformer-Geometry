#!/usr/bin/env python3
"""Restore the submitted Figure-6 trajectories with paper-consistent labels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SIZE_ORDER = ("296m", "436m", "528m")
SIZE_COLORS = {
    "296m": "#ff7f0e",
    "436m": "#2ca02c",
    "528m": "#d62728",
}
METHODS = (
    ("Baseline", "Baseline", "-"),
    ("Residual Rotation", "Attn Para-Rem.", "--"),
    ("Value-Space Rotation", "V-Para Rem.", ":"),
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LOCAL_INPUT = SCRIPT_DIR / "loss_csv_by_size_raw3_cleaned.csv"
DEFAULT_INPUT = (
    LOCAL_INPUT
    if LOCAL_INPUT.is_file()
    else REPO_ROOT
    / "analysis/visualization/loss_curves/data/loss_csv_by_size_raw3_cleaned.csv"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "output"
    if LOCAL_INPUT.is_file()
    else REPO_ROOT / "results/figures/loss_curves_corrected"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            size = row["model_size"].lower()
            method = row["curve_name"]
            if size in SIZE_ORDER and any(method == item[0] for item in METHODS):
                grouped[size][method].append(
                    (float(row["step"]), float(row["val_loss"]))
                )

    for size in SIZE_ORDER:
        for source_name, _, _ in METHODS:
            points = sorted(grouped[size][source_name])
            if not points:
                raise ValueError(f"Missing {size}/{source_name} trajectory")
            grouped[size][source_name] = points
    return grouped


def plot(
    grouped: dict[str, dict[str, list[tuple[float, float]]]], output_dir: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), constrained_layout=False)

    for panel, (ax, size) in enumerate(zip(axes, SIZE_ORDER, strict=True)):
        color = SIZE_COLORS[size]
        for source_name, display_name, linestyle in METHODS:
            points = grouped[size][source_name]
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=display_name,
            )
        ax.set_xlabel(f"({chr(ord('a') + panel)}) {size.upper()}")
        ax.set_ylabel("Validation loss")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(frameon=True, loc="upper right")
        ax.tick_params(direction="out", length=3, width=0.8)

    fig.tight_layout(w_pad=1.6)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "pretraining_loss_by_size_restored.pdf"
    png_path = output_dir / "pretraining_loss_by_size_restored.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(read_rows(args.input), args.output_dir)


if __name__ == "__main__":
    main()
