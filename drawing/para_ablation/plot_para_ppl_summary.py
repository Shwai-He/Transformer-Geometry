#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# File-first configuration.
DRAWING_DIR = Path(__file__).absolute().parent
PAPER_ROOT = DRAWING_DIR.parents[1]
INPUT_TSV = DRAWING_DIR / "data" / "para_ppl_summary.tsv"
OUTPUT_DIR = PAPER_ROOT / "figs" / "para_ablation"
FIG_DPI = 260


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            rows.append({k: float(v) for k, v in raw.items()})
    rows.sort(key=lambda r: r["para_scale"])
    return rows


def _values(rows: list[dict[str, float]], *keys: str) -> list[float]:
    for key in keys:
        if rows and key in rows[0]:
            return [r[key] for r in rows]
    raise KeyError(f"None of these columns were found: {', '.join(keys)}")


def _plot_overview(rows: list[dict[str, float]], out_path: Path) -> None:
    xs = [r["para_scale"] for r in rows]
    delta_xsa = _values(rows, "mean_per_text_delta_ppl_xsa", "avg_delta_ppl_xsa")
    delta_attn = _values(rows, "mean_per_text_delta_ppl_residual_attn", "avg_delta_ppl_residual_attn")
    delta_mlp = _values(rows, "mean_per_text_delta_ppl_residual_mlp", "avg_delta_ppl_residual_mlp")
    delta_both = _values(rows, "mean_per_text_delta_ppl_residual_both", "avg_delta_ppl_residual_both")

    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.plot(xs, delta_xsa, marker="o", linewidth=2.6, markersize=6.5, label="Value-Based", color="#1f77b4")
    ax.plot(xs, delta_attn, marker="s", linewidth=2.1, markersize=5.8, linestyle="--", label="Residual(attn)", color="#2ca02c")
    ax.plot(xs, delta_mlp, marker="^", linewidth=2.1, markersize=5.8, linestyle="-.", label="Residual(MLP)", color="#d62728")
    ax.plot(xs, delta_both, marker="D", linewidth=2.1, markersize=5.6, linestyle=":", label="Residual(both)", color="#9467bd")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Para. Scale")
    ax.set_ylabel(r"$\Delta \mathrm{PPL}$")
    ax.tick_params(axis="both", which="major", length=4.5, width=0.9)
    ax.grid(alpha=0.28, linestyle=":", linewidth=0.9)
    ax.legend(frameon=True, ncol=2, facecolor="white", edgecolor="#cfcfcf", handlelength=2.1, columnspacing=1.1, borderpad=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_xsa_zoom(rows: list[dict[str, float]], out_path: Path) -> None:
    xs = [r["para_scale"] for r in rows]
    delta_xsa = _values(rows, "mean_per_text_delta_ppl_xsa", "avg_delta_ppl_xsa")

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(xs, delta_xsa, marker="o", linewidth=2.6, markersize=6.5, color="#1f77b4")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Para. Scale")
    ax.set_ylabel(r"$\Delta \mathrm{PPL}$")
    ax.tick_params(axis="both", which="major", length=4.5, width=0.9)
    ax.grid(alpha=0.28, linestyle=":", linewidth=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = _load_rows(INPUT_TSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_overview(rows, OUTPUT_DIR / "para_scale_ablation_overview.png")
    _plot_xsa_zoom(rows, OUTPUT_DIR / "para_scale_ablation_xsa_zoom.png")
    print(f"[INFO] Wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
