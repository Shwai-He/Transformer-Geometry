#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DRAWING_DIR = Path(__file__).absolute().parent
PAPER_ROOT = DRAWING_DIR.parents[1]
INPUT_DIR = DRAWING_DIR / "data"
OUTPUT_DIR = PAPER_ROOT / "figs" / "loss_curves"
FIG_DPI = 220
PLOT_MIN_ITER = 1000.0
PLOT_MAX_ITER = 6000.0
FIGSIZE = (6.2, 3.4)
LINEWIDTH = 2.15
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 9
LINESTYLES = {
    "Baseline": "-",
    "VPR / fixed scale": ":",
    "Trainable parallel scale": "--",
}
CURVE_COLORS = {
    "Baseline": "#4c4c4c",
    "VPR / fixed scale": "#1f77b4",
    "Trainable parallel scale": "#d62728",
}

RUNS = [
    ("1p4_baseline.csv", "Baseline", 1.0),
    ("1p4_xsa.csv", "VPR / fixed scale", 1.0),
    ("1p4_gate.csv", "Trainable parallel scale", 0.1),
]


def _clean(text: str) -> str:
    return (text or "").replace("\ufeff", "").strip()


def _load_single_run(path: Path) -> tuple[str, list[float], list[float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Expected two-column loss CSV: {path}")

    x_name = _clean(rows[0][0]) or "iter"
    xs: list[float] = []
    ys: list[float] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            x = float(_clean(row[0]))
            y = float(_clean(row[1]))
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"No numeric rows found in {path}")
    return x_name, xs, ys


def _make_common_range(raw_series: list[tuple[str, list[float], list[float]]]) -> tuple[tuple[float, float], list[tuple[str, list[float], list[float]]]]:
    max_start = max(xs[0] for _, xs, _ in raw_series)
    min_end = min(xs[-1] for _, xs, _ in raw_series)
    start = max(PLOT_MIN_ITER, max_start)
    end = min(min_end, PLOT_MAX_ITER)
    if end <= start:
        raise ValueError(f"No overlapping plot range: start={start}, end={end}")
    cropped: list[tuple[str, list[float], list[float]]] = []
    for label, xs, ys in raw_series:
        kept = [(x, y) for x, y in zip(xs, ys) if start <= x <= end]
        if not kept:
            raise ValueError(f"No points for {label} in range {start}..{end}")
        cropped.append((label, [x for x, _ in kept], [y for _, y in kept]))
    return (start, end), cropped


def _plot_absolute(series: list[tuple[str, list[float], list[float]]], xlim: tuple[float, float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, ys in series:
        ax.plot(
            xs,
            ys,
            linewidth=LINEWIDTH,
            linestyle=LINESTYLES.get(label, "-"),
            color=CURVE_COLORS.get(label),
            label=label,
        )
    ax.set_xlim(*xlim)
    ax.set_xlabel("Training iteration", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Train loss", fontsize=LABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, facecolor="white", edgecolor="#cfcfcf", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_delta(series: list[tuple[str, list[float], list[float]]], xlim: tuple[float, float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, ys in series:
        base = ys[0]
        ax.plot(
            xs,
            [y - base for y in ys],
            linewidth=LINEWIDTH,
            linestyle=LINESTYLES.get(label, "-"),
            color=CURVE_COLORS.get(label),
            label=label,
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Training iteration", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f"Train loss change from {int(PLOT_MIN_ITER)} iter", fontsize=LABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, facecolor="white", edgecolor="#cfcfcf", loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_summary(series: list[tuple[str, list[float], list[float]]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "n_points", "first_iter", "last_iter", "first_loss", "last_loss", "min_loss"])
        for label, xs, ys in series:
            writer.writerow([label, len(xs), xs[0], xs[-1], ys[0], ys[-1], min(ys)])


def main() -> None:
    raw_series: list[tuple[str, list[float], list[float]]] = []
    for filename, label, x_scale in RUNS:
        path = INPUT_DIR / filename
        if not path.exists():
            raise SystemExit(f"Missing input CSV: {path}")
        _, xs, ys = _load_single_run(path)
        xs = [x * x_scale for x in xs]
        raw_series.append((label, xs, ys))

    xlim, series = _make_common_range(raw_series)
    print(f"[INFO] common xlim: {int(xlim[0])}..{int(xlim[1])}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    abs_path = OUTPUT_DIR / "loss_1p4_gate_vs_xsa_absolute.png"
    delta_path = OUTPUT_DIR / "loss_1p4_gate_vs_xsa_delta.png"
    summary_path = OUTPUT_DIR / "loss_1p4_gate_vs_xsa_summary.csv"

    _plot_absolute(series, xlim, abs_path)
    _plot_delta(series, xlim, delta_path)
    _write_summary(series, summary_path)

    print("[OK] wrote:")
    print(abs_path)
    print(delta_path)
    print(summary_path)


if __name__ == "__main__":
    main()
