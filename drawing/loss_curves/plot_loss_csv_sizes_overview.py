#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# File-first configuration
# =========================
DRAWING_DIR = Path(__file__).absolute().parent
PAPER_ROOT = DRAWING_DIR.parents[1]
INPUT_DIR = DRAWING_DIR / "data"
OUTPUT_DIR = PAPER_ROOT / "figs" / "loss_curves"
FIG_DPI = 220
FIGSIZE = (13.5, 9.0)
USE_LOG_Y = False
PLOT_DELTA = True
IGNORE_SMOOTH_COLUMNS = True


def _clean(text: str) -> str:
    return (text or "").replace("\ufeff", "").strip()


def _short_run_name(col: str) -> str:
    parts = [p for p in _clean(col).split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return _clean(col)


def _load_multirun_csv(path: Path) -> tuple[str, list[float], list[tuple[str, list[float]]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")

    x_name = _clean(rows[0][0]) or "step"
    all_cols = [_clean(c) for c in rows[0][1:]]
    kept_indices: list[int] = []
    run_names: list[str] = []
    for idx, col in enumerate(all_cols):
        if IGNORE_SMOOTH_COLUMNS and "(smooth)" in col:
            continue
        kept_indices.append(idx)
        run_names.append(_short_run_name(col))

    xs: list[float] = []
    buckets = [[] for _ in run_names]

    for row in rows[1:]:
        if len(row) < len(all_cols) + 1:
            continue
        try:
            x = float(_clean(row[0]))
        except ValueError:
            continue
        ys: list[float] = []
        ok = True
        for idx in kept_indices:
            raw = row[idx + 1]
            try:
                ys.append(float(_clean(raw)))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        xs.append(x)
        for bucket, y in zip(buckets, ys):
            bucket.append(y)

    named_series = list(zip(run_names, buckets))
    return x_name, xs, named_series


def _plot_grid(data: list[tuple[str, str, list[float], list[tuple[str, list[float]]]]], out_path: Path, delta: bool) -> None:
    n = len(data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE, squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, (size_label, x_name, xs, named_series) in zip(axes_flat, data):
        for name, ys in named_series:
            plot_ys = ys
            ylabel = "val loss"
            if delta:
                base = ys[0]
                plot_ys = [y - base for y in ys]
                ylabel = "val loss - first point"
            ax.plot(xs, plot_ys, linewidth=1.8, label=name)
        ax.set_title(size_label)
        ax.set_xlabel(x_name)
        ax.set_ylabel(ylabel)
        if USE_LOG_Y and not delta:
            ax.set_yscale("log")
        if delta:
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(frameon=True, fontsize=7, facecolor="white", edgecolor="#cfcfcf")

    title = "Loss Curves by Model Size"
    if delta:
        title += " (relative to first point)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    csv_paths = sorted(path for path in INPUT_DIR.glob("*.csv") if path.stem.endswith("m"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {INPUT_DIR}")

    loaded: list[tuple[str, str, list[float], list[tuple[str, list[float]]]]] = []
    for path in csv_paths:
        size_label = path.stem
        x_name, xs, named_series = _load_multirun_csv(path)
        named_series = [(name, ys) for name, ys in named_series if ys]
        if not xs or not named_series:
            print(f"[WARN] skip empty CSV: {path}")
            continue
        loaded.append((size_label, x_name, xs, named_series))
    if not loaded:
        raise SystemExit(f"No plottable size CSV files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_logy" if USE_LOG_Y else ""
    abs_path = OUTPUT_DIR / f"loss_csv_sizes_overview_absolute{suffix}.png"
    _plot_grid(loaded, abs_path, delta=False)
    print(f"[OK] wrote: {abs_path}")

    if PLOT_DELTA:
        delta_path = OUTPUT_DIR / f"loss_csv_sizes_overview_delta{suffix}.png"
        _plot_grid(loaded, delta_path, delta=True)
        print(f"[OK] wrote: {delta_path}")


if __name__ == "__main__":
    main()
