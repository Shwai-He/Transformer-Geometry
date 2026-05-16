#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
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
FIGSIZE = (9, 5.6)
DROP_FIRST_POINT_FOR_ZOOM = True
CLIP_TO_COMMON_MAX_STEP = True
USE_LOG_X = False
USE_LOG_Y = True


@dataclass
class LossSeries:
    label: str
    x_name: str
    y_name: str
    x: list[float]
    y: list[float]
    source: Path


def _clean_header(text: str) -> str:
    return (text or "").replace("\ufeff", "").strip()


def _short_label(path: Path, y_name: str) -> str:
    stem = path.stem
    if y_name:
        tail = y_name.split("/")[-2:] if "/" in y_name else [y_name]
        y_short = "/".join(tail)
        return f"{stem[-15:]} | {y_short}"
    return stem


def _load_csv(path: Path) -> LossSeries:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")

    x_name = _clean_header(rows[0][0])
    y_name = _clean_header(rows[0][1])
    x: list[float] = []
    y: list[float] = []

    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            xv = float(str(row[0]).strip())
            yv = float(str(row[1]).strip())
        except ValueError:
            continue
        x.append(xv)
        y.append(yv)

    if not x:
        raise ValueError(f"No numeric rows found in {path}")

    return LossSeries(
        label=_short_label(path, y_name),
        x_name=x_name or "step",
        y_name=y_name or "loss",
        x=x,
        y=y,
        source=path,
    )


def _load_csv_series(path: Path) -> list[LossSeries]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows or len(rows[0]) <= 2:
        return [_load_csv(path)]

    x_name = _clean_header(rows[0][0]) or "step"
    y_names = [_clean_header(name) for name in rows[0][1:]]
    xs: list[float] = []
    buckets: list[list[float]] = [[] for _ in y_names]

    for row in rows[1:]:
        if len(row) < len(y_names) + 1:
            continue
        try:
            x = float(str(row[0]).strip())
            ys = [float(str(value).strip()) for value in row[1 : len(y_names) + 1]]
        except ValueError:
            continue
        xs.append(x)
        for bucket, y in zip(buckets, ys):
            bucket.append(y)

    series: list[LossSeries] = []
    for y_name, ys in zip(y_names, buckets):
        series.append(
            LossSeries(
                label=_short_label(path, y_name),
                x_name=x_name,
                y_name=y_name or "loss",
                x=xs,
                y=ys,
                source=path,
            )
        )
    return series


def _plot_combined(series_list: list[LossSeries], out_path: Path, zoom: bool) -> None:
    plt.figure(figsize=FIGSIZE)

    for series in series_list:
        xs = series.x
        ys = series.y
        if CLIP_TO_COMMON_MAX_STEP and series_list:
            common_max_x = min(max(s.x) for s in series_list)
            clipped = [(x, y) for x, y in zip(xs, ys) if x <= common_max_x]
            xs = [x for x, _ in clipped]
            ys = [y for _, y in clipped]
        if zoom and DROP_FIRST_POINT_FOR_ZOOM and len(xs) > 1:
            xs = xs[1:]
            ys = ys[1:]
        plt.plot(xs, ys, linewidth=2.0, label=series.label)

    plt.xlabel(series_list[0].x_name if series_list else "step")
    plt.ylabel("val loss")
    title = "296M Validation Loss Curves"
    if CLIP_TO_COMMON_MAX_STEP and series_list:
        title += f" (clipped to common max step={int(min(max(s.x) for s in series_list))})"
    if zoom:
        title += " (zoom without first point)"
    plt.title(title)
    if USE_LOG_X:
        plt.xscale("log")
    if USE_LOG_Y:
        plt.yscale("log")
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend(frameon=True, fontsize=9, facecolor="white", edgecolor="#cfcfcf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def main() -> None:
    csv_paths = [INPUT_DIR / "296m.csv"]
    if not csv_paths[0].exists():
        csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {INPUT_DIR}")

    series_list: list[LossSeries] = []
    for path in csv_paths:
        series_list.extend(_load_csv_series(path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix_parts = []
    if CLIP_TO_COMMON_MAX_STEP:
        suffix_parts.append("clipped")
    if USE_LOG_X:
        suffix_parts.append("logx")
    if USE_LOG_Y:
        suffix_parts.append("logy")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    full_path = OUTPUT_DIR / f"loss_curves_296m_full{suffix}.png"
    zoom_path = OUTPUT_DIR / f"loss_curves_296m_zoom{suffix}.png"

    _plot_combined(series_list, full_path, zoom=False)
    _plot_combined(series_list, zoom_path, zoom=True)

    print("[OK] wrote plots:")
    print(full_path)
    print(zoom_path)


if __name__ == "__main__":
    main()
