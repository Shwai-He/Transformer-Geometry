#!/usr/bin/env python3
from __future__ import annotations

import csv
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
INPUT_CSV = INPUT_DIR / "296m.csv"
OUTPUT_DIR = PAPER_ROOT / "figs" / "loss_curves"
FIG_DPI = 220
FIGSIZE = (9.5, 5.8)
USE_LOG_Y = False


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
    run_names = [_short_run_name(c) for c in rows[0][1:]]
    xs: list[float] = []
    series = [[] for _ in run_names]

    for row in rows[1:]:
        if len(row) < len(run_names) + 1:
            continue
        try:
            x = float(_clean(row[0]))
        except ValueError:
            continue
        ys: list[float] = []
        ok = True
        for raw in row[1 : len(run_names) + 1]:
            try:
                ys.append(float(_clean(raw)))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        xs.append(x)
        for bucket, y in zip(series, ys):
            bucket.append(y)

    named_series = list(zip(run_names, series))
    return x_name, xs, named_series


def _plot_absolute(x_name: str, xs: list[float], named_series: list[tuple[str, list[float]]], out_path: Path) -> None:
    plt.figure(figsize=FIGSIZE)
    for name, ys in named_series:
        plt.plot(xs, ys, linewidth=2.0, label=name)
    plt.xlabel(x_name)
    plt.ylabel("val loss")
    plt.title("Validation Loss by Run")
    if USE_LOG_Y:
        plt.yscale("log")
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend(frameon=True, fontsize=8, facecolor="white", edgecolor="#cfcfcf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def _plot_delta(x_name: str, xs: list[float], named_series: list[tuple[str, list[float]]], out_path: Path) -> None:
    plt.figure(figsize=FIGSIZE)
    for name, ys in named_series:
        base = ys[0]
        deltas = [y - base for y in ys]
        plt.plot(xs, deltas, linewidth=2.0, label=name)
    plt.xlabel(x_name)
    plt.ylabel("val loss - first point")
    plt.title("Relative Validation-Loss Change by Run")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend(frameon=True, fontsize=8, facecolor="white", edgecolor="#cfcfcf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def main() -> None:
    input_csv = INPUT_CSV
    if not input_csv.exists():
        candidates = sorted(INPUT_DIR.glob("*.csv"))
        if not candidates:
            raise SystemExit(f"Missing INPUT_CSV: {INPUT_CSV} and no fallback CSVs in {INPUT_DIR}")
        input_csv = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[INFO] INPUT_CSV missing, fallback to latest CSV: {input_csv}")

    x_name, xs, named_series = _load_multirun_csv(input_csv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = input_csv.stem
    abs_path = OUTPUT_DIR / f"{stem}_multirun_absolute.png"
    delta_path = OUTPUT_DIR / f"{stem}_multirun_delta.png"

    _plot_absolute(x_name, xs, named_series, abs_path)
    _plot_delta(x_name, xs, named_series, delta_path)

    print("[OK] wrote plots:")
    print(abs_path)
    print(delta_path)


if __name__ == "__main__":
    main()
