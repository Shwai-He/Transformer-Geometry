#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# =========================
# File-first configuration
# =========================
DRAWING_DIR = Path(__file__).absolute().parent
PAPER_ROOT = DRAWING_DIR.parents[1]
CLEANED_CSV = DRAWING_DIR / "data" / "loss_csv_by_size_raw3_cleaned.csv"
RAW_LOSS_DIR = DRAWING_DIR / "data"
OUTPUT_DIR = PAPER_ROOT / "figs" / "loss_curves"
FIG_DPI = 220
USE_LOG_Y = False
PLOT_DELTA = True
FIGSIZE = (11.0, 8.0)
ONE_ROW_FIGSIZE = (14.5, 3.1)
LINESTYLES = {
    "Baseline": "-",
    "Residual Rotation": "--",
    "Value-Space Rotation": ":",
}
SIZE_ORDER = ["194m", "296m", "436m", "528m"]
SIZE_COLORS = {
    "194m": "#1f77b4",
    "296m": "#ff7f0e",
    "436m": "#2ca02c",
    "528m": "#d62728",
}
FIXED_MIN_STEP = 600
FIXED_MAX_STEP = 4242


def _plot_from_cleaned_csv(df: pd.DataFrame, out_prefix: str) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(_plot_one_row_absolute(df, out_prefix))

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, squeeze=False)
    axes_flat = axes.flatten()

    for ax, size in zip(axes_flat, SIZE_ORDER):
        sdf = df[df["model_size"] == size].copy()
        if sdf.empty:
            ax.axis("off")
            continue
        for curve_name in ["Baseline", "Residual Rotation", "Value-Space Rotation"]:
            cdf = sdf[sdf["curve_name"] == curve_name].sort_values("step")
            if cdf.empty:
                continue
            ax.plot(
                cdf["step"],
                cdf["val_loss"],
                linewidth=2.0,
                linestyle=LINESTYLES.get(curve_name, "-"),
                color=SIZE_COLORS.get(size),
                label=curve_name,
            )
        clip_min = sdf["clip_min_step"].iloc[0]
        clip_max = sdf["clip_max_step"].iloc[0]
        ax.set_title(f"{size} ({int(float(clip_min))} to {int(float(clip_max))})")
        ax.set_xlabel(sdf["x_name"].iloc[0])
        ax.set_ylabel("val loss")
        if USE_LOG_Y:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(frameon=True, fontsize=8, facecolor="white", edgecolor="#cfcfcf")

    fig.suptitle("Loss Curves by Size (cleaned CSV)", y=0.995)
    fig.tight_layout()
    out = OUTPUT_DIR / f"{out_prefix}_absolute.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    outputs.append(out)

    if PLOT_DELTA:
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, squeeze=False)
        axes_flat = axes.flatten()
        for ax, size in zip(axes_flat, SIZE_ORDER):
            sdf = df[df["model_size"] == size].copy()
            if sdf.empty:
                ax.axis("off")
                continue
            for curve_name in ["Baseline", "Residual Rotation", "Value-Space Rotation"]:
                cdf = sdf[sdf["curve_name"] == curve_name].sort_values("step")
                if cdf.empty:
                    continue
                base = float(cdf["val_loss"].iloc[0])
                ax.plot(
                    cdf["step"],
                    cdf["val_loss"] - base,
                    linewidth=2.0,
                    linestyle=LINESTYLES.get(curve_name, "-"),
                    color=SIZE_COLORS.get(size),
                    label=curve_name,
                )
            clip_min = sdf["clip_min_step"].iloc[0]
            clip_max = sdf["clip_max_step"].iloc[0]
            ax.set_title(f"{size} ({int(float(clip_min))} to {int(float(clip_max))})")
            ax.set_xlabel(sdf["x_name"].iloc[0])
            ax.set_ylabel("val loss - first point")
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(frameon=True, fontsize=8, facecolor="white", edgecolor="#cfcfcf")

        fig.suptitle("Loss Delta by Size (cleaned CSV)", y=0.995)
        fig.tight_layout()
        out = OUTPUT_DIR / f"{out_prefix}_delta.png"
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
        outputs.append(out)

    return outputs


def _plot_one_row_absolute(df: pd.DataFrame, out_prefix: str) -> list[Path]:
    fig, axes = plt.subplots(1, len(SIZE_ORDER), figsize=ONE_ROW_FIGSIZE, squeeze=False)
    axes_flat = axes.flatten()

    for ax, size in zip(axes_flat, SIZE_ORDER):
        sdf = df[df["model_size"] == size].copy()
        if sdf.empty:
            ax.axis("off")
            continue
        for curve_name in ["Baseline", "Residual Rotation", "Value-Space Rotation"]:
            cdf = sdf[sdf["curve_name"] == curve_name].sort_values("step")
            if cdf.empty:
                continue
            ax.plot(
                cdf["step"],
                cdf["val_loss"],
                linewidth=2.0,
                linestyle=LINESTYLES.get(curve_name, "-"),
                color=SIZE_COLORS.get(size),
                label=curve_name,
            )
        ax.set_xlabel(size.upper())
        ax.set_ylabel("val loss")
        if USE_LOG_Y:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout(w_pad=1.6)
    out_png = OUTPUT_DIR / f"{out_prefix}_one_row_absolute.png"
    out_pdf = OUTPUT_DIR / f"{out_prefix}_one_row_absolute.pdf"
    fig.savefig(out_png, dpi=FIG_DPI)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def _clean(text: str) -> str:
    return (text or "").replace("\ufeff", "").strip()


def _short_run_name(col: str) -> str:
    parts = [p for p in _clean(col).split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return _clean(col)


def _load_raw_size_series(csv_path: Path) -> pd.DataFrame:
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [_clean(c) for c in rows[0]]
    x_name = header[0] or "step"
    size = csv_path.stem
    raw_cols = [c for c in header[1:] if "(smooth)" not in c]
    series_rows = []
    display_names = ["Baseline", "Residual Rotation", "Value-Space Rotation"]
    for idx, col in enumerate(raw_cols[:3]):
        curve_name = display_names[idx] if idx < len(display_names) else f"Curve {idx+1}"
        col_idx = header.index(col)
        for row in rows[1:]:
            if len(row) <= col_idx:
                continue
            try:
                step = float(_clean(row[0]))
                val = float(_clean(row[col_idx]))
            except ValueError:
                continue
            if not (FIXED_MIN_STEP <= step <= FIXED_MAX_STEP):
                continue
            series_rows.append(
                {
                    "model_size": size,
                    "curve_name": curve_name,
                    "original_run_name": _short_run_name(col),
                    "x_name": x_name,
                    "step": step,
                    "val_loss": val,
                    "clip_min_step": FIXED_MIN_STEP,
                    "clip_max_step": FIXED_MAX_STEP,
                }
            )
    return pd.DataFrame(series_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CLEANED_CSV.exists():
        df = pd.read_csv(CLEANED_CSV)
        outputs = _plot_from_cleaned_csv(df, "loss_csv_by_size_raw3_cleaned_from_script")
        print("[INFO] source=cleaned_csv")
        for path in outputs:
            print(path)
        return

    csv_paths = [RAW_LOSS_DIR / f"{size}.csv" for size in SIZE_ORDER if (RAW_LOSS_DIR / f"{size}.csv").exists()]
    if not csv_paths:
        raise SystemExit(
            "Cleaned CSV not found and no raw size CSVs found under "
            f"{RAW_LOSS_DIR}"
        )
    frames = [_load_raw_size_series(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    outputs = _plot_from_cleaned_csv(df, "loss_csv_by_size_raw3_fallback_raw")
    print("[INFO] source=raw_csv_fallback")
    for path in outputs:
        print(path)
    return


if __name__ == "__main__":
    main()
