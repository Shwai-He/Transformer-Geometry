#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ===== File-first config (edit here) =====
USE_CSV = True

# Option A: load local CSV
CSV_PATH = "/path/to/history.csv"

# Option B: fetch from W&B API
ENTITY = ""
PROJECT = ""
RUN_ID = ""
KEYS = ["_step", "train/loss", "val/loss", "train/lr"]
SAMPLES = 200000

# Plot/output
X_COL = "_step"
Y_COLS = ["train/loss", "val/loss", "train/lr"]
OUT_DIR = "representation-analysis/outputs/wandb_exports"


def _safe_name(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-+' else '_' for c in s)


def fetch_wandb_history(entity: str, project: str, run_id: str, keys: list[str], samples: int) -> pd.DataFrame:
    import wandb  # lazy import

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    df = run.history(keys=keys, pandas=True, samples=samples)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Empty history for run {run_id}")
    return df


def plot_metrics(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(len(y_cols), 1, figsize=(10, 3.2 * len(y_cols)), sharex=True)
    if len(y_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, y_cols):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Missing column: {col}", ha='center', va='center')
            ax.set_ylabel(col)
            continue
        ax.plot(df[x_col], df[col], color='tab:blue', linewidth=1.2)
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(x_col)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    out_dir = Path(OUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if USE_CSV:
        if not CSV_PATH:
            raise SystemExit("USE_CSV=True but CSV_PATH is empty.")
        csv_file = Path(CSV_PATH).expanduser().resolve()
        df = pd.read_csv(csv_file)
        tag = _safe_name(csv_file.stem)
        title = f"CSV metrics: {csv_file.name}"
    else:
        if not (ENTITY and PROJECT and RUN_ID):
            raise SystemExit("Need ENTITY/PROJECT/RUN_ID when USE_CSV=False.")
        df = fetch_wandb_history(ENTITY, PROJECT, RUN_ID, keys=KEYS, samples=SAMPLES)
        csv_path = out_dir / f"{_safe_name(PROJECT)}-{_safe_name(RUN_ID)}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        tag = _safe_name(f"{PROJECT}-{RUN_ID}")
        title = f"W&B run: {ENTITY}/{PROJECT}/{RUN_ID}"

    x_col = X_COL
    if x_col not in df.columns:
        if '_step' in df.columns:
            x_col = '_step'
        else:
            x_col = df.columns[0]

    y_cols = Y_COLS
    out_png = out_dir / f"{tag}_metrics.png"
    plot_metrics(df=df, x_col=x_col, y_cols=y_cols, title=title, out_png=out_png)
    print(f"Saved figure: {out_png}")


if __name__ == '__main__':
    main()
