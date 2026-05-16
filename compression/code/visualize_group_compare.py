#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt


def load_run_csv(run_dir: Path):
    csv_path = run_dir / "layerwise_metrics.csv"
    if not csv_path.is_file():
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                {
                    "method": r["method"],
                    "component": r["component"],
                    "layer": int(r["layer"]),
                    "alpha": float(r["alpha"]),
                    "para_abs": float(r["para_abs"]),
                    "perp_abs": float(r["perp_abs"]),
                    "perp_ratio": float(r["perp_ratio"]),
                }
            )
    return rows


def select_component(rows, component):
    out = [r for r in rows if r["component"] == component]
    out.sort(key=lambda x: x["layer"])
    return out


def plot_metric(series_by_label, metric, component, out_path):
    plt.figure(figsize=(10, 4.5))
    for label, rows in series_by_label.items():
        xs = [r["layer"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=label)
    plt.xlabel("Layer")
    plt.ylabel(metric)
    plt.title(f"{component} | {metric}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Compare multiple layerwise run folders.")
    ap.add_argument("--run_dirs", type=str, required=True, help="Comma-separated run directories")
    ap.add_argument("--labels", type=str, default="", help="Comma-separated labels; default=folder names")
    ap.add_argument("--component", type=str, default="block_out", choices=["block_out", "attn_out", "mlp_out"])
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    run_dirs = [Path(x.strip()) for x in args.run_dirs.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else []
    if labels and len(labels) != len(run_dirs):
        raise ValueError("labels count must equal run_dirs count")
    if not labels:
        labels = [p.name for p in run_dirs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    for p, label in zip(run_dirs, labels):
        rows = load_run_csv(p)
        if rows is None:
            print(f"[WARN] skip missing csv: {p}")
            continue
        comp_rows = select_component(rows, args.component)
        if not comp_rows:
            print(f"[WARN] skip empty component={args.component}: {p}")
            continue
        series[label] = comp_rows

    if len(series) < 2:
        raise RuntimeError("Need at least 2 valid runs for comparison.")

    for metric in ("perp_ratio", "alpha", "para_abs", "perp_abs"):
        out_path = out_dir / f"compare_{args.component}_{metric}.png"
        plot_metric(series, metric, args.component, out_path)
    print(f"[INFO] saved figures: {out_dir}")


if __name__ == "__main__":
    main()
