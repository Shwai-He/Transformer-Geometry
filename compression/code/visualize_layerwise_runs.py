#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
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


def component_series(rows, metric):
    out = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["layer"]):
        out[r["component"]].append((r["layer"], r[metric]))
    return out


def plot_single(csv_path, out_dir):
    rows = load_csv(csv_path)
    method = rows[0]["method"]
    for metric in ("perp_ratio", "alpha", "para_abs", "perp_abs"):
        comp = component_series(rows, metric)
        plt.figure(figsize=(10, 4.5))
        for cname, pts in comp.items():
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=cname)
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.title(f"{method} | {metric}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"viz_single_{metric}.png")
        plt.savefig(save_path, dpi=180)
        plt.close()


def plot_compare(csv_paths, out_dir, component):
    series = {}
    for p in csv_paths:
        rows = load_csv(p)
        method = rows[0]["method"]
        by_layer = defaultdict(dict)
        for r in rows:
            if r["component"] != component:
                continue
            by_layer[r["layer"]] = r
        series[method] = by_layer

    for metric in ("perp_ratio", "alpha", "para_abs", "perp_abs"):
        plt.figure(figsize=(10, 4.5))
        for method, by_layer in series.items():
            xs = sorted(by_layer.keys())
            ys = [by_layer[i][metric] for i in xs]
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=method)
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.title(f"Compare ({component}) | {metric}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"viz_compare_{component}_{metric}.png")
        plt.savefig(save_path, dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize layerwise_metrics.csv outputs.")
    parser.add_argument("--mode", choices=["single", "compare"], required=True)
    parser.add_argument("--csv", type=str, default="", help="single mode: path to one csv")
    parser.add_argument("--csvs", type=str, default="", help="compare mode: comma-separated csv paths")
    parser.add_argument(
        "--component",
        type=str,
        default="block_out",
        choices=["block_out", "attn_out", "mlp_out"],
        help="compare mode: component to overlay",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "single":
        if not args.csv:
            raise ValueError("--csv is required in single mode")
        plot_single(args.csv, args.out_dir)
        return

    if not args.csvs:
        raise ValueError("--csvs is required in compare mode")
    csv_paths = [x.strip() for x in args.csvs.split(",") if x.strip()]
    if len(csv_paths) < 2:
        raise ValueError("compare mode requires at least 2 csv files")
    plot_compare(csv_paths, args.out_dir, args.component)


if __name__ == "__main__":
    main()
