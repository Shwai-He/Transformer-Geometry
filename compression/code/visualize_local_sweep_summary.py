#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


LAYER_PAT = re.compile(r"_l(\d+)$")


def parse_layer_from_method(method_name):
    m = LAYER_PAT.search(method_name)
    if not m:
        return None
    return int(m.group(1))


def load_all_csvs(root_dir):
    csvs = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        cp = os.path.join(p, "layerwise_metrics.csv")
        if os.path.isfile(cp):
            csvs.append(cp)
    return sorted(csvs)


def extract_focus_row(csv_path, component):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    method = rows[0]["method"]
    focus_layer = parse_layer_from_method(method)
    if focus_layer is None:
        return None
    for r in rows:
        if r["component"] == component and int(r["layer"]) == focus_layer:
            return {
                "method": method,
                "focus_layer": focus_layer,
                "alpha": float(r["alpha"]),
                "para_abs": float(r["para_abs"]),
                "perp_abs": float(r["perp_abs"]),
                "perp_ratio": float(r["perp_ratio"]),
                "csv_path": csv_path,
            }
    return None


def write_summary_tsv(rows, out_path):
    keys = ["method", "focus_layer", "alpha", "para_abs", "perp_abs", "perp_ratio", "csv_path"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_rows(rows, out_dir, component):
    rows = sorted(rows, key=lambda x: x["focus_layer"])
    xs = [r["focus_layer"] for r in rows]
    for metric in ("perp_ratio", "alpha", "para_abs", "perp_abs"):
        ys = [r[metric] for r in rows]
        plt.figure(figsize=(10, 4.5))
        plt.plot(xs, ys, marker="o", markersize=4, linewidth=1.4)
        plt.xlabel("Focus Layer")
        plt.ylabel(metric)
        plt.title(f"Local Sweep Summary | {component} | {metric}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"viz_local_sweep_{component}_{metric}.png")
        plt.savefig(save_path, dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize local sweep runs into layer-index curves.")
    parser.add_argument("--runs_root", type=str, required=True, help="Directory containing run subdirs.")
    parser.add_argument(
        "--component",
        type=str,
        default="block_out",
        choices=["block_out", "attn_out", "mlp_out"],
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_paths = load_all_csvs(args.runs_root)
    if not csv_paths:
        raise RuntimeError(f"No run subdirs with layerwise_metrics.csv found in: {args.runs_root}")

    rows = []
    for cp in csv_paths:
        r = extract_focus_row(cp, args.component)
        if r is not None:
            rows.append(r)
    if not rows:
        raise RuntimeError("No local-sweep rows found. Ensure method names end with _l<layer>.")

    rows = sorted(rows, key=lambda x: x["focus_layer"])
    write_summary_tsv(rows, os.path.join(args.out_dir, f"local_sweep_{args.component}_summary.tsv"))
    plot_rows(rows, args.out_dir, args.component)
    print(f"Saved summary + figures to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
