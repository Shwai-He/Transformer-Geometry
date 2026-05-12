#!/usr/bin/env python3
import argparse
import csv
import os
import re

import matplotlib.pyplot as plt


LAYER_PAT = re.compile(r"_l(\d+)$")


def parse_layer_from_method(method_name):
    m = LAYER_PAT.search(method_name)
    return int(m.group(1)) if m else None


def load_rows_from_root(runs_root, component):
    rows = []
    for run_name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue
        csv_path = os.path.join(run_dir, "layerwise_metrics.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        method = data[0]["method"]
        layer = parse_layer_from_method(method)
        if layer is None:
            continue
        hit = None
        for r in data:
            if r["component"] == component and int(r["layer"]) == layer:
                hit = r
                break
        if hit is None:
            continue
        rows.append(
            {
                "focus_layer": layer,
                "alpha": float(hit["alpha"]),
                "para_abs": float(hit["para_abs"]),
                "perp_abs": float(hit["perp_abs"]),
                "perp_ratio": float(hit["perp_ratio"]),
            }
        )
    rows.sort(key=lambda x: x["focus_layer"])
    return rows


def to_map(rows):
    return {r["focus_layer"]: r for r in rows}


def save_tsv(rows_a, rows_b, label_a, label_b, out_path):
    layers = sorted(set([r["focus_layer"] for r in rows_a] + [r["focus_layer"] for r in rows_b]))
    ma = to_map(rows_a)
    mb = to_map(rows_b)
    fields = [
        "focus_layer",
        f"{label_a}_alpha",
        f"{label_a}_para_abs",
        f"{label_a}_perp_abs",
        f"{label_a}_perp_ratio",
        f"{label_b}_alpha",
        f"{label_b}_para_abs",
        f"{label_b}_perp_abs",
        f"{label_b}_perp_ratio",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for li in layers:
            ra = ma.get(li)
            rb = mb.get(li)
            row = {"focus_layer": li}
            for m in ("alpha", "para_abs", "perp_abs", "perp_ratio"):
                row[f"{label_a}_{m}"] = "" if ra is None else ra[m]
                row[f"{label_b}_{m}"] = "" if rb is None else rb[m]
            w.writerow(row)


def plot_compare(rows_a, rows_b, label_a, label_b, out_dir, component):
    for metric in ("perp_ratio", "alpha", "para_abs", "perp_abs"):
        plt.figure(figsize=(10, 4.5))
        if rows_a:
            xa = [r["focus_layer"] for r in rows_a]
            ya = [r[metric] for r in rows_a]
            plt.plot(xa, ya, marker="o", markersize=4, linewidth=1.4, label=label_a)
        if rows_b:
            xb = [r["focus_layer"] for r in rows_b]
            yb = [r[metric] for r in rows_b]
            plt.plot(xb, yb, marker="o", markersize=4, linewidth=1.4, label=label_b)
        plt.xlabel("Focus Layer")
        plt.ylabel(metric)
        plt.title(f"Local Sweep Compare | {component} | {metric}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"viz_local_compare_{component}_{metric}.png"), dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare two local sweep result roots.")
    parser.add_argument("--runs_root_a", type=str, required=True, help="e.g. prune local sweep outputs root")
    parser.add_argument("--runs_root_b", type=str, required=True, help="e.g. quant local sweep outputs root")
    parser.add_argument("--label_a", type=str, default="prune")
    parser.add_argument("--label_b", type=str, default="quant")
    parser.add_argument("--component", choices=["block_out", "attn_out", "mlp_out"], default="block_out")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows_a = load_rows_from_root(args.runs_root_a, args.component)
    rows_b = load_rows_from_root(args.runs_root_b, args.component)
    if not rows_a and not rows_b:
        raise RuntimeError("No valid local sweep rows found in both roots.")

    save_tsv(
        rows_a,
        rows_b,
        args.label_a,
        args.label_b,
        os.path.join(args.out_dir, f"local_compare_{args.component}.tsv"),
    )
    plot_compare(rows_a, rows_b, args.label_a, args.label_b, args.out_dir, args.component)
    print(f"Saved compare TSV + figures to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
