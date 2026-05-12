#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path


LAYER_PAT = re.compile(r"_l(\d+)$")
METRIC_FIELDS = [
    "alpha",
    "alpha_std",
    "para_abs",
    "para_abs_std",
    "perp_abs",
    "perp_abs_std",
    "x_norm",
    "x_norm_std",
    "delta_norm",
    "delta_norm_std",
    "delta_over_x",
    "delta_over_x_std",
    "para_over_x",
    "para_over_x_std",
    "perp_over_x",
    "perp_over_x_std",
    "perp_ratio",
    "perp_ratio_std",
    "base_update_norm",
    "base_update_norm_std",
    "error_norm",
    "error_norm_std",
    "error_over_base_update",
    "error_over_base_update_std",
    "para_over_base_update",
    "para_over_base_update_std",
    "perp_over_base_update",
    "perp_over_base_update_std",
    "perp_ratio_to_base_update",
    "perp_ratio_to_base_update_std",
    "hidden_state_norm",
    "hidden_state_norm_std",
    "base_update_over_hidden_state",
    "base_update_over_hidden_state_std",
]
METRIC_ALIASES = {
    "base_update_norm": "x_norm",
    "base_update_norm_std": "x_norm_std",
    "error_norm": "delta_norm",
    "error_norm_std": "delta_norm_std",
    "error_over_base_update": "delta_over_x",
    "error_over_base_update_std": "delta_over_x_std",
    "para_over_base_update": "para_over_x",
    "para_over_base_update_std": "para_over_x_std",
    "perp_over_base_update": "perp_over_x",
    "perp_over_base_update_std": "perp_over_x_std",
    "perp_ratio_to_base_update": "perp_ratio",
    "perp_ratio_to_base_update_std": "perp_ratio_std",
}


def parse_layer(method: str):
    m = LAYER_PAT.search(method)
    return int(m.group(1)) if m else None


def load_csv(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser("Summarize local-flip runs for one setting; no plotting.")
    ap.add_argument("--runs_root", type=str, required=True, help="Directory containing run subdirs")
    ap.add_argument("--method_prefix", type=str, required=True, help="Base method prefix, e.g. wanda_2_4")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect one row per focus layer, per component.
    rows = []
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / "layerwise_metrics.csv"
        if not csv_path.is_file():
            continue
        data = load_csv(csv_path)
        if not data:
            continue
        method = data[0].get("method", "")
        if not method.startswith(args.method_prefix + "_l"):
            continue
        focus_layer = parse_layer(method)
        if focus_layer is None:
            continue

        for r in data:
            layer = int(r["layer"])
            if layer != focus_layer:
                continue
            row = {
                "method": method,
                "focus_layer": focus_layer,
                "component": r["component"],
                "count": float(r.get("count", 0.0)),
                "run_dir": str(sub),
            }
            for field in METRIC_FIELDS:
                fallback = METRIC_ALIASES.get(field)
                row[field] = float(r.get(field, r.get(fallback, float("nan")) if fallback else float("nan")))
            rows.append(row)

    rows.sort(key=lambda x: (x["focus_layer"], x["component"]))
    if not rows:
        raise RuntimeError(
            f"No local-flip rows found for method_prefix={args.method_prefix} under {runs_root}"
        )

    out_tsv = out_dir / f"{args.method_prefix}_local_flip_summary.tsv"
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "focus_layer",
                "component",
                *METRIC_FIELDS,
                "count",
                "run_dir",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[INFO] wrote summary: {out_tsv}")


if __name__ == "__main__":
    main()
