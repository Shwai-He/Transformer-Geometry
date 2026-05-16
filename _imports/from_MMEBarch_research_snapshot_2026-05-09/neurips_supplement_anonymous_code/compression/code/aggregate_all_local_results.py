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


def parse_focus_layer(method: str):
    m = LAYER_PAT.search(method or "")
    return int(m.group(1)) if m else None


def infer_setting(method: str):
    li = parse_focus_layer(method)
    if li is None:
        return method
    return method[: method.rfind(f"_l{li}")]


def main():
    ap = argparse.ArgumentParser("Aggregate all local run CSVs into one master TSV.")
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--out_tsv", type=str, required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / "layerwise_metrics.csv"
        if not csv_path.is_file():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            data = list(rd)
        if not data:
            continue
        method = data[0].get("method", "")
        focus_layer = parse_focus_layer(method)
        setting = infer_setting(method)
        for r in data:
            row = {
                "setting": setting,
                "method": method,
                "focus_layer": focus_layer if focus_layer is not None else "",
                "component": r["component"],
                "layer": int(r["layer"]),
                "count": float(r.get("count", 0.0)),
                "run_dir": str(sub),
            }
            for field in METRIC_FIELDS:
                fallback = METRIC_ALIASES.get(field)
                row[field] = float(r.get(field, r.get(fallback, float("nan")) if fallback else float("nan")))
            rows.append(row)

    rows.sort(key=lambda x: (x["setting"], str(x["focus_layer"]), x["component"]))
    if not rows:
        raise RuntimeError(f"No layerwise_metrics.csv found under: {runs_root}")

    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "method",
                "focus_layer",
                "component",
                "layer",
                *METRIC_FIELDS,
                "count",
                "run_dir",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[INFO] wrote master summary: {out_tsv}")
    print(f"[INFO] rows={len(rows)}")


if __name__ == "__main__":
    main()
