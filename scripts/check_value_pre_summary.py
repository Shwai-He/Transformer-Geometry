#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CORE_METRICS = (
    "value_pre_para_ratio",
    "value_pre_perp_ratio",
    "value_pre_para_perp_ratio",
    "value_pre_projected_para_over_z",
    "value_pre_scale10_projected_delta_over_z",
    "value_pre_scale10_post_rms_delta_over_normed_z",
    "value_pre_scale10_rms_damping_ratio",
    "value_pre_scale10_cos_normed_z_edit",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect value_pre summary CSV outputs.")
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="results/value_pre_ratio/Qwen3-4B-Instruct-2507/summary",
        help="Directory containing manifest.json and value_pre_*_summary.csv.",
    )
    args = parser.parse_args()

    root = Path(args.summary_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Summary directory not found: {root}")

    print("files:")
    for path in sorted(root.iterdir()):
        print(f"  {path.name}")

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("\ninputs:")
        for path in manifest.get("inputs", []):
            print(f"  {path}")
    else:
        print("\ninputs:\n  <manifest.json missing>")

    prompt_path = root / "value_pre_prompt_summary.csv"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt summary not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\nrows: {len(rows)}")
    print("columns:", list(rows[0].keys()) if rows else [])
    if not rows:
        return

    sources = sorted(set(row["source"] for row in rows))
    print("\nsources:")
    for source in sources:
        print(f"  {source}")

    print("\ncore metrics by source:")
    for source in sources:
        sub = [row for row in rows if row["source"] == source]
        lengths = [int(float(row["prompt_token_count"])) for row in sub if row.get("prompt_token_count")]
        if lengths:
            length_desc = f"{min(lengths)}..{max(lengths)}"
            prompt_count = len(set(lengths))
        else:
            length_desc = "<missing>"
            prompt_count = 0
        print(f"\n{source}: prompt_token_count range={length_desc}, distinct_lengths={prompt_count}, rows={len(sub)}")

        for metric in CORE_METRICS:
            metric_rows = [row for row in sub if row["metric"] == metric]
            if not metric_rows:
                print(f"  {metric}: MISSING")
                continue
            n_total = sum(int(float(row["n"])) for row in metric_rows)
            mean_avg = sum(float(row["mean"]) for row in metric_rows) / len(metric_rows)
            median_avg = sum(float(row["median"]) for row in metric_rows) / len(metric_rows)
            p10_avg = sum(float(row["p10"]) for row in metric_rows) / len(metric_rows)
            p90_avg = sum(float(row["p90"]) for row in metric_rows) / len(metric_rows)
            print(
                f"  {metric}: "
                f"n_total={n_total} "
                f"mean_avg={mean_avg:.6g} "
                f"median_avg={median_avg:.6g} "
                f"p10_avg={p10_avg:.6g} "
                f"p90_avg={p90_avg:.6g}"
            )


if __name__ == "__main__":
    main()
