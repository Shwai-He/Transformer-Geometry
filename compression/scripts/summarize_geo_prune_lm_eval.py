#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _metric(results: dict, task: str, key: str) -> str:
    task_results = results.get("results", {}).get(task, {})
    value = task_results.get(key)
    return "" if value is None else str(value)


def _resolve_output_path(path_text: str) -> Path | None:
    path = Path(path_text)
    if path.exists():
        return path
    matches = sorted(path.parent.glob(f"{path.stem}_*.json"))
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize geo-prune lm-eval JSON outputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = []
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") not in {"ok", "exists"}:
                continue
            output_path = row.get("output_path", "")
            if not output_path:
                continue
            resolved_output_path = _resolve_output_path(output_path)
            if resolved_output_path is None:
                continue
            with resolved_output_path.open(encoding="utf-8") as jf:
                data = json.load(jf)
            task = row["task"]
            rows.append(
                {
                    "variant": row["variant"],
                    "strategy": row["strategy"],
                    "sparsity_type": row["sparsity_type"],
                    "sparsity_ratio": row["sparsity_ratio"],
                    "task": task,
                    "acc": _metric(data, task, "acc,none"),
                    "acc_norm": _metric(data, task, "acc_norm,none"),
                    "sample_len": _metric(data, task, "sample_len"),
                    "output_path": str(resolved_output_path),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "strategy",
                "sparsity_type",
                "sparsity_ratio",
                "task",
                "acc",
                "acc_norm",
                "sample_len",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
