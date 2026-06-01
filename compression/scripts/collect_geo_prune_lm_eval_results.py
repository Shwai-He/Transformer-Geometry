#!/usr/bin/env python3
"""Collect geometry-prune lm-eval outputs into compact CSV summaries."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOTS = [
    REPO_ROOT / "compression" / "outputs" / "geo_prune_lm_eval_fast" / "by_model",
    REPO_ROOT / "compression" / "outputs" / "grad_geo_prune" / "by_model",
]
RESULT_ROOT = REPO_ROOT / "results" / "quality_eval" / "by_model" / "geo_prune_fast"
README = RESULT_ROOT / "README.md"
SUMMARY_CSV = RESULT_ROOT / "geo_prune_lm_eval_summary.csv"
SOURCE_CSV = RESULT_ROOT / "geo_prune_lm_eval_sources.csv"

TASK_ORDER = [
    "openbookqa",
    "piqa",
    "rte",
    "winogrande",
    "boolq",
    "arc_challenge",
    "hellaswag",
    "mmlu",
]

TASK_METRICS = {
    "openbookqa": ["acc_norm,none", "acc,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "rte": ["acc,none"],
    "winogrande": ["acc,none"],
    "boolq": ["acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "mmlu": ["acc,none"],
}

INCLUDE_4TO8 = os.environ.get("INCLUDE_4TO8", "true").lower() in {"1", "true", "yes"}

MODEL_LABELS = {
    "qwen3_0p6b_base": "Qwen3-0.6B",
    "qwen3_1p7b_base": "Qwen3-1.7B",
    "qwen3_4b_base": "Qwen3-4B",
}


def pick_metric(task: str, data: dict[str, Any]) -> float | None:
    containers: list[dict[str, Any]] = []
    for top_key in ("results", "groups"):
        top = data.get(top_key)
        if not isinstance(top, dict):
            continue
        if isinstance(top.get(task), dict):
            containers.append(top[task])
        for task_key, metrics in top.items():
            if isinstance(task_key, str) and task_key.startswith(task) and isinstance(metrics, dict):
                containers.append(metrics)

    for metrics in containers:
        for key in TASK_METRICS.get(task, []):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    for metrics in containers:
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            lowered = key.lower()
            if "stderr" in lowered or lowered == "sample_len":
                continue
            return float(value)
    return None


def variant_label(variant: str) -> str:
    label = variant
    if label.startswith("wanda_"):
        label = label[len("wanda_") :]
    if label.startswith("grad_"):
        label = label[len("grad_") :]
    label = label.replace("_s0.5_", "-50-")
    label = label.replace("_s0p5_", "-50-")
    label = label.replace("_unstructured", "-unstruct")
    label = label.replace("_2to4", "-2:4")
    label = label.replace("_4to8", "-4:8")
    return label


def iter_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for output_root in OUTPUT_ROOTS:
        for path in sorted(output_root.glob("*/*/lm_eval_fast/*/*.json")):
            parts = path.relative_to(output_root).parts
            if len(parts) < 5:
                continue
            model, variant, eval_kind, task = parts[0], parts[1], parts[2], parts[3]
            if not INCLUDE_4TO8 and "4to8" in variant:
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            value = pick_metric(task, data)
            if value is None:
                continue
            rows.append(
                {
                    "eval": eval_kind.replace("lm_eval_", ""),
                    "model": MODEL_LABELS.get(model, model),
                    "variant": variant_label(variant),
                    "task": task,
                    "value": f"{value * 100:.2f}",
                    "source": str(path.relative_to(REPO_ROOT)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    sources: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda item: item["source"]):
        key = (row["eval"], row["model"], row["variant"])
        cells.setdefault(key, {})[row["task"]] = row["value"]
        sources[(row["eval"], row["model"], row["variant"], row["task"])] = row

    with SUMMARY_CSV.open("w", newline="") as f:
        fieldnames = ["eval", "model", "compression"] + TASK_ORDER + ["avg", "done"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eval_kind, model, variant in sorted(cells):
            task_values = cells[(eval_kind, model, variant)]
            numeric = [float(task_values[task]) for task in TASK_ORDER if task in task_values]
            out = {"eval": eval_kind, "model": model, "compression": variant}
            for task in TASK_ORDER:
                out[task] = task_values.get(task, "")
            out["avg"] = f"{sum(numeric) / len(numeric):.2f}" if numeric else ""
            out["done"] = f"{len(numeric)}/{len(TASK_ORDER)}"
            writer.writerow(out)

    with SOURCE_CSV.open("w", newline="") as f:
        fieldnames = ["eval", "model", "compression", "task", "value", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(sources):
            row = sources[key]
            writer.writerow(
                {
                    "eval": row["eval"],
                    "model": row["model"],
                    "compression": row["variant"],
                    "task": row["task"],
                    "value": row["value"],
                    "source": row["source"],
                }
            )


def refresh_readme(rows: list[dict[str, str]]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Geometry-Prune Fast LM-Eval Results",
        "",
        f"Last auto-updated: {now}. Values are percentages.",
        "",
        f"- Summary CSV: `{SUMMARY_CSV.name}`",
        f"- Source manifest: `{SOURCE_CSV.name}`",
        "",
        "Tasks: `" + ",".join(TASK_ORDER) + "`.",
        "",
    ]
    if not rows:
        lines.append("No results collected yet.")
    else:
        lines.append("Run `column -s, -t < geo_prune_lm_eval_summary.csv | less -S` in this directory for the compact table.")
    README.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = iter_rows()
    write_csv(rows)
    refresh_readme(rows)
    print(f"[OK] collected {len(rows)} result files")
    print(f"[OK] wrote {SUMMARY_CSV}")
    print(f"[OK] wrote {SOURCE_CSV}")


if __name__ == "__main__":
    main()
