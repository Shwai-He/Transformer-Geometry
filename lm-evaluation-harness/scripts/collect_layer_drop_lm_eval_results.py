#!/usr/bin/env python3
"""Collect layer-drop lm-eval outputs into a compact CSV/README summary."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "compression" / "outputs" / "layer_drop_lm_eval"
SUMMARY_ROOT = REPO_ROOT / "results" / "quality_eval" / "by_model" / "layer_drop_geometry"
SUMMARY_CSV = SUMMARY_ROOT / "layer_drop_lm_eval_summary.csv"
SOURCE_CSV = SUMMARY_ROOT / "layer_drop_lm_eval_sources.csv"
README = SUMMARY_ROOT / "README.md"

TASK_METRICS = {
    "mmlu": ["acc,none"],
    "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"],
    "gsm8k_cot": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"],
}

TASK_ORDER = ["gsm8k_cot", "gsm8k", "mmlu"]

MODEL_LABELS = {
    "qwen3_0p6b": "Qwen3-0.6B",
    "qwen3_1p7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
}


def pick_metric(task: str, data: dict[str, Any]) -> float | None:
    containers = []
    results = data.get("results")
    groups = data.get("groups")
    for obj in (results, groups):
        if not isinstance(obj, dict):
            continue
        if task in obj and isinstance(obj[task], dict):
            containers.append(obj[task])
        for task_key, metrics in obj.items():
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
            lk = key.lower()
            if "stderr" in lk or lk == "alias":
                continue
            return float(value)
    return None


def iter_rows() -> list[dict[str, str]]:
    rows = []
    for path in sorted(OUTPUT_ROOT.glob("qwen3_*/*/*/**/*.json")):
        try:
            rel = path.relative_to(OUTPUT_ROOT)
            model, setting, task = rel.parts[0], rel.parts[1], rel.parts[2]
        except Exception:
            continue
        if task not in TASK_METRICS:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        value = pick_metric(task, data)
        if value is None:
            continue
        parts = setting.rsplit("_", 2)
        if len(parts) == 3:
            rank_metric, component, drop_tag = parts
        else:
            rank_metric, component, drop_tag = setting, "", ""
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "setting": setting,
                "rank_metric": rank_metric,
                "component": component,
                "drop_count": drop_tag.replace("drop", ""),
                "task": task,
                "value": f"{value * 100:.2f}",
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    rows.sort(key=lambda r: (r["model"], r["rank_metric"], r["component"], int(r["drop_count"] or 0), r["task"], r["source"]))
    return rows


def latest_cells(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    cells = {}
    for row in rows:
        key = (row["model"], row["rank_metric"], row["component"], row["drop_count"], row["task"])
        cells[key] = row
    return cells


def wide_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cells = latest_cells(rows)
    row_keys = sorted(
        {(m, r, c, d) for (m, r, c, d, _t) in cells},
        key=lambda k: (MODEL_LABELS.get(k[0], k[0]), k[1], k[2], int(k[3] or 0)),
    )
    out_rows: list[dict[str, str]] = []
    for model, rank_metric, component, drop_count in row_keys:
        out = {
            "model": MODEL_LABELS.get(model, model),
            "rank_metric": rank_metric,
            "component": component,
            "drop_count": drop_count,
            "setting": f"{rank_metric}_{component}_drop{drop_count}",
        }
        numeric: list[float] = []
        done = 0
        for task in TASK_ORDER:
            row = cells.get((model, rank_metric, component, drop_count, task))
            if row:
                out[task] = row["value"]
                numeric.append(float(row["value"]))
                done += 1
            else:
                out[task] = ""
        out["avg"] = f"{sum(numeric) / len(numeric):.2f}" if numeric else ""
        out["done"] = f"{done}/{len(TASK_ORDER)}"
        out_rows.append(out)
    return out_rows


def write_csv(rows: list[dict[str, str]]) -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = wide_rows(rows)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "rank_metric", "component", "drop_count", "setting"] + TASK_ORDER + ["avg", "done"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "setting", "rank_metric", "component", "drop_count", "task", "value", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model_label"],
                    "setting": row["setting"],
                    "rank_metric": row["rank_metric"],
                    "component": row["component"],
                    "drop_count": row["drop_count"],
                    "task": row["task"],
                    "value": row["value"],
                    "source": row["source"],
                }
            )


def markdown(rows: list[dict[str, str]]) -> str:
    cells = latest_cells(rows)
    row_keys = sorted(
        {(m, r, c, d) for (m, r, c, d, _t) in cells},
        key=lambda k: (MODEL_LABELS.get(k[0], k[0]), k[1], k[2], int(k[3] or 0)),
    )
    tasks = [t for t in TASK_ORDER if any(key[-1] == t for key in cells)]
    lines = [
        "# Layer-Drop Geometry LM-Eval Summary",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Summary CSV: `{SUMMARY_CSV.relative_to(REPO_ROOT)}`",
        f"Source manifest: `{SOURCE_CSV.relative_to(REPO_ROOT)}`",
        "",
    ]
    if not row_keys:
        lines.append("No completed layer-drop lm-eval JSON files found yet.")
        lines.append("")
        return "\n".join(lines)

    header = ["Model", "Rank metric", "Component", "Drop"] + tasks
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for model, rank_metric, component, drop_count in row_keys:
        vals = []
        for task in tasks:
            row = cells.get((model, rank_metric, component, drop_count, task))
            vals.append(row["value"] if row else "")
        lines.append("| " + " | ".join([MODEL_LABELS.get(model, model), rank_metric, component, drop_count] + vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = iter_rows()
    write_csv(rows)
    README.write_text(markdown(rows), encoding="utf-8")
    print(f"[COLLECT] rows={len(rows)} csv={SUMMARY_CSV} sources={SOURCE_CSV} readme={README}")


if __name__ == "__main__":
    main()
