#!/usr/bin/env python3
"""Summarize layer-drop selections and duplicate layer lists."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_ROOT = REPO_ROOT / "compression" / "outputs" / "layer_drop_geometry"
SUMMARY_ROOT = REPO_ROOT / "results" / "quality_eval" / "by_model" / "layer_drop_geometry"
SELECTION_CSV = SUMMARY_ROOT / "layer_drop_selection_summary.csv"
DEDUP_CSV = SUMMARY_ROOT / "layer_drop_selection_duplicates.csv"
METHOD_DIFF_CSV = SUMMARY_ROOT / "layer_drop_method_differences.csv"
SELECTION_MD = SUMMARY_ROOT / "layer_drop_selection_summary.md"


def iter_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = ("qwen3_*/*/drop_selection.json", "llama*/*/drop_selection.json", "gemma*/*/drop_selection.json")
    for path in sorted({p for pattern in patterns for p in SELECTION_ROOT.glob(pattern)}):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = data.get("model_tag") or path.parents[1].name
        method_tag = data.get("output_tag") or path.parent.name
        metric = data.get("rank_metric") or method_tag
        recs = data.get("recommendations", {})
        if not isinstance(recs, dict):
            continue
        for component, by_drop in recs.items():
            if not isinstance(by_drop, dict):
                continue
            for drop_key, layers in sorted(by_drop.items()):
                if not isinstance(layers, list):
                    continue
                drop_count = str(drop_key).replace("drop_", "").replace("drop", "")
                layer_list = [int(x) for x in layers]
                rows.append(
                    {
                        "model": model,
                        "method_tag": method_tag,
                        "rank_metric": metric,
                        "score_name": data.get("score_name", ""),
                        "rank_order": data.get("rank_order", ""),
                        "component": component,
                        "drop_count": drop_count,
                        "layers": layer_list,
                        "layer_count": len(layer_list),
                        "layers_csv": ",".join(str(x) for x in layer_list),
                        "layers_sorted_csv": ",".join(str(x) for x in sorted(layer_list)),
                        "selection_json": str(path.relative_to(REPO_ROOT)),
                    }
                )
    rows.sort(
        key=lambda r: (
            r["model"],
            r["component"],
            int(r["drop_count"]),
            r["layers_sorted_csv"],
            r["method_tag"],
        )
    )
    return rows


def annotate_duplicate_groups(rows: list[dict[str, Any]]) -> None:
    global_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    model_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        global_groups[(row["component"], row["drop_count"], row["layers_sorted_csv"])].append(row)
        model_groups[(row["model"], row["component"], row["drop_count"], row["layers_sorted_csv"])].append(row)

    for row in rows:
        row["duplicate_group_global"] = ""
        row["duplicate_group_model"] = ""
        row["same_as_in_model"] = ""

    for idx, key in enumerate(sorted(global_groups), start=1):
        group = global_groups[key]
        if len(group) < 2:
            continue
        group_id = f"g{idx:03d}"
        for row in group:
            row["duplicate_group_global"] = group_id

    for idx, key in enumerate(sorted(model_groups), start=1):
        group = model_groups[key]
        if len(group) < 2:
            continue
        group_id = f"m{idx:03d}"
        names = [row["method_tag"] for row in group]
        for row in group:
            row["duplicate_group_model"] = group_id
            row["same_as_in_model"] = ";".join(name for name in names if name != row["method_tag"])


def duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["component"], row["drop_count"], row["layers_sorted_csv"])].append(row)

    out: list[dict[str, str]] = []
    for (component, drop_count, layers_sorted), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        out.append(
            {
                "component": component,
                "drop_count": drop_count,
                "same_layer_set": layers_sorted,
                "configs": "; ".join(f"{r['model']}:{r['method_tag']}:{r['layers_csv']}" for r in group),
            }
        )
    return out


def method_difference_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["component"], row["drop_count"])].append(row)

    out: list[dict[str, str]] = []
    for (model, component, drop_count), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda r: r["rank_metric"])
        layer_sets = {row["layers_sorted_csv"] for row in group}
        all_same = len(layer_sets) == 1
        out.append(
            {
                "model": model,
                "component": component,
                "drop_count": drop_count,
                "all_methods_same": "yes" if all_same else "no",
                "configs": "; ".join(f"{r['method_tag']}:{r['layers_csv']}" for r in group),
            }
        )
    return out


def write_csvs(rows: list[dict[str, Any]], dups: list[dict[str, str]], method_diffs: list[dict[str, str]]) -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    with SELECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "component",
                "drop_count",
                "method_tag",
                "rank_metric",
                "score_name",
                "rank_order",
                "layer_count",
                "layers",
                "layers_sorted",
                "duplicate_group_global",
                "duplicate_group_model",
                "same_as_in_model",
                "selection_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "component": row["component"],
                    "drop_count": row["drop_count"],
                    "method_tag": row["method_tag"],
                    "rank_metric": row["rank_metric"],
                    "score_name": row["score_name"],
                    "rank_order": row["rank_order"],
                    "layer_count": row["layer_count"],
                    "layers": row["layers_csv"],
                    "layers_sorted": row["layers_sorted_csv"],
                    "duplicate_group_global": row["duplicate_group_global"],
                    "duplicate_group_model": row["duplicate_group_model"],
                    "same_as_in_model": row["same_as_in_model"],
                    "selection_json": row["selection_json"],
                }
            )
    with DEDUP_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "drop_count", "same_layer_set", "configs"])
        writer.writeheader()
        writer.writerows(dups)
    with METHOD_DIFF_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "component", "drop_count", "all_methods_same", "configs"],
        )
        writer.writeheader()
        writer.writerows(method_diffs)


def markdown(rows: list[dict[str, Any]], dups: list[dict[str, str]], method_diffs: list[dict[str, str]]) -> str:
    lines = [
        "# Layer-Drop Selection Summary",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Selection CSV: `{SELECTION_CSV.relative_to(REPO_ROOT)}`",
        f"Duplicate CSV: `{DEDUP_CSV.relative_to(REPO_ROOT)}`",
        f"Method-difference CSV: `{METHOD_DIFF_CSV.relative_to(REPO_ROOT)}`",
        "",
        "## Selections",
        "",
        "| Model | Component | Drop | Method | Metric | Layers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['component']} | {row['drop_count']} | "
            f"{row['method_tag']} | {row['rank_metric']} | {row['layers_csv']} |"
        )
    lines.extend(["", "## Duplicate Layer Sets", ""])
    if not dups:
        lines.extend(["No duplicate layer sets found.", ""])
    else:
        lines.append("| Component | Drop | Same layer set | Configs |")
        lines.append("| --- | --- | --- | --- |")
        for row in dups:
            lines.append(
                f"| {row['component']} | {row['drop_count']} | {row['same_layer_set']} | {row['configs']} |"
            )
        lines.append("")
    lines.extend(["", "## Same-Model Method Differences", ""])
    if not method_diffs:
        lines.extend(["No same-model method comparisons found.", ""])
    else:
        lines.append("| Model | Component | Drop | All methods same | Configs |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in method_diffs:
            lines.append(
                f"| {row['model']} | {row['component']} | {row['drop_count']} | "
                f"{row['all_methods_same']} | {row['configs']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = iter_rows()
    annotate_duplicate_groups(rows)
    dups = duplicate_rows(rows)
    method_diffs = method_difference_rows(rows)
    write_csvs(rows, dups, method_diffs)
    SELECTION_MD.write_text(markdown(rows, dups, method_diffs), encoding="utf-8")
    print(
        f"[SELECTIONS] rows={len(rows)} duplicates={len(dups)} "
        f"method_diffs={len(method_diffs)} md={SELECTION_MD}"
    )


if __name__ == "__main__":
    main()
