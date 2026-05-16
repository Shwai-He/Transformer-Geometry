#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent
INPUT_TSV = RESULTS_DIR / "nanogpt_curated.tsv"
OUTPUT_TSV = RESULTS_DIR / "nanogpt_paper_summary.tsv"
OUTPUT_MD = RESULTS_DIR / "nanogpt_paper_summary.md"

METRIC_COLUMNS = [
    "avg",
    "lambada_openai",
    "arc_easy",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "rte",
    "winogrande",
]


def parse_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def canonical_method(model_name: str) -> str:
    if "baseline" in model_name:
        return "baseline"
    if "attn_para_removal" in model_name and "xftattn" in model_name:
        return "residual_attn_control"
    if "attn_para_removal" in model_name and "xftmlp" in model_name:
        return "residual_mlp_control"
    if "axon-hard-kftrue" in model_name:
        return "gated_value_control"
    if "xsa-" in model_name or "-xsa-" in model_name:
        return "xsa_value_control"
    return "other"


def choose_best(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: parse_float(row["avg"]) or float("-inf"))


def main() -> None:
    with INPUT_TSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    keep_rows = [row for row in rows if row["status"] == "keep"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in keep_rows:
        grouped[(row["family"], canonical_method(row["model"]))].append(row)

    best_by_group = {key: choose_best(group_rows) for key, group_rows in grouped.items()}

    families = sorted({family for family, _ in best_by_group})
    methods = [
        "baseline",
        "residual_attn_control",
        "residual_mlp_control",
        "xsa_value_control",
        "gated_value_control",
    ]

    out_rows: list[dict[str, str]] = []
    for family in families:
        baseline_row = best_by_group.get((family, "baseline"))
        baseline_avg = parse_float(baseline_row["avg"]) if baseline_row else None
        for method in methods:
            row = best_by_group.get((family, method))
            if row is None:
                out_rows.append(
                    {
                        "family": family,
                        "method": method,
                        "present": "no",
                        "delta_vs_baseline_avg": "",
                        **{metric: "" for metric in METRIC_COLUMNS},
                        "model": "",
                        "status_detail": "",
                    }
                )
                continue

            avg_value = parse_float(row["avg"])
            delta = ""
            if baseline_avg is not None and avg_value is not None:
                delta = f"{avg_value - baseline_avg:.2f}"

            out_rows.append(
                {
                    "family": family,
                    "method": method,
                    "present": "yes",
                    "delta_vs_baseline_avg": delta,
                    **{metric: row.get(metric, "") for metric in METRIC_COLUMNS},
                    "model": row["model"],
                    "status_detail": row["status_detail"],
                }
            )

    fieldnames = [
        "family",
        "method",
        "present",
        "delta_vs_baseline_avg",
        *METRIC_COLUMNS,
        "status_detail",
        "model",
    ]
    with OUTPUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        "# nanoGPT Paper Summary",
        "",
        "This file summarizes the best kept run for each `(family, method)` pair.",
        "",
    ]
    for family in families:
        lines.append(f"## {family}")
        lines.append("")
        family_rows = [row for row in out_rows if row["family"] == family]
        for row in family_rows:
            if row["present"] != "yes":
                lines.append(f"- `{row['method']}`: missing stable kept run")
                continue
            delta = row["delta_vs_baseline_avg"] or "n/a"
            lines.append(
                f"- `{row['method']}`: avg={row['avg']} "
                f"(delta vs baseline={delta}), model=`{row['model']}`"
            )
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"[OK] wrote {OUTPUT_TSV}")
    print(f"[OK] wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
