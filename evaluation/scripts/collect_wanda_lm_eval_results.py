#!/usr/bin/env python3
"""Collect compression lm-eval outputs and refresh the compression README.

The collector is intentionally idempotent: it scans result JSONs under
``compression/outputs/wanda_layerwise_pruned``,
``compression/outputs/geo_prune_lm_eval_fast``, and
``compression/outputs/grad_geo_prune``. It rewrites one auto-generated section
in the curated README and can be called after every task without duplicating
rows.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WANDA_OUTPUT_ROOT = REPO_ROOT / "compression" / "outputs" / "wanda_layerwise_pruned"
EXTRA_OUTPUT_ROOTS = [
    REPO_ROOT / "compression" / "outputs" / "geo_prune_lm_eval_fast" / "by_model",
    REPO_ROOT / "compression" / "outputs" / "grad_geo_prune" / "by_model",
]
README = REPO_ROOT / "results" / "quality_eval" / "by_model" / "baseline_full_compression" / "README.md"
SUMMARY_CSV = README.with_name("wanda_lm_eval_summary.csv")
SOURCE_CSV = README.with_name("wanda_lm_eval_sources.csv")

TASK_ORDER = [
    "openbookqa",
    "piqa",
    "rte",
    "winogrande",
    "boolq",
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "gsm8k_cot",
    "humaneval",
    "nq_open",
    "drop",
    "mbpp",
    "bbh_cot_zeroshot",
]

FAST_TASKS = TASK_ORDER[:8]

TASK_METRICS = {
    "openbookqa": ["acc_norm,none", "acc,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "rte": ["acc,none"],
    "winogrande": ["acc,none"],
    "boolq": ["acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "mmlu": ["acc,none"],
    "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"],
    "gsm8k_cot": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"],
    "humaneval": ["pass@1,create_test", "pass@1,none", "pass_at_1,none"],
    "nq_open": ["exact_match,remove_whitespace", "exact_match,none"],
    "drop": ["f1,none", "em,none"],
    "mbpp": ["pass_at_1,none", "pass@1,sanitized", "pass@1,none"],
    "bbh_cot_zeroshot": ["exact_match,flexible-extract", "acc_norm,none", "acc,none"],
}

VARIANT_LABELS = {
    "dense": "dense",
    "all_linear_unstructured_s0p5_c4_ns128_seq2048": "unstruct-50",
    "all_linear_unstructured_s0p7_c4_ns128_seq2048": "unstruct-70",
    "all_linear_2to4_s0p5_c4_ns128_seq2048": "2:4-50",
    "all_linear_4to8_s0p5_c4_ns128_seq2048": "4:8-50",
    "all_linear_unstructured_s0p3_c4_ns128_seq2048": "unstruct-30",
}

MODEL_LABELS = {
    "qwen3_0p6b": "Qwen3-0.6B",
    "qwen3_0p6b_base": "Qwen3-0.6B",
    "qwen3_1p7b": "Qwen3-1.7B",
    "qwen3_1p7b_base": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_4b_base": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_8b_base": "Qwen3-8B",
    "qwen3p5_9b": "Qwen3.5-9B",
    "qwen3_9b": "Qwen3.5-9B",
}

START = "<!-- AUTO_WANDA_RESULTS_START -->"
END = "<!-- AUTO_WANDA_RESULTS_END -->"


def pick_metric(task: str, data: dict[str, Any]) -> float | None:
    keys = TASK_METRICS.get(task, [])
    containers = []
    results = data.get("results")
    groups = data.get("groups")
    if isinstance(results, dict):
        if task in results and isinstance(results[task], dict):
            containers.append(results[task])
        for task_key, metrics in results.items():
            if isinstance(task_key, str) and task_key.startswith(task) and isinstance(metrics, dict):
                containers.append(metrics)
    if isinstance(groups, dict):
        if task in groups and isinstance(groups[task], dict):
            containers.append(groups[task])
        for task_key, metrics in groups.items():
            if isinstance(task_key, str) and task_key.startswith(task) and isinstance(metrics, dict):
                containers.append(metrics)

    for metrics in containers:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    for metrics in containers:
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            lk = key.lower()
            if "stderr" in lk or lk in {"sample_len"}:
                continue
            return float(value)
    return None


def infer_task(path: Path) -> str | None:
    # Expected: <variant>/lm_eval_full/<task>/<json>
    try:
        task = path.parent.name
        split = path.parent.parent.name
    except IndexError:
        return None
    if split.startswith("lm_eval_debug"):
        # Expected debug leaf: gsm8k_limit100_nochat
        task = task.split("_limit", 1)[0]
    return task if task else None


def variant_label(variant: str) -> str:
    if variant in VARIANT_LABELS:
        return VARIANT_LABELS[variant]
    label = variant
    if label.startswith("wanda_"):
        label = label[len("wanda_") :]
    if label.startswith("grad_"):
        label = label[len("grad_") :]
    label = label.replace("_s0.3_", "-30-")
    label = label.replace("_s0.5_", "-50-")
    label = label.replace("_s0.7_", "-70-")
    label = label.replace("_s0p3_", "-30-")
    label = label.replace("_s0p5_", "-50-")
    label = label.replace("_s0p7_", "-70-")
    label = label.replace("_unstructured", "-unstruct")
    label = label.replace("_2to4", "-2:4")
    label = label.replace("_4to8", "-4:8")
    return label


def sparsity_label(label: str) -> str:
    if label == "dense":
        return "dense"
    if "unstruct-30" in label or "-30-unstruct" in label or "unstructured_s0p3" in label:
        return "unstruct-30"
    if "unstruct-70" in label or "-70-unstruct" in label or "unstructured_s0p7" in label:
        return "unstruct-70"
    if "unstruct-50" in label or "-50-unstruct" in label or "unstructured" in label or "unstruct" in label:
        return "unstruct-50"
    if "2:4-50" in label or "2to4" in label or "2:4" in label:
        return "2:4-50"
    if "4:8-50" in label or "4to8" in label or "4:8" in label:
        return "4:8-50"
    return label


SPARSITY_ORDER = {
    "dense": 0,
    "unstruct-30": 1,
    "unstruct-50": 2,
    "unstruct-70": 3,
    "2:4-50": 4,
    "4:8-50": 5,
}

EVAL_ORDER = {
    "lm_eval_full": 0,
    "lm_eval_fast": 1,
    "lm_eval_debug": 2,
}


def iter_wanda_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for path in sorted(WANDA_OUTPUT_ROOT.glob("qwen3_*/*/lm_eval_*/**/*.json")):
        parts = path.relative_to(WANDA_OUTPUT_ROOT).parts
        if len(parts) < 5:
            continue
        model, variant, eval_kind = parts[0], parts[1], parts[2]
        if variant not in VARIANT_LABELS:
            continue
        task = infer_task(path)
        if not task:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        value = pick_metric(task, data)
        if value is None:
            continue
        key = (model, variant, eval_kind, task, str(path))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "eval_kind": eval_kind,
                "task": task,
                "value": f"{value * 100:.2f}",
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def iter_extra_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for output_root in EXTRA_OUTPUT_ROOTS:
        for path in sorted(output_root.glob("*/*/lm_eval_fast/*/*.json")):
            parts = path.relative_to(output_root).parts
            if len(parts) < 5:
                continue
            model, variant, eval_kind, task = parts[0], parts[1], parts[2], parts[3]
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            value = pick_metric(task, data)
            if value is None:
                continue
            key = (model, variant, eval_kind, task, str(path))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "variant": variant,
                    "variant_label": variant_label(variant),
                    "eval_kind": eval_kind,
                    "task": task,
                    "value": f"{value * 100:.2f}",
                    "source": str(path.relative_to(REPO_ROOT)),
                }
            )
    return rows


def iter_rows() -> list[dict[str, str]]:
    return iter_wanda_rows() + iter_extra_rows()


def latest_by_cell(rows: list[dict[str, str]], eval_kind: str) -> dict[tuple[str, str, str], dict[str, str]]:
    selected = [row for row in rows if row["eval_kind"] == eval_kind]
    # Paths include timestamps where lm-eval creates them; lexical sort is enough
    # for the current output naming and keeps this collector dependency-free.
    selected.sort(key=lambda row: row["source"])
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in selected:
        table[(row["model"], row["variant_label"], row["task"])] = row
    return table


def wide_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    task_order = task_columns(rows)
    cells: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, str]]] = {}
    labels: dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
    for row in sorted(rows, key=lambda r: r["source"]):
        key = (row["eval_kind"], row["model"], row["variant_label"])
        cells.setdefault(key, {})[(row["eval_kind"], row["task"])] = row
        labels[key] = (
            row["eval_kind"],
            row["model_label"],
            row["variant_label"],
            row["variant"],
        )

    output: list[dict[str, str]] = []
    def sort_key(key: tuple[str, str, str]) -> tuple[Any, ...]:
        eval_kind, model, variant = key
        group = sparsity_label(variant)
        return (
            MODEL_LABELS.get(model, model),
            SPARSITY_ORDER.get(group, 99),
            group,
            EVAL_ORDER.get(eval_kind, 99),
            variant,
        )

    for key in sorted(cells, key=sort_key):
        eval_kind, model_label, variant_label, variant = labels[key]
        group = sparsity_label(variant_label)
        out = {
            "eval": eval_kind.replace("lm_eval_", ""),
            "model": model_label,
            "sparsity": group,
            "compression": variant_label,
        }
        numeric: list[float] = []
        done = 0
        for task in task_order:
            row = cells[key].get((eval_kind, task))
            if row:
                out[task] = row["value"]
                numeric.append(float(row["value"]))
                done += 1
            else:
                out[task] = ""
        out["avg"] = f"{sum(numeric) / len(numeric):.2f}" if numeric else ""
        if eval_kind == "lm_eval_fast":
            fast_done = sum(1 for task in FAST_TASKS if cells[key].get((eval_kind, task)))
            out["done"] = f"{fast_done}/{len(FAST_TASKS)}"
        else:
            out["done"] = f"{done}/{len(task_order)}"
        output.append(out)
    return output


def task_columns(rows: list[dict[str, str]]) -> list[str]:
    seen = {row["task"] for row in rows}
    ordered = TASK_ORDER.copy()
    extras = sorted(seen.difference(TASK_ORDER))
    return ordered + extras


def write_csv(rows: list[dict[str, str]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = wide_rows(rows)
    task_order = task_columns(rows)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval", "model", "sparsity", "compression"] + task_order + ["avg", "done"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with SOURCE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval", "model", "sparsity", "compression", "task", "value", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "eval": row["eval_kind"].replace("lm_eval_", ""),
                    "model": row["model_label"],
                    "sparsity": sparsity_label(row["variant_label"]),
                    "compression": row["variant_label"],
                    "task": row["task"],
                    "value": row["value"],
                    "source": row["source"],
                }
            )


def markdown_table(rows: list[dict[str, str]], eval_kind: str, title: str) -> str:
    task_order = task_columns([row for row in rows if row["eval_kind"] == eval_kind])
    cells = latest_by_cell(rows, eval_kind)
    row_keys = sorted(
        {(m, v) for (m, v, _t) in cells},
        key=lambda k: (MODEL_LABELS.get(k[0], k[0]), SPARSITY_ORDER.get(sparsity_label(k[1]), 99), sparsity_label(k[1]), k[1]),
    )
    if not row_keys:
        return f"### {title}\n\nNo results collected yet.\n"

    headers = ["Model", "Sparsity", "Variant"] + task_order + ["Avg", "Done"]
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(headers) - 2)) + " |",
    ]
    for model, variant in row_keys:
        values: list[str] = []
        numeric: list[float] = []
        done = 0
        for task in task_order:
            row = cells.get((model, variant, task))
            if row:
                value = row["value"]
                values.append(value)
                numeric.append(float(value))
                done += 1
            else:
                values.append("")
        avg = f"{sum(numeric) / len(numeric):.2f}" if numeric else ""
        if eval_kind == "lm_eval_fast":
            done_text = f"{sum(1 for task in FAST_TASKS if cells.get((model, variant, task)))}/{len(FAST_TASKS)}"
        else:
            done_text = f"{done}/{len(task_order)}"
        lines.append(
            "| "
            + " | ".join([MODEL_LABELS.get(model, model), sparsity_label(variant), variant] + values + [avg, done_text])
            + " |"
        )
    return "\n".join(lines) + "\n"


def refresh_readme(rows: list[dict[str, str]]) -> None:
    if not README.exists():
        raise SystemExit(f"README not found: {README}")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    section = "\n".join(
        [
            START,
            "## Auto-Collected Compression LM-Eval Results",
            "",
            f"Last auto-updated: {now}. Values are percentages. Summary CSV: `wanda_lm_eval_summary.csv`; source manifest: `wanda_lm_eval_sources.csv`.",
            "",
            markdown_table(rows, "lm_eval_debug", "Debug Subset"),
            markdown_table(rows, "lm_eval_full", "Full Task Sweep"),
            markdown_table(rows, "lm_eval_fast", "Fast Geometry/Grad Sweep"),
            END,
            "",
        ]
    )
    text = README.read_text()
    if START in text and END in text:
        before = text.split(START, 1)[0]
        after = text.split(END, 1)[1]
        text = before + section + after.lstrip("\n")
    else:
        anchor = "## Full 0.6B Task Sweep"
        if anchor in text:
            text = text.replace(anchor, section + "\n" + anchor, 1)
        else:
            text = text.rstrip() + "\n\n" + section
    README.write_text(text)


def main() -> None:
    rows = iter_rows()
    write_csv(rows)
    refresh_readme(rows)
    print(f"[OK] collected {len(rows)} result files")
    print(f"[OK] wrote {SUMMARY_CSV}")
    print(f"[OK] wrote {SOURCE_CSV}")
    print(f"[OK] refreshed {README}")


if __name__ == "__main__":
    main()
