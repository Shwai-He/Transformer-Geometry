#!/usr/bin/env python3
"""Collect XSA lm-eval results into tidy CSV summaries.

Directory layout expected:
  <input_root>/<model_name>/<task>-<setting>.json

This collector now supports multiple result roots, for example:
  - outputs/xsa_lm_eval
  - outputs/xsa_lm_eval_alpha_sweep
  - outputs/xsa_lm_eval_ruler

Output:
  - combined CSV with one row per (source_root, model, setting)
  - per-source CSVs under the summary directory
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Settings to exclude from summaries.
EXCLUDED_SETTINGS = {
    "xsa_middle",
}

# ---------------------------------------------------------------------------
# Task → metric key (as stored in lm-eval JSON: results[task][metric_key])
# ---------------------------------------------------------------------------
TASK_METRICS = {
    "openbookqa":      "acc_norm,none",
    "piqa":            "acc_norm,none",
    "rte":             "acc,none",
    "winogrande":      "acc,none",
    "boolq":           "acc,none",
    "arc_challenge":   "acc_norm,none",
    "hellaswag":       "acc_norm,none",
    "mmlu":            "acc,none",
    "gsm8k":           "exact_match,strict-match",
    "humaneval":       "pass@1,create_test",
    "nq_open":         "exact_match,remove_whitespace",
    "drop":            "f1,none",
    "mbpp":            "pass_at_1,none",
    "bbh_cot_zeroshot": "exact_match,flexible-extract",
}

# Optional fallback aliases for metric-key differences across lm-eval versions.
TASK_METRIC_ALIASES = {
    "mbpp": [
        "pass@1,sanitized",
        "pass@1,none",
    ],
    "bbh_cot_zeroshot": [
        "acc_norm,none",
        "acc,none",
    ],
}

KNOWN_TASKS = set(TASK_METRICS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re
_TIMESTAMP_RE = re.compile(r"_\d{4}-\d{2}-\d{2}T[\d\-\.]+$")

def parse_task_setting(stem: str):
    """Parse '<task>-<setting>[_timestamp]' filename stem."""
    # strip optional timestamp suffix: _2026-04-22T03-38-32.705737
    stem = _TIMESTAMP_RE.sub("", stem)
    for task in sorted(KNOWN_TASKS, key=len, reverse=True):
        prefix = f"{task}-"
        if stem.startswith(prefix):
            setting = stem[len(prefix):]
            return task, setting
    # Fallback for tasks not explicitly listed in TASK_METRICS.
    # Example: truthfulqa_gen-xsa_middle_multihead
    if "-" in stem:
        task, setting = stem.split("-", 1)
        if task and setting:
            return task, setting
    return None, None


def _pick_metric_from_result_dict(metrics: dict):
    """Choose a primary scalar metric from lm-eval task result dict."""
    if not isinstance(metrics, dict):
        return None
    # Preferred order if present.
    preferred_prefixes = [
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc_norm,none",
        "acc,none",
        "pass_at_1,none",
        "pass@1,none",
        "pass@1,sanitized",
    ]
    for k in preferred_prefixes:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return v

    # Generic fallback: first numeric metric that's not stderr or metadata-like.
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        lk = k.lower()
        if "stderr" in lk:
            continue
        if lk in {"sample_len"}:
            continue
        return v
    return None


def extract_value(task: str, data: dict):
    keys = []
    if task in TASK_METRICS:
        keys = [TASK_METRICS[task]] + TASK_METRIC_ALIASES.get(task, [])
    results = data.get("results", {})
    # 1) For mapped tasks, try strict key matching first.
    if keys:
        for task_key, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for key in keys:
                if key in metrics:
                    return metrics[key]

    # 2) If exact task exists in results, auto-pick from its metric dict.
    if task in results and isinstance(results.get(task), dict):
        v = _pick_metric_from_result_dict(results[task])
        if v is not None:
            return v

    # 3) For group-style outputs where task key may differ slightly, try prefix match.
    for task_key, metrics in results.items():
        if isinstance(task_key, str) and task_key.startswith(task) and isinstance(metrics, dict):
            v = _pick_metric_from_result_dict(metrics)
            if v is not None:
                return v

    # 4) For grouped aggregates, also try "groups".
    groups = data.get("groups", {})
    if isinstance(groups, dict):
        if task in groups and isinstance(groups.get(task), dict):
            v = _pick_metric_from_result_dict(groups[task])
            if v is not None:
                return v
        for group_key, metrics in groups.items():
            if isinstance(group_key, str) and group_key.startswith(task) and isinstance(metrics, dict):
                v = _pick_metric_from_result_dict(metrics)
                if v is not None:
                    return v

    # flat format fallback
    for key in keys:
        if key in data:
            return data[key]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "xsa_lm_eval_summary"

# Allow override via env vars
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", OUTPUT_DIR))

default_roots = [
    Path(__file__).resolve().parents[1] / "outputs" / "xsa_lm_eval",
    Path(__file__).resolve().parents[1] / "outputs" / "xsa_lm_eval_alpha_sweep",
    Path(__file__).resolve().parents[1] / "outputs" / "xsa_lm_eval_ruler",
]
input_roots_env = os.environ.get("INPUT_ROOTS", "").strip()
input_root_env = os.environ.get("INPUT_ROOT", "").strip()

if input_roots_env:
    INPUT_ROOTS = [Path(p.strip()) for p in input_roots_env.split(":") if p.strip()]
elif input_root_env:
    INPUT_ROOTS = [Path(input_root_env)]
else:
    INPUT_ROOTS = default_roots

existing_roots = [root for root in INPUT_ROOTS if root.exists()]
missing_roots = [root for root in INPUT_ROOTS if not root.exists()]
for root in missing_roots:
    print(f"[INFO] Input root missing, skip: {root}")

if not existing_roots:
    raise SystemExit(
        "No input roots exist. Checked: " + ", ".join(str(root) for root in INPUT_ROOTS)
    )

rows = []

for input_root in existing_roots:
    source_root = input_root.name
    for model_dir in sorted(input_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # group by setting within one source root + model
        by_setting: dict[str, dict] = {}

        for jpath in sorted(model_dir.glob("*.json")):
            task, setting = parse_task_setting(jpath.stem)
            if task is None:
                print(f"[SKIP] Unrecognized filename under {source_root}: {jpath.name}")
                continue
            if setting in EXCLUDED_SETTINGS:
                print(f"[SKIP] Excluded setting under {source_root}: {jpath.name}")
                continue

            try:
                data = json.loads(jpath.read_text())
            except Exception as e:
                print(f"[ERROR] {jpath}: {e}")
                continue

            val = extract_value(task, data)
            if val is None:
                print(f"[WARN] metric not found: {jpath.name}")
                continue

            row = by_setting.setdefault(
                setting,
                {
                    "source_root": source_root,
                    "model": model_name,
                    "setting": setting,
                },
            )
            row[task] = round(val * 100, 2)

        rows.extend(by_setting.values())

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No results collected — check INPUT_ROOTS / INPUT_ROOT.")

# sort columns: model, setting, then tasks in TASK_METRICS order, then avg
task_cols = [t for t in TASK_METRICS if t in df.columns]
# include dynamically discovered tasks not in TASK_METRICS
extra_task_cols = sorted(
    c for c in df.columns
    if c not in {"source_root", "model", "setting", "avg"} and c not in task_cols
)
task_cols = task_cols + extra_task_cols
df["avg"] = df[task_cols].mean(axis=1).round(2)
df = df[["source_root", "model", "setting"] + task_cols + ["avg"]].sort_values(
    ["model", "source_root", "setting"]
).reset_index(drop=True)

combined_df = df.drop(columns=["source_root"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUTPUT_DIR / "xsa_results_all.csv"
combined_df.to_csv(out_csv, index=False)

for source_root, subdf in df.groupby("source_root", sort=True):
    source_csv = OUTPUT_DIR / f"{source_root}.csv"
    subdf.drop(columns=["source_root"]).to_csv(source_csv, index=False)
    print(f"[OK] {len(subdf)} rows saved → {source_csv}")

print(f"[OK] {len(combined_df)} rows saved → {out_csv}")
print(combined_df.to_string(index=False))
