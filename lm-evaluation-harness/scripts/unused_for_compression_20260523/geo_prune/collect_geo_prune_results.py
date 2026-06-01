#!/usr/bin/env python3
"""Collect geometry-aware pruning lm-eval results into README and CSV files."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HARNESS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HARNESS_DIR.parent

DEFAULT_MODEL_TAG = "qwen3_0p6b_geo_nm_s0p5_mmlu_gsm8k_prefetched"
DEFAULT_MODEL_PATH = (
    "/beacon-projects/traumallm/.cache/huggingface/"
    "models--Qwen--Qwen3-0.6B-Base/snapshots/"
    "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
)

TASKS = [
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

TASK_METRICS = {
    "openbookqa": "acc_norm,none",
    "piqa": "acc_norm,none",
    "rte": "acc,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "mmlu": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "gsm8k_cot": "exact_match,flexible-extract",
    "humaneval": "pass@1,create_test",
    "nq_open": "exact_match,remove_whitespace",
    "drop": "f1,none",
    "mbpp": "pass_at_1,none",
    "bbh_cot_zeroshot": "exact_match,flexible-extract",
}

TASK_METRIC_ALIASES = {
    "gsm8k": ["exact_match,flexible-extract", "exact_match,none", "exact_match,strict-match"],
    "gsm8k_cot": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"],
    "humaneval": ["pass@1,none", "pass_at_1,none"],
    "mbpp": ["pass@1,sanitized", "pass@1,none", "pass_at_1,create_test"],
    "bbh_cot_zeroshot": ["acc_norm,none", "acc,none", "exact_match,none"],
}

SETTINGS = [
    ("dense", "dense", "none"),
    ("wanda_unstructured", "unstructured", "none"),
    ("wanda_unstructured_residual_perp", "unstructured", "residual_perp"),
    ("wanda_unstructured_residual_para", "unstructured", "residual_para"),
    ("wanda_unstructured_residual_perp_over_para", "unstructured", "residual_perp_over_para"),
    ("wanda_2_4", "2:4", "none"),
    ("wanda_2_4_residual_perp", "2:4", "residual_perp"),
    ("wanda_2_4_residual_para", "2:4", "residual_para"),
    ("wanda_2_4_residual_perp_over_para", "2:4", "residual_perp_over_para"),
    ("wanda_4_8", "4:8", "none"),
    ("wanda_4_8_residual_perp", "4:8", "residual_perp"),
    ("wanda_4_8_residual_para", "4:8", "residual_para"),
    ("wanda_4_8_residual_perp_over_para", "4:8", "residual_perp_over_para"),
]

SETTING_INFO = {setting: (sparsity, geometry) for setting, sparsity, geometry in SETTINGS}

SHORT_STATE = {
    "R": "running",
    "PD": "queued",
    "CG": "running",
    "CF": "queued",
    "CA": "cancelled",
    "F": "failed",
    "TO": "failed",
    "NF": "failed",
    "OOM": "failed",
    "CD": "done",
}


def result_root() -> Path:
    default = HARNESS_DIR / "outputs" / "geo_prune_lm_eval" / DEFAULT_MODEL_TAG
    return Path(os.environ.get("RESULT_ROOT", default)).resolve()


def slurm_dir() -> Path:
    default = HARNESS_DIR / "outputs" / "geo_prune_lm_eval_slurm"
    return Path(os.environ.get("SLURM_DIR", default)).resolve()


def model_path() -> str:
    return os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


def run_command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def read_submission_manifest(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    paths = [
        Path(os.environ["JOB_MANIFEST"])
        if os.environ.get("JOB_MANIFEST")
        else root / "submission_manifest.tsv"
    ]
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                setting = (row.get("setting") or "").strip()
                task = (row.get("task") or "").strip()
                job_id = (row.get("job_id") or "").strip()
                if setting and task and job_id:
                    mapping[(setting, task)] = row
    return mapping


def active_settings(submissions: dict[tuple[str, str], dict[str, str]]) -> list[tuple[str, str, str]]:
    if not submissions:
        return SETTINGS
    ordered = []
    seen = set()
    for row in submissions.values():
        setting = (row.get("setting") or "").strip()
        if setting and setting not in seen:
            seen.add(setting)
            ordered.append(setting)
    return [(setting, *SETTING_INFO.get(setting, ("unknown", "unknown"))) for setting in ordered]


def active_tasks(submissions: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    if not submissions:
        return TASKS
    ordered = []
    seen = set()
    for (_setting, task), _row in submissions.items():
        if task and task not in seen:
            seen.add(task)
            ordered.append(task)
    return ordered


def latest_result_file(root: Path, setting: str, task: str) -> Path | None:
    setting_dir = root / setting
    if not setting_dir.exists():
        return None
    candidates = sorted(setting_dir.glob(f"{task}_*.json"))
    if not candidates:
        candidates = sorted(setting_dir.glob(f"{task}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def stderr_key(metric_key: str) -> str:
    if "," not in metric_key:
        return f"{metric_key}_stderr"
    metric, suffix = metric_key.split(",", 1)
    return f"{metric}_stderr,{suffix}"


def pick_metric(metrics: dict[str, Any], preferred: list[str]) -> tuple[str, float, float | None] | None:
    for key in preferred:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            err = metrics.get(stderr_key(key))
            return key, float(val), float(err) if isinstance(err, (int, float)) else None

    fallback_order = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "acc_norm,none",
        "acc,none",
        "f1,none",
        "pass@1,create_test",
        "pass_at_1,none",
        "pass@1,none",
        "pass@1,sanitized",
    ]
    for key in fallback_order:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            err = metrics.get(stderr_key(key))
            return key, float(val), float(err) if isinstance(err, (int, float)) else None

    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            continue
        low = key.lower()
        if "stderr" in low or low in {"sample_len"}:
            continue
        err = metrics.get(stderr_key(key))
        return key, float(val), float(err) if isinstance(err, (int, float)) else None
    return None


def extract_metric(task: str, path: Path | None) -> tuple[str, float, float | None] | tuple[None, None, None]:
    if path is None:
        return None, None, None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None, None, None

    preferred = [TASK_METRICS[task], *TASK_METRIC_ALIASES.get(task, [])]
    results = data.get("results", {})
    if isinstance(results, dict):
        if isinstance(results.get(task), dict):
            picked = pick_metric(results[task], preferred)
            if picked:
                return picked
        for key, metrics in results.items():
            if isinstance(key, str) and key.startswith(task) and isinstance(metrics, dict):
                picked = pick_metric(metrics, preferred)
                if picked:
                    return picked

    groups = data.get("groups", {})
    if isinstance(groups, dict):
        if isinstance(groups.get(task), dict):
            picked = pick_metric(groups[task], preferred)
            if picked:
                return picked
        for key, metrics in groups.items():
            if isinstance(key, str) and key.startswith(task) and isinstance(metrics, dict):
                picked = pick_metric(metrics, preferred)
                if picked:
                    return picked
    return None, None, None


def slurm_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    states: dict[str, dict[str, str]] = {}
    if not job_ids:
        return states

    queue = run_command(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%t|%P|%M"])
    for line in queue.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id, short, part, elapsed = parts[:4]
        states[job_id] = {
            "state": SHORT_STATE.get(short, short.lower()),
            "partition": part,
            "elapsed": elapsed,
            "exit_code": "0:0",
        }

    acct = run_command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobID,State,ExitCode,Elapsed,Partition",
        ]
    )
    for line in acct.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, state, exit_code, elapsed, partition = parts[:5]
        if "." in job_id:
            continue
        normalized = state.lower()
        if normalized == "completed":
            normalized = "done"
        elif normalized in {"pending"}:
            normalized = "queued"
        elif normalized in {"running", "completing"}:
            normalized = "running"
        elif "cancelled" in normalized:
            normalized = "cancelled"
        elif normalized in {"failed", "timeout", "out_of_memory", "node_fail"}:
            normalized = "failed"
        states[job_id] = {
            "state": normalized,
            "partition": partition,
            "elapsed": elapsed,
            "exit_code": exit_code,
        }
    return states


def slurm_out_glob(log_dir: Path, job_id: str | None) -> str:
    if not job_id:
        return ""
    rel = os.path.relpath(log_dir / f"*-{job_id}.out", result_root())
    return rel


def build_rows(root: Path, log_dir: Path) -> list[dict[str, Any]]:
    submissions = read_submission_manifest(root)
    settings = active_settings(submissions)
    tasks = active_tasks(submissions)
    job_ids = sorted({row["job_id"] for row in submissions.values() if row.get("job_id")})
    state_by_job = slurm_states(job_ids)

    rows: list[dict[str, Any]] = []
    for setting, sparsity, geometry in settings:
        for task in tasks:
            submitted = submissions.get((setting, task), {})
            job_id = submitted.get("job_id", "")
            job_state = state_by_job.get(job_id, {})
            json_path = latest_result_file(root, setting, task)
            metric, value, stderr = extract_metric(task, json_path)

            status = job_state.get("state", "planned" if not job_id else "queued")
            if json_path and value is not None:
                status = "done_with_failed" if job_state.get("exit_code") not in {"", "0:0", None} else "done"
            elif status == "done":
                status = "done_missing_result"

            rows.append(
                {
                    "setting": setting,
                    "task": task,
                    "sparsity": sparsity,
                    "geometry_metric": geometry,
                    "status": status,
                    "job_id": job_id,
                    "partition": job_state.get("partition", submitted.get("partition", "")),
                    "elapsed": job_state.get("elapsed", ""),
                    "exit_code": job_state.get("exit_code", ""),
                    "metric": metric or "",
                    "value": value if value is not None else "",
                    "stderr": stderr if stderr is not None else "",
                    "result_json": os.path.relpath(json_path, root) if json_path else "",
                    "slurm_out_glob": slurm_out_glob(log_dir, job_id),
                }
            )
    return rows


def write_csv(root: Path, rows: list[dict[str, Any]]) -> None:
    out = root / "quality_summary.csv"
    fields = [
        "setting",
        "task",
        "sparsity",
        "geometry_metric",
        "status",
        "job_id",
        "partition",
        "elapsed",
        "exit_code",
        "metric",
        "value",
        "stderr",
        "result_json",
        "slurm_out_glob",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return ""


def score_matrix(rows: list[dict[str, Any]], settings: list[tuple[str, str, str]], tasks: list[str]) -> str:
    by_setting_task = {(row["setting"], row["task"]): row for row in rows}
    lines = []
    header = ["Setting", "Sparsity", "Geometry", *tasks]
    aligns = ["---", "---", "---", *["---:" for _ in tasks]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for setting, sparsity, geometry in settings:
        cells = [f"`{setting}`", f"`{sparsity}`", f"`{geometry}`"]
        for task in tasks:
            cells.append(fmt_value(by_setting_task[(setting, task)]["value"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def progress_table(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["status"] for row in rows)
    lines = ["| Status | Count |", "|---|---:|"]
    for status in sorted(counts):
        lines.append(f"| `{status}` | {counts[status]} |")
    return "\n".join(lines)


def checkpoint_diff_section() -> str:
    old_diff = (
        REPO_ROOT
        / "compression"
        / "outputs"
        / "checkpoint_diffs"
        / "qwen3_0p6b"
        / "old_global_wanda_unstructured_vs_dense"
    )
    new_diff = (
        REPO_ROOT
        / "compression"
        / "outputs"
        / "wanda_layerwise_pruned"
        / "qwen3_0p6b"
        / "unstructured_s0p5_c4_ns128_seq2048_compare"
    )
    old_vs_new = (
        REPO_ROOT
        / "compression"
        / "outputs"
        / "wanda_layerwise_pruned"
        / "qwen3_0p6b"
        / "unstructured_s0p5_c4_ns128_seq2048_compare_old_global"
    )
    old_status = "done" if (old_diff / "checkpoint_diff_summary.json").exists() else "planned"
    new_status = "done" if (new_diff / "checkpoint_diff_summary.json").exists() else "queued"
    old_vs_new_status = "done" if (old_vs_new / "checkpoint_diff_summary.json").exists() else "queued"
    return f"""## Checkpoint Diff Tracking

| Comparison | Status | Job | Output | Notes |
|---|---|---:|---|---|
| dense Qwen3-0.6B-Base vs old `wanda_none_s0.5_unstructured` | {old_status} | local | `{old_diff}` | Old checkpoint is module-level 50% sparse on Linear targets, but was generated from only 2 calibration prompts with `token_scope=last`; it is not a faithful WANDA/C4 baseline. |
| dense Qwen3-0.6B-Base vs new layerwise WANDA/C4 checkpoint | {new_status} | 59342 | `{new_diff}` | Runs after checkpoint generation job 59341. |
| old `wanda_none_s0.5_unstructured` vs new layerwise WANDA/C4 checkpoint | {old_vs_new_status} | 59343 | `{old_vs_new}` | Direct mask/weight-difference check between the previous prune checkpoint and the WANDA-style replacement. |"""


def write_readme(root: Path, rows: list[dict[str, Any]]) -> None:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tasks = []
    seen_tasks = set()
    for row in rows:
        task = row["task"]
        if task not in seen_tasks:
            seen_tasks.add(task)
            tasks.append(task)
    task_list = ", ".join(f"`{task}`" for task in tasks)
    settings = []
    seen = set()
    for row in rows:
        setting = row["setting"]
        if setting not in seen:
            seen.add(setting)
            settings.append((setting, row["sparsity"], row["geometry_metric"]))
    settings_text = ", ".join(f"`{setting}`" for setting, _, _ in settings)
    body = f"""# Geometry-Aware Pruning LM-Eval Results

Last updated: {updated}

This directory collects the LM Evaluation Harness run for residual-level geometry-aware pruning on cached `Qwen3-0.6B-Base`. The task surface matches the full baseline/XSA variant evaluation list, not just `mmlu` and `gsm8k`.

## Setup

| Field | Value |
|---|---|
| Model | `{model_path()}` |
| Tasks | {task_list} |
| Backend | `hf-geo-prune` |
| Method | `wanda` |
| Prune targets | `q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj` |
| Geometry targets | `o_proj+down_proj` |
| Sparsity | `0.5`; settings: {settings_text} |
| Calibration | C4 `nsamples=128`, `seqlen=2048`, `token_scope=all` |
| Geometry metrics | none, residual perpendicular, residual parallel, residual perpendicular / parallel |
| Include path | `{REPO_ROOT / "compression" / "lm_eval_tasks"}` |
| Chat template | disabled |
| Precision / device | `bfloat16` on `cuda` |

## Progress

{progress_table(rows)}

## Score Matrix

Blank cells mean no result JSON has been produced yet. Values use the primary task metric selected by lm-eval result keys, for example MMLU `acc,none`.

{score_matrix(rows, settings, tasks)}

## Submitted Grid

- Jobs were submitted on `scavenger --qos=scavenger`; row-level status is refreshed from Slurm accounting.
- Machine-readable row-level status: `quality_summary.csv`.
- Submission mapping: `submission_manifest.tsv`.
- Dataset prefetch manifests: `../../task_cache/lm_eval_task_download_manifest_full_tasks.json` and `../../task_cache/lm_eval_task_download_manifest_full_tasks_retry.json`.
- Refresh command: `PYTHON_BIN=/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python bash lm-evaluation-harness/scripts/collect_geo_prune_results.sh`.

{checkpoint_diff_section()}

## Notes

- `openbookqa`, `arc_challenge`, and `hellaswag` needed a small `datasets==2.19` compatibility patch for cached newer-style `List` features; the patch is now in `lm_eval/__init__.py` and the three tasks load offline.
- WANDA-aligned unstructured runs use C4 calibration and full Linear pruning targets; geometry multipliers are restricted to residual-compatible `o_proj/down_proj`.
"""
    (root / "README.md").write_text(body)


def main() -> None:
    root = result_root()
    root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(root, slurm_dir())
    write_csv(root, rows)
    write_readme(root, rows)
    print(f"[DONE] wrote {root / 'README.md'}")
    print(f"[DONE] wrote {root / 'quality_summary.csv'}")


if __name__ == "__main__":
    main()
