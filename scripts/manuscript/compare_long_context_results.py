#!/usr/bin/env python3
"""Compare two validated formal RULER manifests with matched inputs."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)
FAMILIES = {
    "niah": TASKS[:8],
    "vt": ("ruler_vt",),
    "cwe": ("ruler_cwe",),
    "fwe": ("ruler_fwe",),
    "qa": ("ruler_qa_squad", "ruler_qa_hotpot"),
}
POSITION_METRICS = {
    "self_probability",
    "nonself_over_total",
    "self_over_total",
    "para_nonself_over_total",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["task"]: row for row in payload["tasks"]}
    return payload, rows


def require_complete(label, payload, rows):
    if not payload.get("complete"):
        raise ValueError(
            "{} manifest is incomplete: passed={} failed={} missing={}".format(
                label, payload.get("passed"), payload.get("failed"), payload.get("missing")
            )
        )
    missing = [task for task in TASKS if rows.get(task, {}).get("status") != "pass"]
    if missing:
        raise ValueError("{} lacks passing tasks: {}".format(label, ", ".join(missing)))


def validate_position_receipt(path, intervention_path, intervention_payload, rows, length):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "transformer-geometry-position-diagnostic-v1":
        raise ValueError("position-diagnostic receipt has an unsupported schema")
    if receipt.get("pass") is not True or receipt.get("errors"):
        raise ValueError("position-diagnostic receipt is not passing")
    if receipt.get("length") != length:
        raise ValueError("position-diagnostic receipt length mismatch")
    if receipt.get("snapshot") != "13afe5124825b4f3751f836b40dafda64c1ed062":
        raise ValueError("position-diagnostic receipt snapshot mismatch")
    if receipt.get("transformers_version") != "4.52.4":
        raise ValueError("position-diagnostic receipt Transformers mismatch")
    if receipt.get("resolved_allowed_git_hash") not in intervention_payload.get(
        "allowed_git_hashes", []
    ):
        raise ValueError("position-diagnostic receipt source commit is not allowed")
    if rows.get(receipt.get("task"), {}).get("status") != "pass":
        raise ValueError("position-diagnostic receipt task is not a passing formal task")
    if receipt.get("position_bins", 0) <= 0:
        raise ValueError("position-diagnostic receipt has no positive bin count")
    if receipt.get("active_layers") != list(range(9, 28)):
        raise ValueError("position-diagnostic receipt layer window mismatch")
    if set(receipt.get("metrics", [])) != POSITION_METRICS:
        raise ValueError("position-diagnostic receipt metric contract mismatch")
    if receipt.get("matched_sample_count", 0) < 1:
        raise ValueError("position-diagnostic receipt has no formal-matched sample")
    if receipt.get("formal_audit_sha256") != sha256(intervention_path):
        raise ValueError("position-diagnostic receipt is not bound to this intervention audit")
    for label in ("result", "samples", "formal_audit"):
        source = Path(receipt.get(label, ""))
        if not source.is_file():
            raise ValueError(f"position-diagnostic {label} source is missing")
        if sha256(source) != receipt.get(f"{label}_sha256"):
            raise ValueError(f"position-diagnostic {label} source hash mismatch")
    return receipt


def require_position_diagnostics(rows, receipt=None):
    missing = [
        task
        for task in TASKS
        if rows.get(task, {}).get("position_diagnostics_present") is not True
    ]
    if missing and receipt is None:
        raise ValueError(
            "intervention manifest lacks required position-bin diagnostics: {}. "
            "Do not aggregate for paper integration until a reviewed diagnostic "
            "receipt is supported by this gate.".format(", ".join(missing))
        )


def aggregate(metrics):
    task_mean = statistics.mean(metrics[task] for task in TASKS)
    family_means = {
        family: statistics.mean(metrics[task] for task in tasks)
        for family, tasks in FAMILIES.items()
    }
    return {
        "task_mean_13": task_mean,
        "family_means": family_means,
        "family_macro_5": statistics.mean(family_means.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--position-diagnostic-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_payload, baseline_rows = read_manifest(args.baseline)
    intervention_payload, intervention_rows = read_manifest(args.intervention)
    require_complete("baseline", baseline_payload, baseline_rows)
    require_complete("intervention", intervention_payload, intervention_rows)
    if baseline_payload.get("arm") != "baseline":
        raise ValueError("baseline manifest has wrong arm")
    if intervention_payload.get("arm") != "intervention":
        raise ValueError("intervention manifest has wrong arm")
    position_receipt = None
    if args.position_diagnostic_receipt:
        position_receipt = validate_position_receipt(
            args.position_diagnostic_receipt,
            args.intervention,
            intervention_payload,
            intervention_rows,
            intervention_payload["length"],
        )
    require_position_diagnostics(intervention_rows, position_receipt)
    if baseline_payload["length"] != intervention_payload["length"]:
        raise ValueError("context-length mismatch")

    rows = []
    for task in TASKS:
        baseline = baseline_rows[task]
        intervention = intervention_rows[task]
        if baseline["input_signature_sha256"] != intervention["input_signature_sha256"]:
            raise ValueError("matched-input signature mismatch for {}".format(task))
        rows.append(
            {
                "task": task,
                "baseline": baseline["metric"],
                "intervention": intervention["metric"],
                "delta": intervention["metric"] - baseline["metric"],
                "input_signature_sha256": baseline["input_signature_sha256"],
                "baseline_result_sha256": baseline["result_sha256"],
                "intervention_result_sha256": intervention["result_sha256"],
            }
        )

    baseline_metrics = {row["task"]: row["baseline"] for row in rows}
    intervention_metrics = {row["task"]: row["intervention"] for row in rows}
    baseline_aggregate = aggregate(baseline_metrics)
    intervention_aggregate = aggregate(intervention_metrics)
    output = {
        "length": baseline_payload["length"],
        "baseline_audit_sha256": sha256(args.baseline),
        "intervention_audit_sha256": sha256(args.intervention),
        "position_diagnostic_receipt_sha256": (
            sha256(args.position_diagnostic_receipt)
            if args.position_diagnostic_receipt
            else None
        ),
        "tasks": rows,
        "baseline": baseline_aggregate,
        "intervention": intervention_aggregate,
        "delta": {
            "task_mean_13": intervention_aggregate["task_mean_13"]
            - baseline_aggregate["task_mean_13"],
            "family_macro_5": intervention_aggregate["family_macro_5"]
            - baseline_aggregate["family_macro_5"],
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
