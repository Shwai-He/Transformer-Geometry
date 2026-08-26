#!/usr/bin/env python3
"""Validate an exclude-self position-bin diagnostic against a formal shard."""

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .audit_long_context_results import SNAPSHOT, SOURCE_COMMIT, resolve_git_hash
except ImportError:  # Direct script execution places this directory on sys.path.
    from audit_long_context_results import SNAPSHOT, SOURCE_COMMIT, resolve_git_hash


METRICS = (
    "self_probability",
    "nonself_over_total",
    "self_over_total",
    "para_nonself_over_total",
)
BRANCH = "attn_pre_o_proj_context_excl_self"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def sample_identities(path):
    identities = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            identity = (row.get("doc_id"), row.get("prompt_hash"), row.get("target_hash"))
            if any(item is None for item in identity):
                raise ValueError(f"sample row {line_number} lacks an identity field")
            identities.append(identity)
    if not identities:
        raise ValueError("diagnostic sample file is empty")
    return identities


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--allowed-git-hash", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors = []
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    formal = json.loads(args.formal_audit.read_text(encoding="utf-8"))
    require(finite_tree(payload), "diagnostic result contains nonfinite values", errors)

    result_tasks = list(payload.get("results", {}))
    require(len(result_tasks) == 1, "diagnostic result must contain exactly one task", errors)
    task = result_tasks[0] if len(result_tasks) == 1 else None
    length = payload.get("max_length")
    model_args = payload.get("config", {}).get("model_args", {})
    window = payload.get("config", {}).get("xsa_layer_window", {})
    allowed = set(args.allowed_git_hash or [SOURCE_COMMIT])
    resolved_commit = resolve_git_hash(payload.get("git_hash"), allowed)

    require(str(model_args.get("pretrained", "")).endswith(SNAPSHOT), "snapshot mismatch", errors)
    require(model_args.get("local_files_only") is True, "local_files_only is not true", errors)
    require(payload.get("transformers_version") == "4.52.4", "transformers version mismatch", errors)
    require(resolved_commit is not None, "source commit is not allowed", errors)
    require(length in (65536, 131072), "context length is not 64k or 128k", errors)
    require(model_args.get("max_length") == length, "model/result length mismatch", errors)
    expected_args = {
        "xsa_target": "attn",
        "xsa_start_layer": 9,
        "xsa_end_layer": 28,
        "xsa_forward_alpha": 0.0,
        "xsa_forward_perp_scale": 1.0,
        "xsa_exclude_self": True,
        "xsa_exclude_self_backend": "flash_lse",
        "xsa_track_stats": True,
        "xsa_layerwise_stats": True,
    }
    for key, value in expected_args.items():
        require(model_args.get(key) == value, f"{key} mismatch", errors)
    expected_window = {
        "start_layer": 9,
        "end_layer_exclusive": 28,
        "n_active_layers": 19,
        "intervention_site": "head_specific",
        "track_stats": True,
        "track_layerwise_stats": True,
    }
    for key, value in expected_window.items():
        require(window.get(key) == value, f"runtime {key} mismatch", errors)
    bins = window.get("xsa_stats_position_bins")
    require(isinstance(bins, int) and bins > 0, "position-bin count is not positive", errors)

    stats = payload.get("config", {}).get("xsa_stats", {})
    overall = stats.get("overall", {}).get(BRANCH, {})
    layers = stats.get("layers", {}).get(BRANCH, {})
    require(set(layers) == {str(index) for index in range(9, 28)}, "active-layer stats mismatch", errors)
    if isinstance(bins, int) and bins > 0:
        for location, values in [("overall", overall)] + sorted(layers.items()):
            for metric in METRICS:
                require(metric in values, f"{location} missing {metric}", errors)
                for bin_index in range(1, bins + 1):
                    key = f"{metric}_pos_q{bin_index}"
                    require(key in values, f"{location} missing {key}", errors)

    formal_rows = {row.get("task"): row for row in formal.get("tasks", [])}
    formal_row = formal_rows.get(task, {})
    require(formal.get("arm") == "intervention", "formal audit is not intervention", errors)
    require(formal.get("length") == length, "formal audit length mismatch", errors)
    require(formal_row.get("status") == "pass", "matching formal task is not audited pass", errors)

    matched = []
    try:
        diagnostic_ids = sample_identities(args.samples)
        formal_result = Path(formal_row.get("result", ""))
        formal_sample_paths = sorted(formal_result.parent.glob("samples_*.jsonl"))
        require(len(formal_sample_paths) == 1, "formal shard sample file is not uniquely resolvable", errors)
        formal_ids = set(sample_identities(formal_sample_paths[0])) if len(formal_sample_paths) == 1 else set()
        matched = [identity for identity in diagnostic_ids if identity in formal_ids]
        require(len(matched) == len(diagnostic_ids), "diagnostic sample identity is absent from formal shard", errors)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    receipt = {
        "schema": "transformer-geometry-position-diagnostic-v1",
        "pass": not errors,
        "errors": errors,
        "task": task,
        "length": length,
        "snapshot": SNAPSHOT,
        "git_hash": payload.get("git_hash"),
        "resolved_allowed_git_hash": resolved_commit,
        "transformers_version": payload.get("transformers_version"),
        "position_bins": bins,
        "active_layers": list(range(9, 28)),
        "metrics": list(METRICS),
        "matched_sample_count": len(matched),
        "result": str(args.result.resolve()),
        "result_sha256": sha256(args.result),
        "samples": str(args.samples.resolve()),
        "samples_sha256": sha256(args.samples),
        "formal_audit": str(args.formal_audit.resolve()),
        "formal_audit_sha256": sha256(args.formal_audit),
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
