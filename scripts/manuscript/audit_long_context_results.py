#!/usr/bin/env python3
"""Validate formal RULER artifacts before camera-ready aggregation."""

import argparse
import hashlib
import json
import math
import re
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
SNAPSHOT = "13afe5124825b4f3751f836b40dafda64c1ed062"
SOURCE_COMMIT = "e0cbc87b21c72dc88bfda8b46752eab22bbeff0c"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_finite(value, location, errors):
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"nonfinite value at {location}: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            scan_finite(item, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan_finite(item, f"{location}[{index}]", errors)


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def resolve_git_hash(observed, allowed_git_hashes):
    """Resolve Git's abbreviated hashes without weakening commit binding."""
    if not isinstance(observed, str) or not GIT_HASH_PATTERN.fullmatch(observed):
        return None
    matches = {
        allowed
        for allowed in allowed_git_hashes
        if isinstance(allowed, str)
        and GIT_HASH_PATTERN.fullmatch(allowed)
        and (allowed.startswith(observed) or observed.startswith(allowed))
    }
    return next(iter(matches)) if len(matches) == 1 else None


def find_result(task_dir):
    paths = [
        path
        for path in task_dir.glob("*.json")
        if path.is_file() and ".layer_scales" not in path.as_posix()
    ]
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"expected one result JSON in {task_dir}, found {len(paths)}")
    return paths[0]


def validate_task(root, task, length, arm, allowed_git_hashes):
    task_dir = root / task
    result_path = find_result(task_dir)
    if result_path is None:
        return {"task": task, "status": "missing"}

    errors: list[str] = []
    warnings = []
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    scan_finite(payload, "$", errors)
    model_args = payload.get("config", {}).get("model_args", {})
    task_result = payload.get("results", {}).get(task, {})
    metric_key = f"{length},none"
    position_diagnostics_present = arm == "baseline"

    require(str(model_args.get("pretrained", "")).endswith(SNAPSHOT), "snapshot mismatch", errors)
    require(model_args.get("local_files_only") is True, "local_files_only is not true", errors)
    require(model_args.get("max_length") == length, "model max_length mismatch", errors)
    require(payload.get("max_length") == length, "result max_length mismatch", errors)
    require(payload.get("transformers_version") == "4.52.4", "transformers version mismatch", errors)
    resolved_git_hash = resolve_git_hash(payload.get("git_hash"), allowed_git_hashes)
    require(
        resolved_git_hash is not None,
        "lm-eval source commit mismatch: {} not in {}".format(
            payload.get("git_hash"), sorted(allowed_git_hashes)
        ),
        errors,
    )
    if payload.get("upper_git_hash") != SOURCE_COMMIT:
        warnings.append(
            "parent-worktree hash differs: {} (lm-eval git_hash is {})".format(
                payload.get("upper_git_hash"), payload.get("git_hash")
            )
        )
    require(payload.get("config", {}).get("limit") is None, "result used a sample limit", errors)
    require(task_result.get("sample_len") == 500, "task sample_len is not 500", errors)
    require(
        payload.get("n-samples", {}).get(task, {}).get("effective") == 500,
        "effective sample count is not 500",
        errors,
    )
    require(metric_key in task_result, f"missing metric {metric_key}", errors)

    if arm == "baseline":
        require(model_args.get("xsa_target") == "none", "baseline has active XSA target", errors)
        require(
            model_args.get("attn_implementation") == "flash_attention_2",
            "baseline is not FlashAttention-2 matched",
            errors,
        )
    else:
        expected = {
            "xsa_target": "attn",
            "xsa_start_layer": 9,
            "xsa_end_layer": 28,
            "xsa_forward_alpha": 0.0,
            "xsa_forward_perp_scale": 1.0,
            "xsa_exclude_self": True,
            "xsa_exclude_self_backend": "flash_lse",
        }
        for key, value in expected.items():
            require(model_args.get(key) == value, f"{key} mismatch", errors)
        layer_window = payload.get("config", {}).get("xsa_layer_window", {})
        runtime_expected = {
            "start_layer": 9,
            "end_layer_exclusive": 28,
            "n_active_layers": 19,
            "intervention_site": "head_specific",
            "track_stats": True,
            "track_layerwise_stats": True,
            "xsa_exclude_self": True,
            "xsa_forward_alpha": 0.0,
            "xsa_forward_perp_scale": 1.0,
        }
        for key, value in runtime_expected.items():
            require(layer_window.get(key) == value, f"runtime {key} mismatch", errors)
        overall_stats = (
            payload.get("config", {})
            .get("xsa_stats", {})
            .get("overall", {})
            .get("attn_pre_o_proj_context_excl_self", {})
        )
        for key in (
            "self_probability",
            "nonself_over_total",
            "self_over_total",
            "para_nonself_over_total",
        ):
            require(key in overall_stats, f"missing intervention statistic {key}", errors)
        position_bins = layer_window.get("xsa_stats_position_bins")
        position_diagnostics_present = isinstance(position_bins, int) and position_bins > 0
        if not isinstance(position_bins, int) or position_bins <= 0:
            warnings.append(
                "position-bin diagnostics are disabled; a separate reviewed "
                "position-diagnostic receipt is required before paper integration"
            )

    sample_paths = sorted(task_dir.glob("samples_*.jsonl"))
    require(len(sample_paths) == 1, f"expected one sample JSONL, found {len(sample_paths)}", errors)
    sample_count = None
    sample_hash = None
    input_signature_hash = None
    if len(sample_paths) == 1:
        sample_count = 0
        input_signature = hashlib.sha256()
        doc_ids = set()
        with sample_paths[0].open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                scan_finite(row, f"samples[{line_number}]", errors)
                doc_id = row.get("doc_id")
                prompt_hash = row.get("prompt_hash")
                target_hash = row.get("target_hash")
                require(doc_id is not None, f"samples[{line_number}] missing doc_id", errors)
                require(
                    isinstance(prompt_hash, str) and SHA256_PATTERN.fullmatch(prompt_hash),
                    f"samples[{line_number}] has invalid prompt_hash",
                    errors,
                )
                require(
                    isinstance(target_hash, str) and SHA256_PATTERN.fullmatch(target_hash),
                    f"samples[{line_number}] has invalid target_hash",
                    errors,
                )
                if doc_id is not None:
                    require(
                        doc_id not in doc_ids,
                        f"samples[{line_number}] duplicates doc_id {doc_id!r}",
                        errors,
                    )
                    doc_ids.add(doc_id)
                signature_row = [
                    doc_id,
                    prompt_hash,
                    target_hash,
                ]
                input_signature.update(
                    (json.dumps(signature_row, separators=(",", ":")) + "\n").encode("utf-8")
                )
                sample_count += 1
        require(sample_count == 500, "sample JSONL does not contain 500 rows", errors)
        require(len(doc_ids) == 500, "sample JSONL does not contain 500 unique doc_ids", errors)
        sample_hash = sha256(sample_paths[0])
        input_signature_hash = input_signature.hexdigest()

    return {
        "task": task,
        "status": "pass" if not errors else "fail",
        "metric": task_result.get(metric_key),
        "samples": sample_count,
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "samples_sha256": sample_hash,
        "input_signature_sha256": input_signature_hash,
        "position_diagnostics_present": position_diagnostics_present,
        "git_hash": payload.get("git_hash"),
        "resolved_allowed_git_hash": resolved_git_hash,
        "task_hash": payload.get("task_hashes", {}).get(task),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(65536, 131072), required=True)
    parser.add_argument("--arm", choices=("baseline", "intervention"), required=True)
    parser.add_argument(
        "--allowed-git-hash",
        action="append",
        default=[],
        help="Accepted lm-eval git_hash; repeat only for an explicitly reviewed mixed-source ledger.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    allowed_git_hashes = set(args.allowed_git_hash or [SOURCE_COMMIT[:8]])
    rows = [
        validate_task(args.root, task, args.length, args.arm, allowed_git_hashes)
        for task in TASKS
    ]
    summary = {
        "root": str(args.root.resolve()),
        "length": args.length,
        "arm": args.arm,
        "allowed_git_hashes": sorted(allowed_git_hashes),
        "expected_tasks": len(TASKS),
        "passed": sum(row["status"] == "pass" for row in rows),
        "failed": sum(row["status"] == "fail" for row in rows),
        "missing": sum(row["status"] == "missing" for row in rows),
        "complete": all(row["status"] == "pass" for row in rows),
        "tasks": rows,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
