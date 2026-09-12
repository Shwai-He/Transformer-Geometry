#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset

DEFAULT_TASKS = (
    "openbookqa,piqa,rte,winogrande,boolq,arc_challenge,hellaswag,"
    "mmlu,gsm8k,humaneval,nq_open,drop,mbpp,bbh_cot_zeroshot"
)


def _patch_datasets_list_feature() -> None:
    import datasets.features.features as features

    if "List" not in getattr(features, "_FEATURE_TYPES", {}):
        features._FEATURE_TYPES["List"] = features.Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mmlu_subjects(repo_root: Path) -> list[str]:
    task_dir = repo_root / "evaluation" / "lm_eval" / "tasks" / "mmlu" / "default"
    subjects: list[str] = []
    for path in sorted(task_dir.glob("mmlu_*.yaml")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r'["\']?dataset_name["\']?\s*:\s*["\']([^"\']+)["\']', text)
        if match:
            subjects.append(match.group(1))
    return sorted(set(subjects))


def _bbh_subjects(repo_root: Path) -> list[str]:
    task_dir = repo_root / "evaluation" / "lm_eval" / "tasks" / "bbh" / "cot_zeroshot"
    subjects: list[str] = []
    for path in sorted(task_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r'["\']?dataset_name["\']?\s*:\s*["\']([^"\']+)["\']', text)
        if match:
            subjects.append(match.group(1))
    return sorted(set(subjects))


def _load_one(dataset_path: str, dataset_name: str | None, splits: list[str], cache_dir: str | None) -> dict:
    started = time.time()
    split_rows = {}
    for split in splits:
        ds = load_dataset(dataset_path, dataset_name, split=split, cache_dir=cache_dir)
        split_rows[split] = len(ds)
    return {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "splits": split_rows,
        "elapsed_sec": round(time.time() - started, 3),
        "status": "ok",
    }


def main() -> int:
    _patch_datasets_list_feature()

    parser = argparse.ArgumentParser(description="Prefetch lm-eval datasets into the shared HF cache.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated tasks to prefetch.")
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--cache-dir", default=None, help="Datasets cache dir. Defaults to <hf-home>/datasets.")
    parser.add_argument("--output", default=None, help="JSON manifest path.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    hf_home = Path(args.hf_home).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else hf_home / "datasets"
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else repo_root / "evaluation" / "outputs" / "task_cache" / "lm_eval_task_download_manifest.json"
    )

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    requested = [item.strip() for item in args.tasks.split(",") if item.strip()]
    jobs: list[tuple[str, str, list[str]]] = []
    if "mmlu" in requested:
        subjects = _mmlu_subjects(repo_root)
        if not subjects:
            raise RuntimeError("No MMLU subject YAML files found.")
        local_mmlu = repo_root / "evaluation" / "outputs" / "task_cache" / "cais_mmlu_git_tmp"
        mmlu_path = str(local_mmlu) if local_mmlu.exists() else "cais/mmlu"
        jobs.extend((mmlu_path, subject, ["dev", "test"]) for subject in subjects)
    if "gsm8k" in requested:
        jobs.append(("openai/gsm8k", "main", ["train", "test"]))
    if "openbookqa" in requested:
        jobs.append(("allenai/openbookqa", "main", ["train", "validation", "test"]))
    if "piqa" in requested:
        jobs.append(("baber/piqa", None, ["train", "validation"]))
    if "rte" in requested:
        jobs.append(("nyu-mll/glue", "rte", ["train", "validation"]))
    if "winogrande" in requested:
        jobs.append(("allenai/winogrande", "winogrande_xl", ["train", "validation"]))
    if "boolq" in requested:
        jobs.append(("aps/super_glue", "boolq", ["train", "validation"]))
    if "arc_challenge" in requested:
        jobs.append(("allenai/ai2_arc", "ARC-Challenge", ["train", "validation", "test"]))
    if "hellaswag" in requested:
        jobs.append(("Rowan/hellaswag", None, ["train", "validation"]))
    if "humaneval" in requested:
        jobs.append(("openai/openai_humaneval", None, ["test"]))
    if "nq_open" in requested:
        jobs.append(("google-research-datasets/nq_open", None, ["train", "validation"]))
    if "drop" in requested:
        jobs.append(("EleutherAI/drop", None, ["train", "validation"]))
    if "mbpp" in requested:
        jobs.append(("google-research-datasets/mbpp", "full", ["test"]))
    if "bbh_cot_zeroshot" in requested:
        subjects = _bbh_subjects(repo_root)
        if not subjects:
            raise RuntimeError("No BBH cot_zeroshot subject YAML files found.")
        jobs.extend(("SaylorTwift/bbh", subject, ["test"]) for subject in subjects)

    manifest = {
        "repo_root": str(repo_root),
        "hf_home": str(hf_home),
        "cache_dir": str(cache_dir),
        "tasks": requested,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": [],
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] HF_HOME={hf_home}", flush=True)
    print(f"[INFO] HF_DATASETS_CACHE={cache_dir}", flush=True)
    print(f"[INFO] jobs={len(jobs)}", flush=True)
    if os.environ.get("HF_ENDPOINT"):
        print(f"[INFO] HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)

    failed = 0
    for idx, (dataset_path, dataset_name, splits) in enumerate(jobs, start=1):
        print(f"[{idx:03d}/{len(jobs):03d}] {dataset_path}/{dataset_name} splits={','.join(splits)}", flush=True)
        try:
            record = _load_one(dataset_path, dataset_name, splits, str(cache_dir))
            print(f"  ok rows={record['splits']} elapsed={record['elapsed_sec']}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            record = {
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "splits": splits,
                "status": "failed",
                "error": repr(exc),
            }
            print(f"  failed: {exc!r}", flush=True)
            if not args.continue_on_error:
                manifest["records"].append(record)
                output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return 1
        manifest["records"].append(record)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["failed"] = failed
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[DONE] manifest={output_path} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
