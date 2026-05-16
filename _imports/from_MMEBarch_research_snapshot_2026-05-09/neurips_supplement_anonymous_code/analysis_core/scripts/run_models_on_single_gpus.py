#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _load_models(models: List[str], models_file: str | None) -> List[str]:
    loaded = list(models)
    if models_file:
        path = Path(models_file)
        if not path.exists():
            raise FileNotFoundError(f"models_file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                loaded.append(line)
    seen = set()
    deduped = []
    for item in loaded:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple models, each pinned to a single GPU.")
    parser.add_argument(
        "--run_type",
        type=str,
        choices=["probe", "generation"],
        default="probe",
        help="Choose which script to run: run_probe.py or run_generation_probe.py",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model path/id. Repeat this flag for multiple models.",
    )
    parser.add_argument(
        "--models_file",
        type=str,
        default=None,
        help="Text file with one model path/id per line. '#' lines are ignored.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated GPU ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Unused legacy arg kept for backward compatibility.",
    )
    parser.add_argument(
        "--geometry_space",
        type=str,
        choices=["hidden", "logits"],
        default="hidden",
        help="Geometry space passed to run_probe/run_generation_probe.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run one-by-one instead of one process per GPU in parallel.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to target script (prefix with --).",
    )
    args = parser.parse_args()

    models = _load_models(args.model, args.models_file)
    if not models:
        raise ValueError("No models provided. Use --model and/or --models_file.")

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No valid GPU ids provided in --gpus.")

    target_script = "scripts/run_probe.py" if args.run_type == "probe" else "scripts/run_generation_probe.py"

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    jobs = []
    for idx, model in enumerate(models):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        cmd = [
            sys.executable,
            target_script,
            "--model_name_or_path",
            model,
            "--device",
            "cuda:0",
            "--geometry_space",
            args.geometry_space,
            *extra_args,
        ]
        jobs.append((model, gpu_id, cmd))

    if args.sequential:
        for model, gpu_id, cmd in jobs:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id)
            print(f"[RUN ] model={model} gpu={gpu_id}")
            subprocess.run(cmd, check=True, env=env)
            print(f"[DONE] model={model} gpu={gpu_id}")
        return

    running = {}
    queue = list(jobs)
    while queue or running:
        launched = False
        for i, (model, gpu_id, cmd) in enumerate(queue):
            if gpu_id in running:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id)
            print(f"[RUN ] model={model} gpu={gpu_id}")
            proc = subprocess.Popen(cmd, env=env)
            running[gpu_id] = (proc, model)
            queue.pop(i)
            launched = True
            break

        if launched:
            continue

        finished_gpu = None
        for gpu_id, (proc, model) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            if rc != 0:
                raise RuntimeError(f"Job failed: model={model}, gpu={gpu_id}, exit_code={rc}")
            print(f"[DONE] model={model} gpu={gpu_id}")
            finished_gpu = gpu_id
            break
        if finished_gpu is not None:
            running.pop(finished_gpu)
            continue

        time.sleep(0.2)


if __name__ == "__main__":
    main()
