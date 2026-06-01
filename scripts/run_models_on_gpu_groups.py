#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import torch


def _first_model(model_name_or_path: str | None, models: List[str], models_file: str | None) -> str:
    if model_name_or_path:
        return model_name_or_path

    for model in models:
        model = model.strip()
        if model:
            return model

    if models_file:
        path = Path(models_file)
        if not path.exists():
            raise FileNotFoundError(f"models_file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line

    raise ValueError("No model provided. Use --model_name_or_path (recommended).")


def _has_cli_flag(args: List[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _script_supports_flag(script_path: str, flag: str) -> bool:
    path = Path(script_path)
    if not path.exists():
        return False
    return flag in path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple single-model multi-GPU launcher (all visible GPUs + device_map sharding)."
    )
    parser.add_argument("--run_type", choices=["probe", "generation"], default="generation")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--models_file", type=str, default=None)
    parser.add_argument(
        "--single_model_only",
        action="store_true",
        help="Backward compatible no-op. This launcher always runs one model.",
    )
    parser.add_argument("--geometry_space", choices=["hidden", "logits"], default="logits")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device_map", choices=["auto", "balanced", "balanced_low_0", "sequential"], default="balanced")
    parser.add_argument("--max_memory_per_gpu", type=str, default=None, help='Example: "80GiB"')
    parser.add_argument("--max_memory_gpu0", type=str, default=None, help='Example: "10GiB"')
    parser.add_argument("--offload_folder", type=str, default=None, help='Example: "/tmp/offload"')
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args, unknown_args = parser.parse_known_args()

    model = _first_model(args.model_name_or_path, args.model, args.models_file)
    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

    target_script = "scripts/run_probe.py" if args.run_type == "probe" else "scripts/run_generation_probe.py"

    extra_args = list(args.extra_args) + list(unknown_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = [
        sys.executable,
        target_script,
        "--model_name_or_path",
        model,
        "--geometry_space",
        args.geometry_space,
    ]

    if not _has_cli_flag(extra_args, "--device_map") and _script_supports_flag(target_script, "--device_map"):
        cmd.extend(["--device_map", args.device_map])

    if args.run_type == "generation" and not _has_cli_flag(extra_args, "--max_new_tokens"):
        cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])

    if args.max_memory_per_gpu is not None:
        if _script_supports_flag(target_script, "--max_memory_per_gpu") and not _has_cli_flag(extra_args, "--max_memory_per_gpu"):
            cmd.extend(["--max_memory_per_gpu", args.max_memory_per_gpu])
        else:
            print(f"[WARN] {target_script} does not support --max_memory_per_gpu; ignored.")

    if args.max_memory_gpu0 is not None:
        if _script_supports_flag(target_script, "--max_memory_gpu0") and not _has_cli_flag(extra_args, "--max_memory_gpu0"):
            cmd.extend(["--max_memory_gpu0", args.max_memory_gpu0])
        else:
            print(f"[WARN] {target_script} does not support --max_memory_gpu0; ignored.")

    if args.offload_folder is not None:
        if _script_supports_flag(target_script, "--offload_folder") and not _has_cli_flag(extra_args, "--offload_folder"):
            cmd.extend(["--offload_folder", args.offload_folder])
        else:
            print(f"[WARN] {target_script} does not support --offload_folder; ignored.")

    if gpu_count >= 2 and not _has_cli_flag(extra_args, "--require_multi_gpu") and _script_supports_flag(target_script, "--require_multi_gpu"):
        cmd.append("--require_multi_gpu")

    cmd.extend(extra_args)

    print(f"[INFO] model={model}")
    print(f"[INFO] visible_gpu_count={gpu_count} (set CUDA_VISIBLE_DEVICES in shell)")
    print(f"[INFO] run_type={args.run_type} geometry_space={args.geometry_space}")
    print("[RUN ]", " ".join(cmd))

    subprocess.run(cmd, check=True)

    print("[DONE] finished")


if __name__ == "__main__":
    main()
