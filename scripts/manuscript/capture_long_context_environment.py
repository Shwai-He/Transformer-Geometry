#!/usr/bin/env python3
"""Capture a non-secret environment and model receipt for formal RULER runs."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, cwd=None, env=None):
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=command_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "argv": args,
        "exit_code": completed.returncode,
        "output": completed.stdout.strip(),
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_inventory(snapshot):
    rows = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        rows.append(
            {
                "path": str(path.relative_to(snapshot)),
                "size": resolved.stat().st_size,
                "sha256": sha256(resolved),
                "symlink_target": os.readlink(path) if path.is_symlink() else None,
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rows, hashlib.sha256(canonical).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--core-file", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve(strict=True)
    snapshot = args.model_snapshot.resolve(strict=True)
    inventory, inventory_hash = model_inventory(snapshot)
    torch = importlib.import_module("torch")
    flash_attn = importlib.import_module("flash_attn")
    flash_cuda = importlib.import_module("flash_attn_2_cuda")
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    loader_path = str(torch_lib)
    if os.environ.get("LD_LIBRARY_PATH"):
        loader_path += os.pathsep + os.environ["LD_LIBRARY_PATH"]
    flash_root = Path(flash_attn.__file__).resolve().parent
    shared_object_paths = set(flash_root.rglob("*.so"))
    shared_object_paths.add(Path(flash_cuda.__file__).resolve())
    flash_shared_objects = []
    for shared_object in sorted(shared_object_paths):
        flash_shared_objects.append(
            {
                "path": str(shared_object),
                "size": shared_object.stat().st_size,
                "sha256": sha256(shared_object),
                "ldd": command(
                    ["ldd", str(shared_object)],
                    env={"LD_LIBRARY_PATH": loader_path},
                ),
                "version_info": command(["readelf", "--version-info", str(shared_object)]),
            }
        )
    flash_direct_url = None
    flash_distribution_metadata = None
    try:
        flash_distribution = importlib.metadata.distribution("flash_attn")
        direct_url_path = Path(flash_distribution._path) / "direct_url.json"
        if direct_url_path.is_file():
            flash_direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
        flash_distribution_metadata = {
            "path": str(Path(flash_distribution._path).resolve()),
            "files": [],
        }
        for name in ("METADATA", "WHEEL", "RECORD", "INSTALLER", "direct_url.json"):
            metadata_path = Path(flash_distribution._path) / name
            if metadata_path.is_file():
                flash_distribution_metadata["files"].append(
                    {
                        "path": str(metadata_path.resolve()),
                        "size": metadata_path.stat().st_size,
                        "sha256": sha256(metadata_path),
                    }
                )
    except importlib.metadata.PackageNotFoundError:
        pass

    flash_smoke = {"attempted": False}
    if torch.cuda.is_available():
        flash_smoke["attempted"] = True
        try:
            flash_attn_func = getattr(flash_attn, "flash_attn_func")
            generator = torch.Generator(device="cuda")
            generator.manual_seed(2026)
            shape = (1, 32, 8, 64)
            q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
            k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
            v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
            returned = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                causal=True,
                return_attn_probs=True,
            )
            output, softmax_lse = returned[0], returned[1]
            torch.cuda.synchronize()
            flash_smoke.update(
                {
                    "exit_code": 0,
                    "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "input_shape": list(shape),
                    "return_items": len(returned),
                    "output_shape": list(output.shape),
                    "lse_shape": list(softmax_lse.shape),
                    "output_finite": bool(torch.isfinite(output).all().item()),
                    "lse_finite": bool(torch.isfinite(softmax_lse).all().item()),
                    "causal": True,
                    "dropout_p": 0.0,
                    "return_attn_probs": True,
                }
            )
        except Exception as error:  # receipt must preserve the exact failure
            flash_smoke.update(
                {
                    "exit_code": 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    core_files = []
    for relative in args.core_file:
        path = relative if relative.is_absolute() else repo / relative
        resolved = path.resolve(strict=True)
        core_files.append(
            {
                "path": str(path),
                "size": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )

    launcher = None
    if args.launcher:
        launcher_path = args.launcher.resolve(strict=True)
        launcher = {
            "path": str(launcher_path),
            "size": launcher_path.stat().st_size,
            "sha256": sha256(launcher_path),
        }

    receipt = {
        "schema_version": 1,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "libc": platform.libc_ver(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "datasets",
                "transformers",
                "accelerate",
                "lm_eval",
                "flash_attn",
            )
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "flash_attn": {
            "module_path": str(Path(flash_attn.__file__).resolve()),
            "module_sha256": sha256(Path(flash_attn.__file__).resolve()),
            "direct_url": flash_direct_url,
            "distribution_metadata": flash_distribution_metadata,
            "shared_objects": flash_shared_objects,
            "cuda_smoke": flash_smoke,
        },
        "system_commands": {
            "nvidia_smi": command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": command(["nvcc", "--version"]),
            "ldd": command(["ldd", "--version"]),
        },
        "repository": {
            "root": str(repo),
            "head": command(["git", "rev-parse", "HEAD"], cwd=repo),
            "status_short": command(["git", "status", "--short"], cwd=repo),
            "core_files": core_files,
        },
        "model_snapshot": {
            "path": str(snapshot),
            "commit": snapshot.name,
            "files": inventory,
            "inventory_sha256": inventory_hash,
        },
        "launcher": launcher,
        "receipt_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
