#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense and pruned checkpoint parameter sparsity.")
    parser.add_argument("--dense_model", required=True)
    parser.add_argument("--pruned_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def load_model(path: str, dtype: torch.dtype, device: str, local_files_only: bool):
    return AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def main() -> None:
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    dense = load_model(args.dense_model, dtype, args.device, args.local_files_only)
    pruned = load_model(args.pruned_model, dtype, args.device, args.local_files_only)

    dense_params = dict(dense.named_parameters())
    pruned_params = dict(pruned.named_parameters())
    rows = []
    total = 0
    total_zero_dense = 0
    total_zero_pruned = 0
    total_changed = 0

    for name, dense_tensor in dense_params.items():
        pruned_tensor = pruned_params.get(name)
        if pruned_tensor is None:
            rows.append({"name": name, "status": "missing_in_pruned"})
            continue
        if tuple(dense_tensor.shape) != tuple(pruned_tensor.shape):
            rows.append(
                {
                    "name": name,
                    "status": "shape_mismatch",
                    "dense_shape": tuple(dense_tensor.shape),
                    "pruned_shape": tuple(pruned_tensor.shape),
                }
            )
            continue

        dense_cpu = dense_tensor.detach().float().cpu()
        pruned_cpu = pruned_tensor.detach().float().cpu()
        count = dense_cpu.numel()
        zero_dense = int((dense_cpu == 0).sum().item())
        zero_pruned = int((pruned_cpu == 0).sum().item())
        changed = int((dense_cpu != pruned_cpu).sum().item())
        newly_zero = int(((dense_cpu != 0) & (pruned_cpu == 0)).sum().item())
        total += count
        total_zero_dense += zero_dense
        total_zero_pruned += zero_pruned
        total_changed += changed
        rows.append(
            {
                "name": name,
                "status": "ok",
                "numel": count,
                "dense_zero": zero_dense,
                "pruned_zero": zero_pruned,
                "newly_zero": newly_zero,
                "changed": changed,
                "dense_sparsity": zero_dense / count,
                "pruned_sparsity": zero_pruned / count,
                "newly_zero_ratio": newly_zero / count,
                "changed_ratio": changed / count,
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "checkpoint_diff.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "status",
            "numel",
            "dense_zero",
            "pruned_zero",
            "newly_zero",
            "changed",
            "dense_sparsity",
            "pruned_sparsity",
            "newly_zero_ratio",
            "changed_ratio",
            "dense_shape",
            "pruned_shape",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dense_model": args.dense_model,
        "pruned_model": args.pruned_model,
        "total_numel": total,
        "dense_zero": total_zero_dense,
        "pruned_zero": total_zero_pruned,
        "changed": total_changed,
        "dense_sparsity": total_zero_dense / max(total, 1),
        "pruned_sparsity": total_zero_pruned / max(total, 1),
        "changed_ratio": total_changed / max(total, 1),
        "num_params_checked": len(rows),
        "num_problem_rows": sum(1 for row in rows if row["status"] != "ok"),
    }
    (out / "checkpoint_diff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
