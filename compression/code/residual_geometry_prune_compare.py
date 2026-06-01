#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from geometric_compression_metric import _append_metric, _capture_model, _read_prompts
from geometry_aware_pruning import (
    GeometryScoreCollector,
    _dtype_from_name,
    _find_decoder_layers,
    _model_tag,
    apply_geometry_pruning,
)


def _load_model(path: str, device: str, dtype: str, local_files_only: bool):
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    torch_dtype = _dtype_from_name(dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs).to(device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _encoded_prompts(tokenizer, prompts: list[str], device: str, max_length: int):
    encoded = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        encoded.append(
            {
                "input_ids": enc["input_ids"].to(device),
                "attention_mask": enc["attention_mask"].to(device),
            }
        )
    return encoded


def _collect_dense_context(model, encoded, token_scope: str) -> tuple[dict[str, Any], list[dict[str, list[Any]]]]:
    collector = GeometryScoreCollector(model, token_scope=token_scope)
    collector.install()
    captures = []
    try:
        for enc in encoded:
            captures.append(_capture_model(model, enc["input_ids"], enc["attention_mask"]))
    finally:
        collector.remove()
    return collector.to_tensors(), captures


def _compare_residual_components(
    *,
    method_name: str,
    dense_caps: list[dict[str, list[Any]]],
    comp_caps: list[dict[str, list[Any]]],
    n_layers: int,
) -> list[dict[str, Any]]:
    sum_rows: dict[tuple[str, int], dict[str, float]] = {}
    counts: dict[tuple[str, int], int] = {}

    for dense_cap, comp_cap in zip(dense_caps, comp_caps):
        rows: list[dict[str, Any]] = []
        for li in range(n_layers):
            h0 = dense_cap["hidden"][li]
            h1 = dense_cap["hidden"][li + 1]
            h1c = comp_cap["hidden"][li + 1]
            if h0 is not None and h1 is not None and h1c is not None:
                _append_metric(
                    rows,
                    space="residual",
                    component="block_out",
                    layer=li,
                    error=h1c - h1,
                    ref=h1 - h0,
                )

            attn_dense = dense_cap["attn_out"][li]
            attn_comp = comp_cap["attn_out"][li]
            if attn_dense is not None and attn_comp is not None:
                _append_metric(
                    rows,
                    space="residual",
                    component="attn_out",
                    layer=li,
                    error=attn_comp - attn_dense,
                    ref=attn_dense,
                )

            mlp_dense = dense_cap["mlp_out"][li]
            mlp_comp = comp_cap["mlp_out"][li]
            if mlp_dense is not None and mlp_comp is not None:
                _append_metric(
                    rows,
                    space="residual",
                    component="mlp_out",
                    layer=li,
                    error=mlp_comp - mlp_dense,
                    ref=mlp_dense,
                )

        for row in rows:
            key = (row["component"], int(row["layer"]))
            if key not in sum_rows:
                sum_rows[key] = {k: 0.0 for k in row if k not in {"space", "component", "layer"}}
                counts[key] = 0
            counts[key] += 1
            for k, v in row.items():
                if k not in {"space", "component", "layer"}:
                    sum_rows[key][k] += float(v)

    out = []
    for key in sorted(sum_rows):
        component, layer = key
        count = counts[key]
        row = {
            "method": method_name,
            "space": "residual",
            "component": component,
            "layer": layer,
            "count": count,
        }
        row.update({k: v / count for k, v in sum_rows[key].items()})
        out.append(row)
    return out


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for component in sorted({row["component"] for row in rows}):
        subset = [row for row in rows if row["component"] == component]
        if not subset:
            continue
        summary[component] = {
            "mean_error_over_ref": float(sum(row["error_over_ref"] for row in subset) / len(subset)),
            "mean_parallel_over_ref": float(sum(row["parallel_over_ref"] for row in subset) / len(subset)),
            "mean_perpendicular_over_ref": float(sum(row["perpendicular_over_ref"] for row in subset) / len(subset)),
            "mean_perp_energy_ratio": float(sum(row["perp_energy_ratio"] for row in subset) / len(subset)),
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "space",
        "component",
        "layer",
        "count",
        "error_norm",
        "ref_norm",
        "error_over_ref",
        "parallel_over_ref",
        "perpendicular_over_ref",
        "perp_energy_ratio",
        "signed_alpha",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="In-memory residual-level comparison for geometry-aware pruning.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--max_prompts", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--token_scope", choices=["last", "all"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prune_method", choices=["wanda", "magnitude"], default="wanda")
    parser.add_argument("--strategies", default="none,residual_perp,residual_para,residual_perp_over_para")
    parser.add_argument("--geometry_mode", choices=["contribution", "residual_error"], default="residual_error")
    parser.add_argument("--geometry_alpha", type=float, default=1.0)
    parser.add_argument("--geometry_targets", default="o_proj,down_proj,v_proj")
    parser.add_argument("--sparsity_ratio", type=float, default=0.1)
    parser.add_argument("--sparsity_type", choices=["unstructured", "2:4", "4:8"], default="unstructured")
    parser.add_argument("--threshold_scope", choices=["row", "global"], default="global")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    if not args.output_dir:
        model_tag = _model_tag(args.model_name_or_path)
        args.output_dir = f"compression/outputs/residual_prune_compare/by_model/{model_tag}/{args.prune_method}_s{args.sparsity_ratio:g}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = _read_prompts(args)
    encoded = _encoded_prompts(tokenizer, prompts, args.device, args.max_length)

    dense = _load_model(args.model_name_or_path, args.device, args.dtype, args.local_files_only)
    scores, dense_caps = _collect_dense_context(dense, encoded, token_scope=args.token_scope)
    n_layers = len(_find_decoder_layers(dense))
    torch.save(scores, out_dir / "geometry_scores.pt")
    del dense
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_rows = []
    summaries = {}
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    for strategy in strategies:
        comp = _load_model(args.model_name_or_path, args.device, args.dtype, args.local_files_only)
        prune_args = SimpleNamespace(
            prune_method=args.prune_method,
            geometry_strategy=strategy,
            geometry_mode=args.geometry_mode,
            geometry_alpha=args.geometry_alpha,
            geometry_targets=args.geometry_targets,
            sparsity_ratio=args.sparsity_ratio,
            sparsity_type=args.sparsity_type,
            threshold_scope=args.threshold_scope,
        )
        records = apply_geometry_pruning(comp, scores, prune_args)
        comp_caps = [_capture_model(comp, enc["input_ids"], enc["attention_mask"]) for enc in encoded]
        method_name = f"{args.prune_method}_{strategy}"
        rows = _compare_residual_components(
            method_name=method_name,
            dense_caps=dense_caps,
            comp_caps=comp_caps,
            n_layers=n_layers,
        )
        all_rows.extend(rows)
        summaries[method_name] = {
            "pruned_modules": len(records),
            "mean_module_sparsity": float(sum(item["sparsity"] for item in records) / len(records)) if records else 0.0,
            "components": _method_summary(rows),
        }
        del comp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = out_dir / "residual_layerwise_metrics.csv"
    _write_csv(csv_path, all_rows)
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "num_prompts": len(prompts),
        "num_layers": n_layers,
        "prune_method": args.prune_method,
        "strategies": strategies,
        "geometry_targets": args.geometry_targets,
        "geometry_mode": args.geometry_mode,
        "geometry_alpha": args.geometry_alpha,
        "sparsity_ratio": args.sparsity_ratio,
        "sparsity_type": args.sparsity_type,
        "threshold_scope": args.threshold_scope,
        "csv_path": str(csv_path),
        "methods": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
