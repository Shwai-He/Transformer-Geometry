#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import torch


COMPONENT_GROUPS = {
    "attn_value_head": ("attn", "value_total", "value_para", "value_perp"),
    "attn_residual_head": ("attn", "residual_total", "residual_para", "residual_perp"),
    "attn_residual_col": ("attn", "residual_col_total", "residual_col_para", "residual_col_perp"),
    "mlp_residual_col": ("mlp", "residual_total", "residual_para", "residual_perp"),
    "v_proj_value_col": ("v_proj", "value_col_total", "value_col_para", "value_col_perp"),
}

UNIT_GROUPS = {
    "attn_output_unit": ("output", None, "attn_para_unit", "attn_perp_unit"),
    "mlp_output_unit": ("output", None, "mlp_para_unit", "mlp_perp_unit"),
    "value_output_unit": ("output", None, "value_para_unit", "value_perp_unit"),
    "attn_value_output_unit": ("output", None, "attn_value_para_unit", "attn_value_perp_unit"),
}


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(-1)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _flat(a)
    b = _flat(b)
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def _top_overlap(a: torch.Tensor, b: torch.Tensor, frac: float) -> float:
    a = _flat(a)
    b = _flat(b)
    if a.numel() != b.numel() or a.numel() == 0:
        return float("nan")
    k = max(1, int(a.numel() * frac))
    ia = torch.topk(a, k, largest=True).indices
    ib = torch.topk(b, k, largest=True).indices
    mask = torch.zeros(a.numel(), dtype=torch.bool)
    mask[ia] = True
    return float(mask[ib].float().mean())


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    x = _flat(x)
    qs = torch.quantile(x, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))
    return {
        "min": float(x.min()),
        "p01": float(qs[0]),
        "p10": float(qs[1]),
        "p50": float(qs[2]),
        "p90": float(qs[3]),
        "p99": float(qs[4]),
        "max": float(x.max()),
    }


def _base_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    xf = _flat(x)
    stats = {
        f"{prefix}_mean": float(xf.mean()),
        f"{prefix}_std": float(xf.std()),
        f"{prefix}_cv": float(xf.std() / xf.mean().abs().clamp_min(1e-12)),
    }
    for name, value in _quantiles(xf).items():
        stats[f"{prefix}_{name}"] = value
    return stats


def _component_rows(model: str, group: str, total: torch.Tensor | None, para: torch.Tensor, perp: torch.Tensor) -> tuple[dict[str, object], list[dict[str, object]]]:
    eps = 1e-12
    para = para.float()
    perp = perp.float()
    if total is None:
        total = (para.square() + perp.square()).sqrt()
    else:
        total = total.float()

    para_ratio = para / total.clamp_min(eps)
    perp_ratio = perp / total.clamp_min(eps)
    row = {
        "model": model,
        "group": group,
        "shape": "x".join(str(dim) for dim in total.shape),
        **_base_stats(total, "total"),
        **_base_stats(para, "para"),
        **_base_stats(perp, "perp"),
        **_base_stats(para_ratio, "para_over_total"),
        **_base_stats(perp_ratio, "perp_over_total"),
        "corr_total_para": _pearson(total, para),
        "corr_total_perp": _pearson(total, perp),
        "corr_para_perp": _pearson(para, perp),
        "top01_total_para_overlap": _top_overlap(total, para, 0.01),
        "top01_total_perp_overlap": _top_overlap(total, perp, 0.01),
        "top10_total_para_overlap": _top_overlap(total, para, 0.10),
        "top10_total_perp_overlap": _top_overlap(total, perp, 0.10),
    }

    layer_rows: list[dict[str, object]] = []
    if total.ndim >= 2:
        for layer_idx in range(total.shape[0]):
            layer_total = total[layer_idx]
            layer_para = para[layer_idx]
            layer_perp = perp[layer_idx]
            layer_para_ratio = layer_para / layer_total.clamp_min(eps)
            layer_perp_ratio = layer_perp / layer_total.clamp_min(eps)
            layer_rows.append(
                {
                    "model": model,
                    "group": group,
                    "layer": layer_idx,
                    "shape": "x".join(str(dim) for dim in layer_total.shape),
                    "total_mean": float(_flat(layer_total).mean()),
                    "para_mean": float(_flat(layer_para).mean()),
                    "perp_mean": float(_flat(layer_perp).mean()),
                    "para_over_total_mean": float(_flat(layer_para_ratio).mean()),
                    "perp_over_total_mean": float(_flat(layer_perp_ratio).mean()),
                    "corr_total_para": _pearson(layer_total, layer_para),
                    "corr_total_perp": _pearson(layer_total, layer_perp),
                    "top10_total_para_overlap": _top_overlap(layer_total, layer_para, 0.10),
                    "top10_total_perp_overlap": _top_overlap(layer_total, layer_perp, 0.10),
                }
            )
    return row, layer_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _load_scores(root: Path, model: str, score_dir: str) -> dict:
    path = root / model / score_dir / "geometry_scores.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize total/para/perp geometry score components.")
    parser.add_argument(
        "--root",
        default="compression/outputs/geometry_scores/by_model",
        help="Root containing model/score-dir/geometry_scores.pt files.",
    )
    parser.add_argument("--models", default="qwen3_1p7b,qwen3_4b", help="Comma-separated model score directories.")
    parser.add_argument("--score_dir", default="weight_error_c4_ns128_seq2048")
    parser.add_argument(
        "--output_dir",
        default="results/quality_eval/by_model/baseline_full_compression/score_components",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    summary_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []

    for model in _parse_csv(args.models):
        scores = _load_scores(root, model, args.score_dir)
        for group, (table_name, total_key, para_key, perp_key) in {**COMPONENT_GROUPS, **UNIT_GROUPS}.items():
            table = scores.get(table_name, {})
            if not isinstance(table, dict):
                continue
            para = table.get(para_key)
            perp = table.get(perp_key)
            total = table.get(total_key) if total_key is not None else None
            if para is None or perp is None:
                continue
            row, rows = _component_rows(model, group, total, para, perp)
            summary_rows.append(row)
            layer_rows.extend(rows)

    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["model", "group"]
    layer_fields = list(layer_rows[0].keys()) if layer_rows else ["model", "group", "layer"]
    _write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    _write_csv(output_dir / "by_layer.csv", layer_rows, layer_fields)
    print(f"[OK] wrote {output_dir / 'summary.csv'}")
    print(f"[OK] wrote {output_dir / 'by_layer.csv'}")


if __name__ == "__main__":
    main()
