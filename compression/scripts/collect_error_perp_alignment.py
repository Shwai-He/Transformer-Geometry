#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_ROOT = REPO_ROOT / "compression/outputs/geometry_scores/by_model"
OUT_DIR = REPO_ROOT / "results/analysis/error_perp_alignment"

MODEL_LABELS = {
    "qwen3_0p6b": "Qwen3-0.6B",
    "qwen3_0p6b_base": "Qwen3-0.6B-Base",
    "qwen3_1p7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "phi2": "Phi-2",
    "gemma2_2b_unsloth": "Gemma-2-2B",
    "llama3_2_1b_unsloth": "Llama-3.2-1B",
    "llama3_2_1b_instruct_unsloth": "Llama-3.2-1B-Instruct",
}

SPACES = [
    ("attn_residual_head", ("attn", "residual_total"), ("attn", "residual_perp"), ("attn", "residual_para")),
    (
        "attn_residual_col",
        ("attn", "residual_col_total"),
        ("attn", "residual_col_perp"),
        ("attn", "residual_col_para"),
    ),
    ("attn_value_head", ("attn", "value_total"), ("attn", "value_perp"), ("attn", "value_para")),
    ("mlp_residual_neuron", ("mlp", "residual_total"), ("mlp", "residual_perp"), ("mlp", "residual_para")),
    ("value_col", ("v_proj", "value_col_total"), ("v_proj", "value_col_perp"), ("v_proj", "value_col_para")),
]


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    ok = torch.isfinite(x) & torch.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def ranks(x: torch.Tensor) -> torch.Tensor:
    flat = x.float().flatten()
    order = torch.argsort(flat)
    rank = torch.empty_like(order, dtype=torch.float32)
    rank[order] = torch.arange(flat.numel(), dtype=torch.float32)
    return rank


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    ok = torch.isfinite(x) & torch.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.numel() < 2:
        return float("nan")
    return pearson(ranks(x), ranks(y))


def top_overlap(x: torch.Tensor, y: torch.Tensor, frac: float) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    ok = torch.isfinite(x) & torch.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = x.numel()
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    top_x = torch.topk(x, k, largest=True).indices
    top_y = torch.topk(y, k, largest=True).indices
    mask = torch.zeros(n, dtype=torch.bool)
    mask[top_x] = True
    return float(mask[top_y].float().mean())


def mean_ratio(num: torch.Tensor, denom: torch.Tensor) -> float:
    num = num.float().flatten()
    denom = denom.float().flatten()
    ok = torch.isfinite(num) & torch.isfinite(denom) & (denom > 0)
    if ok.sum() == 0:
        return float("nan")
    return float((num[ok] / denom[ok]).mean())


def mean_layer_spearman(total: torch.Tensor, perp: torch.Tensor) -> float:
    vals = []
    if total.ndim < 2:
        return spearman(total, perp)
    for layer_idx in range(total.shape[0]):
        vals.append(spearman(total[layer_idx], perp[layer_idx]))
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def get_nested(d: dict, path: tuple[str, str]) -> torch.Tensor | None:
    table = d.get(path[0])
    if not isinstance(table, dict):
        return None
    value = table.get(path[1])
    return value if torch.is_tensor(value) else None


def iter_score_files() -> Iterable[tuple[str, Path]]:
    for path in sorted(SCORE_ROOT.glob("*/weight_error_c4_ns128_seq2048/geometry_scores.pt")):
        yield path.parents[1].name, path


def fmt(value: float, digits: int = 3) -> str:
    return "" if value != value else f"{value:.{digits}f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_tag, path in iter_score_files():
        scores = torch.load(path, map_location="cpu")
        for space, total_path, perp_path, para_path in SPACES:
            total = get_nested(scores, total_path)
            perp = get_nested(scores, perp_path)
            para = get_nested(scores, para_path)
            if total is None or perp is None or total.shape != perp.shape:
                continue
            para_total_ratio = mean_ratio(para, total) if para is not None and para.shape == total.shape else float("nan")
            rows.append(
                {
                    "model_tag": model_tag,
                    "model": MODEL_LABELS.get(model_tag, model_tag),
                    "space": space,
                    "num_layers": str(total.shape[0]) if total.ndim >= 2 else "",
                    "num_scores": str(total.numel()),
                    "pearson": fmt(pearson(total, perp)),
                    "spearman": fmt(spearman(total, perp)),
                    "mean_layer_spearman": fmt(mean_layer_spearman(total, perp)),
                    "mean_perp_over_total": fmt(mean_ratio(perp, total)),
                    "mean_para_over_total": fmt(para_total_ratio),
                    "top_1pct_overlap": fmt(top_overlap(total, perp, 0.01)),
                    "top_10pct_overlap": fmt(top_overlap(total, perp, 0.10)),
                    "top_20pct_overlap": fmt(top_overlap(total, perp, 0.20)),
                    "source": str(path.relative_to(REPO_ROOT)),
                }
            )

    csv_path = OUT_DIR / "summary.csv"
    fields = [
        "model",
        "model_tag",
        "space",
        "num_layers",
        "num_scores",
        "pearson",
        "spearman",
        "mean_layer_spearman",
        "mean_perp_over_total",
        "mean_para_over_total",
        "top_1pct_overlap",
        "top_10pct_overlap",
        "top_20pct_overlap",
        "source",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paper_spaces = {"attn_residual_col", "mlp_residual_neuron", "value_col"}
    table_rows = [r for r in rows if r["space"] in paper_spaces]
    md_path = OUT_DIR / "README.md"
    with md_path.open("w") as f:
        f.write("# Overall Error vs. Perpendicular Error Alignment\n\n")
        f.write("Last updated: 2026-05-25.\n\n")
        f.write(
            "This appendix-facing diagnostic compares total/overall geometry error with its perpendicular component. "
            "High Spearman correlation and high top-k overlap indicate that weights or channels with large overall error are largely the same ones with large perpendicular error.\n\n"
        )
        f.write("Calibration: C4, 128 samples, sequence length 2048. Scores are collected from `geometry_scores.pt` files.\n\n")
        f.write("| Model | Space | Spearman | Perp / Total | Para / Total | Top-1% Overlap | Top-10% Overlap |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in table_rows:
            f.write(
                f"| {r['model']} | {r['space']} | {r['spearman']} | {r['mean_perp_over_total']} | {r['mean_para_over_total']} | {r['top_1pct_overlap']} | {r['top_10pct_overlap']} |\n"
            )
        f.write("\nFull machine-readable table: `summary.csv`.\n")

    paper_spaces = {"attn_residual_col", "mlp_residual_neuron", "value_col"}
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if r["space"] in paper_spaces and r["spearman"]:
            grouped.setdefault(r["model"], []).append(r)

    def avg(model_rows: list[dict[str, str]], key: str) -> str:
        vals = [float(r[key]) for r in model_rows if r.get(key)]
        return fmt(sum(vals) / len(vals)) if vals else ""

    paper_rows = []
    for model in sorted(grouped):
        model_rows = grouped[model]
        paper_rows.append(
            {
                "model": model,
                "spaces": str(len(model_rows)),
                "spearman": avg(model_rows, "spearman"),
                "perp_total": avg(model_rows, "mean_perp_over_total"),
                "para_total": avg(model_rows, "mean_para_over_total"),
                "top1": avg(model_rows, "top_1pct_overlap"),
                "top10": avg(model_rows, "top_10pct_overlap"),
            }
        )

    paper_md = OUT_DIR / "paper_table.md"
    with paper_md.open("w") as f:
        f.write(
            "Table: Alignment between overall pruning error and perpendicular error. "
            "Each row averages over attention residual columns, MLP residual neurons, and value-projection columns when available. "
            "Top-k overlap measures the fraction of the largest-k% overall-error entries that also appear among the largest-k% perpendicular-error entries.\n\n"
        )
        f.write("| Model | Spaces | Spearman | Perp./Total | Para./Total | Top-1% Overlap | Top-10% Overlap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in paper_rows:
            f.write(
                f"| {r['model']} | {r['spaces']} | {r['spearman']} | {r['perp_total']} | {r['para_total']} | {r['top1']} | {r['top10']} |\n"
            )

    paper_tex = OUT_DIR / "paper_table.tex"
    with paper_tex.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Spearman $\\uparrow$ & Perp./Total & Para./Total & Top-1\\% & Top-10\\% \\\\\n")
        f.write("\\midrule\n")
        for r in paper_rows:
            f.write(
                f"{r['model']} & {r['spearman']} & {r['perp_total']} & {r['para_total']} & {r['top1']} & {r['top10']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Overall pruning error is nearly aligned with its perpendicular component. "
            "For each model, we average over attention residual columns, MLP residual neurons, and value-projection columns when available. "
            "Top-$k$ reports the overlap between the largest-$k\\%$ entries ranked by overall error and perpendicular error.}\n"
        )
        f.write("\\label{tab:error-perp-alignment}\n")
        f.write("\\end{table}\n")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {paper_md}")
    print(f"[OK] wrote {paper_tex}")


if __name__ == "__main__":
    main()
