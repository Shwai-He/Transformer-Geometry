#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


DEFAULT_VARIANTS = (
    "wanda_residual_para_weight_error_s0.5_unstructured_c4_ns128_seq2048",
    "wanda_residual_perp_weight_error_s0.5_unstructured_c4_ns128_seq2048",
    "wanda_value_para_weight_error_s0.5_unstructured_c4_ns128_seq2048",
    "wanda_value_perp_weight_error_s0.5_unstructured_c4_ns128_seq2048",
)


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _variant_label(name: str) -> str:
    label = name
    if label.startswith("wanda_"):
        label = label[len("wanda_") :]
    label = label.replace("_s0.5_unstructured_c4_ns128_seq2048", "")
    return label


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _summarize_records(model: str, variant: str, records_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _read_records(records_path)
    applied = [row for row in rows if row.get("geometry_applied") == "True"]
    module_rows: list[dict[str, object]] = []
    total_pruned = 0
    total_changed = 0
    jaccards: list[float] = []
    overlaps: list[float] = []

    for row in applied:
        pruned = int(row["pruned"])
        changed = int(row["mask_changed_vs_base"])
        overlap = 1.0 - changed / (2.0 * pruned) if pruned else 1.0
        jaccard = float(row["mask_jaccard_with_base"])
        total_pruned += pruned
        total_changed += changed
        jaccards.append(jaccard)
        overlaps.append(overlap)
        module_rows.append(
            {
                "model": model,
                "variant": _variant_label(variant),
                "layer": row["layer"],
                "linear": row["linear"],
                "geometry_axis": row["geometry_axis"],
                "pruned": pruned,
                "changed_vs_base": changed,
                "overlap_fraction_of_pruned": f"{overlap:.8f}",
                "jaccard_with_base": f"{jaccard:.8f}",
            }
        )

    summary = {
        "model": model,
        "variant": _variant_label(variant),
        "records_path": str(records_path),
        "applied_modules": len(applied),
        "pruned_weights": total_pruned,
        "changed_vs_base": total_changed,
        "overlap_fraction_of_pruned": f"{1.0 - total_changed / (2.0 * total_pruned):.8f}" if total_pruned else "",
        "mean_module_overlap_fraction": f"{mean(overlaps):.8f}" if overlaps else "",
        "mean_module_jaccard": f"{mean(jaccards):.8f}" if jaccards else "",
        "min_module_jaccard": f"{min(jaccards):.8f}" if jaccards else "",
    }
    return summary, module_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze overlap between geometry-pruned masks and baseline WANDA masks.")
    parser.add_argument(
        "--root",
        default="compression/outputs/wanda_layerwise_pruned",
        help="Root containing per-model WANDA/geometry checkpoint directories.",
    )
    parser.add_argument("--models", default="qwen3_1p7b,qwen3_4b", help="Comma-separated model directory names.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated geometry checkpoint directory names to inspect.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/quality_eval/by_model/baseline_full_compression/mask_overlap",
        help="Directory for summary and per-module CSV outputs.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    summary_rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []

    for model in _parse_csv(args.models):
        for variant in _parse_csv(args.variants):
            records_path = root / model / variant / "prune_records.csv"
            if not records_path.exists():
                continue
            summary, modules = _summarize_records(model, variant, records_path)
            summary_rows.append(summary)
            module_rows.extend(modules)

    summary_fields = [
        "model",
        "variant",
        "applied_modules",
        "pruned_weights",
        "changed_vs_base",
        "overlap_fraction_of_pruned",
        "mean_module_overlap_fraction",
        "mean_module_jaccard",
        "min_module_jaccard",
        "records_path",
    ]
    module_fields = [
        "model",
        "variant",
        "layer",
        "linear",
        "geometry_axis",
        "pruned",
        "changed_vs_base",
        "overlap_fraction_of_pruned",
        "jaccard_with_base",
    ]
    _write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    _write_csv(output_dir / "by_module.csv", module_rows, module_fields)

    print(f"[OK] wrote {output_dir / 'summary.csv'}")
    print(f"[OK] wrote {output_dir / 'by_module.csv'}")


if __name__ == "__main__":
    main()
