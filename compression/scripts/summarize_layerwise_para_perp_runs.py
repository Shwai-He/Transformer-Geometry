#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


METRICS = (
    "error_norm",
    "para_abs",
    "perp_abs",
    "para_over_base_update",
    "perp_over_base_update",
    "perp_ratio",
    "alpha",
    "base_update_over_hidden_state",
)


def _float(row: dict[str, str], key: str) -> float | None:
    text = row.get(key, "")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _split_run_name(run_dir: Path) -> tuple[str, str]:
    name = run_dir.name
    if "__" in name:
        model, method = name.split("__", 1)
        return model, method
    return "", name


def _summarize_file(path: Path) -> list[dict[str, object]]:
    model, method = _split_run_name(path.parent)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    components = sorted({row["component"] for row in rows if row.get("component")})
    for component in components:
        subset = [row for row in rows if row.get("component") == component and _float(row, "count") not in (None, 0.0)]
        if not subset:
            continue
        result: dict[str, object] = {
            "model": model,
            "method": method,
            "component": component,
            "layers": len(subset),
            "source": str(path),
        }
        for metric in METRICS:
            values = [_float(row, metric) for row in subset]
            values = [v for v in values if v is not None]
            result[f"{metric}_mean"] = f"{mean(values):.8f}" if values else ""
        para = [_float(row, "para_abs") for row in subset]
        perp = [_float(row, "perp_abs") for row in subset]
        error = [_float(row, "error_norm") for row in subset]
        para = [v for v in para if v is not None]
        perp = [v for v in perp if v is not None]
        error = [v for v in error if v is not None]
        result["para_abs_share"] = f"{mean(para) / max(mean(error), 1e-12):.8f}" if para and error else ""
        result["perp_abs_share"] = f"{mean(perp) / max(mean(error), 1e-12):.8f}" if perp and error else ""
        out.append(result)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize layerwise para/perp comparison runs.")
    parser.add_argument("--root", default="compression/outputs/layerwise_para_perp")
    parser.add_argument("--output", default="results/analysis/by_model/quantization_para_perp/summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/layerwise_metrics.csv")):
        rows.extend(_summarize_file(path))

    fields = [
        "model",
        "method",
        "component",
        "layers",
        *[f"{metric}_mean" for metric in METRICS],
        "para_abs_share",
        "perp_abs_share",
        "source",
    ]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
