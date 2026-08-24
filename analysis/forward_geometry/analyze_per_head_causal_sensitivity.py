from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


MODEL_ORDER = ["qwen3_0p6b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b"]
MODEL_LABELS = {
    "qwen3_0p6b": "Qwen3-0.6B",
    "qwen3_1p7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def paired_bootstrap(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    point = weighted_mean(values, weights)
    n = len(values)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        estimates[idx] = weighted_mean(values[sample], weights[sample])
    low, high = np.percentile(estimates, [2.5, 97.5])
    p_improve = float(np.mean(estimates < 0.0))
    return point, float(low), float(high), p_improve


def paired_sign_flip_pvalue(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    chunk_size: int = 512,
) -> float:
    weighted = values * weights
    observed = abs(float(np.sum(weighted) / np.sum(weights)))
    extreme = 0
    completed = 0
    while completed < n_permutations:
        current = min(chunk_size, n_permutations - completed)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null = np.abs(signs @ weighted / np.sum(weights))
        extreme += int(np.sum(null >= observed))
        completed += current
    return float((extreme + 1) / (n_permutations + 1))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.tolist()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_model(
    model: str,
    model_dir: Path,
    *,
    n_bootstrap: int,
    n_sign_flips: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    head_rows = read_csv(model_dir / "head_sensitivity.csv")
    sample_rows = read_csv(model_dir / "per_sample_head_sensitivity.csv")
    singles = [row for row in head_rows if row["scope"] == "single_head"]
    all_heads = {
        int(row["layer"]): row for row in head_rows if row["scope"] == "all_heads_one_layer"
    }
    samples_by_head: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in sample_rows:
        samples_by_head[(int(row["layer"]), int(row["head"]))].append(row)

    rng = np.random.default_rng(seed)
    detailed = []
    for row in singles:
        layer = int(row["layer"])
        head = int(row["head"])
        paired = samples_by_head[(layer, head)]
        values = np.asarray([float(item["delta_loss"]) for item in paired], dtype=np.float64)
        weights = np.asarray([float(item["tokens"]) for item in paired], dtype=np.float64)
        point, low, high, p_improve = paired_bootstrap(
            values,
            weights,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        sign_flip_p = paired_sign_flip_pvalue(
            values,
            weights,
            n_permutations=n_sign_flips,
            rng=rng,
        )
        if high < 0:
            ci_class = "improve"
        elif low > 0:
            ci_class = "degrade"
        else:
            ci_class = "inconclusive"
        detailed.append(
            {
                "model": model,
                "layer": layer,
                "head": head,
                "delta_loss": float(row["delta_loss"]),
                "bootstrap_delta_loss": point,
                "ci95_low": low,
                "ci95_high": high,
                "p_improve": p_improve,
                "sign_flip_p": sign_flip_p,
                "ci_class": ci_class,
                "para_ratio": float(row["para_ratio"]),
                "para_over_ref": float(row["para_over_ref"]),
                "edit_over_y": float(row["edit_over_y"]),
                "edit_over_ref": float(row["edit_over_ref"]),
                "samples": len(paired),
                "tokens": int(sum(weights)),
            }
        )

    layer_rows = []
    for layer in sorted(all_heads):
        current = [row for row in detailed if row["layer"] == layer]
        deltas = np.asarray([row["delta_loss"] for row in current])
        summed = float(deltas.sum())
        all_delta = float(all_heads[layer]["delta_loss"])
        layer_rows.append(
            {
                "model": model,
                "layer": layer,
                "heads": len(current),
                "improving_heads": int(np.sum(deltas < 0)),
                "degrading_heads": int(np.sum(deltas > 0)),
                "ci_improving_heads": sum(row["ci_class"] == "improve" for row in current),
                "ci_degrading_heads": sum(row["ci_class"] == "degrade" for row in current),
                "ci_inconclusive_heads": sum(row["ci_class"] == "inconclusive" for row in current),
                "mean_single_head_delta_loss": float(deltas.mean()),
                "sum_single_head_delta_loss": summed,
                "all_heads_delta_loss": all_delta,
                "interaction_residual": all_delta - summed,
            }
        )

    ys = np.asarray([row["delta_loss"] for row in detailed])
    correlation_rows = []
    for metric in ["para_ratio", "para_over_ref", "edit_over_y", "edit_over_ref"]:
        xs = np.asarray([row[metric] for row in detailed])
        correlation_rows.append(
            {
                "model": model,
                "metric": metric,
                "pearson_r": float(pearsonr(xs, ys).statistic),
                "pearson_p": float(pearsonr(xs, ys).pvalue),
                "spearman_rho": float(spearmanr(xs, ys).statistic),
                "spearman_p": float(spearmanr(xs, ys).pvalue),
                "heads": len(detailed),
            }
        )
    return detailed, layer_rows, correlation_rows


def plot_distribution(rows: list[dict], layer_rows: list[dict], output: Path) -> None:
    colors = {"improve": "#2878B5", "degrade": "#C9473D", "inconclusive": "#8A8F98"}
    model = "qwen3_4b"
    current = [row for row in rows if row["model"] == model]
    layers = sorted({row["layer"] for row in current})
    depth_names = ["Early", "Middle", "Late"]
    n_heads = max(row["head"] for row in current) + 1
    joint_x = n_heads + 5

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.76, wspace=0.24)
    for panel_idx, (ax, layer, depth_name) in enumerate(zip(axes, layers, depth_names)):
        layer_items = sorted(
            (row for row in current if row["layer"] == layer), key=lambda item: item["head"]
        )
        xs = np.asarray([row["head"] for row in layer_items])
        ys = np.asarray([1000.0 * row["delta_loss"] for row in layer_items])
        low = np.asarray([1000.0 * row["ci95_low"] for row in layer_items])
        high = np.asarray([1000.0 * row["ci95_high"] for row in layer_items])
        for ci_class in colors:
            mask = np.asarray([row["fdr_global_class"] == ci_class for row in layer_items])
            if np.any(mask):
                ax.scatter(xs[mask], ys[mask], s=34, color=colors[ci_class], zorder=3)
        ax.vlines(xs, low, high, color="#B4B7BD", linewidth=0.9, alpha=0.70, zorder=1)
        layer_summary = next(
            row for row in layer_rows if row["model"] == model and row["layer"] == layer
        )
        ax.scatter(
            joint_x,
            1000.0 * layer_summary["all_heads_delta_loss"],
            marker="*",
            s=145,
            color="#222222",
            zorder=4,
        )
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.axvline(n_heads + 1.5, color="#D7D9DD", linewidth=0.9)
        ax.set_xlim(-1.0, joint_x + 1.2)
        ax.set_xticks([0, 10, 20, 30, joint_x], ["0", "10", "20", "30", "All"])
        ax.set_title(f"({chr(97 + panel_idx)}) {depth_name}: layer {layer}", fontsize=16)
        ax.set_xlabel("Head index", fontsize=14)
        if panel_idx == 0:
            ax.set_ylabel(r"$\Delta$ loss ($\times 10^{-3}$)", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", color="#D9DCE1", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors["improve"], label="Significant loss decrease"),
        plt.Line2D([], [], marker="o", linestyle="", color=colors["degrade"], label="Significant loss increase"),
        plt.Line2D([], [], marker="o", linestyle="", color=colors["inconclusive"], label="Not significant"),
        plt.Line2D([], [], marker="*", linestyle="", color="#222222", markersize=11, label="Joint edit (all heads)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "analysis/visualization/per_head_causal_sensitivity/data"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/figures/per_head_causal_sensitivity"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--sign-flips", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    detailed_rows = []
    layer_rows = []
    correlation_rows = []
    for model_idx, model in enumerate(MODEL_ORDER):
        detailed, layers, correlations = analyze_model(
            model,
            args.result_root / model,
            n_bootstrap=args.bootstrap,
            n_sign_flips=args.sign_flips,
            seed=args.seed + model_idx,
        )
        detailed_rows.extend(detailed)
        layer_rows.extend(layers)
        correlation_rows.extend(correlations)

    global_q = benjamini_hochberg([float(row["sign_flip_p"]) for row in detailed_rows])
    for row, qvalue in zip(detailed_rows, global_q):
        row["sign_flip_q_global"] = qvalue
        if qvalue <= 0.05:
            row["fdr_global_class"] = "improve" if float(row["delta_loss"]) < 0 else "degrade"
        else:
            row["fdr_global_class"] = "inconclusive"
    for model in MODEL_ORDER:
        indices = [idx for idx, row in enumerate(detailed_rows) if row["model"] == model]
        model_q = benjamini_hochberg(
            [float(detailed_rows[idx]["sign_flip_p"]) for idx in indices]
        )
        for idx, qvalue in zip(indices, model_q):
            row = detailed_rows[idx]
            row["sign_flip_q_model"] = qvalue
            if qvalue <= 0.05:
                row["fdr_model_class"] = "improve" if float(row["delta_loss"]) < 0 else "degrade"
            else:
                row["fdr_model_class"] = "inconclusive"

    for layer_row in layer_rows:
        current = [
            row
            for row in detailed_rows
            if row["model"] == layer_row["model"] and row["layer"] == layer_row["layer"]
        ]
        layer_row["fdr_model_improving_heads"] = sum(
            row["fdr_model_class"] == "improve" for row in current
        )
        layer_row["fdr_model_degrading_heads"] = sum(
            row["fdr_model_class"] == "degrade" for row in current
        )
        layer_row["fdr_model_inconclusive_heads"] = sum(
            row["fdr_model_class"] == "inconclusive" for row in current
        )
        layer_row["fdr_global_improving_heads"] = sum(
            row["fdr_global_class"] == "improve" for row in current
        )
        layer_row["fdr_global_degrading_heads"] = sum(
            row["fdr_global_class"] == "degrade" for row in current
        )
        layer_row["fdr_global_inconclusive_heads"] = sum(
            row["fdr_global_class"] == "inconclusive" for row in current
        )

    summary_dir = args.result_root / "summary"
    write_csv(
        summary_dir / "head_bootstrap.csv",
        detailed_rows,
        list(detailed_rows[0]),
    )
    write_csv(summary_dir / "layer_interactions.csv", layer_rows, list(layer_rows[0]))
    write_csv(summary_dir / "geometry_correlations.csv", correlation_rows, list(correlation_rows[0]))
    plot_distribution(detailed_rows, layer_rows, args.figure_dir / "head_delta_loss.pdf")

    status = {
        "models": MODEL_ORDER,
        "bootstrap_resamples": args.bootstrap,
        "sign_flip_permutations": args.sign_flips,
        "seed": args.seed,
        "head_rows": len(detailed_rows),
        "layer_rows": len(layer_rows),
        "figure": str(args.figure_dir / "head_delta_loss.pdf"),
    }
    (summary_dir / "analysis_manifest.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
