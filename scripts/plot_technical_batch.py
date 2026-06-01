#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot batch technical reproduction aggregates")
    parser.add_argument("--input", type=str, required=True, help="JSON from reproduce_technical_batch.py")
    parser.add_argument("--output", type=str, default="results/technical_reproduction_batch.png")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    aggregate = data.get("aggregate", {})
    layerwise = aggregate.get("layerwise", [])
    if not layerwise:
        raise ValueError("No aggregate.layerwise found in input JSON.")

    layer_ids = [r["layer"] for r in layerwise]

    def mstd(name: str) -> tuple[list[float], list[float]]:
        means = [float(r[f"{name}_mean"]) for r in layerwise]
        stds = [float(r[f"{name}_std"]) for r in layerwise]
        return means, stds

    dz_para_m, dz_para_s = mstd("dz_para_norm")
    dz_perp_m, dz_perp_s = mstd("dz_perp_norm")
    ratio_m, ratio_s = mstd("log_para_perp_ratio")
    gain_m, gain_s = mstd("scale_gain")
    overlap_m, overlap_s = mstd("topk_overlap")
    ortho_m, ortho_s = mstd("mhc_orthogonality_cos")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].plot(layer_ids, dz_para_m, label="dz_para_norm mean")
    axes[0, 0].fill_between(layer_ids, [m - s for m, s in zip(dz_para_m, dz_para_s)], [m + s for m, s in zip(dz_para_m, dz_para_s)], alpha=0.2)
    axes[0, 0].plot(layer_ids, dz_perp_m, label="dz_perp_norm mean")
    axes[0, 0].fill_between(layer_ids, [m - s for m, s in zip(dz_perp_m, dz_perp_s)], [m + s for m, s in zip(dz_perp_m, dz_perp_s)], alpha=0.2)
    axes[0, 0].set_title("Update Norms (mean ± std)")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].legend()

    axes[0, 1].plot(layer_ids, ratio_m, color="tab:orange")
    axes[0, 1].fill_between(layer_ids, [m - s for m, s in zip(ratio_m, ratio_s)], [m + s for m, s in zip(ratio_m, ratio_s)], color="tab:orange", alpha=0.2)
    axes[0, 1].set_title("log(dz_para / dz_perp)")
    axes[0, 1].set_xlabel("Layer")

    axes[0, 2].plot(layer_ids, gain_m, color="tab:green")
    axes[0, 2].fill_between(layer_ids, [m - s for m, s in zip(gain_m, gain_s)], [m + s for m, s in zip(gain_m, gain_s)], color="tab:green", alpha=0.2)
    axes[0, 2].set_title("Scale Gain (mean ± std)")
    axes[0, 2].set_xlabel("Layer")

    axes[1, 0].plot(layer_ids, overlap_m, color="tab:red")
    axes[1, 0].fill_between(layer_ids, [m - s for m, s in zip(overlap_m, overlap_s)], [m + s for m, s in zip(overlap_m, overlap_s)], color="tab:red", alpha=0.2)
    axes[1, 0].set_title("Top-k Overlap (mean ± std)")
    axes[1, 0].set_xlabel("Layer")

    axes[1, 1].plot(layer_ids, ortho_m, color="tab:purple")
    axes[1, 1].fill_between(layer_ids, [m - s for m, s in zip(ortho_m, ortho_s)], [m + s for m, s in zip(ortho_m, ortho_s)], color="tab:purple", alpha=0.2)
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1, 1].set_title("mHC Orthogonality Cosine")
    axes[1, 1].set_xlabel("Layer")

    trans_m = [
        aggregate["translation_invariance_mean_abs_diff_mean"],
        aggregate["translation_invariance_max_abs_diff_mean"],
    ]
    trans_s = [
        aggregate["translation_invariance_mean_abs_diff_std"],
        aggregate["translation_invariance_max_abs_diff_std"],
    ]
    axes[1, 2].bar(["mean_abs_diff", "max_abs_diff"], trans_m, yerr=trans_s, capsize=4, color=["tab:blue", "tab:gray"])
    axes[1, 2].set_title("Softmax Translation Invariance")

    fig.suptitle(f"Batch Technical Reproduction (n_prompts={data.get('n_prompts', 0)})")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
