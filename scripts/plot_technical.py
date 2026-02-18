#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot technical reproduction results")
    parser.add_argument("--input", type=str, required=True, help="JSON from reproduce_technical.py")
    parser.add_argument("--output", type=str, default="results/technical_reproduction.png")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    layers = data["layer_records"]
    if not layers:
        raise ValueError("No layer_records found in input JSON.")

    layer_ids = [r["layer"] for r in layers]
    dz_para = [r["dz_para_norm"] for r in layers]
    dz_perp = [r["dz_perp_norm"] for r in layers]
    ratio = [r["log_para_perp_ratio"] for r in layers]
    gain = [r["scale_gain"] for r in layers]
    overlap = [r["topk_overlap"] for r in layers]
    mhc_ortho = [r["mhc_orthogonality_cos"] for r in layers]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    axes[0, 0].plot(layer_ids, dz_para, label="dz_para_norm")
    axes[0, 0].plot(layer_ids, dz_perp, label="dz_perp_norm")
    axes[0, 0].set_title("Update Norm Decomposition")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].legend()

    axes[0, 1].plot(layer_ids, ratio, color="tab:orange")
    axes[0, 1].set_title("log(dz_para / dz_perp)")
    axes[0, 1].set_xlabel("Layer")

    axes[0, 2].plot(layer_ids, gain, color="tab:green")
    axes[0, 2].set_title("Scale Gain (z_out / z_in)")
    axes[0, 2].set_xlabel("Layer")

    axes[1, 0].plot(layer_ids, overlap, color="tab:red")
    axes[1, 0].set_title("Top-k Overlap (in vs out)")
    axes[1, 0].set_xlabel("Layer")

    axes[1, 1].plot(layer_ids, mhc_ortho, color="tab:purple")
    axes[1, 1].set_title("mHC Orthogonality Cosine")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1)

    trans = data["translation_invariance"]
    axes[1, 2].bar(
        ["mean_abs_diff", "max_abs_diff"],
        [trans["mean_abs_diff"], trans["max_abs_diff"]],
        color=["tab:blue", "tab:gray"],
    )
    axes[1, 2].set_title(f"Softmax Translation Invariance (shift={trans['shift']})")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()

