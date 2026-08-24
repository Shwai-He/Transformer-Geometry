#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot layer-wise metrics from probe output")
    parser.add_argument("--input", type=str, default="results/probe_result.json", help="JSON from run_probe.py")
    parser.add_argument("--output", type=str, default="results/layer_metrics.png")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics = data["layer_metrics"]

    layers = [m["layer"] for m in metrics if m["dz_para_norm"] is not None]
    dz_para = [m["dz_para_norm"] for m in metrics if m["dz_para_norm"] is not None]
    dz_perp = [m["dz_perp_norm"] for m in metrics if m["dz_perp_norm"] is not None]
    gains = [m["scale_gain_to_next"] for m in metrics if m["scale_gain_to_next"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(layers, dz_para, label="dz_para_norm")
    axes[0].plot(layers, dz_perp, label="dz_perp_norm")
    axes[0].set_title("Parallel vs Orthogonal Update Norm")
    axes[0].set_xlabel("Layer")
    axes[0].legend()

    axes[1].plot(layers, gains, label="scale_gain_to_next", color="tab:green")
    axes[1].set_title("Layer-wise Scale Gain")
    axes[1].set_xlabel("Layer")
    axes[1].legend()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
