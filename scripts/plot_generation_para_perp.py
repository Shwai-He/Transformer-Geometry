#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _collect_step_stats(steps: List[Dict[str, Any]], metric_key: str) -> Dict[str, List[float]]:
    x: List[int] = []
    means: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []

    for i, step in enumerate(steps):
        sub = step.get("sublayer_metrics") or []
        vals: List[float] = []
        for layer_item in sub:
            metric = layer_item.get(metric_key) or {}
            ratio = metric.get("para_perp_ratio")
            if ratio is None:
                continue
            vals.append(float(ratio))
        if not vals:
            continue
        x.append(i)
        means.append(sum(vals) / len(vals))
        mins.append(min(vals))
        maxs.append(max(vals))

    return {
        "x": x,
        "mean": means,
        "min": mins,
        "max": maxs,
    }


def _plot_one_run(run_data: Dict[str, Any], output_path: Path, title_prefix: str = "") -> None:
    steps = run_data.get("steps") or []
    if not steps:
        raise ValueError("No steps found in input JSON.")

    block = _collect_step_stats(steps, "block_update")
    attn = _collect_step_stats(steps, "attn_update")
    mlp = _collect_step_stats(steps, "mlp_update")

    if not block["x"] and not attn["x"] and not mlp["x"]:
        raise ValueError(
            "No sublayer_metrics found. Re-run generation probe with --include_sublayer_metrics."
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    specs = [
        ("block", block, "tab:blue"),
        ("attn", attn, "tab:orange"),
        ("mlp", mlp, "tab:green"),
    ]
    for ax, (name, stat, color) in zip(axes, specs):
        if stat["x"]:
            ax.plot(stat["x"], stat["mean"], color=color, label=f"{name} mean")
            ax.fill_between(stat["x"], stat["min"], stat["max"], color=color, alpha=0.2, label=f"{name} range")
        ax.set_title(f"{name.upper()} para_perp_ratio")
        ax.set_ylabel("para_perp_ratio")
        ax.grid(alpha=0.25)
        ax.legend()

    title = "Generation Step para_perp_ratio (mean + min/max range)"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    fig.suptitle(title)
    fig.supxlabel("Generation Step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("Saved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot step-wise para_perp_ratio (mean + min/max range) for block/attn/mlp."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSON output from scripts/run_generation_probe.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/generation_para_perp_ratio.png",
        help="Output image path for single-run input. For multi-run input, suffix _prompt{i}.png is used.",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = Path(args.output)

    if "runs" in data:
        for i, run in enumerate(data["runs"]):
            run_out = out.with_name(f"{out.stem}_prompt{i}{out.suffix}")
            _plot_one_run(run, run_out, title_prefix=f"prompt {i}")
    else:
        _plot_one_run(data, out)


if __name__ == "__main__":
    main()
