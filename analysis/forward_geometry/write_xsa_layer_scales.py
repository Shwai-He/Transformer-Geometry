#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_num_layers(model_name: str) -> int:
    path = Path(model_name)
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is not None and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            for key in ("num_hidden_layers", "n_layer", "num_layers"):
                value = text_config.get(key)
                if value is not None:
                    return int(value)
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            value = config.get(key)
            if value is not None:
                return int(value)
    raise ValueError(f"Could not infer number of layers from {model_name!r}")


def sample_para_scales(mode: str, n_layers: int, seed: int, default_scale: float, lo: float, hi: float):
    rng = random.Random(int(seed))
    mode = str(mode).lower().strip()
    rows = []
    if mode.startswith("sample_"):
        mode = "none"
    for layer_idx in range(n_layers):
        scale = float(default_scale)
        if mode == "layer_para_uniform":
            scale = rng.uniform(float(lo), float(hi))
        elif mode == "layer_para_choice":
            scale = rng.choice([float(lo), float(hi)])
        elif mode != "none":
            raise ValueError(f"Unsupported mode={mode!r}")
        rows.append({"layer": layer_idx, "para_scale": scale, "alpha": 1.0 - scale})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--para_scale_min", type=float, default=-20.0)
    parser.add_argument("--para_scale_max", type=float, default=20.0)
    parser.add_argument("--default_para_scale", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    n_layers = load_num_layers(args.model_name)
    rows = sample_para_scales(
        args.mode,
        n_layers,
        args.seed,
        args.default_para_scale,
        args.para_scale_min,
        args.para_scale_max,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "n_layers": n_layers,
        "mode": args.mode,
        "seed": args.seed,
        "para_scale_min": args.para_scale_min,
        "para_scale_max": args.para_scale_max,
        "default_para_scale": args.default_para_scale,
        "tag": args.tag,
        "sample_scale_granularity": (
            "per_forward_batch_item" if str(args.mode).lower().strip().startswith("sample_") else "none"
        ),
        "layer_para_scales": {str(row["layer"]): row["para_scale"] for row in rows},
    }
    (out_dir / "layer_scales.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "layer_scales.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "para_scale", "alpha"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {out_dir / 'layer_scales.json'}")


if __name__ == "__main__":
    main()
