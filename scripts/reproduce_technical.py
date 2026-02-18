#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repgeo import TechnicalReproducer


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce technical claims from representation-geometry paper")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--translation_shift", type=float, default=100.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default="results/technical_reproduction.json")
    args = parser.parse_args()

    rep = TechnicalReproducer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )
    result = rep.reproduce(
        prompt=args.prompt,
        top_k=args.top_k,
        translation_shift=args.translation_shift,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("n_layers_observed:", result["n_layers_observed"])
    print("translation_invariance:", result["translation_invariance"])
    if result["layer_records"]:
        print("example layer record keys:", list(result["layer_records"][0].keys()))


if __name__ == "__main__":
    main()

