#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repgeo import TechnicalReproducer


def _load_prompts(prompts_file: str) -> list[str]:
    lines = Path(prompts_file).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise ValueError(f"No valid prompts found in: {prompts_file}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch technical reproduction for representation geometry")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--translation_shift", type=float, default=100.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default="results/technical_reproduction_batch.json")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)

    rep = TechnicalReproducer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )
    result = rep.reproduce_batch(
        prompts=prompts,
        top_k=args.top_k,
        translation_shift=args.translation_shift,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("n_prompts:", result["n_prompts"])
    print("top_k:", result["top_k"])
    print("translation_shift:", result["translation_shift"])
    print("aggregate keys:", list(result["aggregate"].keys()))


if __name__ == "__main__":
    main()
