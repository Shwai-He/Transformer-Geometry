#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repgeo import GeometryAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-wise generation geometry probe")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="results/generation_probe_result.json")
    args = parser.parse_args()

    analyzer = GeometryAnalyzer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )
    result = analyzer.analyze_generation(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("Generated text:")
    print(result["generated_text"])


if __name__ == "__main__":
    main()
