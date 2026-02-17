#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repgeo import GeometryAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch representation geometry probe")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True, help="JSON list file or txt file (one prompt per line)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default="results/batch_probe_result.json")
    args = parser.parse_args()

    path = Path(args.prompts_file)
    if path.suffix.lower() == ".json":
        prompts = json.loads(path.read_text(encoding="utf-8"))
    else:
        prompts = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    analyzer = GeometryAnalyzer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )

    result = analyzer.analyze_batch(prompts=prompts, top_k=args.top_k)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("n_prompts:", result["n_prompts"])
    print("aggregate:")
    for k, v in result["aggregate"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
