#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repgeo import GeometryAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation geometry probe for Transformer LMs")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--granularity",
        type=str,
        default="block",
        choices=["block", "sublayer", "both"],
        help="block: only block-level hidden states; sublayer/both: capture attn+mlp states via forward hooks.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default="results/probe_result.json")
    args = parser.parse_args()

    analyzer = GeometryAnalyzer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )
    result = analyzer.analyze(prompt=args.prompt, top_k=args.top_k, granularity=args.granularity)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print("Predicted next token:", repr(result["predicted_next_token"]))
    if "sublayer_summary" in result:
        print("Sublayer summary:")
        for k, v in result["sublayer_summary"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
