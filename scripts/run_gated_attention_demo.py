#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repgeo import compare_vanilla_vs_gated


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic geometry comparison: vanilla attention vs gated attention")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="results/gated_attention_demo.json")
    args = parser.parse_args()

    result = compare_vanilla_vs_gated(
        d_model=args.d_model,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("\nVanilla:")
    for k, v in result["vanilla"].items():
        print(f"  {k}: {v}")

    print("\nGated:")
    for k, v in result["gated"].items():
        print(f"  {k}: {v}")

    print("\nDelta (gated - vanilla):")
    for k, v in result["delta_gated_minus_vanilla"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
