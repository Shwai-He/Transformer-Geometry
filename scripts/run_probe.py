#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repgeo import GeometryAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation geometry probe for Transformer LMs")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Direct prompt text. If omitted, load from --prompt_file.",
    )
    parser.add_argument("--prompt_file", type=str, default="results/prompts.txt", help="Text file with one prompt per line.")
    parser.add_argument("--prompt_index", type=int, default=4, help="Zero-based line index from --prompt_file.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default="results/probe_result.json")
    args = parser.parse_args()

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompts = [ln.strip() for ln in prompt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in: {prompt_path}")
        if not (0 <= args.prompt_index < len(prompts)):
            raise IndexError(f"prompt_index out of range: {args.prompt_index}. Valid range: 0..{len(prompts)-1}")
        prompt = prompts[args.prompt_index]

    analyzer = GeometryAnalyzer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )
    result = analyzer.analyze(prompt=prompt, top_k=args.top_k)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print("Predicted next token:", repr(result["predicted_next_token"]))


if __name__ == "__main__":
    main()
