#!/usr/bin/env python3
"""Inspect matched generation samples for appendix tables.

Example:
  python inspect_generation_samples.py --prompt-idx 8
  python inspect_generation_samples.py --prompt-idx 8 --full
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "representation-analysis"
    / "outputs"
    / "generate"
)

PARA_SETTINGS = ["-1", "0", "2"]
PERP_SETTINGS = ["0", "0.5", "1.5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the prompt, baseline, and matched edit outputs for one prompt_idx."
    )
    parser.add_argument("--prompt-idx", type=int, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Generation-output root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=420,
        help="Maximum characters per output excerpt unless --full is set.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full prompt and outputs instead of clipped excerpts.",
    )
    parser.add_argument(
        "--keep-think-tags",
        action="store_true",
        help="Keep generated <think> markers in the printed text.",
    )
    parser.add_argument(
        "--include-mlp-both",
        action="store_true",
        help="Also print residual MLP and residual both outputs.",
    )
    parser.add_argument(
        "--include-counter-settings",
        action="store_true",
        help="Also print Residual(attn) for parallel sweeps and Value-Based for perpendicular sweeps.",
    )
    return parser.parse_args()


def load_prompt_row(path: Path, prompt_idx: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("prompt_idx") == prompt_idx:
                return row
    raise KeyError(f"prompt_idx={prompt_idx} not found in {path}")


def clean_text(text: str, keep_think_tags: bool) -> str:
    if not keep_think_tags:
        text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def maybe_clip(text: str, max_chars: int, full: bool) -> str:
    if full or len(text) <= max_chars:
        return text
    return text[: max_chars - 4].rstrip() + " ..."


def get_nested(row: dict[str, Any], *keys: str) -> str:
    value: Any = row["generations"]
    for key in keys:
        value = value[key]
    if not isinstance(value, str):
        raise TypeError(f"Expected string at generations/{'/'.join(keys)}")
    return value


def print_block(title: str, text: str) -> None:
    print(f"\n### {title}")
    print(text)


def perp_file_scale(scale: str) -> str:
    return "0.0" if scale == "0" else scale


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()

    baseline_path = root / "para" / "generate_para_scale_1.jsonl"
    baseline_row = load_prompt_row(baseline_path, args.prompt_idx)

    prompt = clean_text(baseline_row["prompt"], keep_think_tags=True)
    print_block("Prompt", maybe_clip(prompt, args.max_chars, args.full))

    baseline = clean_text(
        get_nested(baseline_row, "residual_output", "none"),
        keep_think_tags=args.keep_think_tags,
    )
    print_block(
        "Baseline | no edit",
        maybe_clip(baseline, args.max_chars, args.full),
    )

    for scale in PARA_SETTINGS:
        row = load_prompt_row(root / "para" / f"generate_para_scale_{scale}.jsonl", args.prompt_idx)
        value_based = clean_text(
            get_nested(row, "xsa_middle_multihead", "attn"),
            keep_think_tags=args.keep_think_tags,
        )
        print_block(
            f"Value-Based | s_parallel={scale}",
            maybe_clip(value_based, args.max_chars, args.full),
        )

        if args.include_counter_settings:
            residual_attn = clean_text(
                get_nested(row, "residual_output", "attn"),
                keep_think_tags=args.keep_think_tags,
            )
            print_block(
                f"Residual(attn) | s_parallel={scale}",
                maybe_clip(residual_attn, args.max_chars, args.full),
            )

        if args.include_mlp_both:
            for branch in ("mlp", "both"):
                residual = clean_text(
                    get_nested(row, "residual_output", branch),
                    keep_think_tags=args.keep_think_tags,
                )
                print_block(
                    f"Residual({branch}) | s_parallel={scale}",
                    maybe_clip(residual, args.max_chars, args.full),
                )

    for scale in PERP_SETTINGS:
        row = load_prompt_row(
            root / "perp" / f"generate_para_scale_1.0_perp_scale_{perp_file_scale(scale)}.jsonl",
            args.prompt_idx,
        )
        residual_attn = clean_text(
            get_nested(row, "residual_output", "attn"),
            keep_think_tags=args.keep_think_tags,
        )
        print_block(
            f"Residual(attn) | s_parallel=1.0, s_perp={scale}",
            maybe_clip(residual_attn, args.max_chars, args.full),
        )

        if args.include_counter_settings:
            value_based = clean_text(
                get_nested(row, "xsa_middle_multihead", "attn"),
                keep_think_tags=args.keep_think_tags,
            )
            print_block(
                f"Value-Based | s_parallel=1.0, s_perp={scale}",
                maybe_clip(value_based, args.max_chars, args.full),
            )

        if args.include_mlp_both:
            for branch in ("mlp", "both"):
                residual = clean_text(
                    get_nested(row, "residual_output", branch),
                    keep_think_tags=args.keep_think_tags,
                )
                print_block(
                    f"Residual({branch}) | s_parallel=1.0, s_perp={scale}",
                    maybe_clip(residual, args.max_chars, args.full),
                )


if __name__ == "__main__":
    main()
