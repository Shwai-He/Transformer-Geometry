#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


TARGET_LENGTHS = (128, 512, 1024, 2048)


PROMPT_BLUEPRINTS = [
    {
        "kind": "narrative",
        "instruction": (
            "Read the following story carefully. At the end, answer the final question in one short sentence.\n\n"
        ),
        "unit": (
            "In the coastal town of Marrow Bay, Lina kept a small notebook of unusual events. "
            "Each morning she walked past the harbor, the clock tower, and the old library, writing down "
            "who arrived, what they carried, and which doors were left open. The details seemed ordinary, "
            "but Lina noticed that the same blue bicycle appeared near a different building every day. "
        ),
        "question": "\nQuestion: What object kept appearing near different buildings?",
    },
    {
        "kind": "retrieval",
        "instruction": (
            "Use the notes below to answer the final question. The important fact may appear only once.\n\n"
        ),
        "unit": (
            "Field note: the orchard team checked irrigation lines before sunrise. "
            "Field note: the east greenhouse had twelve trays of basil and eight trays of mint. "
            "Field note: the storage cabinet beside the west gate contained the copper key for the archive room. "
            "Field note: the afternoon inventory listed rope, chalk, seed packets, and spare labels. "
        ),
        "question": "\nQuestion: Where was the copper key kept?",
    },
    {
        "kind": "reasoning",
        "instruction": (
            "Solve the final problem. The preceding examples are included only to create a longer context.\n\n"
        ),
        "unit": (
            "Example: Mira packed 6 red blocks, 4 blue blocks, and 3 green blocks into a box. "
            "After giving 2 red blocks away, she had 11 blocks left. "
            "Example: A train had 9 empty seats, then 5 passengers sat down, leaving 4 empty seats. "
            "Example: The bakery made 18 rolls and sold half before noon, leaving 9 rolls. "
        ),
        "question": (
            "\nFinal problem: Omar has 24 marbles. He gives 7 to Ana and 5 to Rui. "
            "How many marbles does Omar have left?"
        ),
    },
    {
        "kind": "expository",
        "instruction": (
            "Read the explanation and then answer the final question.\n\n"
        ),
        "unit": (
            "A small weather station records temperature, wind direction, and rainfall every hour. "
            "The operator checks the instruments, copies the numbers into a log, and compares them with "
            "yesterday's measurements. Over many days, these repeated observations reveal patterns that "
            "are hard to see from a single measurement. "
        ),
        "question": "\nQuestion: Why does the station record measurements repeatedly?",
    },
]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def build_prompt(tokenizer, blueprint: Dict[str, str], target_tokens: int) -> str:
    prefix = blueprint["instruction"]
    unit = blueprint["unit"]
    question = blueprint["question"]
    text = prefix + unit + question
    while token_len(tokenizer, text) < target_tokens:
        text = prefix + (unit * (text.count(unit) + 1)) + question
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manually designed value_pre prompt buckets.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/value_pre_prompts_manual")
    parser.add_argument(
        "--target_lengths",
        type=int,
        nargs="+",
        default=list(TARGET_LENGTHS),
        help="Approximate token-length buckets to generate.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    manifest_rows: List[Dict[str, object]] = []
    for target in args.target_lengths:
        bucket = f"manual_len{target}"
        bucket_path = output_dir / f"{bucket}.txt"
        with bucket_path.open("w", encoding="utf-8") as f:
            for prompt_idx, blueprint in enumerate(PROMPT_BLUEPRINTS):
                prompt = build_prompt(tokenizer, blueprint, target)
                actual_tokens = token_len(tokenizer, prompt)
                f.write(prompt + "\n")
                manifest_rows.append(
                    {
                        "bucket": bucket,
                        "target_tokens": target,
                        "prompt_idx": prompt_idx,
                        "kind": blueprint["kind"],
                        "actual_tokens": actual_tokens,
                        "prompt_file": str(bucket_path),
                        "prompt": prompt,
                    }
                )
        print(f"[SAVE] {bucket_path}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "target_tokens",
                "prompt_idx",
                "kind",
                "actual_tokens",
                "prompt_file",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[SAVE] {manifest_path}")

    print("\n[NEXT] Run:")
    print(f'PROMPT_FILES="{output_dir}/manual_len*.txt" bash scripts/run_value_pre_ratio_probe.sh')


if __name__ == "__main__":
    main()
