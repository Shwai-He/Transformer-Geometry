#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


FIXED_BUCKETS: List[Tuple[str, int, int]] = [
    ("len0001_0128", 1, 128),
    ("len0129_0256", 129, 256),
    ("len0257_0512", 257, 512),
    ("len0513_1024", 513, 1024),
    ("len1025_2048", 1025, 2048),
    ("len2049_plus", 2049, 10**12),
]


def read_prompts(path: Path, max_prompts: int) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            out.append((line_idx, text))
            if max_prompts > 0 and len(out) >= max_prompts:
                break
    return out


def assign_fixed_bucket(token_count: int) -> str:
    for name, lo, hi in FIXED_BUCKETS:
        if lo <= token_count <= hi:
            return name
    return "len_unknown"


def assign_quantile_buckets(rows: List[Dict[str, object]], num_buckets: int) -> None:
    sorted_rows = sorted(rows, key=lambda r: (int(r["token_count"]), int(r["source_line_idx"])))
    n = len(sorted_rows)
    for rank, row in enumerate(sorted_rows):
        bucket_idx = min(int(rank * num_buckets / max(n, 1)), num_buckets - 1)
        row["quantile_bucket"] = f"q{bucket_idx + 1:02d}"


def write_prompt_file(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row["prompt"]).replace("\n", " ").strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build value_pre prompt bucket files from para ablation prompt samples."
    )
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/value_pre_prompts")
    parser.add_argument("--max_prompts", type=int, default=0, help="0 means use all prompts.")
    parser.add_argument("--num_quantile_buckets", type=int, default=4)
    parser.add_argument(
        "--write_fixed_buckets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write fixed length bucket files such as len0257_0512.txt.",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    prompts = read_prompts(prompts_path, max_prompts=args.max_prompts)
    if not prompts:
        raise RuntimeError(f"No prompts loaded from {prompts_path}")

    rows: List[Dict[str, object]] = []
    for prompt_idx, (source_line_idx, prompt) in enumerate(prompts):
        token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        rows.append(
            {
                "prompt_idx": prompt_idx,
                "source_line_idx": source_line_idx,
                "token_count": token_count,
                "fixed_bucket": assign_fixed_bucket(token_count),
                "quantile_bucket": "",
                "prompt": prompt,
            }
        )
    assign_quantile_buckets(rows, num_buckets=args.num_quantile_buckets)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_idx",
                "source_line_idx",
                "token_count",
                "fixed_bucket",
                "quantile_bucket",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_quantile: Dict[str, List[Dict[str, object]]] = {}
    by_fixed: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_quantile.setdefault(str(row["quantile_bucket"]), []).append(row)
        by_fixed.setdefault(str(row["fixed_bucket"]), []).append(row)

    print(f"[SAVE] {manifest_path}")
    print("[INFO] quantile buckets:")
    for bucket, bucket_rows in sorted(by_quantile.items()):
        path = output_dir / f"{bucket}.txt"
        write_prompt_file(path, bucket_rows)
        lengths = [int(r["token_count"]) for r in bucket_rows]
        print(f"  {bucket}: n={len(bucket_rows)} min={min(lengths)} max={max(lengths)} -> {path}")

    if args.write_fixed_buckets:
        print("[INFO] fixed buckets:")
        for bucket, bucket_rows in sorted(by_fixed.items()):
            if not bucket_rows:
                continue
            path = output_dir / f"{bucket}.txt"
            write_prompt_file(path, bucket_rows)
            lengths = [int(r["token_count"]) for r in bucket_rows]
            print(f"  {bucket}: n={len(bucket_rows)} min={min(lengths)} max={max(lengths)} -> {path}")


if __name__ == "__main__":
    main()
