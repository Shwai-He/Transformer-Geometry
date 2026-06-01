#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from transformers import AutoTokenizer


DEFAULT_BUCKETS: List[Tuple[str, int, int]] = [
    ("c4_len128", 96, 160),
    ("c4_len512", 448, 576),
    ("c4_len1024", 896, 1152),
    ("c4_len2048", 1792, 2304),
]


def parse_buckets(raw: str) -> List[Tuple[str, int, int]]:
    buckets: List[Tuple[str, int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, lo, hi = item.split(":")
        buckets.append((name, int(lo), int(hi)))
    if not buckets:
        raise ValueError("No buckets parsed.")
    return buckets


def iter_local_json(paths: List[str], text_key: str) -> Iterator[Dict[str, object]]:
    for raw_path in paths:
        path = Path(raw_path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if text_key in obj:
                    yield obj


def iter_hf_c4(dataset_name: str, dataset_config: str, split: str) -> Iterable[Dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Please install datasets: python3 -m pip install datasets") from e
    return load_dataset(dataset_name, dataset_config, split=split, streaming=True)


def clean_text(text: str) -> str:
    return " ".join(text.replace("\u0000", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample C4 prompts into tokenizer-length buckets for value_pre probes.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/value_pre_prompts_c4")
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--dataset_config", type=str, default="en")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--local_files", nargs="*", default=None, help="Optional local C4 json/jsonl/json.gz files.")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--samples_per_bucket", type=int, default=8)
    parser.add_argument("--max_scan", type=int, default=200000)
    parser.add_argument(
        "--buckets",
        type=str,
        default=",".join(f"{name}:{lo}:{hi}" for name, lo, hi in DEFAULT_BUCKETS),
        help="Comma-separated name:min:max token buckets.",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=100,
        help="Skip very short raw documents before tokenization.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = parse_buckets(args.buckets)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.local_files:
        stream = iter_local_json(args.local_files, text_key=args.text_key)
        source = "local"
    else:
        stream = iter_hf_c4(args.dataset_name, args.dataset_config, args.split)
        source = f"{args.dataset_name}/{args.dataset_config}/{args.split}"

    selected: Dict[str, List[Dict[str, object]]] = {name: [] for name, _, _ in buckets}
    manifest_rows: List[Dict[str, object]] = []
    scanned = 0

    for sample_idx, item in enumerate(stream):
        if scanned >= args.max_scan:
            break
        scanned += 1
        text = clean_text(str(item.get(args.text_key, "")))
        if len(text) < args.min_chars:
            continue
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        token_count = len(token_ids)
        for bucket_name, lo, hi in buckets:
            if lo <= token_count <= hi and len(selected[bucket_name]) < args.samples_per_bucket:
                row = {
                    "bucket": bucket_name,
                    "sample_idx": sample_idx,
                    "token_count": token_count,
                    "source": source,
                    "prompt": text,
                }
                selected[bucket_name].append(row)
                manifest_rows.append(row)
                break
        if all(len(rows) >= args.samples_per_bucket for rows in selected.values()):
            break

    for bucket_name, _, _ in buckets:
        path = output_dir / f"{bucket_name}.txt"
        with path.open("w", encoding="utf-8") as f:
            for row in selected[bucket_name]:
                f.write(str(row["prompt"]) + "\n")
        lengths = [int(row["token_count"]) for row in selected[bucket_name]]
        if lengths:
            print(f"[SAVE] {path} n={len(lengths)} min={min(lengths)} max={max(lengths)}")
        else:
            print(f"[WARN] {bucket_name}: no samples found -> {path}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "sample_idx", "token_count", "source", "prompt"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[SAVE] {manifest_path}")
    print(f"[INFO] scanned={scanned}")
    print("\n[NEXT] Run:")
    print(f'PROMPT_FILES="{output_dir}/c4_len*.txt" bash scripts/run_value_pre_ratio_probe.sh')


if __name__ == "__main__":
    main()
