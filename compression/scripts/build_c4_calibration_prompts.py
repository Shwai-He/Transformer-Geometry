#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


C4_TRAIN_SHARD = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WANDA-style C4 calibration prompts.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_scan", type=int, default=20000)
    parser.add_argument("--local_shard", default="")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def iter_c4_texts(local_shard: str):
    if local_shard:
        with gzip.open(local_shard, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line).get("text", "")
        return

    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": C4_TRAIN_SHARD},
        split="train",
    )
    for row in dataset:
        yield row.get("text", "")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    candidates: list[str] = []
    scanned = 0
    for text_raw in iter_c4_texts(args.local_shard):
        scanned += 1
        if scanned > args.max_scan:
            break
        text = str(text_raw).replace("\n", " ").strip()
        if not text:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids
        if tokens.shape[1] <= args.seqlen:
            continue
        start = random.randint(0, tokens.shape[1] - args.seqlen - 1)
        sample_ids = tokens[:, start : start + args.seqlen]
        sample = tokenizer.decode(sample_ids[0], skip_special_tokens=True).replace("\n", " ").strip()
        if sample:
            candidates.append(sample)
        if len(candidates) >= args.nsamples:
            break

    if len(candidates) < args.nsamples:
        raise RuntimeError(
            f"Only found {len(candidates)} C4 samples with at least {args.seqlen} tokens "
            f"after scanning {len(indices)} rows."
        )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(candidates) + "\n", encoding="utf-8")
    meta = {
        "source": "allenai/c4",
        "data_file": C4_TRAIN_SHARD,
        "local_shard": args.local_shard,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "scanned_rows": scanned,
        "model_name_or_path": args.model_name_or_path,
        "output_file": str(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
