#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _build_loader_kwargs(dataset_name: str, dataset_config: str | None, split: str, streaming: bool) -> dict:
    kwargs = {"split": split, "streaming": streaming}
    if dataset_config:
        kwargs["name"] = dataset_config
    return kwargs


def _iter_texts(args) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "datasets is required to prepare real evaluation texts. "
            "Install it or run inside the representation-analysis environment."
        ) from exc

    kwargs = _build_loader_kwargs(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        streaming=args.streaming,
    )
    print(
        "[INFO] Loading dataset: "
        f"name={args.dataset_name} config={args.dataset_config or '<none>'} "
        f"split={args.dataset_split} streaming={args.streaming}",
        flush=True,
    )
    ds = load_dataset(args.dataset_name, **kwargs)

    if args.streaming:
        ds = ds.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
        iterator = iter(ds)
    else:
        ds = ds.shuffle(seed=args.shuffle_seed)
        iterator = iter(ds)

    rows = []
    seen = set()
    scanned = 0
    kept = 0
    max_to_scan = args.max_to_scan if args.max_to_scan > 0 else None
    progress = None

    print(
        "[INFO] Sampling constraints: "
        f"max_texts={args.max_texts} min_chars={args.min_chars} max_chars={args.max_chars} "
        f"shuffle_seed={args.shuffle_seed} max_to_scan={args.max_to_scan}",
        flush=True,
    )
    if tqdm is not None:
        progress = tqdm(
            total=args.max_texts,
            desc="Collecting prompts",
            unit="prompt",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        for ex in iterator:
            scanned += 1
            if max_to_scan is not None and scanned > max_to_scan:
                break
            text = ex.get(args.text_key)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            if len(text) < args.min_chars or len(text) > args.max_chars:
                continue
            if text in seen:
                continue
            seen.add(text)
            kept += 1
            rows.append(
                {
                    "text": text,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config or "",
                    "dataset_split": args.dataset_split,
                    "source_index": scanned - 1,
                }
            )
            if progress is not None:
                progress.update(1)
                progress.set_postfix(scanned=scanned, latest_chars=len(text))
            if len(rows) >= args.max_texts:
                break

            if kept % 8 == 0:
                print(
                    f"[INFO] Progress: scanned={scanned} kept={kept} latest_chars={len(text)}",
                    flush=True,
                )
    finally:
        if progress is not None:
            progress.close()

    if not rows:
        raise RuntimeError(
            "No texts were collected from the dataset. "
            "Try relaxing min/max char thresholds or changing the dataset split."
        )
    print(f"[INFO] Sampling finished: scanned={scanned} kept={len(rows)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample real texts from an HF dataset and write them as JSONL.")
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--dataset_config", type=str, default="en")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_texts", type=int, default=16)
    parser.add_argument("--min_chars", type=int, default=160)
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--shuffle_seed", type=int, default=1337)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--max_to_scan", type=int, default=200000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    rows = _iter_texts(args)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(rows)} texts to {out_path}")
    print(f"[INFO] First source_index={rows[0]['source_index']} last_source_index={rows[-1]['source_index']}")
    print(
        "[INFO] Dataset="
        f"{args.dataset_name} config={args.dataset_config or '<none>'} split={args.dataset_split} "
        f"text_key={args.text_key} streaming={args.streaming}"
    )


if __name__ == "__main__":
    main()
