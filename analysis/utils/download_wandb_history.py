#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import wandb


DEFAULT_KEYS = [
    "_step",
    "_runtime",
    "train/loss",
    "train/lr",
    "train/tokens",
    "val/loss",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download W&B run history to CSV.")
    p.add_argument("--entity", type=str, required=True, help="W&B entity/user/team")
    p.add_argument("--project", type=str, required=True, help="W&B project name")
    p.add_argument(
        "--run_ids",
        type=str,
        default="",
        help="Comma-separated run ids, e.g. abc123,def456",
    )
    p.add_argument(
        "--run_names",
        type=str,
        default="",
        help="Comma-separated exact run names (fallback selector)",
    )
    p.add_argument(
        "--keys",
        type=str,
        default=",".join(DEFAULT_KEYS),
        help="Comma-separated history keys to fetch",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="Max samples per run from history",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="representation-analysis/outputs/wandb_exports",
        help="Output directory for CSV files",
    )
    p.add_argument(
        "--list_only",
        action="store_true",
        help="Only list runs in project, do not download",
    )
    return p.parse_args()


def split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def sanitize_filename(name: str) -> str:
    out = name.replace("/", "_")
    out = "".join(c if c.isalnum() or c in "._-+" else "_" for c in out)
    return out


def iter_target_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    run_ids: Iterable[str],
    run_names: Iterable[str],
):
    path = f"{entity}/{project}"
    runs = api.runs(path)
    by_id = {r.id: r for r in runs}
    by_name = {}
    for r in runs:
        by_name.setdefault(r.name, []).append(r)

    selected = []
    for rid in run_ids:
        if rid in by_id:
            selected.append(by_id[rid])
        else:
            print(f"[WARN] run id not found: {rid}")
    for name in run_names:
        matches = by_name.get(name, [])
        if not matches:
            print(f"[WARN] run name not found: {name}")
            continue
        if len(matches) > 1:
            print(f"[WARN] run name duplicated: {name}, using latest created one")
            matches = sorted(matches, key=lambda r: r.created_at or "", reverse=True)
        selected.append(matches[0])

    # de-dup by id, stable order
    seen = set()
    uniq = []
    for r in selected:
        if r.id in seen:
            continue
        seen.add(r.id)
        uniq.append(r)
    return uniq


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"

    if args.list_only:
        for r in api.runs(project_path):
            print(f"{r.id}\t{r.name}\t{r.state}\t{r.created_at}")
        return

    run_ids = split_csv(args.run_ids)
    run_names = split_csv(args.run_names)
    if not run_ids and not run_names:
        raise SystemExit("Need at least one of --run_ids or --run_names")

    keys = split_csv(args.keys)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = iter_target_runs(api, args.entity, args.project, run_ids, run_names)
    if not targets:
        raise SystemExit("No target runs selected.")

    for run in targets:
        print(f"[INFO] Downloading: id={run.id} name={run.name}")
        df = run.history(keys=keys, pandas=True, samples=args.samples)
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"[WARN] empty history: {run.id}")
            continue
        base = sanitize_filename(f"{run.name}-{run.id}")
        out_path = out_dir / f"{base}.csv"
        df.to_csv(out_path, index=False)
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
