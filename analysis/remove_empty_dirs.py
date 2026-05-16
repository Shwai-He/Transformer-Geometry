#!/usr/bin/env python3
"""Remove empty directories under a given root path."""
import argparse
import os
from pathlib import Path


def remove_empty_dirs(root: Path, dry_run: bool = False) -> list[Path]:
    removed = []
    # Walk bottom-up so inner empty dirs are removed before their parents.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        if not any(p.iterdir()):
            if dry_run:
                print(f"[DRY] would remove: {p}")
            else:
                p.rmdir()
                print(f"[REMOVED] {p}")
            removed.append(p)
    return removed


def main():
    parser = argparse.ArgumentParser(description="Remove empty directories recursively.")
    parser.add_argument("root", nargs="?", default="/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out",
                        help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without deleting")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}")
        raise SystemExit(1)

    removed = remove_empty_dirs(root, dry_run=args.dry_run)
    print(f"\n{'[DRY] Would remove' if args.dry_run else 'Removed'} {len(removed)} empty director{'y' if len(removed)==1 else 'ies'}.")


if __name__ == "__main__":
    main()
