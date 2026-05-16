#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nltk


DEFAULT_DOWNLOAD_DIR = "/mnt/hdfs/shwai.he/DepthBoost/nltk_data"
DEFAULT_RESOURCES = (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab/english"),
)


def have_resource(resource_path: str) -> bool:
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False


def ensure_resource(resource_name: str, resource_path: str, download_dir: Path) -> None:
    if have_resource(resource_path):
        print(f"[INFO] Resource already present: {resource_name} ({resource_path})")
        return

    print(f"[INFO] Downloading {resource_name} -> {download_dir}")
    ok = nltk.download(resource_name, download_dir=str(download_dir))
    if not ok:
        raise RuntimeError(f"nltk.download({resource_name!r}) returned False")
    if not have_resource(resource_path):
        raise RuntimeError(
            f"Downloaded {resource_name}, but resource path still missing: {resource_path}"
        )
    print(f"[INFO] Ready: {resource_name} ({resource_path})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify NLTK resources needed by RULER/NIAH."
    )
    parser.add_argument(
        "--download-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Target NLTK data directory (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional subset of resources to download, e.g. --only punkt punkt_tab",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Make sure NLTK looks here first during both download and verification.
    if str(download_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(download_dir))

    selected = set(args.only)
    resources = [
        item for item in DEFAULT_RESOURCES if not selected or item[0] in selected
    ]
    if not resources:
        raise ValueError(f"No known resources matched --only={args.only}")

    print(f"[INFO] NLTK data dir: {download_dir}")
    print("[INFO] Resources:", ", ".join(name for name, _ in resources))

    for resource_name, resource_path in resources:
        ensure_resource(resource_name, resource_path, download_dir)

    print("[INFO] All requested NLTK resources are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
