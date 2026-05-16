#!/usr/bin/env python3
"""
Encode a directory or file into copyable base64 text chunks.

This is meant for machines where downloads are blocked but terminal/text copy
still works. Run this on the source machine, copy the generated text files to
the destination machine, then run restore_from_text.py there.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from datetime import datetime, timezone


DEFAULT_CHUNK_CHARS = 900_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def safe_slug(name: str) -> str:
    cleaned = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "archive"


def make_tar_gz(source: Path, archive_path: Path, arcname: str) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=arcname, recursive=True)


def write_chunks(encoded: bytes, output_dir: Path, chunk_chars: int) -> list[dict[str, object]]:
    chunks = []
    total = (len(encoded) + chunk_chars - 1) // chunk_chars
    for index, start in enumerate(range(0, len(encoded), chunk_chars), start=1):
        chunk = encoded[start : start + chunk_chars]
        chunk_name = f"chunk_{index:04d}_of_{total:04d}.b64.txt"
        chunk_path = output_dir / chunk_name
        chunk_path.write_bytes(chunk + b"\n")
        chunks.append(
            {
                "index": index,
                "file": chunk_name,
                "chars": len(chunk),
                "sha256": sha256_bytes(chunk),
            }
        )
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack a directory/file into base64 text chunks for copy-paste transfer."
    )
    parser.add_argument("source", help="Directory or file to encode")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output bundle directory. Default: ./<source-name>_text_bundle",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Archive root name after restore. Default: source basename",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"Base64 characters per chunk file. Default: {DEFAULT_CHUNK_CHARS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    if args.chunk_chars < 1024:
        raise SystemExit("--chunk-chars must be at least 1024")

    arcname = args.name or source.name
    output_dir = Path(args.output_dir or f"{safe_slug(source.name)}_text_bundle").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir() and is_relative_to(output_dir, source):
        raise SystemExit(
            "Output directory is inside the source directory. Choose another --output-dir "
            "to avoid packing the generated chunks into themselves."
        )

    with tempfile.TemporaryDirectory(prefix="copy_text_archive_") as tmp:
        archive_path = Path(tmp) / f"{safe_slug(arcname)}.tar.gz"
        make_tar_gz(source, archive_path, arcname)
        archive_bytes = archive_path.read_bytes()

    encoded = base64.b64encode(archive_bytes)
    chunks = write_chunks(encoded, output_dir, args.chunk_chars)

    manifest = {
        "format": "copyable-tar-gz-base64-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "archive_root_name": arcname,
        "archive_size_bytes": len(archive_bytes),
        "archive_sha256": sha256_bytes(archive_bytes),
        "base64_size_chars": len(encoded),
        "base64_sha256": sha256_bytes(encoded),
        "chunk_count": len(chunks),
        "chunk_chars": args.chunk_chars,
        "chunks": chunks,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote text bundle: {output_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Archive bytes: {len(archive_bytes)}")
    print(f"Archive SHA256: {manifest['archive_sha256']}")
    print()
    print("Copy manifest.json and every chunk_*.b64.txt file to the destination,")
    print("then run restore_from_text.py on the destination bundle directory.")


if __name__ == "__main__":
    main()
