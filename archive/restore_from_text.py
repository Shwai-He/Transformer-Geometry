#!/usr/bin/env python3
"""
Restore a directory/file from copyable base64 text chunks.

Place manifest.json and all chunk_*.b64.txt files in one directory, then run:
    python restore_from_text.py <bundle_dir> -o <restore_parent_dir>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def safe_extract_tar_gz(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if not is_relative_to(target, output_dir):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(output_dir)


def load_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def read_encoded_bundle(bundle_dir: Path, manifest: dict) -> bytes:
    parts = []
    chunks = manifest.get("chunks", [])
    expected_count = manifest.get("chunk_count")
    if expected_count != len(chunks):
        raise SystemExit(f"Manifest chunk_count={expected_count}, but chunks list has {len(chunks)} entries")

    for chunk in chunks:
        chunk_path = bundle_dir / chunk["file"]
        if not chunk_path.exists():
            raise SystemExit(f"Missing chunk file: {chunk_path}")
        data = b"".join(chunk_path.read_bytes().split())
        digest = sha256_bytes(data)
        if digest != chunk["sha256"]:
            raise SystemExit(
                f"Chunk checksum mismatch for {chunk_path.name}: expected {chunk['sha256']}, got {digest}"
            )
        parts.append(data)

    encoded = b"".join(parts)
    digest = sha256_bytes(encoded)
    if digest != manifest["base64_sha256"]:
        raise SystemExit(
            f"Combined base64 checksum mismatch: expected {manifest['base64_sha256']}, got {digest}"
        )
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a base64 text bundle created by encode_dir_to_text.py.")
    parser.add_argument("bundle_dir", help="Directory containing manifest.json and chunk_*.b64.txt")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Parent directory where the archive root will be extracted. Default: current directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(bundle_dir)
    if manifest.get("format") != "copyable-tar-gz-base64-v1":
        raise SystemExit(f"Unsupported bundle format: {manifest.get('format')}")

    encoded = read_encoded_bundle(bundle_dir, manifest)
    try:
        archive_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise SystemExit(f"Base64 decode failed: {exc}") from exc

    digest = sha256_bytes(archive_bytes)
    if digest != manifest["archive_sha256"]:
        raise SystemExit(f"Archive checksum mismatch: expected {manifest['archive_sha256']}, got {digest}")

    with tempfile.TemporaryDirectory(prefix="copy_text_restore_") as tmp:
        archive_path = Path(tmp) / "archive.tar.gz"
        archive_path.write_bytes(archive_bytes)
        safe_extract_tar_gz(archive_path, output_dir)

    restored_root = output_dir / manifest["archive_root_name"]
    print(f"Restored to: {restored_root}")
    print(f"Archive SHA256 verified: {digest}")


if __name__ == "__main__":
    main()
