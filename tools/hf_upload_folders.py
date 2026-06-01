#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi

# =========================
# File-first configuration
# =========================
# Fill these values directly in this file, then run:
# python hf_upload_folders.py
REPO_ID = "shwai-he/llm-rotator"
REPO_TYPE = "dataset"  # dataset | model | space
REVISION = "main"
PRIVATE = True
REPO_EXIST_OK = True
INCLUDE_PATTERNS: list[str] | None = None
EXCLUDE_PATTERNS: list[str] | None = None
FOLDERS: list[str] = [
    
    # "/abs/path/to/folder_a",
    # "/abs/path/to/folder_b",
]


def _iter_dirs(items: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Expected directory, got file: {p}")
        out.append(p)
    return out


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Please export HF_TOKEN first.")
    if not REPO_ID or "yourname/" in REPO_ID:
        raise RuntimeError("Please set REPO_ID in this file before running.")
    if len(FOLDERS) == 0:
        raise RuntimeError("Please fill FOLDERS in this file before running.")

    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=bool(PRIVATE),
        exist_ok=bool(REPO_EXIST_OK),
    )

    dirs = _iter_dirs(FOLDERS)
    for d in dirs:
        path_in_repo = d.name
        print(f"[INFO] Uploading directory: {d}")
        print(f"[INFO] -> repo: {REPO_TYPE}:{REPO_ID} /{path_in_repo}")
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(d),
            path_in_repo=path_in_repo,
            revision=REVISION,
            allow_patterns=INCLUDE_PATTERNS,
            ignore_patterns=EXCLUDE_PATTERNS,
            commit_message=f"Upload folder {d.name}",
        )
        print(f"[INFO] Uploaded: {d}")


if __name__ == "__main__":
    main()
