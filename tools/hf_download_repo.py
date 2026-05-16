#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

# =========================
# File-first configuration
# =========================
# Fill these values directly in this file, then run:
# python hf_download_repo.py
REPO_ID = "yourname/your-repo"
REPO_TYPE = "dataset"  # dataset | model | space
REVISION = "main"
INCLUDE_PATTERNS: list[str] | None = None
EXCLUDE_PATTERNS: list[str] | None = None
LOCAL_DIR = "/abs/path/to/local_download"


def main() -> None:
    if not REPO_ID or "yourname/" in REPO_ID:
        raise RuntimeError("Please set REPO_ID in this file before running.")
    if not LOCAL_DIR or LOCAL_DIR.startswith("/abs/path/to/"):
        raise RuntimeError("Please set LOCAL_DIR in this file before running.")

    local_dir = Path(LOCAL_DIR).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {REPO_TYPE}:{REPO_ID} -> {local_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        revision=REVISION,
        allow_patterns=INCLUDE_PATTERNS,
        ignore_patterns=EXCLUDE_PATTERNS,
    )
    print(f"[INFO] Download complete: {local_dir}")


if __name__ == "__main__":
    main()
