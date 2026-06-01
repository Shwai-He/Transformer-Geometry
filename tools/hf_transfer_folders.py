#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download


def _iter_paths(items: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Expected directory, got file: {p}")
        out.append(p)
    return out


def upload_dirs(
    *,
    repo_id: str,
    repo_type: str,
    dirs: list[Path],
    revision: str,
    private: bool,
    exist_ok: bool,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok)

    for d in dirs:
        path_in_repo = d.name
        print(f"[INFO] Uploading directory: {d}")
        print(f"[INFO] -> repo: {repo_type}:{repo_id} /{path_in_repo}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(d),
            path_in_repo=path_in_repo,
            revision=revision,
            allow_patterns=include_patterns,
            ignore_patterns=exclude_patterns,
            commit_message=f"Upload folder {d.name}",
        )
        print(f"[INFO] Uploaded: {d}")


def download_repo(
    *,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    revision: str,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {repo_type}:{repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        revision=revision,
        allow_patterns=include_patterns,
        ignore_patterns=exclude_patterns,
    )
    print(f"[INFO] Download complete: {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload/download folders to/from Hugging Face Hub.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Upload one or more local folders to a HF repo.")
    up.add_argument("--repo-id", required=True, help="HF repo id, e.g. username/my-repo")
    up.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    up.add_argument("--revision", default="main")
    up.add_argument("--private", action="store_true")
    up.add_argument("--no-exist-ok", action="store_true", help="Fail if repo already exists.")
    up.add_argument("--include", nargs="*", default=None, help="Glob allowlist patterns.")
    up.add_argument("--exclude", nargs="*", default=None, help="Glob denylist patterns.")
    up.add_argument(
        "folders",
        nargs="+",
        help="Local directory paths to upload. Each folder is uploaded under /<folder_name> in repo.",
    )

    down = sub.add_parser("download", help="Download a HF repo snapshot to local.")
    down.add_argument("--repo-id", required=True)
    down.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    down.add_argument("--revision", default="main")
    down.add_argument("--include", nargs="*", default=None)
    down.add_argument("--exclude", nargs="*", default=None)
    down.add_argument("--local-dir", required=True, help="Local directory to place snapshot.")

    args = parser.parse_args()

    if args.cmd == "upload":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set. Please export HF_TOKEN first.")
        dirs = _iter_paths(args.folders)
        upload_dirs(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            dirs=dirs,
            revision=args.revision,
            private=bool(args.private),
            exist_ok=not bool(args.no_exist_ok),
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )
        return

    if args.cmd == "download":
        download_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=Path(args.local_dir).expanduser().resolve(),
            revision=args.revision,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )
        return

    raise RuntimeError(f"Unsupported cmd={args.cmd}")


if __name__ == "__main__":
    main()
