#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any


# =========================
# File-first configuration
# =========================
ROOT_DIR = Path(
    "/mnt/bn/seed-aws-va/shwai.he/demystifying-transformers-main/lm-evaluation-harness/nanoGPT/out"
)
CKPT_PATTERNS = (
    "ckpt.pt",
    "ckpt_best.pt",
    "ckpt_iter_*.pt",
)
DELETE_IF_ITER_NUM_LE = 0
DELETE_SIDECAR_JSON = True
DRY_RUN = True
PRINT_EVERY = 50


def _iter_checkpoint_paths(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in CKPT_PATTERNS:
        found.extend(root.rglob(pattern))
    # de-duplicate while preserving a stable order
    unique = sorted({p.resolve() for p in found if p.is_file()})
    return unique


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        print(f"[WARN] failed to read checkpoint {path}: {exc}")
        return None


def _iter_num_from_payload(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("iter_num")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _tokens_seen_from_payload(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("tokens_seen")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _delete_path(path: Path) -> None:
    if DRY_RUN:
        return
    path.unlink(missing_ok=True)


def main() -> None:
    if not ROOT_DIR.exists():
        raise SystemExit(f"Missing ROOT_DIR: {ROOT_DIR}")

    ckpt_paths = _iter_checkpoint_paths(ROOT_DIR)
    print(f"[INFO] ROOT_DIR={ROOT_DIR}")
    print(f"[INFO] DRY_RUN={DRY_RUN}")
    print(f"[INFO] DELETE_IF_ITER_NUM_LE={DELETE_IF_ITER_NUM_LE}")
    print(f"[INFO] DELETE_SIDECAR_JSON={DELETE_SIDECAR_JSON}")
    print(f"[INFO] found checkpoint files: {len(ckpt_paths)}")

    kept = 0
    deleted = 0
    unreadable = 0
    deleted_sidecars = 0

    for idx, ckpt_path in enumerate(ckpt_paths, start=1):
        if idx % PRINT_EVERY == 0:
            print(f"[INFO] scanned {idx}/{len(ckpt_paths)} checkpoints")

        payload = _load_checkpoint(ckpt_path)
        if payload is None:
            unreadable += 1
            continue

        iter_num = _iter_num_from_payload(payload)
        tokens_seen = _tokens_seen_from_payload(payload)

        if iter_num is None:
            print(f"[WARN] missing iter_num, keep: {ckpt_path}")
            kept += 1
            continue

        if iter_num <= DELETE_IF_ITER_NUM_LE:
            sidecar = Path(str(ckpt_path) + ".json")
            print(
                f"[DELETE] iter_num={iter_num} tokens_seen={tokens_seen} ckpt={ckpt_path}"
            )
            _delete_path(ckpt_path)
            deleted += 1
            if DELETE_SIDECAR_JSON and sidecar.exists():
                print(f"[DELETE] sidecar={sidecar}")
                _delete_path(sidecar)
                deleted_sidecars += 1
        else:
            kept += 1

    print("-" * 60)
    print(
        f"[DONE] kept={kept} deleted={deleted} deleted_sidecars={deleted_sidecars} unreadable={unreadable}"
    )


if __name__ == "__main__":
    main()
