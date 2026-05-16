#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent
INPUT_CSV = RESULTS_DIR / "nanogpt.csv"
OUTPUT_CSV = RESULTS_DIR / "nanogpt_clean.csv"
REMOVED_CSV = RESULTS_DIR / "nanogpt_early_removed.csv"

# Treat very early or clearly incomplete checkpoints as noisy for model-to-model comparison.
MIN_SAVE_STEP = 5000
DROP_ZERO_OR_MISSING_TOKENS = False
DROP_MISSING_SAVE_STEP = False


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _should_drop(row: dict[str, str]) -> tuple[bool, str]:
    save_step = _parse_float(row.get("save_step"))
    tokens_seen_b = _parse_float(row.get("tokens_seen_b"))

    if DROP_MISSING_SAVE_STEP and save_step is None:
        return True, "missing_save_step"
    if save_step is not None and save_step <= MIN_SAVE_STEP:
        return True, f"save_step<={MIN_SAVE_STEP}"
    if DROP_ZERO_OR_MISSING_TOKENS:
        if tokens_seen_b is None:
            return True, "missing_tokens_seen_b"
        if tokens_seen_b <= 0:
            return True, "tokens_seen_b<=0"
    elif tokens_seen_b is not None and tokens_seen_b <= 0:
        return True, "tokens_seen_b<=0"
    return False, ""


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input CSV: {INPUT_CSV}")

    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    kept: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []

    for row in rows:
        drop, reason = _should_drop(row)
        if drop:
            out = dict(row)
            out["drop_reason"] = reason
            removed.append(out)
        else:
            kept.append(row)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    removed_fields = fieldnames + ["drop_reason"]
    with REMOVED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=removed_fields)
        writer.writeheader()
        writer.writerows(removed)

    print(f"[OK] kept={len(kept)} -> {OUTPUT_CSV}")
    print(f"[OK] removed={len(removed)} -> {REMOVED_CSV}")
    print(
        "[INFO] rule: "
        f"save_step > {MIN_SAVE_STEP}, "
        f"tokens_seen_b {'required and > 0' if DROP_ZERO_OR_MISSING_TOKENS else 'not enforced'}"
    )


if __name__ == "__main__":
    main()
