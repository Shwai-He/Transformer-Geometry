#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


# =========================
# File-first configuration
# =========================
# Put your fresh result roots here. Each entry can be a directory or a file.
INPUT_PATHS = [
    "/path/to/resource",
]

# 1 = recursive search under directories; 0 = only top-level files.
CFG_RECURSIVE = 1

# If empty, auto-pick: common_parent(INPUT_PATHS) / "ppl_ablation_summary"
OUTPUT_DIR = ""

# Max prompt cases to keep in the "largest regressions" summary.
TOP_K_CASES = 5


COMPARE_PATTERNS = [
    ("alpha", re.compile(r"^compare_alpha_(.+)\.tsv$")),
    ("gamma", re.compile(r"^compare_gamma_(.+)\.tsv$")),
    ("para", re.compile(r"^compare_para_(.+)\.tsv$")),
    ("perp", re.compile(r"^compare_perp_(.+)\.tsv$")),
]

SUMMARY_PATTERNS = [
    ("para_summary", re.compile(r"^para_ppl_summary\.tsv$")),
    ("perp_summary", re.compile(r"^perp_ppl_summary\.tsv$")),
]

STATS_PATTERNS = [
    ("alpha", re.compile(r"^generate_alpha_(.+)\.stats\.json$")),
    ("gamma", re.compile(r"^generate_gamma_(.+)\.stats\.json$")),
    ("para", re.compile(r"^generate_para_(.+)\.stats\.json$")),
    ("perp", re.compile(r"^generate_perp_(.+)\.stats\.json$")),
]


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _try_float_token(token: str) -> float | None:
    token = token.strip()
    if token == "":
        return None
    try:
        return float(token)
    except Exception:
        pass
    try:
        return float(token.replace("p", "."))
    except Exception:
        return None


def _flatten_numeric_dict(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_numeric_dict(v, key))
        else:
            fv = _to_float(v)
            if fv is not None:
                out[key] = fv
    return out


def _guess_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
    if first.count("\t") > first.count(","):
        return "\t"
    return ","


def _find_first(d: dict[str, str], keys: list[str]) -> str | None:
    lower_to_key = {k.lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in lower_to_key:
            return d[lower_to_key[k.lower()]]
    return None


def _parse_setting(setting_type: str, raw: str) -> tuple[float | None, float | None, float | None]:
    # Returns: primary, alpha_value, perp_or_gamma_value
    if setting_type in {"alpha", "para"}:
        m = re.match(r"^alpha_fixed_(.+)_perp_keep_(.+)$", raw)
        if m:
            a = _try_float_token(m.group(1))
            p = _try_float_token(m.group(2))
            return a, a, p
        m = re.match(r"^(.+)_perp_(.+)$", raw)
        if m:
            a = _try_float_token(m.group(1))
            p = _try_float_token(m.group(2))
            return a, a, p
        a = _try_float_token(raw)
        return a, a, None
    if setting_type in {"gamma", "perp"}:
        v = _try_float_token(raw)
        return v, None, v
    v = _try_float_token(raw)
    return v, None, None


def _scan_files(input_paths: list[Path], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in input_paths:
        if not p.exists():
            continue
        if p.is_file():
            out.append(p)
            continue
        if recursive:
            out.extend([x for x in p.rglob("*") if x.is_file()])
        else:
            out.extend([x for x in p.iterdir() if x.is_file()])
    return sorted(set(out))


def _match_pattern(name: str, specs: list[tuple[str, re.Pattern[str]]]) -> tuple[str, str] | None:
    for kind, rgx in specs:
        m = rgx.match(name)
        if m:
            return kind, m.group(1) if m.groups() else ""
    return None


def _load_compare_rows(path: Path, setting_type: str, setting_raw: str) -> list[dict[str, Any]]:
    delim = _guess_delimiter(path)
    setting_value, alpha_value, aux_value = _parse_setting(setting_type, setting_raw)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            prompt_idx = _find_first(row, ["prompt_idx", "idx", "index", "id"]) or ""
            prompt_preview = _find_first(row, ["prompt_preview", "prompt", "text"]) or ""
            rows.append(
                {
                    "source_file": str(path),
                    "source_dir": str(path.parent),
                    "setting_type": setting_type,
                    "setting_raw": setting_raw,
                    "setting_value": setting_value,
                    "alpha_value": alpha_value,
                    "aux_value": aux_value,
                    "prompt_idx": prompt_idx,
                    "prompt_preview": prompt_preview,
                    "baseline_loss": _to_float(_find_first(row, ["baseline_loss"])),
                    "baseline_ppl": _to_float(_find_first(row, ["baseline_ppl"])),
                    "xsa_loss": _to_float(_find_first(row, ["xsa_loss"])),
                    "xsa_ppl": _to_float(_find_first(row, ["xsa_ppl"])),
                    "delta_loss_xsa": _to_float(_find_first(row, ["delta_loss_xsa", "delta_loss"])),
                    "delta_ppl_xsa": _to_float(_find_first(row, ["delta_ppl_xsa", "delta_ppl"])),
                    "residual_attn_loss": _to_float(_find_first(row, ["residual_attn_loss"])),
                    "residual_attn_ppl": _to_float(_find_first(row, ["residual_attn_ppl"])),
                    "delta_loss_residual_attn": _to_float(_find_first(row, ["delta_loss_residual_attn"])),
                    "delta_ppl_residual_attn": _to_float(_find_first(row, ["delta_ppl_residual_attn"])),
                    "residual_mlp_loss": _to_float(_find_first(row, ["residual_mlp_loss"])),
                    "residual_mlp_ppl": _to_float(_find_first(row, ["residual_mlp_ppl"])),
                    "delta_loss_residual_mlp": _to_float(_find_first(row, ["delta_loss_residual_mlp"])),
                    "delta_ppl_residual_mlp": _to_float(_find_first(row, ["delta_ppl_residual_mlp"])),
                    "residual_both_loss": _to_float(_find_first(row, ["residual_both_loss"])),
                    "residual_both_ppl": _to_float(_find_first(row, ["residual_both_ppl"])),
                    "delta_loss_residual_both": _to_float(_find_first(row, ["delta_loss_residual_both"])),
                    "delta_ppl_residual_both": _to_float(_find_first(row, ["delta_ppl_residual_both"])),
                }
            )
    return rows


def _load_summary_table(path: Path, summary_kind: str) -> list[dict[str, Any]]:
    delim = _guess_delimiter(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            out = {
                "source_file": str(path),
                "source_dir": str(path.parent),
                "summary_kind": summary_kind,
            }
            out.update(row)
            rows.append(out)
    return rows


def _load_stats_json(path: Path, setting_type: str, setting_raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    setting_value, alpha_value, aux_value = _parse_setting(setting_type, setting_raw)
    row = {
        "source_file": str(path),
        "source_dir": str(path.parent),
        "setting_type": setting_type,
        "setting_raw": setting_raw,
        "setting_value": setting_value,
        "alpha_value": alpha_value,
        "aux_value": aux_value,
    }
    row.update(_flatten_numeric_dict(data))
    return row


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_mean(values: list[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return mean(xs) if xs else None


def _safe_median(values: list[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return median(xs) if xs else None


def _build_overview(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in compare_rows:
        key = (row["source_dir"], row["setting_type"], str(row["setting_raw"]))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (source_dir, setting_type, setting_raw), rows in sorted(groups.items()):
        out.append(
            {
                "source_dir": source_dir,
                "setting_type": setting_type,
                "setting_raw": setting_raw,
                "setting_value": rows[0]["setting_value"],
                "alpha_value": rows[0]["alpha_value"],
                "aux_value": rows[0]["aux_value"],
                "num_prompts": len(rows),
                "avg_delta_ppl_xsa": _safe_mean([r["delta_ppl_xsa"] for r in rows]),
                "avg_delta_ppl_residual_attn": _safe_mean([r["delta_ppl_residual_attn"] for r in rows]),
                "avg_delta_ppl_residual_mlp": _safe_mean([r["delta_ppl_residual_mlp"] for r in rows]),
                "avg_delta_ppl_residual_both": _safe_mean([r["delta_ppl_residual_both"] for r in rows]),
                "median_delta_ppl_xsa": _safe_median([r["delta_ppl_xsa"] for r in rows]),
                "median_delta_ppl_residual_attn": _safe_median([r["delta_ppl_residual_attn"] for r in rows]),
                "median_delta_ppl_residual_mlp": _safe_median([r["delta_ppl_residual_mlp"] for r in rows]),
                "median_delta_ppl_residual_both": _safe_median([r["delta_ppl_residual_both"] for r in rows]),
            }
        )
    return out


def _build_top_cases(compare_rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = [
        row for row in compare_rows
        if row.get("delta_ppl_xsa") is not None
    ]
    scored.sort(key=lambda r: float(r["delta_ppl_xsa"]), reverse=True)
    return scored[:top_k]


def _common_parent(paths: list[Path]) -> Path:
    parts = [p.resolve() for p in paths if p.exists()]
    if not parts:
        return Path.cwd()
    common = Path(parts[0])
    for p in parts[1:]:
        while not str(p).startswith(str(common)):
            common = common.parent
    return common


def main() -> None:
    input_paths = [Path(x) for x in INPUT_PATHS]
    recursive = bool(CFG_RECURSIVE)
    files = _scan_files(input_paths, recursive=recursive)

    compare_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    file_index: list[dict[str, Any]] = []

    for path in files:
        rec = _match_pattern(path.name, COMPARE_PATTERNS)
        if rec is not None:
            setting_type, setting_raw = rec
            rows = _load_compare_rows(path, setting_type, setting_raw)
            compare_rows.extend(rows)
            file_index.append({"kind": "compare", "path": str(path), "rows": len(rows)})
            continue

        rec = _match_pattern(path.name, SUMMARY_PATTERNS)
        if rec is not None:
            summary_kind, _ = rec
            rows = _load_summary_table(path, summary_kind)
            summary_rows.extend(rows)
            file_index.append({"kind": summary_kind, "path": str(path), "rows": len(rows)})
            continue

        rec = _match_pattern(path.name, STATS_PATTERNS)
        if rec is not None:
            setting_type, setting_raw = rec
            row = _load_stats_json(path, setting_type, setting_raw)
            if row is not None:
                stats_rows.append(row)
                file_index.append({"kind": "stats_json", "path": str(path), "rows": 1})

    if OUTPUT_DIR:
        out_dir = Path(OUTPUT_DIR)
    else:
        out_dir = _common_parent(input_paths) / "ppl_ablation_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = _build_overview(compare_rows)
    top_case_rows = _build_top_cases(compare_rows, TOP_K_CASES)

    _write_tsv(out_dir / "file_index.tsv", file_index)
    _write_tsv(out_dir / "compare_prompt_level.tsv", compare_rows)
    _write_tsv(out_dir / "compare_setting_overview.tsv", overview_rows)
    _write_tsv(out_dir / "top_regression_cases.tsv", top_case_rows)
    _write_tsv(out_dir / "summary_tables_merged.tsv", summary_rows)
    _write_tsv(out_dir / "stats_json_merged.tsv", stats_rows)

    meta = {
        "input_paths": [str(p) for p in input_paths],
        "recursive": recursive,
        "num_scanned_files": len(files),
        "num_compare_rows": len(compare_rows),
        "num_summary_rows": len(summary_rows),
        "num_stats_rows": len(stats_rows),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Summary complete.")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] compare_rows={len(compare_rows)} summary_rows={len(summary_rows)} stats_rows={len(stats_rows)}")


if __name__ == "__main__":
    main()
