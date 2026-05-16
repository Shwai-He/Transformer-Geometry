#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

# =========================
# File-first configuration
# =========================
# Set INPUT_DIRS to your ablation result directories, then run:
# python summarize_alpha_gamma_ablation.py
INPUT_DIRS = [
    "/mnt/hdfs/shwai.he/DepthBoost/representation-analysis/outputs/xsa_alpha_sweep",
    "/mnt/hdfs/shwai.he/DepthBoost/representation-analysis/outputs/xsa_perp_ablation",
]
# If empty string, script will auto-pick:
# common_parent_of_inputs / "xsa_ablation_summary"
OUTPUT_DIR = ""
TOP_K_CASES = 3


COMPARE_PATTERNS = [
    ("alpha", re.compile(r"^compare_alpha_(.+)\.tsv$")),
    ("gamma", re.compile(r"^compare_gamma_(.+)\.tsv$")),
]
STATS_PATTERNS = [
    ("alpha", re.compile(r"^generate_alpha_(.+)\.stats\.json$")),
    ("gamma", re.compile(r"^generate_gamma_(.+)\.stats\.json$")),
]


@dataclass
class RowRecord:
    setting_type: str
    setting_raw: str
    setting_value: float
    alpha_value: float | None
    perp_value: float | None
    gamma_value: float | None
    prompt_idx: str
    prompt_preview: str
    delta_ppl: float | None
    delta_loss: float | None
    row: dict[str, str]


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
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


def _decode_setting_token(token: str) -> float:
    # "1p5" -> 1.5, "-100p0" -> -100.0
    return float(token.replace("p", "."))


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


def _parse_setting(setting_type: str, raw: str) -> tuple[float, float | None, float | None, float | None]:
    """
    Return:
      (primary_setting_value, alpha_value, perp_value, gamma_value)

    Supported patterns from your screenshots:
      - alpha sweep: raw="0p5" / "-1p0"
      - alpha+perp: raw="1.0_perp_0.3" or "1p0_perp_0p3"
      - gamma sweep: raw same token style as alpha
    """
    if setting_type == "alpha":
        # New perp ablation naming:
        # compare_alpha_fixed_<alpha>_perp_keep_<perp>.tsv
        m = re.match(r"^alpha_fixed_(.+)_perp_keep_(.+)$", raw)
        if m:
            a = _try_float_token(m.group(1))
            p = _try_float_token(m.group(2))
            if a is None or p is None:
                raise ValueError(f"Cannot parse alpha_fixed/perp_keep from: {raw}")
            return (a, a, p, None)

        m = re.match(r"^(.+)_perp_(.+)$", raw)
        if m:
            a = _try_float_token(m.group(1))
            p = _try_float_token(m.group(2))
            if a is None or p is None:
                raise ValueError(f"Cannot parse alpha/perp from: {raw}")
            return (a, a, p, None)
        a = _try_float_token(raw)
        if a is None:
            raise ValueError(f"Cannot parse alpha from: {raw}")
        return (a, a, None, None)

    if setting_type == "gamma":
        g = _try_float_token(raw)
        if g is None:
            raise ValueError(f"Cannot parse gamma from: {raw}")
        return (g, None, None, g)

    v = _try_float_token(raw)
    if v is None:
        raise ValueError(f"Cannot parse setting value from: {raw}")
    return (v, None, None, None)


def _find_first(d: dict[str, str], keys: list[str]) -> str | None:
    lower_to_key = {k.lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in lower_to_key:
            return d[lower_to_key[k.lower()]]
    return None


def _extract_row_record(
    setting_type: str,
    setting_raw: str,
    setting_value: float,
    alpha_value: float | None,
    perp_value: float | None,
    gamma_value: float | None,
    row: dict[str, str],
) -> RowRecord:
    prompt_idx = _find_first(row, ["prompt_idx", "idx", "index", "id"]) or ""
    prompt_preview = _find_first(row, ["prompt_preview", "prompt", "text"]) or ""

    delta_ppl = _to_float(
        _find_first(
            row,
            [
                "delta_ppl",
                "ppl_delta",
                "delta_ppl_mean",
                "ppl_diff",
            ],
        )
    )
    delta_loss = _to_float(
        _find_first(
            row,
            [
                "delta_loss",
                "loss_delta",
                "delta_loss_mean",
                "loss_diff",
            ],
        )
    )
    return RowRecord(
        setting_type=setting_type,
        setting_raw=setting_raw,
        setting_value=setting_value,
        alpha_value=alpha_value,
        perp_value=perp_value,
        gamma_value=gamma_value,
        prompt_idx=prompt_idx,
        prompt_preview=prompt_preview,
        delta_ppl=delta_ppl,
        delta_loss=delta_loss,
        row=row,
    )


def _scan_compare_records(input_dir: Path) -> list[RowRecord]:
    records: list[RowRecord] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        matched = None
        for setting_type, rgx in COMPARE_PATTERNS:
            m = rgx.match(p.name)
            if m:
                matched = (setting_type, m.group(1))
                break
        if matched is None:
            continue
        setting_type, raw = matched
        try:
            setting_value, alpha_value, perp_value, gamma_value = _parse_setting(setting_type, raw)
        except Exception:
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                records.append(
                    _extract_row_record(
                        setting_type, raw, setting_value, alpha_value, perp_value, gamma_value, row
                    )
                )
    return records


def _collect_numeric_top_level(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, (int, float)):
            fv = _to_float(v)
            if fv is not None:
                out[key] = fv
        elif isinstance(v, dict):
            out.update(_collect_numeric_top_level(v, key))
    return out


def _scan_stats_files(input_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        matched = None
        for setting_type, rgx in STATS_PATTERNS:
            m = rgx.match(p.name)
            if m:
                matched = (setting_type, m.group(1))
                break
        if matched is None:
            continue
        setting_type, raw = matched
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            out[(setting_type, raw)] = _collect_numeric_top_level(data)
    return out


def _common_parent(paths: list[Path]) -> Path:
    if not paths:
        return Path(".").resolve()
    common = Path(paths[0]).resolve()
    for p in paths[1:]:
        p = p.resolve()
        while common != common.parent and not str(p).startswith(str(common)):
            common = common.parent
    return common


def _group_summary(records: list[RowRecord]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, float, float | None, float | None, float | None], list[RowRecord]] = {}
    for r in records:
        k = (r.setting_type, r.setting_raw, r.setting_value, r.alpha_value, r.perp_value, r.gamma_value)
        by_key.setdefault(k, []).append(r)

    rows: list[dict[str, Any]] = []
    for (setting_type, raw, val, alpha_value, perp_value, gamma_value), group in by_key.items():
        d_ppl = [x.delta_ppl for x in group if x.delta_ppl is not None]
        d_loss = [x.delta_loss for x in group if x.delta_loss is not None]
        row: dict[str, Any] = {
            "setting_type": setting_type,
            "setting_raw": raw,
            "setting_value": f"{val:.6g}",
            "alpha": "" if alpha_value is None else f"{alpha_value:.6g}",
            "perp": "" if perp_value is None else f"{perp_value:.6g}",
            "gamma": "" if gamma_value is None else f"{gamma_value:.6g}",
            "n_rows": len(group),
            "n_delta_ppl": len(d_ppl),
            "n_delta_loss": len(d_loss),
            "mean_delta_ppl": f"{mean(d_ppl):.8f}" if d_ppl else "",
            "median_delta_ppl": f"{median(d_ppl):.8f}" if d_ppl else "",
            "win_rate_delta_ppl_lt_0": f"{(sum(1 for x in d_ppl if x < 0.0) / len(d_ppl)):.8f}" if d_ppl else "",
            "worst_case_delta_ppl": f"{max(d_ppl):.8f}" if d_ppl else "",
            "best_case_delta_ppl": f"{min(d_ppl):.8f}" if d_ppl else "",
            "mean_delta_loss": f"{mean(d_loss):.8f}" if d_loss else "",
            "median_delta_loss": f"{median(d_loss):.8f}" if d_loss else "",
            "worst_case_delta_loss": f"{max(d_loss):.8f}" if d_loss else "",
            "best_case_delta_loss": f"{min(d_loss):.8f}" if d_loss else "",
        }
        rows.append(row)
    rows.sort(
        key=lambda x: (
            x["setting_type"],
            float(x["alpha"]) if x["alpha"] != "" else float("inf"),
            float(x["perp"]) if x["perp"] != "" else float("inf"),
            float(x["gamma"]) if x["gamma"] != "" else float("inf"),
            float(x["setting_value"]),
        )
    )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _pareto_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in summary_rows:
        by_type.setdefault(r["setting_type"], []).append(r)

    out: list[dict[str, Any]] = []
    for setting_type, rows in by_type.items():
        candidates = []
        for r in rows:
            m = _to_float(r.get("mean_delta_ppl"))
            w = _to_float(r.get("worst_case_delta_ppl"))
            if m is None or w is None:
                continue
            candidates.append((r, m, w))
        for i, (ri, mi, wi) in enumerate(candidates):
            dominated = False
            for j, (rj, mj, wj) in enumerate(candidates):
                if i == j:
                    continue
                if (mj <= mi and wj <= wi) and (mj < mi or wj < wi):
                    dominated = True
                    break
            if not dominated:
                out.append(ri)
    out.sort(key=lambda x: (x["setting_type"], float(x["setting_value"])))
    return out


def _grid_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows where both alpha and perp are present (alpha-perp ablation grid)."""
    out = [r for r in summary_rows if r.get("alpha", "") != "" and r.get("perp", "") != ""]
    out.sort(key=lambda x: (float(x["alpha"]), float(x["perp"])))
    return out


def _attach_stats(summary_rows: list[dict[str, Any]], stats_map: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    all_stat_keys: set[str] = set()
    for v in stats_map.values():
        all_stat_keys.update(v.keys())
    stat_keys = sorted(all_stat_keys)

    out: list[dict[str, Any]] = []
    for row in summary_rows:
        r = dict(row)
        k = (row["setting_type"], row["setting_raw"])
        stats = stats_map.get(k, {})
        for sk in stat_keys:
            v = stats.get(sk)
            r[f"stats.{sk}"] = f"{v:.8f}" if v is not None else ""
        out.append(r)
    return out


def _trim_text(s: str, n: int = 280) -> str:
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _write_casebook(path: Path, records: list[RowRecord], top_k: int) -> None:
    by_type: dict[str, list[RowRecord]] = {}
    for r in records:
        by_type.setdefault(r.setting_type, []).append(r)

    lines: list[str] = []
    lines.append("# Alpha/Gamma Ablation Casebook")
    lines.append("")

    for setting_type, rs in sorted(by_type.items()):
        rs_ppl = [x for x in rs if x.delta_ppl is not None]
        rs_ppl.sort(key=lambda x: x.delta_ppl if x.delta_ppl is not None else 0.0)
        best = rs_ppl[:top_k]
        worst = rs_ppl[-top_k:] if top_k > 0 else []
        worst = list(reversed(worst))

        lines.append(f"## {setting_type.upper()} - Best {top_k} (lowest delta_ppl)")
        lines.append("")
        for r in best:
            lines.append(
                f"- {setting_type}={r.setting_value:.6g} | prompt_idx={r.prompt_idx} | "
                f"delta_ppl={r.delta_ppl:.8f} | delta_loss={'' if r.delta_loss is None else f'{r.delta_loss:.8f}'}"
            )
            if r.prompt_preview:
                lines.append(f"  - prompt: {_trim_text(r.prompt_preview)}")
        lines.append("")

        lines.append(f"## {setting_type.upper()} - Worst {top_k} (highest delta_ppl)")
        lines.append("")
        for r in worst:
            lines.append(
                f"- {setting_type}={r.setting_value:.6g} | prompt_idx={r.prompt_idx} | "
                f"delta_ppl={r.delta_ppl:.8f} | delta_loss={'' if r.delta_loss is None else f'{r.delta_loss:.8f}'}"
            )
            if r.prompt_preview:
                lines.append(f"  - prompt: {_trim_text(r.prompt_preview)}")
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    input_dirs = [Path(x).expanduser().resolve() for x in INPUT_DIRS]
    if not input_dirs:
        raise RuntimeError("INPUT_DIRS is empty. Please set at least one input directory.")
    for d in input_dirs:
        if not d.exists() or not d.is_dir():
            raise RuntimeError(f"Input directory not found or not a directory: {d}")

    if OUTPUT_DIR.strip():
        output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    else:
        output_dir = _common_parent(input_dirs) / "xsa_ablation_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[RowRecord] = []
    for d in input_dirs:
        records.extend(_scan_compare_records(d))
    if not records:
        dirs_str = ", ".join(str(x) for x in input_dirs)
        raise RuntimeError(f"No compare_alpha_*.tsv or compare_gamma_*.tsv found in: {dirs_str}")

    # De-duplicate repeated rows from multi-dir scans:
    # same setting + same prompt => keep one.
    uniq: dict[tuple[Any, ...], RowRecord] = {}
    for r in records:
        key = (
            r.setting_type,
            r.setting_raw,
            r.setting_value,
            r.alpha_value,
            r.perp_value,
            r.gamma_value,
            r.prompt_idx,
            r.prompt_preview,
        )
        if key not in uniq:
            uniq[key] = r
    records = list(uniq.values())

    summary = _group_summary(records)
    stats_map: dict[tuple[str, str], dict[str, float]] = {}
    for d in input_dirs:
        stats_map.update(_scan_stats_files(d))
    summary_with_stats = _attach_stats(summary, stats_map)
    pareto = _pareto_rows(summary_with_stats)
    alpha_perp_grid = _grid_rows(summary_with_stats)

    summary_path = output_dir / "alpha_gamma_summary.tsv"
    pareto_path = output_dir / "alpha_gamma_pareto.tsv"
    grid_path = output_dir / "alpha_perp_grid_summary.tsv"
    casebook_path = output_dir / "alpha_gamma_casebook.md"

    _write_tsv(summary_path, summary_with_stats)
    _write_tsv(pareto_path, pareto)
    _write_tsv(grid_path, alpha_perp_grid)
    _write_casebook(casebook_path, records, TOP_K_CASES)

    print("[INFO] Input dirs:")
    for d in input_dirs:
        print(f"  {d}")
    print("Logs:")
    print(f"  {summary_path}")
    print(f"  {pareto_path}")
    print(f"  {grid_path}")
    print(f"  {casebook_path}")
    print(
        f"[INFO] Done. summary_rows={len(summary_with_stats)} pareto_rows={len(pareto)} "
        f"alpha_perp_rows={len(alpha_perp_grid)}"
    )


if __name__ == "__main__":
    main()
