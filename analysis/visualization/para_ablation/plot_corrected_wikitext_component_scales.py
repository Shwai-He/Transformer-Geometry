#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
PARA_ROOT = ROOT / "eval/lm-evaluation-harness/outputs/xsa_ppl_wikitext_alpha_sweep_20260606/remove_parallel/by_model/qwen3_1p7b"
PERP_ROOT = ROOT / "eval/lm-evaluation-harness/outputs/xsa_ppl_perp_retained_scale_qwen1p7b_20260711/by_model/qwen3_1p7b"
OUTPUT_DIR = ROOT / "_ARR_May/manuscript_working_copy/figs/para_ablation"
AUDIT_DIR = ROOT / "_ARR_May/experimental/corrected_ppl_curves"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_ppl(path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["results"]["wikitext"]["word_perplexity,none"]), payload


def one_result(directory: Path) -> Path:
    paths = [p for p in directory.glob("*.json") if ".layer_scales" not in p.as_posix()]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one result in {directory}, found {len(paths)}")
    return paths[0]


def collect_para() -> list[dict[str, object]]:
    baseline_path = one_result(PARA_ROOT / "baseline")
    baseline, baseline_payload = load_ppl(baseline_path)
    rows = [{
        "component": "parallel", "retained_scale": 1.0,
        "ppl": baseline, "delta_ppl": 0.0,
        "source_json": str(baseline_path.relative_to(ROOT)),
        "harness_git_hash": baseline_payload.get("git_hash", ""),
    }]
    for path in sorted((PARA_ROOT / "kv2_group").glob("wikitext-kv2_group_para_edit*_*.json")):
        ppl, payload = load_ppl(path)
        args = payload["config"]["model_args"]
        removal_strength = float(args["xsa_forward_alpha"])
        if args.get("xsa_exclude_self") is not True:
            raise RuntimeError(f"Expected exclude-self para result: {path}")
        rows.append({
            "component": "parallel", "retained_scale": 1.0 - removal_strength,
            "ppl": ppl, "delta_ppl": ppl - baseline,
            "source_json": str(path.relative_to(ROOT)),
            "harness_git_hash": payload.get("git_hash", ""),
        })
    return sorted(rows, key=lambda r: float(r["retained_scale"]))


def collect_perp() -> list[dict[str, object]]:
    baseline_path = one_result(PERP_ROOT / "baseline")
    independent_baseline, baseline_payload = load_ppl(baseline_path)
    rows = []
    for op_dir in sorted((PERP_ROOT / "kv2_group").glob("perp_*")):
        path = one_result(op_dir / "exclude_self")
        ppl, payload = load_ppl(path)
        args = payload["config"]["model_args"]
        if args.get("xsa_exclude_self") is not True:
            raise RuntimeError(f"Expected exclude-self perp result: {path}")
        scale = float(args["xsa_forward_perp_scale"])
        rows.append({
            "component": "perpendicular", "retained_scale": scale,
            "ppl": ppl, "delta_ppl": 0.0,
            "source_json": str(path.relative_to(ROOT)),
            "harness_git_hash": payload.get("git_hash", ""),
        })
    if not any(float(r["retained_scale"]) == 1.0 for r in rows):
        raise RuntimeError("Perpendicular sweep is missing the scale-1 no-op gate")
    no_op = next(r for r in rows if float(r["retained_scale"]) == 1.0)
    hook_baseline = float(no_op["ppl"])
    raw_no_op_path = one_result(PERP_ROOT / "kv2_group/perp_1p0/raw_full")
    raw_no_op, _ = load_ppl(raw_no_op_path)
    if abs(raw_no_op - independent_baseline) > 1e-12:
        raise RuntimeError(
            f"Raw perpendicular scale-1 no-op failed: {raw_no_op - independent_baseline}"
        )
    hook_drift = hook_baseline - independent_baseline
    if abs(hook_drift / independent_baseline) > 1e-3:
        raise RuntimeError(f"Exclude-self hook baseline drift is too large: {hook_drift}")
    for row in rows:
        row["delta_ppl"] = float(row["ppl"]) - hook_baseline
        row["independent_baseline_ppl"] = independent_baseline
        row["hook_baseline_drift"] = hook_drift
    return sorted(rows, key=lambda r: float(r["retained_scale"]))


def plot(rows: list[dict[str, object]], component: str, filename: str) -> None:
    xs = [float(r["retained_scale"]) for r in rows]
    ys = [float(r["delta_ppl"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.55))
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.3, label="Baseline")
    ax.plot(xs, ys, color="#1f77b4", marker="o", linewidth=2.2, markersize=5.2,
            label="Excl.-self V")
    ax.set_xlabel(f"{component.capitalize()} retained scale", fontsize=13)
    ax.set_ylabel(r"$\Delta\mathrm{PPL}$", fontsize=13)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0] if component == "parallel" else [0.0, 0.5, 1.0, 1.5]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="both", labelsize=10.5, length=3.5)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)
    ax.legend(loc="upper right", fontsize=10.5, frameon=True, facecolor="white")
    ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_DIR / filename.replace(".pdf", ".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    para = collect_para()
    perp = collect_perp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = para + perp
    with (AUDIT_DIR / "curve_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    plot(para, "parallel", "para_scale_weighted_delta_ppl.pdf")
    plot(perp, "perpendicular", "perp_scale_weighted_delta_ppl.pdf")
    print(f"Wrote corrected curves with {len(para)} para and {len(perp)} perp points")


if __name__ == "__main__":
    main()
