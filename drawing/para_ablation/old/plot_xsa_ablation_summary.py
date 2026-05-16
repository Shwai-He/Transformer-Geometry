#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =========================
# File-first configuration
# =========================
DRAWING_DIR = Path(__file__).absolute().parent
PAPER_ROOT = DRAWING_DIR.parents[1]
SUMMARY_TSV = DRAWING_DIR / "data" / "alpha_gamma_summary.tsv"
OUTPUT_DIR = PAPER_ROOT / "figs" / "para_ablation"
FIG_DPI = 220
PERP_PARA_SCALE_FIXED = 1.0
HEATMAP_MAX_ABS_DPPL = 15.0
LOW_IMPACT_ABS_DPPL = 1.0
LOW_IMPACT_ABS_DLOSS = 0.02
EXCLUDED_ALPHA_VALUES = {-100.0, 100.0}
USE_SYMLOG_FOR_SWEEP_Y = True
SYMLOG_LINTHRESH_DPPL = 1.0
SYMLOG_LINTHRESH_DLOSS = 0.02


def _to_float(x: str) -> float | None:
    s = (x or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _para_scale_from_alpha(alpha: float) -> float:
    # Internal summaries still store alpha; public plots use para scale.
    return 1.0 - alpha


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _select_alpha_sweep(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for r in rows:
        if r.get("setting_type", "") != "alpha":
            continue
        alpha = _to_float(r.get("alpha", ""))
        perp = _to_float(r.get("perp", ""))
        gamma = _to_float(r.get("gamma", ""))
        mean_dppl = _to_float(r.get("mean_delta_ppl", ""))
        worst_dppl = _to_float(r.get("worst_case_delta_ppl", ""))
        mean_dloss = _to_float(r.get("mean_delta_loss", ""))
        win_rate = _to_float(r.get("win_rate_delta_ppl_lt_0", ""))
        if alpha is None or perp is not None or gamma is not None:
            continue
        if alpha in EXCLUDED_ALPHA_VALUES:
            continue
        if mean_dppl is None or worst_dppl is None or mean_dloss is None:
            continue
        out.append(
            {
                "alpha": alpha,
                "para_scale": _para_scale_from_alpha(alpha),
                "mean_delta_ppl": mean_dppl,
                "worst_case_delta_ppl": worst_dppl,
                "mean_delta_loss": mean_dloss,
                "win_rate": 0.0 if win_rate is None else win_rate,
            }
        )
    out.sort(key=lambda x: x["alpha"])
    return out


def _select_perp_sweep_para_fixed(
    rows: list[dict[str, str]], para_scale_target: float = 1.0
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for r in rows:
        if r.get("setting_type", "") != "alpha":
            continue
        alpha = _to_float(r.get("alpha", ""))
        perp = _to_float(r.get("perp", ""))
        gamma = _to_float(r.get("gamma", ""))
        mean_dppl = _to_float(r.get("mean_delta_ppl", ""))
        worst_dppl = _to_float(r.get("worst_case_delta_ppl", ""))
        mean_dloss = _to_float(r.get("mean_delta_loss", ""))
        win_rate = _to_float(r.get("win_rate_delta_ppl_lt_0", ""))
        if alpha is None or perp is None or gamma is not None:
            continue
        if abs(_para_scale_from_alpha(alpha) - para_scale_target) > 1e-8:
            continue
        if mean_dppl is None or worst_dppl is None or mean_dloss is None:
            continue
        out.append(
            {
                "perp": perp,
                "mean_delta_ppl": mean_dppl,
                "worst_case_delta_ppl": worst_dppl,
                "mean_delta_loss": mean_dloss,
                "win_rate": 0.0 if win_rate is None else win_rate,
            }
        )
    out.sort(key=lambda x: x["perp"])
    return out


def _select_alpha_perp_grid(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for r in rows:
        if r.get("setting_type", "") != "alpha":
            continue
        alpha = _to_float(r.get("alpha", ""))
        perp = _to_float(r.get("perp", ""))
        gamma = _to_float(r.get("gamma", ""))
        mean_dppl = _to_float(r.get("mean_delta_ppl", ""))
        mean_dloss = _to_float(r.get("mean_delta_loss", ""))
        if alpha is None or perp is None or gamma is not None:
            continue
        if alpha in EXCLUDED_ALPHA_VALUES:
            continue
        if mean_dppl is None or mean_dloss is None:
            continue
        out.append(
            {
                "alpha": alpha,
                "perp": perp,
                "mean_delta_ppl": mean_dppl,
                "mean_delta_loss": mean_dloss,
            }
        )
    out.sort(key=lambda x: (x["alpha"], x["perp"]))
    return out


def _low_impact_count(values: list[float], threshold: float) -> int:
    return sum(1 for v in values if abs(v) <= threshold)


def _plot_alpha_sweep(alpha_rows: list[dict[str, float]], out_dir: Path) -> Path:
    xs = [r["para_scale"] for r in alpha_rows]
    ys_mean = [r["mean_delta_ppl"] for r in alpha_rows]
    ys_loss = [r["mean_delta_loss"] for r in alpha_rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(xs, ys_mean, marker="o", linewidth=2.2, color="#1f77b4")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axhspan(-LOW_IMPACT_ABS_DPPL, LOW_IMPACT_ABS_DPPL, color="#9fd3c7", alpha=0.25)
    if USE_SYMLOG_FOR_SWEEP_Y:
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH_DPPL)
    ax.set_ylabel("Mean ΔPPL")
    ax.set_title("Para Scale: Mean ΔPPL")
    ax.grid(alpha=0.25, linestyle=":")

    ax = axes[1]
    ax.plot(xs, ys_loss, marker="o", linewidth=2.2, color="#d95f02")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axhspan(-LOW_IMPACT_ABS_DLOSS, LOW_IMPACT_ABS_DLOSS, color="#fee08b", alpha=0.3)
    if USE_SYMLOG_FOR_SWEEP_Y:
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH_DLOSS)
    ax.set_ylabel("Mean ΔLoss")
    ax.set_xlabel("Para scale")
    ax.set_title("Para Scale: Mean ΔLoss")
    ax.grid(alpha=0.25, linestyle=":")

    fig.suptitle("Para Scale Sweep", y=1.01)
    out = out_dir / "fig_alpha_sweep_observed_effects.png"
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def _plot_perp_sweep(perp_rows: list[dict[str, float]], out_dir: Path, para_scale_fixed: float) -> Path:
    xs = [r["perp"] for r in perp_rows]
    ys_mean = [r["mean_delta_ppl"] for r in perp_rows]
    ys_loss = [r["mean_delta_loss"] for r in perp_rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(xs, ys_mean, marker="o", linewidth=2.2, color="#1f77b4")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1)
    ax.axhspan(-LOW_IMPACT_ABS_DPPL, LOW_IMPACT_ABS_DPPL, color="#9fd3c7", alpha=0.25)
    if USE_SYMLOG_FOR_SWEEP_Y:
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH_DPPL)
    ax.set_ylabel("Mean ΔPPL")
    ax.set_title(
        f"Perpendicular Scale at para_scale={para_scale_fixed:g}: Mean ΔPPL"
    )
    ax.grid(alpha=0.25, linestyle=":")

    ax = axes[1]
    ax.plot(xs, ys_loss, marker="o", linewidth=2.2, color="#d95f02")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1)
    ax.axhspan(-LOW_IMPACT_ABS_DLOSS, LOW_IMPACT_ABS_DLOSS, color="#fee08b", alpha=0.3)
    if USE_SYMLOG_FOR_SWEEP_Y:
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH_DLOSS)
    ax.set_ylabel("Mean ΔLoss")
    ax.set_xlabel("Perpendicular scale")
    ax.set_title(
        f"Perpendicular Scale at para_scale={para_scale_fixed:g}: Mean ΔLoss"
    )
    ax.grid(alpha=0.25, linestyle=":")

    fig.suptitle("Perpendicular Component Scale Sweep", y=1.01)
    out = out_dir / "fig_perp_sweep_observed_effects.png"
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def _plot_alpha_perp_heatmap(grid_rows: list[dict[str, float]], out_dir: Path) -> Path:
    para_vals = sorted({_para_scale_from_alpha(r["alpha"]) for r in grid_rows})
    perp_vals = sorted({r["perp"] for r in grid_rows})
    matrix = []
    for perp in perp_vals:
        row = []
        for para_scale in para_vals:
            cell = next(
                (
                    r["mean_delta_ppl"]
                    for r in grid_rows
                    if abs(_para_scale_from_alpha(r["alpha"]) - para_scale) < 1e-8
                    and abs(r["perp"] - perp) < 1e-8
                ),
                None,
            )
            row.append(float("nan") if cell is None else cell)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    norm = TwoSlopeNorm(vmin=-HEATMAP_MAX_ABS_DPPL, vcenter=0.0, vmax=HEATMAP_MAX_ABS_DPPL)
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ΔPPL")

    ax.set_xticks(range(len(para_vals)))
    ax.set_xticklabels([f"{v:g}" for v in para_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(perp_vals)))
    ax.set_yticklabels([f"{v:g}" for v in perp_vals])
    ax.set_xlabel("Para scale")
    ax.set_ylabel("Perpendicular scale")
    ax.set_title("Para-Scale / Perpendicular-Scale Grid: Mean ΔPPL")

    # Mark cells inside the small-effect band.
    for yi, perp in enumerate(perp_vals):
        for xi, para_scale in enumerate(para_vals):
            val = matrix[yi][xi]
            if val == val and abs(val) <= LOW_IMPACT_ABS_DPPL:
                ax.text(xi, yi, "·", ha="center", va="center", color="black", fontsize=12)

    fig.tight_layout()
    out = out_dir / "fig_alpha_perp_heatmap_mean_dppl.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def _plot_tolerance_bar(alpha_rows: list[dict[str, float]], perp_rows: list[dict[str, float]], out_dir: Path) -> Path:
    alpha_ppl = _low_impact_count([r["mean_delta_ppl"] for r in alpha_rows], LOW_IMPACT_ABS_DPPL)
    alpha_loss = _low_impact_count([r["mean_delta_loss"] for r in alpha_rows], LOW_IMPACT_ABS_DLOSS)
    perp_ppl = _low_impact_count([r["mean_delta_ppl"] for r in perp_rows], LOW_IMPACT_ABS_DPPL)
    perp_loss = _low_impact_count([r["mean_delta_loss"] for r in perp_rows], LOW_IMPACT_ABS_DLOSS)

    labels = [
        f"para scale\n|ΔPPL| <= {LOW_IMPACT_ABS_DPPL:g}",
        f"para scale\n|ΔLoss| <= {LOW_IMPACT_ABS_DLOSS:g}",
        f"perpendicular scale\n|ΔPPL| <= {LOW_IMPACT_ABS_DPPL:g}",
        f"perpendicular scale\n|ΔLoss| <= {LOW_IMPACT_ABS_DLOSS:g}",
    ]
    values = [alpha_ppl, alpha_loss, perp_ppl, perp_loss]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of settings in low-effect band")
    ax.set_title(
        f"Observed Tolerance Width\nGreen/Red bars use |ΔPPL| <= {LOW_IMPACT_ABS_DPPL:g} and |ΔLoss| <= {LOW_IMPACT_ABS_DLOSS:g}"
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    for i, v in enumerate(values):
        ax.text(i, v + 0.15, str(v), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out = out_dir / "fig_alpha_perp_tolerance_counts.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def main() -> None:
    summary_path = Path(SUMMARY_TSV).expanduser().resolve()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(summary_path)
    alpha_rows = _select_alpha_sweep(rows)
    perp_rows = _select_perp_sweep_para_fixed(rows, para_scale_target=PERP_PARA_SCALE_FIXED)
    grid_rows = _select_alpha_perp_grid(rows)

    if not alpha_rows:
        raise RuntimeError("No alpha-only sweep rows found. Check SUMMARY_TSV.")
    if not perp_rows:
        raise RuntimeError(
            f"No perp-sweep rows found at para_scale={PERP_PARA_SCALE_FIXED:g}. Check SUMMARY_TSV."
        )
    if not grid_rows:
        raise RuntimeError("No alpha-perp grid rows found. Check SUMMARY_TSV.")

    p1 = _plot_alpha_sweep(alpha_rows, out_dir)
    p2 = _plot_perp_sweep(perp_rows, out_dir, para_scale_fixed=PERP_PARA_SCALE_FIXED)
    p3 = _plot_alpha_perp_heatmap(grid_rows, out_dir)
    p4 = _plot_tolerance_bar(alpha_rows, perp_rows, out_dir)

    print("[INFO] Summary file:")
    print(f"  {summary_path}")
    print("[INFO] Figures:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print(f"  {p4}")
    print(
        f"[INFO] Done. para_points={len(alpha_rows)} "
        f"perp_points(para_scale={PERP_PARA_SCALE_FIXED:g})={len(perp_rows)} "
        f"grid_points={len(grid_rows)}"
    )


if __name__ == "__main__":
    main()
