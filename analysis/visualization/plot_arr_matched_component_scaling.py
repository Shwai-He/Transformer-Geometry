#!/usr/bin/env python3
"""Draw Figure 3 only from protocol-matched Qwen3-1.7B evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALUE_SOURCE = (
    REPO_ROOT / "_ARR_May/experimental/corrected_ppl_curves/curve_rows.csv"
)
DEFAULT_RESIDUAL_ROOT = (
    REPO_ROOT
    / "eval/lm-evaluation-harness/outputs/xsa_figure3_matched_residual_20260809"
    / "by_model/qwen3_1p7b"
)
DEFAULT_VALUE_PARALLEL_ROOT = (
    REPO_ROOT
    / "eval/lm-evaluation-harness/outputs/xsa_figure3_matched_component_scaling"
    / "by_model/qwen3_1p7b"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "_ARR_May/manuscript_working_copy/figs/para_ablation_corrected"
    / "ppl_component_scaling_corrected.pdf"
)
EXPECTED_VALUE_CSV = {
    "perpendicular": {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0},
}
EXPECTED_PARALLEL = {-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0}
EXPECTED_RESIDUAL = {
    "parallel": {-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0},
    "perpendicular": {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5},
}
SERIES = [
    ("value", "Value-space (excl.-self)", "#1f77b4", "o", "-"),
    ("residual_attn", "Residual(Attn)", "#ff7f0e", "s", "--"),
    ("residual_mlp", "Residual(MLP)", "#2ca02c", "^", "-."),
]


def load_source_manifest(
    path: Path,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Load the paper's lightweight, already-audited plotting bundle.

    This mode intentionally does not claim to revalidate the original lm-eval
    JSONs.  It makes the publication asset reproducible when those large raw
    output trees are unavailable, while retaining their relative paths in the
    manifest for provenance and later source-level verification.
    """
    panels: dict[str, dict[str, list[dict[str, object]]]] = {
        axis: {key: [] for key, *_ in SERIES}
        for axis in ("parallel", "perpendicular")
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            axis = row["panel"]
            series = row["series"]
            if axis not in panels or series not in panels[axis]:
                raise ValueError(f"Unexpected panel/series in {path}: {axis}/{series}")
            panels[axis][series].append(
                {
                    "scale": round(float(row["scale"]), 8),
                    "ppl": float(row["ppl"]),
                    "delta_ppl": float(row["delta_ppl"]),
                    "source": row["source"],
                }
            )

    for axis, series_rows in panels.items():
        for series, rows in series_rows.items():
            expected = (
                EXPECTED_VALUE_CSV[axis]
                if axis == "perpendicular" and series == "value"
                else EXPECTED_PARALLEL
                if axis == "parallel"
                else EXPECTED_RESIDUAL[axis]
            )
            observed = [float(row["scale"]) for row in rows]
            if len(observed) != len(set(observed)):
                raise ValueError(f"Duplicate {axis}/{series} scales in {path}: {observed}")
            _assert_scales(f"{axis}/{series}", set(observed), expected)
            noop = next(row for row in rows if float(row["scale"]) == 1.0)
            if abs(float(noop["delta_ppl"])) > 1e-9:
                raise ValueError(f"{axis}/{series} scale-1 no-op failed: {noop}")
            rows.sort(key=lambda row: float(row["scale"]))
    return panels


def _assert_scales(label: str, observed: set[float], expected: set[float]) -> None:
    if observed != expected:
        raise ValueError(
            f"Incomplete {label}: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )


def load_value_rows(path: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            axis = row["component"]
            source = REPO_ROOT / row["source_json"]
            source_ppl, payload = _load_ppl(source)
            model_args = payload["config"]["model_args"]
            expected_ppl = float(row["ppl"])
            if abs(source_ppl - expected_ppl) > 1e-9:
                raise ValueError(f"Value-space PPL/source mismatch in {source}")
            if not str(model_args["pretrained"]).endswith(
                "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
            ):
                raise ValueError(f"Value-space snapshot mismatch in {source}")
            if model_args["xsa_intervention_site"] != "head_specific":
                raise ValueError(f"Wrong value-space intervention site in {source}")
            if not bool(model_args["xsa_exclude_self"]):
                raise ValueError(f"Expected exclude-self value reference in {source}")
            if int(model_args["xsa_skip_first_n"]) != 1:
                raise ValueError(f"Wrong value-space layer range in {source}")
            grouped[axis].append(
                {
                    "scale": round(float(row["retained_scale"]), 8),
                    "delta_ppl": float(row["delta_ppl"]),
                    "ppl": expected_ppl,
                    "source": row["source_json"],
                }
            )
    for axis, expected in EXPECTED_VALUE_CSV.items():
        _assert_scales(f"value-space {axis}", {r["scale"] for r in grouped[axis]}, expected)
        noop = next(r for r in grouped[axis] if r["scale"] == 1.0)
        if abs(noop["delta_ppl"]) > 1e-9:
            raise ValueError(f"Value-space {axis} scale-1 no-op failed: {noop}")
        grouped[axis].sort(key=lambda row: row["scale"])
    return grouped


def load_value_parallel_rows(root: Path) -> list[dict[str, object]]:
    baseline_files = list((root / "baseline").glob("wikitext-baseline_*.json"))
    if len(baseline_files) != 1:
        raise ValueError(f"Expected exactly one matched value baseline, found {baseline_files}")
    _, baseline_payload = _load_ppl(baseline_files[0])
    expected_snapshot = baseline_payload["config"]["model_args"]["pretrained"]

    rows: list[dict[str, object]] = []
    result_files = sorted((root / "kv2_group").glob("wikitext-kv2_group_para_edit*.json"))
    for path in result_files:
        ppl, payload = _load_ppl(path)
        model_args = payload["config"]["model_args"]
        if model_args["pretrained"] != expected_snapshot:
            raise ValueError(f"Value-space snapshot mismatch in {path}")
        if model_args["xsa_intervention_site"] != "head_specific":
            raise ValueError(f"Wrong value-space intervention site in {path}")
        if not bool(model_args["xsa_exclude_self"]):
            raise ValueError(f"Expected exclude-self value reference in {path}")
        if str(model_args["xsa_value_proj_group_size"]) != "kv2":
            raise ValueError(f"Expected KV2 grouping in {path}")
        if int(model_args["xsa_skip_first_n"]) != 1:
            raise ValueError(f"Wrong value-space layer range in {path}")
        if abs(float(model_args["xsa_forward_perp_scale"]) - 1.0) > 1e-12:
            raise ValueError(f"Not a parallel-only value-space point: {path}")
        retained_scale = round(float(model_args["xsa_forward_alpha"]), 8)
        rows.append(
            {
                "scale": retained_scale,
                "delta_ppl": 0.0,
                "ppl": ppl,
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )

    scales = [row["scale"] for row in rows]
    if len(scales) != len(set(scales)):
        raise ValueError(f"Duplicate value-space parallel scales: {scales}")
    _assert_scales("value-space parallel", set(scales), EXPECTED_PARALLEL)
    noop = next(row for row in rows if row["scale"] == 1.0)
    noop_ppl = float(noop["ppl"])
    for row in rows:
        row["delta_ppl"] = float(row["ppl"]) - noop_ppl
    rows.sort(key=lambda row: row["scale"])
    return rows


def _load_ppl(path: Path) -> tuple[float, dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    result = payload["results"]["wikitext"]
    if int(result["sample_len"]) != 62:
        raise ValueError(f"Expected all 62 WikiText documents in {path}")
    return float(result["word_perplexity,none"]), payload


def load_residual_rows(root: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    baseline_files = list((root / "baseline").glob("wikitext-baseline_*.json"))
    if len(baseline_files) != 1:
        raise ValueError(f"Expected exactly one matched baseline, found {baseline_files}")
    baseline_ppl, baseline_payload = _load_ppl(baseline_files[0])
    expected_snapshot = baseline_payload["config"]["model_args"]["pretrained"]
    output: dict[str, dict[str, list[dict[str, object]]]] = {}

    for method in ("residual_attn", "residual_mlp"):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        result_files = sorted((root / method).glob("wikitext-*.json"))
        for path in result_files:
            ppl, payload = _load_ppl(path)
            model_args = payload["config"]["model_args"]
            if model_args["pretrained"] != expected_snapshot:
                raise ValueError(f"Snapshot mismatch in {path}")
            if model_args["xsa_intervention_site"] != "residual_output":
                raise ValueError(f"Wrong intervention site in {path}")
            if int(model_args["xsa_skip_first_n"]) != 1:
                raise ValueError(f"Wrong layer range in {path}")
            if model_args["xsa_target"] != ("attn" if method.endswith("attn") else "mlp"):
                raise ValueError(f"Wrong residual target in {path}")

            para = round(float(model_args["xsa_forward_alpha"]), 8)
            perp = round(float(model_args["xsa_forward_perp_scale"]), 8)
            if abs(perp - 1.0) < 1e-12:
                axis, scale = "parallel", para
            elif abs(para - 1.0) < 1e-12:
                axis, scale = "perpendicular", perp
            else:
                raise ValueError(f"Not a one-axis sweep point: {path}")
            grouped[axis].append(
                {
                    "scale": scale,
                    "delta_ppl": ppl - baseline_ppl,
                    "ppl": ppl,
                    "source": str(path.relative_to(REPO_ROOT)),
                }
            )

        # The scale-1 canary belongs to both axes, although it is stored once.
        para_noop = [r for r in grouped["parallel"] if r["scale"] == 1.0]
        if len(para_noop) != 1:
            raise ValueError(f"Expected one {method} scale-1 canary")
        grouped["perpendicular"].append(dict(para_noop[0]))
        for axis, expected in EXPECTED_RESIDUAL.items():
            scales = [r["scale"] for r in grouped[axis]]
            if len(scales) != len(set(scales)):
                raise ValueError(f"Duplicate {method} {axis} scales: {scales}")
            _assert_scales(f"{method} {axis}", set(scales), expected)
            noop = next(r for r in grouped[axis] if r["scale"] == 1.0)
            if abs(noop["delta_ppl"]) > 1e-9:
                raise ValueError(f"{method} {axis} scale-1 no-op failed: {noop}")
            grouped[axis].sort(key=lambda row: row["scale"])
        output[method] = grouped
    return output


def draw_panel(ax, data: dict[str, list[dict[str, object]]]) -> None:
    for key, label, color, marker, linestyle in SERIES:
        rows = data[key]
        ax.plot(
            [row["scale"] for row in rows],
            [row["delta_ppl"] for row in rows],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.15,
            markersize=5.0,
            markeredgewidth=0.5,
        )
    ax.set_yscale("symlog", base=10, linthresh=1.0, linscale=0.9)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, zorder=0)
    ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1.1, zorder=0)
    ax.annotate(
        "no-op", xy=(1.0, 0.0), xytext=(5, -14), textcoords="offset points",
        ha="left", va="top", fontsize=9.5,
    )
    ax.grid(axis="y", which="major", color="#d5d5d5", linewidth=0.75, alpha=0.9)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-source", type=Path, default=DEFAULT_VALUE_SOURCE)
    parser.add_argument(
        "--value-parallel-root", type=Path, default=DEFAULT_VALUE_PARALLEL_ROOT
    )
    parser.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL_ROOT)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Replot directly from a previously audited *_sources.csv bundle; "
            "raw lm-eval JSON validation is skipped in this mode."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.source_manifest is not None:
        panels = load_source_manifest(args.source_manifest)
    else:
        value = load_value_rows(args.value_source)
        value["parallel"] = load_value_parallel_rows(args.value_parallel_root)
        residual = load_residual_rows(args.residual_root)
        panels = {
            axis: {
                "value": value[axis],
                "residual_attn": residual["residual_attn"][axis],
                "residual_mlp": residual["residual_mlp"][axis],
            }
            for axis in ("parallel", "perpendicular")
        }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.9, 3.8))
    draw_panel(axes[0], panels["parallel"])
    draw_panel(axes[1], panels["perpendicular"])
    axes[0].set_title("(a) Parallel scaling")
    axes[1].set_title("(b) Perpendicular scaling")
    axes[0].set_xlabel(r"Parallel retained scale $s_{\parallel}$")
    axes[1].set_xlabel(r"Perpendicular retained scale $s_{\perp}$")
    axes[0].set_ylabel(r"$\Delta$PPL (matched no-op $=0$)")
    axes[0].set_xlim(-1.1, 3.1)
    axes[1].set_xlim(-0.05, 2.05)
    axes[1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    # Symlog's default locator labels every decade, which is visually crowded
    # at the paper's two-column scale. Keep the zero and negative-unit anchors,
    # then label only alternating positive decades.
    axes[0].yaxis.set_major_locator(FixedLocator([-1.0, 0.0, 1e2, 1e4, 1e6]))
    axes[1].yaxis.set_major_locator(
        FixedLocator([-1.0, 0.0, 1e2, 1e4, 1e6, 1e8])
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=3, frameon=True, framealpha=0.9, handlelength=2.0,
        columnspacing=1.2, borderpad=0.35,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.85), pad=0.55, w_pad=1.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    manifest = args.output.with_name(args.output.stem + "_sources.csv")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "scale", "ppl", "delta_ppl", "source"])
        for axis, series_rows in panels.items():
            for key, rows in series_rows.items():
                for row in rows:
                    writer.writerow(
                        [axis, key, row["scale"], row["ppl"], row["delta_ppl"], row["source"]]
                    )
    print(args.output)


if __name__ == "__main__":
    main()
