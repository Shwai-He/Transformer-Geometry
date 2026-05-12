#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import pandas as pd


HERE = Path(__file__).absolute().parent
PAPER_ROOT = HERE.parents[1]
DATA_PATH = HERE / "data" / "all_settings_master_v2.tsv"
OUT_DIR = PAPER_ROOT / "figs" / "comp_analysis"

KEEP_SETTINGS = ["awq_native", "wanda_2_4", "wanda_4_8", "wanda_unstructured"]
PLOT_ORDER = ["awq_native", "wanda_unstructured", "wanda_4_8", "wanda_2_4"]
COMPONENTS = ["block_out", "attn_out", "mlp_out"]
PLOT_PARALLEL_ERROR = True

DISPLAY_NAME = {
    "awq_native": "Quantization",
    "wanda_2_4": "2:4",
    "wanda_4_8": "4:8",
    "wanda_unstructured": "Unstructured",
}

COMPONENT_TITLE = {
    "block_out": "Block",
    "attn_out": "Attention",
    "mlp_out": "MLP",
}

STYLE_MAP = {
    "awq_native": {"color": "#4C78A8", "marker": "o"},
    "wanda_unstructured": {"color": "#222222", "marker": "s"},
    "wanda_4_8": {"color": "#F58518", "marker": "^"},
    "wanda_2_4": {"color": "#E45756", "marker": "D"},
}

PLOT_RC = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.05,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
}


def load_local_rows():
    df = pd.read_csv(DATA_PATH, sep="\t")
    if "focus_layer" in df.columns:
        df = df[df["layer"] == df["focus_layer"]].copy()
    for col in (
        "perp_over_base_update",
        "perp_over_base_update_std",
        "base_update_over_hidden_state",
        "base_update_over_hidden_state_std",
        "hidden_state_norm",
        "hidden_state_norm_std",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    baseline_cols = [
        "hidden_state_norm",
        "hidden_state_norm_std",
        "base_update_over_hidden_state",
        "base_update_over_hidden_state_std",
    ]
    for col in baseline_cols:
        if col in df.columns:
            fill = df.groupby(["component", "layer"])[col].transform(
                lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan")
            )
            df[col] = df[col].fillna(fill)
    return df


def exclude_edge_layers(df):
    layers = sorted(df["layer"].dropna().unique().tolist())
    if len(layers) < 3:
        return df
    return df[(df["layer"] >= layers[1]) & (df["layer"] <= layers[-2])].copy()


def nice_ylim(series, pad=0.12):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    hi = float(s.quantile(0.98))
    if hi <= 0:
        hi = float(s.max())
    return (0.0, hi * (1.0 + pad))


def centered_ylim(lo_series, hi_series=None, pad=0.16, min_span=0.05):
    lo_s = pd.to_numeric(lo_series, errors="coerce").dropna()
    hi_s = lo_s if hi_series is None else pd.to_numeric(hi_series, errors="coerce").dropna()
    if lo_s.empty or hi_s.empty:
        return None
    lo = float(lo_s.quantile(0.02))
    hi = float(hi_s.quantile(0.98))
    if hi < lo:
        lo, hi = hi, lo
    span = max(hi - lo, min_span)
    center = 0.5 * (lo + hi)
    half = 0.5 * span * (1.0 + pad)
    return (center - half, center + half)


Y_LIM = {
    ("orthogonal_error", "block_out"): (0.0, 0.6137837491035462),
    ("orthogonal_error", "attn_out"): (0.0, 1.619879617691047),
    ("orthogonal_error", "mlp_out"): (0.0, 0.5234566307067871),
    ("baseline_update_ratio", "block_out"): None,
    ("baseline_update_ratio", "attn_out"): None,
    ("baseline_update_ratio", "mlp_out"): None,
}

PARA_Y_LIM = {
    ("parallel_error", "block_out"): (0.0, 0.48),
    ("parallel_error", "attn_out"): (0.0, 0.44),
    ("parallel_error", "mlp_out"): (0.0, 0.52),
}


def plot_orthogonal_error(df):
    metric = "perp_over_base_update"
    ylabel = r"$\|\Delta_{\perp \Delta_{\mathrm{base}}}\| / \|\Delta_{\mathrm{base}}\|$"
    df = exclude_edge_layers(df[df["setting"].isin(KEEP_SETTINGS)].copy())

    for comp in COMPONENTS:
        d = df[df["component"] == comp].copy()
        if d.empty or d[metric].notna().sum() == 0:
            continue

        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=200, constrained_layout=True)
            valid_settings = [s for s in PLOT_ORDER if d[d["setting"] == s][metric].notna().any()]

            for setting in valid_settings:
                g = d[d["setting"] == setting].sort_values("layer")
                style = STYLE_MAP.get(setting, {"color": "#4C78A8", "marker": "o"})
                line = ax.plot(
                    g["layer"],
                    g[metric],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=3.0,
                    markersize=7.2,
                    markeredgewidth=1.0,
                    label=DISPLAY_NAME.get(setting, setting),
                    zorder=3,
                )[0]
                std_col = f"{metric}_std"
                if std_col in g.columns and g[std_col].notna().any():
                    y = g[metric].astype(float)
                    ystd = g[std_col].fillna(0.0).astype(float)
                    ax.fill_between(
                        g["layer"],
                        y - ystd,
                        y + ystd,
                        color=line.get_color(),
                        alpha=0.08,
                        linewidth=0,
                        zorder=1,
                    )

            layers = sorted(d["layer"].dropna().unique().tolist())
            if layers:
                ax.set_xlim(min(layers), max(layers))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=min(10, len(layers)), integer=True))
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#d0d0d0", linewidth=0.9, alpha=0.50)
            ax.tick_params(axis="both", which="major", length=4.8, width=0.9)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ylim = Y_LIM.get(("orthogonal_error", comp), nice_ylim(d[metric]))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=min(4, len(valid_settings)),
                frameon=True,
                facecolor="white",
                edgecolor="#cfcfcf",
                borderpad=0.42,
                handlelength=2.5,
                columnspacing=1.45,
            )

            for suffix in ("pdf", "png"):
                path = OUT_DIR / f"local_flip_compare-{comp}-{metric}.{suffix}"
                fig.savefig(path, bbox_inches="tight", dpi=220 if suffix == "png" else None)
                print(f"Saved: {path}")
            plt.close(fig)


def plot_parallel_error(df):
    metric = "para_over_base_update"
    ylabel = r"$\|\Delta_{\parallel \Delta_{\mathrm{base}}}\| / \|\Delta_{\mathrm{base}}\|$"
    df = exclude_edge_layers(df[df["setting"].isin(KEEP_SETTINGS)].copy())

    for comp in COMPONENTS:
        d = df[df["component"] == comp].copy()
        if d.empty or d[metric].notna().sum() == 0:
            continue

        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=200, constrained_layout=True)
            valid_settings = [s for s in PLOT_ORDER if d[d["setting"] == s][metric].notna().any()]

            for setting in valid_settings:
                g = d[d["setting"] == setting].sort_values("layer")
                style = STYLE_MAP.get(setting, {"color": "#4C78A8", "marker": "o"})
                line = ax.plot(
                    g["layer"],
                    g[metric],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=3.0,
                    markersize=7.2,
                    markeredgewidth=1.0,
                    label=DISPLAY_NAME.get(setting, setting),
                    zorder=3,
                )[0]
                std_col = f"{metric}_std"
                if std_col in g.columns and g[std_col].notna().any():
                    y = g[metric].astype(float)
                    ystd = g[std_col].fillna(0.0).astype(float)
                    ax.fill_between(
                        g["layer"],
                        y - ystd,
                        y + ystd,
                        color=line.get_color(),
                        alpha=0.08,
                        linewidth=0,
                        zorder=1,
                    )

            layers = sorted(d["layer"].dropna().unique().tolist())
            if layers:
                ax.set_xlim(min(layers), max(layers))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=min(10, len(layers)), integer=True))
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#d0d0d0", linewidth=0.9, alpha=0.50)
            ax.tick_params(axis="both", which="major", length=4.8, width=0.9)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ylim = PARA_Y_LIM.get(("parallel_error", comp), nice_ylim(d[metric]))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=min(4, len(valid_settings)),
                frameon=True,
                facecolor="white",
                edgecolor="#cfcfcf",
                borderpad=0.42,
                handlelength=2.5,
                columnspacing=1.45,
            )

            for suffix in ("pdf", "png"):
                path = OUT_DIR / f"local_flip_compare-{comp}-{metric}.{suffix}"
                fig.savefig(path, bbox_inches="tight", dpi=220 if suffix == "png" else None)
                print(f"Saved: {path}")
            plt.close(fig)


def plot_baseline_update_ratio(df):
    metric = "base_update_over_hidden_state"
    ylabel = r"$\|\Delta_{\mathrm{base}}\| / \|x\|$"
    d = df[df[metric].notna()].copy()
    d = exclude_edge_layers(d)
    if d.empty:
        return

    # The ratio is a dense-baseline property. Multiple settings duplicate it,
    # so collapse to one curve per component/layer.
    d = (
        d.groupby(["component", "layer"], as_index=False)
        .agg(
            base_update_over_hidden_state=(metric, "mean"),
            base_update_over_hidden_state_std=(f"{metric}_std", "mean"),
        )
    )

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=200, constrained_layout=True, sharey=False)
        for ax, comp in zip(axes, COMPONENTS):
            g = d[d["component"] == comp].sort_values("layer")
            if g.empty:
                ax.set_visible(False)
                continue
            y = g[metric].astype(float)
            std_col = f"{metric}_std"
            ystd = g[std_col].fillna(0.0).astype(float) if std_col in g.columns else pd.Series(0.0, index=g.index)
            line = ax.plot(
                g["layer"],
                y,
                color="#4C78A8",
                marker="o",
                linewidth=3.0,
                markersize=7.0,
                markeredgewidth=1.0,
                zorder=3,
            )[0]
            if std_col in g.columns and g[std_col].notna().any():
                ax.fill_between(
                    g["layer"],
                    y - ystd,
                    y + ystd,
                    color=line.get_color(),
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )
            ax.text(
                0.97,
                0.94,
                COMPONENT_TITLE.get(comp, comp),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=20,
            )
            ax.set_xlabel("Layer")
            ax.grid(True, color="#d0d0d0", linewidth=0.9, alpha=0.50)
            ax.tick_params(axis="both", which="major", length=4.8, width=0.9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            auto_ylim = centered_ylim(y - ystd, y + ystd)
            ylim = Y_LIM.get(("baseline_update_ratio", comp), auto_ylim)
            if ylim is not None:
                ax.set_ylim(*ylim)

        axes[0].set_ylabel(ylabel)

        for suffix in ("pdf", "png"):
            path = OUT_DIR / f"baseline_update_over_hidden_state.{suffix}"
            fig.savefig(path, bbox_inches="tight", dpi=220 if suffix == "png" else None)
            print(f"Saved: {path}")
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_local_rows()
    print(f"Loaded: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print("Settings:", sorted(df["setting"].dropna().unique().tolist()))
    missing_hidden = df.groupby("setting")["hidden_state_norm"].apply(lambda s: int(s.isna().sum()))
    print("hidden_state_norm NaNs by setting:")
    print(missing_hidden.to_string())
    plot_orthogonal_error(df)
    if PLOT_PARALLEL_ERROR:
        plot_parallel_error(df)
    plot_baseline_update_ratio(df)


if __name__ == "__main__":
    main()
