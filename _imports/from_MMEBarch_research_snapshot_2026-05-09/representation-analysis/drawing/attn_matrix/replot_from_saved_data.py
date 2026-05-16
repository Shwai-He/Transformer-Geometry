from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np


SHARED_VALUE_CMAP = 'viridis'


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    return json.loads(summary_path.read_text(encoding='utf-8'))


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _load_plot_bundle_json(bundle_path: Path) -> Dict[str, np.ndarray]:
    raw = json.loads(bundle_path.read_text(encoding='utf-8'))
    out: Dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            out[key] = np.asarray(value, dtype=object if key == 'token_labels' else float)
        else:
            out[key] = np.asarray(value)
    return out


def _load_tsv_manifest(manifest_path: Path) -> Dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    out: Dict[str, np.ndarray] = {}
    for key, meta in manifest.get("fields", {}).items():
        tsv_path = manifest_path.parent / meta["path"]
        rows = [line.rstrip("\n") for line in tsv_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            continue
        data_rows = rows[1:]
        kind = meta.get("kind")
        shape = tuple(meta.get("shape", []))
        if kind == "scalar":
            out[key] = np.asarray(float(data_rows[0]))
        elif kind == "vector_text":
            vals = []
            for row in data_rows:
                _, label = row.split("	", 1)
                vals.append(label.replace("\\n", "\n"))
            out[key] = np.asarray(vals, dtype=object)
        elif kind == "vector":
            vals = [float(row.split("	")[1]) for row in data_rows]
            out[key] = np.asarray(vals, dtype=float)
        elif kind == "matrix":
            arr = np.zeros(shape, dtype=float)
            for row in data_rows:
                i, j, val = row.split("	")
                arr[int(i), int(j)] = float(val)
            out[key] = arr
        elif kind == "tensor3":
            arr = np.zeros(shape, dtype=float)
            for row in data_rows:
                i, j, k, val = row.split("	")
                arr[int(i), int(j), int(k)] = float(val)
            out[key] = arr
        else:
            raise ValueError(f"Unsupported TSV field kind: {kind} for {key}")
    return out


def _load_bundle(path_str: str):
    path = Path(path_str)
    if path.suffix == '.json':
        if path.name == 'plot_bundle.json' or path.name.endswith('_plot_bundle.json'):
            summary_path = path.parent / 'summary.json'
            summary = _load_summary(summary_path) if summary_path.exists() else {}
            return summary_path if summary_path.exists() else None, summary, path, _load_plot_bundle_json(path)
        summary = _load_summary(path)
        bundle_path = summary.get('plot_bundle_path')
        manifest_path = summary.get('plot_data_manifest_path')
        plot_data_path = summary.get('plot_data_path')
        if bundle_path:
            json_path = Path(bundle_path)
            if not json_path.is_absolute():
                json_path = (path.parent / json_path).resolve()
            return path, summary, json_path, _load_plot_bundle_json(json_path)
        if manifest_path:
            tsv_path = Path(manifest_path)
            if not tsv_path.is_absolute():
                tsv_path = (path.parent / tsv_path).resolve()
            return path, summary, tsv_path, _load_tsv_manifest(tsv_path)
        if plot_data_path:
            npz_path = Path(plot_data_path)
            if not npz_path.is_absolute():
                npz_path = (path.parent / npz_path).resolve()
            return path, summary, npz_path, _load_npz(npz_path)
        raise ValueError(f'summary has no plot_bundle_path, plot_data_manifest_path, or plot_data_path: {path}')
    if path.suffix == '.npz':
        npz = _load_npz(path)
        summary_path = path.with_name('summary.json')
        summary = _load_summary(summary_path) if summary_path.exists() else {}
        return summary_path if summary_path.exists() else None, summary, path, npz
    if path.name == 'plot_data_manifest.json':
        summary_path = path.parent.parent / 'summary.json'
        summary = _load_summary(summary_path) if summary_path.exists() else {}
        return summary_path if summary_path.exists() else None, summary, path, _load_tsv_manifest(path)
    raise ValueError(f'Unsupported input path: {path}')


def _to_labels(npz: Dict[str, np.ndarray], summary: Dict[str, Any]) -> List[str]:
    if 'token_labels' in npz:
        raw = npz['token_labels']
        return [str(x) for x in raw.tolist()]
    if 'tokens' in summary:
        return [str(x) for x in summary['tokens']]
    raise ValueError('Could not find token labels in npz or summary.')


def _overlay_diagonal_boxes(ax, n: int) -> None:
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor='black',
                linewidth=1.1,
                alpha=0.85,
            )
        )
        ax.add_patch(
            Rectangle(
                (i - 0.47, i - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor='white',
                linewidth=0.8,
                alpha=0.9,
            )
        )


def _plot_matrix(ax, matrix: np.ndarray, title: str, labels: List[str], cmap, vmin=None, vmax=None, norm=None):
    arr = np.asarray(matrix)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        upper_mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
        arr = np.ma.array(arr, mask=upper_mask)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, 'copy') else cmap_obj
        cmap_obj.set_bad(color='#e6e6e6')
    else:
        cmap_obj = cmap
    im = ax.imshow(arr, cmap=cmap_obj, aspect='auto', vmin=vmin, vmax=vmax, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    _overlay_diagonal_boxes(ax, len(labels))
    ax.text(
        0.99,
        0.01,
        f'min={arr.min():.3g}\nmax={arr.max():.3g}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=7,
        color='black',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, linewidth=0.3),
    )
    return im


def _plot_signed_matrix(ax, matrix: np.ndarray, title: str, labels: List[str]):
    arr = np.asarray(matrix)
    matrix_min = float(arr.min())
    matrix_max = float(arr.max())
    if matrix_min < 0.0 < matrix_max:
        norm = colors.TwoSlopeNorm(vmin=matrix_min, vcenter=0.0, vmax=matrix_max)
        return _plot_matrix(ax, arr, title, labels, 'coolwarm', norm=norm)
    delta_abs = float(np.abs(arr).max())
    return _plot_matrix(ax, arr, title, labels, 'coolwarm', -delta_abs, delta_abs)


def _plot_shared_value_matrix_auto(ax, matrix: np.ndarray, title: str, labels: List[str], shared_vmax: float):
    arr = np.asarray(matrix)
    matrix_min = float(arr.min())
    matrix_max = float(arr.max())
    if matrix_min < 0.0:
        signed_max = max(abs(matrix_min), abs(matrix_max))
        return _plot_matrix(ax, arr, title, labels, 'coolwarm', -signed_max, signed_max)
    return _plot_matrix(ax, arr, title, labels, SHARED_VALUE_CMAP, 0.0, shared_vmax)


def _plot_curve_panel(ax, diag: np.ndarray, offdiag_sum: np.ndarray, total_sum: np.ndarray, title: str):
    xs = np.arange(len(diag))
    ax.plot(xs, diag, color='tab:orange', linewidth=1.5, label='diag')
    ax.plot(xs, offdiag_sum, color='tab:blue', linewidth=1.5, label='offdiag-sum')
    ax.plot(xs, total_sum, color='tab:green', linewidth=1.5, label='total-sum')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('token', fontsize=8)
    ax.set_ylabel('value', fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7, loc='best', frameon=True, facecolor='white', edgecolor='#cfcfcf')


def replot_xparallel(npz: Dict[str, np.ndarray], summary: Dict[str, Any], out_path: Path, dpi: int):
    labels = _to_labels(npz, summary)
    baseline = npz['panel1_baseline']
    compare = npz['panel2_effective_compare']
    delta = npz['panel3_effective_delta']
    diag = npz['panel4_diag']
    offdiag_sum = npz['panel4_offdiag_sum']
    total_sum = npz['panel4_total_sum']

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    shared_vmax = float(max(baseline.max(), compare.max()))
    ims = [
        _plot_shared_value_matrix_auto(axes[0], baseline, summary.get('panel1_title', f"L{summary.get('flip_layer', '?')} merged baseline"), labels, shared_vmax),
        _plot_shared_value_matrix_auto(axes[1], compare, summary.get('panel2_title', f"L{summary.get('flip_layer', '?')} merged effective x-removal"), labels, shared_vmax),
        _plot_signed_matrix(axes[2], delta, summary.get('panel3_title', 'x-reference effective delta'), labels),
    ]
    for ax, im in zip(axes[:3], ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _plot_curve_panel(axes[3], diag, offdiag_sum, total_sum, summary.get('panel4_title', f"L{summary.get('flip_layer', '?')} merged effective x-removal: diag/offdiag"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def replot_single_layer_flip(npz: Dict[str, np.ndarray], summary: Dict[str, Any], out_path: Path, dpi: int):
    labels = _to_labels(npz, summary)
    baseline = npz['panel1_flip_head_baseline']
    compare = npz['panel2_flip_head_effective_compare'] if 'panel2_flip_head_effective_compare' in npz else npz['panel2_flip_head_raw_xsa']
    delta = npz['panel3_flip_head_effective_delta'] if 'panel3_flip_head_effective_delta' in npz else (compare - baseline)
    if 'panel4_effective_diag' in npz and 'panel4_effective_offdiag_sum' in npz and 'panel4_effective_total_sum' in npz:
        diag = npz['panel4_effective_diag']
        offdiag_sum = npz['panel4_effective_offdiag_sum']
        total_sum = npz['panel4_effective_total_sum']
        compare_label = 'effective xsa'
    else:
        diag = npz['panel4_raw_diag']
        offdiag_sum = npz['panel4_raw_offdiag_sum']
        total_sum = npz['panel4_raw_total_sum']
        compare_label = 'raw xsa'

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    shared_vmax = float(max(baseline.max(), compare.max()))
    ims = [
        _plot_shared_value_matrix_auto(axes[0], baseline, summary.get('panel1_title', f"L{summary.get('flip_layer', '?')} H{summary.get('flip_head', '?')} baseline"), labels, shared_vmax),
        _plot_shared_value_matrix_auto(axes[1], compare, summary.get('panel2_title', f"L{summary.get('flip_layer', '?')} H{summary.get('flip_head', '?')} {compare_label}"), labels, shared_vmax),
        _plot_signed_matrix(axes[2], delta, summary.get('panel3_title', 'effective delta'), labels),
    ]
    for ax, im in zip(axes[:3], ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _plot_curve_panel(axes[3], diag, offdiag_sum, total_sum, summary.get('panel4_title', f"L{summary.get('flip_layer', '?')} H{summary.get('flip_head', '?')} {compare_label}: diag/offdiag"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def infer_mode(npz: Dict[str, np.ndarray], summary: Dict[str, Any]) -> str:
    if 'panel1_baseline' in npz and 'panel4_total_sum' in npz:
        return 'xparallel'
    if 'panel1_flip_head_baseline' in npz:
        return 'single_layer_flip'
    mode = summary.get('local_replot_mode')
    if mode:
        return str(mode)
    raise ValueError(f'Could not infer plot mode from keys: {sorted(npz.keys())}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Replot attention-matrix figures from saved cloud-side TSV or NPZ data.')
    parser.add_argument('--input', type=str, required=True, help='Path to summary.json, plot_bundle.json, plot_data_manifest.json, or plot_data.npz downloaded from server.')
    parser.add_argument('--output', type=str, default=None, help='Optional output PNG path. Defaults to <input_dir>/replot.png.')
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args()

    summary_path, summary, npz_path, npz = _load_bundle(args.input)
    mode = infer_mode(npz, summary)
    base_dir = npz_path.parent
    out_path = Path(args.output) if args.output else (base_dir / 'replot.png')

    if mode == 'xparallel':
        replot_xparallel(npz, summary, out_path, args.dpi)
    elif mode == 'single_layer_flip':
        replot_single_layer_flip(npz, summary, out_path, args.dpi)
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    print(f'[INFO] mode={mode} re-rendered figure to {out_path}')
    if summary_path is not None:
        print(f'[INFO] summary={summary_path}')
    print(f'[INFO] plot_data={npz_path}')


if __name__ == '__main__':
    main()
