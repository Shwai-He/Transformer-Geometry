#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_result_path(
    root: Path,
    run_type: str,
    space: str,
    model_name: str,
    random_init: bool = False,
    mode: str = "both",
    max_new_tokens: int = 32,
) -> Path:
    subdir = f"{run_type}_{space}" + ("_random_init" if random_init else "")
    if run_type == "probe":
        filename = f"{model_name}.json"
    else:
        filename = f"{model_name}-mode{mode}-tokens{max_new_tokens}.json"
    return root / subdir / filename


def load_runs(result_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    runs = data["runs"] if "runs" in data else [data]
    print(f"loaded runs: {len(runs)} from {result_path}")
    return runs


def pick_steps(run: Dict[str, Any], phase: str = "decode") -> List[Dict[str, Any]]:
    timeline = run.get("timeline") or []
    if timeline:
        picked = [s for s in timeline if s.get("phase") == phase]
        return picked if picked else timeline

    if phase == "prefill":
        prefill = run.get("prefill_steps") or []
        if prefill:
            return prefill
    return run.get("steps", [])


def collect_step_layer_values(
    runs: Iterable[Dict[str, Any]],
    part: str = "block",
    metric: str = "para_perp_ratio",
    phase: str = "decode",
) -> Dict[int, Dict[int, List[float]]]:
    out: Dict[int, Dict[int, List[float]]] = {}
    for run in runs:
        for step in pick_steps(run, phase=phase):
            step_idx = step.get("step")
            if step_idx is None:
                continue

            if part == "block":
                sub_items = step.get("sublayer_metrics") or []
                if sub_items:
                    for item in sub_items:
                        layer = item.get("layer")
                        value = (item.get("block_update") or {}).get(metric)
                        if layer is None or value is None:
                            continue
                        out.setdefault(int(layer), {}).setdefault(int(step_idx), []).append(float(value))
                else:
                    for item in step.get("layer_metrics") or []:
                        layer = item.get("layer")
                        value = item.get(metric)
                        if layer is None or value is None:
                            continue
                        out.setdefault(int(layer), {}).setdefault(int(step_idx), []).append(float(value))
            else:
                subkey = "attn_update" if part == "attn" else "mlp_update"
                for item in step.get("sublayer_metrics") or []:
                    layer = item.get("layer")
                    value = (item.get(subkey) or {}).get(metric)
                    if layer is None or value is None:
                        continue
                    out.setdefault(int(layer), {}).setdefault(int(step_idx), []).append(float(value))
    return out


def collect_step_layer_ratios(
    runs: Iterable[Dict[str, Any]],
    part: str,
    numerator: str,
    denominator: str,
    phase: str = "decode",
) -> Dict[int, Dict[int, List[float]]]:
    out: Dict[int, Dict[int, List[float]]] = {}
    for run in runs:
        for step in pick_steps(run, phase=phase):
            step_idx = step.get("step")
            if step_idx is None:
                continue

            if part == "block":
                sub_items = step.get("sublayer_metrics") or []
                if sub_items:
                    for item in sub_items:
                        layer = item.get("layer")
                        block_update = item.get("block_update") or {}
                        if numerator == "dz_para_norm" and denominator in {"z_post_norm", "z_mean"}:
                            value = _para_over_residual(block_update)
                        else:
                            value = _ratio_from_item(block_update, numerator, denominator)
                        if value is None and denominator in {"z_mean", "z_post_norm"}:
                            # block_update does not store its input residual norm in older outputs.
                            # attn_update z_post_norm is the residual state entering the block.
                            value = _ratio_from_items(
                                block_update,
                                item.get("attn_update") or {},
                                numerator,
                                "z_post_norm",
                            )
                        if layer is None or value is None:
                            continue
                        out.setdefault(int(layer), {}).setdefault(int(step_idx), []).append(value)
                else:
                    for item in step.get("layer_metrics") or []:
                        layer = item.get("layer")
                        if numerator == "dz_para_norm" and denominator in {"z_post_norm", "z_mean"}:
                            value = _para_over_residual(item)
                        else:
                            value = _ratio_from_item(item, numerator, denominator)
                        if layer is None or value is None:
                            continue
                        out.setdefault(int(layer), {}).setdefault(int(step_idx), []).append(value)
            else:
                subkey = "attn_update" if part == "attn" else "mlp_update"
                for item in step.get("sublayer_metrics") or []:
                    layer = item.get("layer")
                    sub_update = item.get(subkey) or {}
                    if numerator == "dz_para_norm" and denominator in {"z_post_norm", "z_mean"}:
                        value = _para_over_residual(sub_update)
                    else:
                        value = _ratio_from_item(sub_update, numerator, denominator)
                    if layer is None or value is None:
                        continue
                    out.setdefault(int(layer), {}).setdefault(int(step_idx), []).append(value)
    return out


def build_mean_min_max_series(step_map: Dict[int, List[float]]) -> Tuple[List[int], List[float], List[float], List[float]]:
    steps = sorted(step_map.keys())
    means = [float(np.mean(step_map[s])) for s in steps]
    mins = [float(np.min(step_map[s])) for s in steps]
    maxs = [float(np.max(step_map[s])) for s in steps]
    return steps, means, mins, maxs


def trim_layers(layers: List[int], trim_edge_layers: bool, start_layer: Optional[int] = None) -> List[int]:
    if not trim_edge_layers or len(layers) <= 2:
        return layers
    if start_layer is None:
        return layers[1:-1]
    return layers[start_layer:-1]


def collect_layerwise_from_layer_metrics(
    steps: Iterable[Dict[str, Any]],
    value_key: str,
    trim_edge_layers: bool = False,
    start_layer: Optional[int] = None,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    by_layer: Dict[int, List[float]] = {}
    for step in steps:
        for item in step.get("layer_metrics") or []:
            layer = item.get("layer")
            value = item.get(value_key)
            if layer is None or value is None:
                continue
            by_layer.setdefault(int(layer), []).append(float(value))

    layers = trim_layers(sorted(by_layer), trim_edge_layers, start_layer)
    means = [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]
    mins = [min(by_layer[layer]) for layer in layers]
    maxs = [max(by_layer[layer]) for layer in layers]
    return layers, means, mins, maxs


def collect_layerwise_from_sublayer(
    steps: Iterable[Dict[str, Any]],
    metric_key: str,
    value_key: str,
    trim_edge_layers: bool = False,
    start_layer: Optional[int] = None,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    by_layer: Dict[int, List[float]] = {}
    for step in steps:
        for item in step.get("sublayer_metrics") or []:
            layer = item.get("layer")
            value = (item.get(metric_key) or {}).get(value_key)
            if layer is None or value is None:
                continue
            by_layer.setdefault(int(layer), []).append(float(value))

    layers = trim_layers(sorted(by_layer), trim_edge_layers, start_layer)
    means = [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]
    mins = [min(by_layer[layer]) for layer in layers]
    maxs = [max(by_layer[layer]) for layer in layers]
    return layers, means, mins, maxs


def _ratio_from_item(item: Dict[str, Any], numerator: str, denominator: str, eps: float = 1e-12) -> Optional[float]:
    num = item.get(numerator)
    den = item.get(denominator)
    if num is None or den is None:
        return None
    return float(num) / max(float(den), eps)


def _ratio_from_items(
    numerator_item: Dict[str, Any],
    denominator_item: Dict[str, Any],
    numerator: str,
    denominator: str,
    eps: float = 1e-12,
) -> Optional[float]:
    num = numerator_item.get(numerator)
    den = denominator_item.get(denominator)
    if num is None or den is None:
        return None
    return float(num) / max(float(den), eps)


def _para_over_residual(item: Dict[str, Any], eps: float = 1e-12) -> Optional[float]:
    para = item.get("dz_para_norm")
    if para is None:
        return None

    residual = item.get("z_post_norm")
    if residual is None:
        residual = item.get("z_mean")
    if residual is not None:
        return float(para) / max(float(residual), eps)

    perp = item.get("dz_perp_norm")
    perp_over_residual_plus_para = item.get("dz_perp_over_z_plus_dz_para")
    if perp is None or perp_over_residual_plus_para is None:
        return None

    residual = float(perp) / max(float(perp_over_residual_plus_para), eps) - float(para)
    if residual <= eps:
        return None
    return float(para) / residual


def collect_layerwise_ratio_from_layer_metrics(
    steps: Iterable[Dict[str, Any]],
    numerator: str,
    denominator: str,
    trim_edge_layers: bool = False,
    start_layer: Optional[int] = None,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    by_layer: Dict[int, List[float]] = {}
    for step in steps:
        for item in step.get("layer_metrics") or []:
            layer = item.get("layer")
            if numerator == "dz_para_norm" and denominator in {"z_post_norm", "z_mean"}:
                value = _para_over_residual(item)
            else:
                value = _ratio_from_item(item, numerator, denominator)
            if layer is None or value is None:
                continue
            by_layer.setdefault(int(layer), []).append(value)

    layers = trim_layers(sorted(by_layer), trim_edge_layers, start_layer)
    means = [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]
    mins = [min(by_layer[layer]) for layer in layers]
    maxs = [max(by_layer[layer]) for layer in layers]
    return layers, means, mins, maxs


def collect_layerwise_ratio_from_sublayer(
    steps: Iterable[Dict[str, Any]],
    metric_key: str,
    numerator: str,
    denominator: str,
    trim_edge_layers: bool = False,
    start_layer: Optional[int] = None,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    by_layer: Dict[int, List[float]] = {}
    for step in steps:
        for item in step.get("sublayer_metrics") or []:
            layer = item.get("layer")
            metric_item = item.get(metric_key) or {}
            if numerator == "dz_para_norm" and denominator in {"z_post_norm", "z_mean"}:
                value = _para_over_residual(metric_item)
            else:
                value = _ratio_from_item(metric_item, numerator, denominator)
            if value is None and metric_key == "block_update" and denominator in {"z_mean", "z_post_norm"}:
                value = _ratio_from_items(
                    metric_item,
                    item.get("attn_update") or {},
                    numerator,
                    "z_post_norm",
                )
            if layer is None or value is None:
                continue
            by_layer.setdefault(int(layer), []).append(value)

    layers = trim_layers(sorted(by_layer), trim_edge_layers, start_layer)
    means = [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]
    mins = [min(by_layer[layer]) for layer in layers]
    maxs = [max(by_layer[layer]) for layer in layers]
    return layers, means, mins, maxs
