#!/usr/bin/env python3
"""Build layer-drop selection JSONs from existing residual metric CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "compression" / "outputs" / "layer_drop_geometry"


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.replace("+", ",").split(",") if x.strip()]


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def _theta(row: dict[str, str]) -> float:
    value = _float(row, "output_angle")
    if math.isfinite(value):
        return value
    cosine = 1.0 - _float(row, "one_minus_cosine")
    if not math.isfinite(cosine):
        return math.nan
    return math.acos(max(-1.0, min(1.0, cosine)))


def _phi(row: dict[str, str]) -> float:
    value = _float(row, "update_angle")
    if math.isfinite(value):
        return value
    perp_ratio = _float(row, "perp_ratio")
    if not math.isfinite(perp_ratio):
        return math.nan
    return math.asin(math.sqrt(max(0.0, min(1.0, perp_ratio))))


def _score(row: dict[str, str], score_name: str) -> float:
    if score_name == "cos_plus_perp_ratio_energy":
        return _float(row, "one_minus_cosine") + _float(row, "perp_ratio_energy")
    if score_name in {
        "angle_l2",
        "update_scaled_angle_l2",
        "update_scaled_angle_sum",
        "theta_perp_l2",
        "theta_area_l2",
    }:
        theta = _theta(row)
        if score_name == "theta_perp_l2":
            perp = _float(row, "perp_over_ref")
            if not math.isfinite(theta) or not math.isfinite(perp):
                return math.nan
            return math.sqrt(theta**2 + perp**2)
        if score_name == "theta_area_l2":
            area = _score(row, "final_area")
            if not math.isfinite(theta) or not math.isfinite(area):
                return math.nan
            return math.sqrt(theta**2 + area**2)
        phi = _phi(row)
        if not math.isfinite(theta) or not math.isfinite(phi):
            return math.nan
        if score_name == "angle_l2":
            return math.sqrt(theta**2 + phi**2)
        update_over_ref = _float(row, "update_norm_over_ref")
        if not math.isfinite(update_over_ref):
            return math.nan
        if score_name == "update_scaled_angle_l2":
            return update_over_ref * math.sqrt(theta**2 + phi**2)
        return update_over_ref * (theta + phi)
    if score_name == "final_area":
        value = _float(row, "final_area")
        if math.isfinite(value):
            return value
        alpha = _float(row, "signed_alpha")
        perp = _float(row, "perp_over_ref")
        if not math.isfinite(alpha) or not math.isfinite(perp):
            return math.nan
        # Normalized area of the triangle spanned by x and x + Delta:
        # 2A / ||x||^2 = ||x + Delta_parallel|| / ||x|| * ||Delta_perp|| / ||x||.
        return abs(1.0 + alpha) * perp
    if score_name == "final_area_dense":
        value = _float(row, "final_area_dense")
        if math.isfinite(value):
            return value
        alpha_error = _float(row, "signed_alpha")
        perp = _float(row, "perp_over_ref")
        if not math.isfinite(alpha_error) or not math.isfinite(perp):
            return math.nan
        return abs(1.0 - alpha_error) * perp
    if score_name == "perp_over_parallel":
        value = _float(row, "perp_over_parallel")
        if math.isfinite(value):
            return value
        alpha_error = _float(row, "signed_alpha")
        perp = _float(row, "perp_over_ref")
        if not math.isfinite(alpha_error) or not math.isfinite(perp):
            return math.nan
        return perp / max(abs(1.0 - alpha_error), 1e-12)
    if score_name in {
        "error_over_ref",
        "para_over_ref",
        "perp_over_ref",
        "perp_ratio",
        "perp_ratio_energy",
        "perp_energy",
        "perp_energy_over_ref",
        "one_minus_cosine",
        "update_norm_over_ref",
        "output_angle",
        "update_angle",
        "angle_sum",
        "final_area",
        "final_area_dense",
        "perp_over_parallel",
        "angle_l2_mean",
        "update_scaled_angle_l2_mean",
        "update_scaled_angle_sum_mean",
        "theta_perp_l2_mean",
        "theta_area_l2_mean",
        "theta_beta_l1_mean",
        "theta_area_l1_mean",
        "update_scaled_theta_perp_l2_mean",
        "update_scaled_theta_area_l2_mean",
        "update_scaled_theta_beta_l1_mean",
        "update_scaled_theta_area_l1_mean",
    }:
        return _float(row, score_name)
    if score_name == "perp_dualsum":
        return _float(row, "perp_over_ref") + math.sqrt(max(_float(row, "perp_ratio"), 0.0))
    if score_name.startswith("perp_e"):
        lam = float(score_name.removeprefix("perp_e").replace("p", "."))
        return _float(row, "perp_over_ref") + lam * _float(row, "error_over_ref")
    if score_name.startswith("perp_alpha"):
        lam = float(score_name.removeprefix("perp_alpha").replace("p", "."))
        return _float(row, "perp_over_ref") + lam * abs(_float(row, "signed_alpha"))
    raise ValueError(f"Unsupported score_name={score_name}")


def _percentile_ranks(items: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    denom = max(len(ordered) - 1, 1)
    return {layer: idx / denom for idx, (layer, _value) in enumerate(ordered)}


def _decode_float(text: str) -> float:
    return float(text.replace("p", "."))


def _rank_combo_scores(rows: list[dict[str, str]], score_name: str) -> dict[int, float]:
    if (
        score_name not in {"perp_rankmax", "perp_rankmean", "perp_compw", "para_perp_rankmean"}
        and not score_name.startswith("perp_rankw")
        and not (score_name.startswith("perp_dual") and score_name != "perp_dualsum")
        and not score_name.startswith("perp_scaleguard")
        and not score_name.startswith("perp_scaletail")
        and not score_name.startswith("perp_tworef")
        and not score_name.startswith("para_rankw")
    ):
        return {}
    layers = []
    for row in rows:
        layer = int(row["layer"])
        ratio = _float(row, "perp_ratio")
        abs_perp = _float(row, "perp_over_ref")
        abs_para = _float(row, "para_over_ref")
        cosine = _float(row, "one_minus_cosine")
        if math.isfinite(ratio) and math.isfinite(abs_perp) and math.isfinite(abs_para):
            layers.append((layer, ratio, abs_perp, abs_para, cosine))
    ratio_rank = _percentile_ranks([(layer, ratio) for layer, ratio, _abs_perp, _abs_para, _cosine in layers])
    perp_rank = _percentile_ranks([(layer, abs_perp) for layer, _ratio, abs_perp, _abs_para, _cosine in layers])
    para_rank = _percentile_ranks([(layer, abs_para) for layer, _ratio, _abs_perp, abs_para, _cosine in layers])
    cosine_items = [(layer, cosine) for layer, _ratio, _abs_perp, _abs_para, cosine in layers if math.isfinite(cosine)]
    cosine_rank = _percentile_ranks(cosine_items) if cosine_items else {}
    if score_name == "perp_rankmax":
        return {layer: max(ratio_rank[layer], perp_rank[layer]) for layer, _ratio, _abs_perp, _abs_para, _cosine in layers}
    if score_name == "perp_rankmean":
        return {
            layer: 0.5 * (ratio_rank[layer] + perp_rank[layer])
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name == "perp_compw":
        component = rows[0].get("component", "") if rows else ""
        ratio_weight = 0.10 if component == "attn" else 0.50
        cosine_weight = 0.05
        return {
            layer: perp_rank[layer]
            + ratio_weight * ratio_rank[layer]
            + cosine_weight * cosine_rank.get(layer, perp_rank[layer])
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name.startswith("perp_dual"):
        ratio_weight = _decode_float(score_name.removeprefix("perp_dual"))
        # Existing perp_ratio stores ||e_perp||^2 / ||e||^2. Use the square
        # root so both terms are first-order magnitudes before ranking.
        mag_ratio_rank = _percentile_ranks(
            [(layer, math.sqrt(max(ratio, 0.0))) for layer, ratio, _abs_perp, _abs_para, _cosine in layers]
        )
        return {
            layer: perp_rank[layer] + ratio_weight * mag_ratio_rank[layer]
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name == "para_perp_rankmean":
        return {
            layer: 0.5 * (para_rank[layer] + perp_rank[layer])
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name.startswith("perp_scaleguard"):
        alpha_weight = _decode_float(score_name.removeprefix("perp_scaleguard"))
        component = rows[0].get("component", "") if rows else ""
        ratio_weight = 0.10 if component == "attn" else 0.50
        alpha_rank = _percentile_ranks(
            [(layer, abs(_float(row, "signed_alpha"))) for row in rows if math.isfinite(_float(row, "signed_alpha"))]
        )
        return {
            layer: perp_rank[layer]
            + ratio_weight * ratio_rank[layer]
            + alpha_weight * alpha_rank.get(layer, para_rank[layer])
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name.startswith("perp_scaletail"):
        alpha_weight = _decode_float(score_name.removeprefix("perp_scaletail"))
        component = rows[0].get("component", "") if rows else ""
        ratio_weight = 0.10 if component == "attn" else 0.50
        alpha_tail_start = 0.80
        alpha_rank = _percentile_ranks(
            [(layer, abs(_float(row, "signed_alpha"))) for row in rows if math.isfinite(_float(row, "signed_alpha"))]
        )
        return {
            layer: perp_rank[layer]
            + ratio_weight * ratio_rank[layer]
            + alpha_weight * max(0.0, alpha_rank.get(layer, para_rank[layer]) - alpha_tail_start)
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name.startswith("perp_tworef"):
        norm_weight = _decode_float(score_name.removeprefix("perp_tworef"))
        component = rows[0].get("component", "") if rows else ""
        ratio_weight = 0.10 if component == "attn" else 0.50
        norm_tail_start = 0.80

        def state_norm_drift(row: dict[str, str]) -> float:
            alpha = _float(row, "signed_alpha")
            perp = _float(row, "perp_over_ref")
            if not math.isfinite(alpha) or not math.isfinite(perp):
                return math.nan
            # Dense state after applying the dropped update is h - e, where
            # e = alpha h + e_perp and ||e_perp|| / ||h|| = perp.
            norm_ratio = math.sqrt(max((1.0 - alpha) ** 2 + perp**2, 1e-24))
            return abs(math.log(norm_ratio))

        norm_rank = _percentile_ranks(
            [(int(row["layer"]), state_norm_drift(row)) for row in rows if math.isfinite(state_norm_drift(row))]
        )
        return {
            layer: perp_rank[layer]
            + ratio_weight * ratio_rank[layer]
            + norm_weight * max(0.0, norm_rank.get(layer, para_rank[layer]) - norm_tail_start)
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    if score_name.startswith("para_rankw"):
        para_weight = _decode_float(score_name.removeprefix("para_rankw"))
        return {
            layer: perp_rank[layer] + para_weight * para_rank[layer]
            for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
        }
    ratio_weight = _decode_float(score_name.removeprefix("perp_rankw"))
    return {
        layer: perp_rank[layer] + ratio_weight * ratio_rank[layer]
        for layer, _ratio, _abs_perp, _abs_para, _cosine in layers
    }


def _parse_theta_guard_score(score_name: str) -> tuple[str, float, str]:
    suffix = score_name.removeprefix("angle_thetaguard")
    if "_" not in suffix:
        raise ValueError(f"Unsupported theta-guard score_name={score_name}")
    guard_text, rank_name = suffix.rsplit("_", 1)
    if rank_name not in {"phi", "perp", "sum"}:
        raise ValueError(f"Unsupported theta-guard rank={rank_name} score_name={score_name}")
    if guard_text.endswith("x"):
        return "mult", _decode_float(guard_text.removesuffix("x") or "1"), rank_name
    return "extra", int(guard_text or "0"), rank_name


def _select_theta_guard(
    rows: list[dict[str, str]],
    component: str,
    drop_counts: list[int],
    score_name: str,
    exclude_first: int,
    min_gap: int,
) -> dict[str, list[int]]:
    guard_mode, guard_value, rank_name = _parse_theta_guard_score(score_name)
    subset = [row for row in rows if row.get("component") == component]
    theta_ranked = []
    for row in subset:
        layer = int(row["layer"])
        if layer < exclude_first:
            continue
        theta = _theta(row)
        phi = _phi(row)
        perp = _float(row, "perp_over_ref")
        if math.isfinite(theta) and math.isfinite(phi) and math.isfinite(perp):
            theta_ranked.append((theta, phi, perp, layer))
    theta_ranked = sorted(theta_ranked, key=lambda item: (item[0], item[2]))
    if not theta_ranked:
        raise ValueError(
            f"No finite theta/phi scores for component={component} score_name={score_name} "
            f"source_columns={sorted(rows[0].keys()) if rows else []}"
        )
    selections = {}
    for count in drop_counts:
        guard_size = math.ceil(count * guard_value) if guard_mode == "mult" else count + int(guard_value)
        guard_size = min(len(theta_ranked), guard_size)
        if rank_name == "phi":
            pool = sorted(theta_ranked[:guard_size], key=lambda item: (item[1], item[0], item[3]))
        elif rank_name == "perp":
            pool = sorted(theta_ranked[:guard_size], key=lambda item: (item[2], item[0], item[3]))
        else:
            pool = sorted(theta_ranked[:guard_size], key=lambda item: (item[0] + item[1], item[0], item[1], item[3]))
        selected: list[int] = []
        for theta_value, phi_value, perp_value, layer in pool:
            if min_gap > 0 and any(abs(layer - prev) <= min_gap for prev in selected):
                continue
            selected.append(layer)
            if len(selected) == count:
                break
        if len(selected) < count:
            for theta_value, phi_value, perp_value, layer in pool:
                if layer in selected:
                    continue
                selected.append(layer)
                if len(selected) == count:
                    break
        selections[f"drop_{count}"] = selected
    return selections


def _select(
    rows: list[dict[str, str]],
    component: str,
    drop_counts: list[int],
    score_name: str,
    exclude_first: int,
    min_gap: int,
) -> dict[str, list[int]]:
    if score_name.startswith("angle_thetaguard"):
        return _select_theta_guard(rows, component, drop_counts, score_name, exclude_first, min_gap)
    subset = [row for row in rows if row.get("component") == component]
    combo_scores = _rank_combo_scores(subset, score_name)
    candidates = []
    for row in subset:
        layer = int(row["layer"])
        if layer < exclude_first:
            continue
        score = combo_scores[layer] if combo_scores else _score(row, score_name)
        if not math.isfinite(score):
            continue
        candidates.append((score, layer))
    ranked = sorted(candidates, key=lambda item: (item[0], item[1]))
    if not ranked:
        raise ValueError(
            f"No finite layer scores for component={component} score_name={score_name} "
            f"source_columns={sorted(rows[0].keys()) if rows else []}"
        )
    selections = {}
    for count in drop_counts:
        selected: list[int] = []
        for _score_value, layer in ranked:
            if min_gap > 0 and any(abs(layer - prev) <= min_gap for prev in selected):
                continue
            selected.append(layer)
            if len(selected) == count:
                break
        if len(selected) < count:
            for _score_value, layer in ranked:
                if layer in selected:
                    continue
                selected.append(layer)
                if len(selected) == count:
                    break
        selections[f"drop_{count}"] = selected
    return selections


def build_one(args: argparse.Namespace, model_dir: Path, score_name: str, exclude_first: int, min_gap: int) -> Path:
    source_csv = model_dir / args.source_tag / "layer_drop_geometry_metrics.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    source_json = model_dir / args.source_tag / "drop_selection.json"
    source_meta: dict[str, Any] = {}
    if source_json.is_file():
        source_meta = json.loads(source_json.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(source_csv.open(encoding="utf-8")))
    drop_counts = _parse_ints(args.drop_counts)
    tag = f"hybrid_{score_name}_alltok"
    if exclude_first:
        tag = f"{tag}_skip{exclude_first}"
    if min_gap:
        tag = f"{tag}_gap{min_gap}"
    out_dir = model_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    recommendations = {
        component: _select(rows, component, drop_counts, score_name, exclude_first, min_gap)
        for component in ("attn", "mlp")
    }
    summary = {
        "model_name": source_meta.get("model_name"),
        "model_tag": source_meta.get("model_tag", model_dir.name),
        "output_tag": tag,
        "source_tag": args.source_tag,
        "source_csv": str(source_csv),
        "rank_metric": tag,
        "rank_order": "ascending",
        "score_name": score_name,
        "score_formula": _score_formula(score_name),
        "exclude_first_layers": exclude_first,
        "min_gap": min_gap,
        "token_scope": source_meta.get("token_scope", "all"),
        "drop_counts": drop_counts,
        "recommendations": recommendations,
    }
    for key in (
        "prompts_file",
        "num_prompts",
        "max_prompts",
        "max_length",
        "calibration_source",
        "calibration_nsamples",
        "calibration_seqlen",
        "calibration_seed",
    ):
        if key in source_meta:
            summary[key] = source_meta[key]
    out_path = out_dir / "drop_selection.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_path


def _score_formula(score_name: str) -> str:
    if score_name == "perp_rankmax":
        return "max(percentile_rank(perp_ratio), percentile_rank(perp_over_ref))"
    if score_name == "perp_rankmean":
        return "0.5 * percentile_rank(perp_ratio) + 0.5 * percentile_rank(perp_over_ref)"
    if score_name == "perp_compw":
        return "percentile_rank(perp_over_ref) + component_weight * percentile_rank(perp_ratio) + 0.05 * percentile_rank(one_minus_cosine), with component_weight=0.10 for attention and 0.50 for MLP"
    if score_name.startswith("perp_rankw"):
        weight = _decode_float(score_name.removeprefix("perp_rankw"))
        return f"percentile_rank(perp_over_ref) + {weight:g} * percentile_rank(perp_ratio)"
    if score_name == "perp_dualsum":
        return "perp_over_ref + sqrt(perp_ratio)"
    if score_name == "perp_ratio_energy":
        return "sum(||e_perp||^2) / sum(||e||^2)"
    if score_name == "perp_energy":
        return "sum(||e_perp||^2) / N"
    if score_name == "perp_energy_over_ref":
        return "sum(||e_perp||^2) / sum(||h||^2)"
    if score_name == "cos_plus_perp_ratio_energy":
        return "(1 - cosine(h + update, h)) + sum(||e_perp||^2) / sum(||e||^2)"
    if score_name == "output_angle":
        return "mean arccos(cosine(x, x + Delta))"
    if score_name == "update_angle":
        return "mean asin(sqrt(||Delta_perp||^2 / ||Delta||^2))"
    if score_name == "angle_sum":
        return "mean arccos(cosine(x, x + Delta)) + mean asin(sqrt(||Delta_perp||^2 / ||Delta||^2))"
    if score_name == "angle_l2":
        return "sqrt(theta^2 + phi^2), where theta=angle(x, x+Delta) and phi=angle(Delta, Delta_parallel)"
    if score_name == "update_scaled_angle_l2":
        return "(||Delta||/||x||) * sqrt(theta^2 + phi^2)"
    if score_name == "update_scaled_angle_sum":
        return "(||Delta||/||x||) * (theta + phi)"
    if score_name == "final_area":
        return "mean(abs(1 + alpha_i) * beta_i), where beta_i=||Delta_perp_i||/||x_i||; falls back to abs(1 + mean(alpha)) * mean(beta) for legacy CSVs"
    if score_name == "final_area_dense":
        return "mean(abs(1 + alpha_update_i) * beta_i), using alpha_update for Delta rather than drop error"
    if score_name == "perp_over_parallel":
        return "mean(beta_i / abs(1 + alpha_update_i)), the vertical component relative to the dense parallel component"
    if score_name == "angle_l2_mean":
        return "mean(sqrt(theta_i^2 + phi_i^2))"
    if score_name == "update_scaled_angle_l2_mean":
        return "mean((||Delta_i||/||x_i||) * sqrt(theta_i^2 + phi_i^2))"
    if score_name == "update_scaled_angle_sum_mean":
        return "mean((||Delta_i||/||x_i||) * (theta_i + phi_i))"
    if score_name == "theta_perp_l2_mean":
        return "mean(sqrt(theta_i^2 + (||Delta_perp_i||/||x_i||)^2))"
    if score_name == "theta_area_l2_mean":
        return "mean(sqrt(theta_i^2 + final_area_i^2)), where final_area_i=abs(1 + alpha_i) * ||Delta_perp_i||/||x_i||"
    if score_name == "theta_beta_l1_mean":
        return "mean(theta_i + beta_i), where beta_i=||Delta_perp_i||/||x_i||"
    if score_name == "theta_area_l1_mean":
        return "mean(theta_i + final_area_i), where final_area_i=abs(1 + alpha_i) * ||Delta_perp_i||/||x_i||"
    if score_name == "update_scaled_theta_perp_l2_mean":
        return "mean((||Delta_i||/||x_i||) * sqrt(theta_i^2 + beta_i^2))"
    if score_name == "update_scaled_theta_area_l2_mean":
        return "mean((||Delta_i||/||x_i||) * sqrt(theta_i^2 + final_area_i^2))"
    if score_name == "update_scaled_theta_beta_l1_mean":
        return "mean((||Delta_i||/||x_i||) * (theta_i + beta_i))"
    if score_name == "update_scaled_theta_area_l1_mean":
        return "mean((||Delta_i||/||x_i||) * (theta_i + final_area_i))"
    if score_name == "theta_perp_l2":
        return "sqrt(theta^2 + perp_over_ref^2), where theta=angle(x, x+Delta)"
    if score_name == "theta_area_l2":
        return "sqrt(theta^2 + final_area^2), where final_area=abs(1 + signed_alpha) * perp_over_ref"
    if score_name.startswith("angle_thetaguard"):
        guard_mode, guard_value, rank_name = _parse_theta_guard_score(score_name)
        guard_desc = f"{guard_value:g}k" if guard_mode == "mult" else f"k+{int(guard_value)}"
        rank_desc = (
            "phi=asin(sqrt(||Delta_perp||^2 / ||Delta||^2))"
            if rank_name == "phi"
            else "perp_over_ref=||Delta_perp||/||x||"
            if rank_name == "perp"
            else "theta+phi"
        )
        return (
            f"for drop count k, keep the {guard_desc} layers with smallest "
            "theta=angle(x, x+Delta), then select k layers with smallest "
            f"{rank_desc}"
        )
    if score_name.startswith("perp_dual"):
        weight = _decode_float(score_name.removeprefix("perp_dual"))
        return (
            "percentile_rank(perp_over_ref) "
            f"+ {weight:g} * percentile_rank(sqrt(perp_ratio)); "
            "perp_over_ref measures ||e_perp||/||h|| and sqrt(perp_ratio) measures ||e_perp||/||e||"
        )
    if score_name.startswith("perp_scaleguard"):
        weight = _decode_float(score_name.removeprefix("perp_scaleguard"))
        return (
            "percentile_rank(perp_over_ref) + component_weight * percentile_rank(perp_ratio) "
            f"+ {weight:g} * percentile_rank(abs(signed_alpha)), "
            "with component_weight=0.10 for attention and 0.50 for MLP"
        )
    if score_name.startswith("perp_scaletail"):
        weight = _decode_float(score_name.removeprefix("perp_scaletail"))
        return (
            "percentile_rank(perp_over_ref) + component_weight * percentile_rank(perp_ratio) "
            f"+ {weight:g} * max(0, percentile_rank(abs(signed_alpha)) - 0.80), "
            "with component_weight=0.10 for attention and 0.50 for MLP"
        )
    if score_name.startswith("perp_tworef"):
        weight = _decode_float(score_name.removeprefix("perp_tworef"))
        return (
            "percentile_rank(perp_over_ref) + component_weight * percentile_rank(perp_ratio) "
            f"+ {weight:g} * max(0, percentile_rank(abs(log(||h - e|| / ||h||))) - 0.80), "
            "where e is the layer-drop error and component_weight=0.10 for attention and 0.50 for MLP"
        )
    if score_name == "para_perp_rankmean":
        return "0.5 * percentile_rank(para_over_ref) + 0.5 * percentile_rank(perp_over_ref)"
    if score_name.startswith("para_rankw"):
        weight = _decode_float(score_name.removeprefix("para_rankw"))
        return f"percentile_rank(perp_over_ref) + {weight:g} * percentile_rank(para_over_ref)"
    if score_name == "para_over_ref":
        return "para_over_ref"
    if score_name.startswith("perp_e"):
        return "perp_over_ref + lambda * error_over_ref"
    if score_name.startswith("perp_alpha"):
        return "perp_over_ref + lambda * abs(signed_alpha)"
    return score_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--models", default="qwen3_0p6b,qwen3_1p7b,qwen3_4b")
    parser.add_argument("--source_tag", default="perp_over_ref_alltok")
    parser.add_argument("--score_names", default="perp_e0p05,perp_e0p10,perp_e0p20")
    parser.add_argument("--drop_counts", default="4,8")
    parser.add_argument("--exclude_first", type=int, default=0)
    parser.add_argument("--min_gap", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    outputs = []
    for model in [x.strip() for x in args.models.split(",") if x.strip()]:
        model_dir = root / model
        for score_name in [x.strip() for x in args.score_names.split(",") if x.strip()]:
            outputs.append(build_one(args, model_dir, score_name, args.exclude_first, args.min_gap))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
