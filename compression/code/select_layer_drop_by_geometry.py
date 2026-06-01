#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from layerwise_para_perp_compare import derive_model_tag, read_prompts


def _dtype_from_name(name: str) -> torch.dtype | None:
    return {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = _mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / len(values)))


def _component_metrics(update: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    update = update.float()
    ref = ref.float().to(update.device)
    error = -update
    ref_sq = (ref * ref).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = (error * ref).sum(dim=-1, keepdim=True) / ref_sq
    para = coeff * ref
    perp = error - para

    ref_norm = ref.norm(dim=-1).clamp_min(eps)
    error_norm = error.norm(dim=-1)
    para_norm = para.norm(dim=-1)
    perp_norm = perp.norm(dim=-1)
    error_sq_raw = (error * error).sum(dim=-1)
    perp_sq_raw = (perp * perp).sum(dim=-1)
    error_sq = error_sq_raw.clamp_min(eps)
    update_norm = update.norm(dim=-1)
    dropped_state = ref
    dense_state = ref + update
    cosine_dense_drop = torch.nn.functional.cosine_similarity(dense_state.float(), dropped_state.float(), dim=-1)
    # Geometry-angle view for x -> x + Delta.  theta is exactly the cosine
    # angle between the dropped and dense residual states, so the signed
    # parallel component is already handled by the cosine.  phi is the angle
    # form of the perpendicular update ratio: asin(||Delta_perp|| / ||Delta||).
    output_angle = torch.acos(cosine_dense_drop.clamp(-1.0, 1.0))
    update_angle = torch.asin(torch.sqrt((perp_sq_raw / error_sq).clamp(0.0, 1.0)))
    angle_sum = output_angle + update_angle
    update_over_ref = update_norm / ref_norm
    beta = perp_norm / ref_norm
    # coeff is the projection coefficient of the drop error e = -Delta onto x.
    # If the paper-side update is Delta = alpha_update x + Delta_perp, then
    # alpha_update = -coeff and x + Delta has parallel scale 1 - coeff.
    error_alpha = coeff.squeeze(-1)
    update_alpha = -error_alpha
    dense_parallel_scale = torch.abs(1.0 - error_alpha).clamp_min(eps)
    # Legacy area kept for backward compatibility with existing selections.
    final_area = torch.abs(1.0 + error_alpha) * beta
    final_area_dense = dense_parallel_scale * beta
    perp_over_parallel = beta / dense_parallel_scale
    angle_l2 = torch.sqrt(output_angle**2 + update_angle**2)
    update_scaled_angle_l2 = update_over_ref * angle_l2
    update_scaled_angle_sum = update_over_ref * angle_sum
    theta_perp_l2 = torch.sqrt(output_angle**2 + beta**2)
    theta_area_l2 = torch.sqrt(output_angle**2 + final_area**2)
    theta_beta_l1 = output_angle + beta
    theta_area_l1 = output_angle + final_area
    update_scaled_theta_perp_l2 = update_over_ref * theta_perp_l2
    update_scaled_theta_area_l2 = update_over_ref * theta_area_l2
    update_scaled_theta_beta_l1 = update_over_ref * theta_beta_l1
    update_scaled_theta_area_l1 = update_over_ref * theta_area_l1

    return {
        "error_norm": float(error_norm.mean().item()),
        "para_norm": float(para_norm.mean().item()),
        "perp_norm": float(perp_norm.mean().item()),
        "ref_norm": float(ref_norm.mean().item()),
        "error_over_ref": float((error_norm / ref_norm).mean().item()),
        "para_over_ref": float((para_norm / ref_norm).mean().item()),
        "perp_over_ref": float((perp_norm / ref_norm).mean().item()),
        "perp_ratio": float((perp_sq_raw / error_sq).mean().item()),
        "perp_sq_sum": float(perp_sq_raw.sum().item()),
        "error_sq_sum": float(error_sq_raw.sum().item()),
        "ref_sq_sum": float(ref_sq.squeeze(-1).sum().item()),
        "position_count": float(error_sq_raw.numel()),
        "signed_alpha": float(coeff.squeeze(-1).mean().item()),
        "update_alpha": float(update_alpha.mean().item()),
        "dense_parallel_scale": float(dense_parallel_scale.mean().item()),
        "update_norm_over_ref": float(update_over_ref.mean().item()),
        "cosine_dense_drop": float(cosine_dense_drop.mean().item()),
        "one_minus_cosine": float((1.0 - cosine_dense_drop).mean().item()),
        "output_angle": float(output_angle.mean().item()),
        "update_angle": float(update_angle.mean().item()),
        "angle_sum": float(angle_sum.mean().item()),
        "final_area": float(final_area.mean().item()),
        "final_area_dense": float(final_area_dense.mean().item()),
        "perp_over_parallel": float(perp_over_parallel.mean().item()),
        "angle_l2_mean": float(angle_l2.mean().item()),
        "update_scaled_angle_l2_mean": float(update_scaled_angle_l2.mean().item()),
        "update_scaled_angle_sum_mean": float(update_scaled_angle_sum.mean().item()),
        "theta_perp_l2_mean": float(theta_perp_l2.mean().item()),
        "theta_area_l2_mean": float(theta_area_l2.mean().item()),
        "theta_beta_l1_mean": float(theta_beta_l1.mean().item()),
        "theta_area_l1_mean": float(theta_area_l1.mean().item()),
        "update_scaled_theta_perp_l2_mean": float(update_scaled_theta_perp_l2.mean().item()),
        "update_scaled_theta_area_l2_mean": float(update_scaled_theta_area_l2.mean().item()),
        "update_scaled_theta_beta_l1_mean": float(update_scaled_theta_beta_l1.mean().item()),
        "update_scaled_theta_area_l1_mean": float(update_scaled_theta_area_l1.mean().item()),
    }


def _select_token_scope(x: torch.Tensor, token_scope: str, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if token_scope == "last":
        return x[:, -1, :]
    if attention_mask is None:
        return x.reshape(-1, x.shape[-1])
    mask = attention_mask.to(device=x.device, dtype=torch.bool)
    if mask.shape[-1] != x.shape[1]:
        mask = mask[:, -x.shape[1] :]
    return x[mask]


def _count_positions(x: torch.Tensor | None) -> int:
    if x is None:
        return 0
    if x.ndim <= 2:
        return int(x.shape[0])
    return int(torch.tensor(x.shape[:-1]).prod().item())


def _find_decoder_layers(model) -> list[torch.nn.Module]:
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
    ):
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise ValueError("Could not find decoder layers for layer-drop geometry selector.")


def collect_sublayer_residual_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_scope: str,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], list[torch.Tensor | None], list[torch.Tensor | None]]:
    layers = _find_decoder_layers(model)
    num_layers = len(layers)
    attn_out: list[torch.Tensor | None] = [None] * num_layers
    mlp_out: list[torch.Tensor | None] = [None] * num_layers
    attn_ref: list[torch.Tensor | None] = [None] * num_layers
    mlp_ref: list[torch.Tensor | None] = [None] * num_layers
    handles = []

    def capture(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        scoped = _select_token_scope(x, token_scope, attention_mask)
        return scoped.detach().to(device="cpu", dtype=torch.float32)

    for li, layer in enumerate(layers):
        def _layer_pre_hook(_mod, args, kwargs, _li=li):
            h = kwargs.get("hidden_states", args[0] if args else None)
            attn_ref[_li] = capture(h)
            return args, kwargs

        handles.append(layer.register_forward_pre_hook(_layer_pre_hook, with_kwargs=True))

        if hasattr(layer, "self_attn") and layer.self_attn is not None:
            def _attn_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                attn_out[_li] = capture(y)

            handles.append(layer.self_attn.register_forward_hook(_attn_hook))

        mlp_ref_module = (
            getattr(layer, "post_attention_layernorm", None)
            or getattr(layer, "post_attention_norm", None)
            or getattr(layer, "pre_feedforward_layernorm", None)
            or getattr(layer, "ffn_norm", None)
        )
        if mlp_ref_module is not None:
            def _mlp_pre_hook(_mod, args, _li=li):
                h = args[0] if args else None
                mlp_ref[_li] = capture(h)

            handles.append(mlp_ref_module.register_forward_pre_hook(_mlp_pre_hook))

        if hasattr(layer, "mlp") and layer.mlp is not None:
            def _mlp_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                mlp_out[_li] = capture(y)

            handles.append(layer.mlp.register_forward_hook(_mlp_hook))

    forward_params = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "output_hidden_states": False,
        "use_cache": False,
        "return_dict": True,
    }
    if "attention_mask" in forward_params:
        kwargs["attention_mask"] = attention_mask
    try:
        with torch.no_grad():
            model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    for li in range(num_layers):
        if mlp_out[li] is not None and mlp_ref[li] is None and attn_ref[li] is not None:
            # Phi-style parallel decoder blocks feed attention and MLP from the
            # same normalized layer input, with no post-attention norm module.
            # For layer-drop geometry, the residual reference is therefore the
            # layer input captured for attention.
            mlp_ref[li] = attn_ref[li]

    return attn_out, mlp_out, attn_ref, mlp_ref


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).replace("+", ",").split(",") if x.strip()]


def _select_layers(
    rows: list[dict[str, Any]],
    component: str,
    drop_counts: list[int],
    rank_metric: str,
    rank_order: str,
) -> dict[str, list[int]]:
    subset = [row for row in rows if row["component"] == component and int(row["count"]) > 0]
    reverse = rank_order == "descending"
    ranked = sorted(subset, key=lambda row: (float(row[rank_metric]), int(row["layer"])), reverse=reverse)
    out = {}
    for count in drop_counts:
        out[f"drop_{count}"] = [int(row["layer"]) for row in ranked[:count]]
    return out


def _load_calibration_meta(prompts_file: str) -> dict[str, Any]:
    if not prompts_file:
        return {}
    meta_path = Path(f"{prompts_file}.meta.json")
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {f"calibration_{key}": value for key, value in meta.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select attn/MLP layers to drop using residual para/perp error.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_tag", default="")
    parser.add_argument("--output_tag", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_prompts", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--token_scope", choices=["all", "last"], default="all")
    parser.add_argument("--drop_counts", default="4,8")
    parser.add_argument(
        "--rank_metric",
        choices=[
            "perp_ratio",
            "perp_ratio_energy",
            "perp_energy",
            "perp_energy_over_ref",
            "perp_over_ref",
            "error_over_ref",
            "para_over_ref",
            "update_norm_over_ref",
            "cosine_dense_drop",
            "one_minus_cosine",
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
            "cos_plus_perp_ratio_energy",
        ],
        default="perp_ratio",
    )
    parser.add_argument("--rank_order", choices=["ascending", "descending"], default="ascending")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="compression/outputs/layer_drop_geometry")
    args = parser.parse_args()

    model_tag = args.model_tag or derive_model_tag(args.model_name)
    output_tag = args.output_tag or model_tag
    out_dir = Path(args.output_dir) / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    dtype = _dtype_from_name(args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(args.device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    prompts = read_prompts(args)
    num_layers = len(model.model.layers)
    accum: dict[tuple[str, int], dict[str, list[float]]] = {
        (component, layer): {} for component in ("attn", "mlp") for layer in range(num_layers)
    }
    counts: dict[tuple[str, int], int] = {(component, layer): 0 for component in ("attn", "mlp") for layer in range(num_layers)}

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        attn_out, mlp_out, attn_ref, mlp_ref = collect_sublayer_residual_tokens(
            model, input_ids, attention_mask, args.token_scope
        )
        for layer in range(num_layers):
            for component, updates, refs in (
                ("attn", attn_out, attn_ref),
                ("mlp", mlp_out, mlp_ref),
            ):
                if updates[layer] is None or refs[layer] is None:
                    continue
                metrics = _component_metrics(updates[layer], refs[layer])
                key = (component, layer)
                for name, value in metrics.items():
                    accum[key].setdefault(name, []).append(value)
                counts[key] += _count_positions(updates[layer])

    rows: list[dict[str, Any]] = []
    metric_names = [
        "error_norm",
        "para_norm",
        "perp_norm",
        "ref_norm",
        "error_over_ref",
        "para_over_ref",
        "perp_over_ref",
        "perp_ratio",
        "perp_ratio_energy",
        "perp_energy",
        "perp_energy_over_ref",
        "signed_alpha",
        "update_alpha",
        "dense_parallel_scale",
        "update_norm_over_ref",
        "cosine_dense_drop",
        "one_minus_cosine",
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
        "cos_plus_perp_ratio_energy",
    ]
    for component in ("attn", "mlp"):
        for layer in range(num_layers):
            key = (component, layer)
            row: dict[str, Any] = {"component": component, "layer": layer, "count": counts[key]}
            for name in metric_names:
                if name == "perp_ratio_energy":
                    perp_sq_sum = sum(accum[key].get("perp_sq_sum", []))
                    error_sq_sum = sum(accum[key].get("error_sq_sum", []))
                    position_count = sum(accum[key].get("position_count", []))
                    row["perp_sq_sum"] = perp_sq_sum
                    row["error_sq_sum"] = error_sq_sum
                    row["position_count"] = position_count
                    row[name] = perp_sq_sum / max(error_sq_sum, 1e-12)
                    batch_vals = []
                    for perp_sq, error_sq in zip(
                        accum[key].get("perp_sq_sum", []),
                        accum[key].get("error_sq_sum", []),
                    ):
                        batch_vals.append(perp_sq / max(error_sq, 1e-12))
                    row[f"{name}_std"] = _std(batch_vals)
                    continue
                if name == "perp_energy":
                    perp_sq_sum = sum(accum[key].get("perp_sq_sum", []))
                    row[name] = perp_sq_sum / max(int(row["count"]), 1)
                    batch_vals = []
                    for perp_sq, batch_count in zip(
                        accum[key].get("perp_sq_sum", []),
                        accum[key].get("position_count", []),
                    ):
                        batch_vals.append(perp_sq / max(batch_count, 1.0))
                    row[f"{name}_std"] = _std(batch_vals)
                    continue
                if name == "perp_energy_over_ref":
                    perp_sq_sum = sum(accum[key].get("perp_sq_sum", []))
                    ref_sq_sum = sum(accum[key].get("ref_sq_sum", []))
                    row["ref_sq_sum"] = ref_sq_sum
                    row[name] = perp_sq_sum / max(ref_sq_sum, 1e-12)
                    batch_vals = []
                    for perp_sq, ref_sq in zip(
                        accum[key].get("perp_sq_sum", []),
                        accum[key].get("ref_sq_sum", []),
                    ):
                        batch_vals.append(perp_sq / max(ref_sq, 1e-12))
                    row[f"{name}_std"] = _std(batch_vals)
                    continue
                if name == "cos_plus_perp_ratio_energy":
                    row[name] = row.get("one_minus_cosine", float("nan")) + row.get(
                        "perp_ratio_energy", float("nan")
                    )
                    row[f"{name}_std"] = float("nan")
                    continue
                vals = accum[key].get(name, [])
                row[name] = _mean(vals)
                row[f"{name}_std"] = _std(vals)
            rows.append(row)

    drop_counts = _parse_ints(args.drop_counts)
    recommendations = {
        component: _select_layers(rows, component, drop_counts, args.rank_metric, args.rank_order)
        for component in ("attn", "mlp")
    }

    csv_path = out_dir / "layer_drop_geometry_metrics.csv"
    fieldnames = ["component", "layer", "count"]
    for name in metric_names:
        if name == "perp_ratio_energy":
            fieldnames.extend(["perp_sq_sum", "error_sq_sum", "ref_sq_sum", "position_count"])
        fieldnames.extend([name, f"{name}_std"])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_name": args.model_name,
        "model_tag": model_tag,
        "output_tag": output_tag,
        "prompts_file": args.prompts_file,
        "num_prompts": len(prompts),
        "max_prompts": args.max_prompts,
        "max_length": args.max_length,
        "token_scope": args.token_scope,
        "num_layers": num_layers,
        "rank_metric": args.rank_metric,
        "rank_order": args.rank_order,
        "drop_counts": drop_counts,
        "recommendations": recommendations,
        "csv_path": str(csv_path),
    }
    summary.update(_load_calibration_meta(args.prompts_file))
    json_path = out_dir / "drop_selection.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
