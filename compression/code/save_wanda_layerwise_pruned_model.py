#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils as modeling_utils

from geometry_aware_pruning import (
    _apply_error_component,
    _apply_metric_multiplier,
    _geometry_error_multiplier_for_linear,
    _geometry_multiplier_for_linear,
)


def find_decoder_layers(model: nn.Module):
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
            return obj
    raise AttributeError("Could not locate decoder layers.")


def find_linears(module: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    found = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            found[full] = child
        else:
            found.update(find_linears(child, full))
    return found


def parse_targets(text: str) -> set[str]:
    text = text.replace("+", ",").replace(";", ",")
    return {item.strip() for item in text.split(",") if item.strip()}


def is_target(linear_name: str, targets: set[str]) -> bool:
    short = linear_name.split(".")[-1]
    if "all" in targets or short in targets:
        return True
    if short in {"q_proj", "k_proj", "v_proj", "o_proj", "dense"} and "attn" in targets:
        return True
    if short in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"} and "mlp" in targets:
        return True
    return False


class WandaAccumulator:
    def __init__(self, width: int, device: torch.device):
        self.scaler_row = torch.zeros(width, device=device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp_t = inp.float().t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp_t, p=2, dim=1).square() / self.nsamples


class ExactProjectionAccumulator:
    """Accumulate per-weight projected pruning-error costs for N:M pruning."""

    def __init__(self, out_features: int, in_features: int, device: torch.device):
        self.sum_x2_c2 = {
            "para": torch.zeros(out_features, in_features, device=device, dtype=torch.float32),
            "perp": torch.zeros(out_features, in_features, device=device, dtype=torch.float32),
        }
        self.ntokens = {"para": 0, "perp": 0}

    def add_batch(self, inp: torch.Tensor, coeff: torch.Tensor, component: str) -> None:
        inp = inp.detach().float().reshape(-1, inp.shape[-1])
        coeff = coeff.detach().float().reshape(-1, coeff.shape[-1])
        n = min(inp.shape[0], coeff.shape[0])
        if n <= 0:
            return
        inp = inp[:n]
        coeff = coeff[:n]
        self.sum_x2_c2[component].add_(coeff.square().t().matmul(inp.square()))
        self.ntokens[component] += n

    def metric(self, weight: torch.Tensor, component: str) -> torch.Tensor:
        denom = max(self.ntokens[component], 1)
        return weight.detach().abs().float() * (self.sum_x2_c2[component] / denom).clamp_min(0.0).sqrt()


class JointProjectionAccumulator:
    """Accumulate block-local projected second moments for joint N:M pruning."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        block_m: int,
        device: torch.device,
        block_chunk: int = 128,
        out_chunk: int = 128,
    ):
        self.block_m = block_m
        self.n_blocks = in_features // block_m
        self.trimmed_in_features = self.n_blocks * block_m
        self.block_chunk = block_chunk
        self.out_chunk = out_chunk
        self.sum_xx_c2 = torch.zeros(
            out_features,
            self.n_blocks,
            block_m,
            block_m,
            device=device,
            dtype=torch.float32,
        )
        self.ntokens = 0

    def add_batch(self, inp: torch.Tensor, coeff: torch.Tensor) -> None:
        if self.n_blocks <= 0:
            return
        inp = inp.detach().float().reshape(-1, inp.shape[-1])
        coeff = coeff.detach().float().reshape(-1, coeff.shape[-1])
        n = min(inp.shape[0], coeff.shape[0])
        if n <= 0:
            return
        inp = inp[:n, : self.trimmed_in_features].reshape(n, self.n_blocks, self.block_m)
        coeff2 = coeff[:n].square()
        for b0 in range(0, self.n_blocks, self.block_chunk):
            b1 = min(b0 + self.block_chunk, self.n_blocks)
            xb = inp[:, b0:b1, :]
            for o0 in range(0, coeff2.shape[1], self.out_chunk):
                o1 = min(o0 + self.out_chunk, coeff2.shape[1])
                self.sum_xx_c2[o0:o1, b0:b1].add_(
                    torch.einsum("to,tbm,tbn->obmn", coeff2[:, o0:o1], xb, xb)
                )
        self.ntokens += n

    def build_mask(self, weight: torch.Tensor, prune_n: int) -> torch.Tensor:
        if self.n_blocks <= 0:
            return torch.zeros_like(weight, dtype=torch.bool)
        denom = max(self.ntokens, 1)
        weight = weight.detach().float()
        mask = torch.zeros_like(weight, dtype=torch.bool)
        combos = torch.combinations(
            torch.arange(self.block_m, device=weight.device),
            r=prune_n,
        )
        for block_idx in range(self.n_blocks):
            start = block_idx * self.block_m
            w_block = weight[:, start : start + self.block_m]
            gram = (self.sum_xx_c2[:, block_idx] / denom).to(weight.device)
            costs = []
            for combo in combos:
                w_sub = w_block[:, combo]
                gram_sub = gram[:, combo][:, :, combo]
                costs.append(torch.einsum("on,onm,om->o", w_sub, gram_sub, w_sub))
            cost_matrix = torch.stack(costs, dim=1)
            chosen = torch.argmin(cost_matrix, dim=1)
            row_idx = torch.arange(weight.shape[0], device=weight.device)
            for combo_idx, combo in enumerate(combos):
                rows = row_idx[chosen == combo_idx]
                if rows.numel() > 0:
                    mask[rows[:, None], start + combo[None, :]] = True
        return mask


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def component_coeff(ref: torch.Tensor, component: str) -> torch.Tensor:
    ref = flatten_tokens(ref).float()
    ref_norm = ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    para = (ref / ref_norm).abs()
    if component == "para":
        return para
    if component == "perp":
        return (1.0 - para.square()).clamp_min(0.0).sqrt()
    raise ValueError(f"Unsupported exact projection component: {component}")


def component_coeff_pair(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref = flatten_tokens(ref).float()
    ref_norm = ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    para = (ref / ref_norm).abs()
    perp = (1.0 - para.square()).clamp_min(0.0).sqrt()
    return para, perp


def strategy_component(strategy: str) -> str | None:
    if strategy.endswith("_perp") or strategy.endswith("_perp_over_para") or strategy.endswith("_perp_minus_para"):
        return "perp"
    if strategy.endswith("_para"):
        return "para"
    return None


def exact_projection_kind(linear_name: str, strategy: str, geometry_targets: set[str]) -> str | None:
    short = linear_name.split(".")[-1]
    if strategy.startswith("residual_") and short in {"o_proj", "dense"} and (
        short in geometry_targets or "o_proj" in geometry_targets or "attn" in geometry_targets
    ):
        return "attn_residual"
    if strategy.startswith("residual_") and short in {"down_proj", "fc2"} and (
        short in geometry_targets or "down_proj" in geometry_targets or "mlp" in geometry_targets
    ):
        return "mlp_residual"
    if strategy.startswith("value_") and short == "v_proj" and (
        "v_proj" in geometry_targets or "value" in geometry_targets or "attn" in geometry_targets
    ):
        return "value_self"
    return None


def exact_metric_from_components(
    acc: ExactProjectionAccumulator,
    weight: torch.Tensor,
    component: str,
    objective: str,
    margin_lambda: float,
    eps: float,
) -> torch.Tensor:
    primary = acc.metric(weight, component)
    if objective == "absolute":
        return primary
    other_component = "para" if component == "perp" else "perp"
    other = acc.metric(weight, other_component)
    if objective == "ratio":
        return primary / other.clamp_min(eps)
    if objective == "margin":
        return primary - margin_lambda * other
    raise ValueError(f"Unsupported exact projection objective: {objective}")


def build_mask(
    metric: torch.Tensor,
    sparsity_ratio: float,
    sparsity_type: str,
    threshold_scope: str,
    sparsity_axis: str = "input",
) -> torch.Tensor:
    if sparsity_type != "unstructured":
        prune_n, prune_m = [int(x) for x in sparsity_type.split(":")]
        if not math.isclose(sparsity_ratio, 0.5, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("N:M sparsity expects sparsity_ratio=0.5")
        if sparsity_axis == "output":
            return build_mask(metric.t(), sparsity_ratio, sparsity_type, threshold_scope, sparsity_axis="input").t()
        if sparsity_axis != "input":
            raise ValueError(f"Unsupported sparsity_axis: {sparsity_axis}")
        mask = torch.zeros_like(metric, dtype=torch.bool)
        for start in range(0, metric.shape[1], prune_m):
            chunk = metric[:, start : start + prune_m].float()
            if chunk.shape[1] < prune_m:
                continue
            idx = torch.topk(chunk, prune_n, dim=1, largest=False).indices
            mask.scatter_(1, start + idx, True)
        return mask

    if threshold_scope == "global":
        k = int(metric.numel() * sparsity_ratio)
        flat_idx = torch.topk(metric.flatten().float(), k, largest=False).indices
        mask = torch.zeros(metric.numel(), device=metric.device, dtype=torch.bool)
        mask[flat_idx] = True
        return mask.view_as(metric)

    if threshold_scope == "row":
        sort_idx = torch.sort(metric.float(), dim=-1, stable=True).indices
        cols = int(metric.shape[1] * sparsity_ratio)
        mask = torch.zeros_like(metric, dtype=torch.bool)
        mask.scatter_(1, sort_idx[:, :cols], True)
        return mask

    raise ValueError(f"Unsupported threshold_scope: {threshold_scope}")


def parse_nm_sparsity(sparsity_type: str, sparsity_ratio: float) -> tuple[int, int]:
    if sparsity_type == "unstructured":
        raise ValueError("Joint block exact projection requires N:M sparsity, e.g. 2:4 or 4:8")
    prune_n, prune_m = [int(x) for x in sparsity_type.split(":")]
    if not math.isclose(sparsity_ratio, 0.5, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("N:M sparsity expects sparsity_ratio=0.5")
    return prune_n, prune_m


def load_prompts(path: Path, nsamples: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < nsamples:
        raise ValueError(f"Need {nsamples} prompts, found {len(prompts)} in {path}")
    return prompts[:nsamples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a WANDA-pruned model using locuslab-style layerwise propagation.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--calib_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--sparsity_type", choices=["unstructured", "2:4", "4:8"], default="unstructured")
    parser.add_argument(
        "--sparsity_axis",
        choices=["input", "output"],
        default="input",
        help="Axis used for N:M blocks. input is standard row-wise grouping over input columns; output groups rows per input column.",
    )
    parser.add_argument("--threshold_scope", choices=["row", "global"], default="global")
    parser.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--geometry_scores", default="")
    parser.add_argument("--geometry_strategy", default="none")
    parser.add_argument(
        "--geometry_mode",
        choices=["contribution", "residual_error", "weight_error", "exact_projection"],
        default="residual_error",
    )
    parser.add_argument("--geometry_alpha", type=float, default=1.0)
    parser.add_argument("--geometry_targets", default="o_proj,down_proj,v_proj")
    parser.add_argument(
        "--geometry_exact_objective",
        choices=["absolute", "ratio", "margin", "joint_absolute"],
        default="absolute",
        help="Objective for exact_projection. N:M pruning still selects within each block.",
    )
    parser.add_argument("--geometry_exact_margin_lambda", type=float, default=1.0)
    parser.add_argument("--geometry_exact_eps", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local_files_only",
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        use_fast=False,
    )
    if not callable(tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            use_fast=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    prompts = load_prompts(Path(args.calib_file), args.nsamples)
    targets = parse_targets(args.targets)
    geometry_targets = parse_targets(args.geometry_targets)
    geometry_scores = None
    if args.geometry_strategy != "none":
        if args.geometry_mode != "exact_projection" and not args.geometry_scores:
            raise ValueError("--geometry_scores is required when --geometry_strategy is not none")
        if args.geometry_scores:
            geometry_scores = torch.load(args.geometry_scores, map_location="cpu")
        if args.geometry_mode in {"residual_error", "weight_error"} and (
            args.threshold_scope == "row" or args.sparsity_type != "unstructured"
        ):
            warnings.warn(
                "weight_error/residual_error geometry decomposes the per-weight WANDA pruning error along output coordinates; "
                "row-wise unstructured or N:M pruning is invariant to row-constant geometry factors. Use "
                "--threshold_scope global for unstructured pruning, or --geometry_mode contribution for N:M pruning.",
                RuntimeWarning,
            )
        if args.geometry_mode == "exact_projection" and args.geometry_exact_objective == "joint_absolute":
            parse_nm_sparsity(args.sparsity_type, args.sparsity_ratio)
            if args.sparsity_axis != "input":
                raise ValueError("joint_absolute currently supports only --sparsity_axis input")
    layers = find_decoder_layers(model)
    records = []

    for layer_idx, layer in enumerate(layers):
        subset = {name: mod for name, mod in find_linears(layer).items() if is_target(name, targets)}
        accum = {name: WandaAccumulator(mod.in_features, mod.weight.device) for name, mod in subset.items()}
        layer_context: dict[str, torch.Tensor] = {}
        exact_accum: dict[str, ExactProjectionAccumulator] = {}
        joint_accum: dict[str, JointProjectionAccumulator] = {}
        exact_component = strategy_component(args.geometry_strategy)
        if args.geometry_mode == "exact_projection" and exact_component is not None:
            joint_prune_m = None
            if args.geometry_exact_objective == "joint_absolute":
                _, joint_prune_m = parse_nm_sparsity(args.sparsity_type, args.sparsity_ratio)
            for name, mod in subset.items():
                if exact_projection_kind(name, args.geometry_strategy, geometry_targets) is not None:
                    if joint_prune_m is None:
                        exact_accum[name] = ExactProjectionAccumulator(
                            mod.out_features,
                            mod.in_features,
                            mod.weight.device,
                        )
                    else:
                        joint_accum[name] = JointProjectionAccumulator(
                            mod.out_features,
                            mod.in_features,
                            joint_prune_m,
                            mod.weight.device,
                        )

        def make_hook(name: str):
            def hook(_mod, inp, _out):
                if inp and torch.is_tensor(inp[0]):
                    accum[name].add_batch(inp[0].detach())
                    if args.geometry_mode == "exact_projection" and name in exact_accum:
                        kind = exact_projection_kind(name, args.geometry_strategy, geometry_targets)
                        if kind == "attn_residual" and "attn_residual_ref" in layer_context:
                            para, perp = component_coeff_pair(layer_context["attn_residual_ref"])
                            exact_accum[name].add_batch(inp[0], para, "para")
                            exact_accum[name].add_batch(inp[0], perp, "perp")
                        elif kind == "mlp_residual" and "mlp_residual_ref" in layer_context:
                            para, perp = component_coeff_pair(layer_context["mlp_residual_ref"])
                            exact_accum[name].add_batch(inp[0], para, "para")
                            exact_accum[name].add_batch(inp[0], perp, "perp")
                    elif args.geometry_mode == "exact_projection" and name in joint_accum:
                        kind = exact_projection_kind(name, args.geometry_strategy, geometry_targets)
                        if kind == "attn_residual" and "attn_residual_ref" in layer_context:
                            coeff = component_coeff(layer_context["attn_residual_ref"], exact_component)
                            joint_accum[name].add_batch(inp[0], coeff)
                        elif kind == "mlp_residual" and "mlp_residual_ref" in layer_context:
                            coeff = component_coeff(layer_context["mlp_residual_ref"], exact_component)
                            joint_accum[name].add_batch(inp[0], coeff)
            return hook

        def layer_pre_hook(_mod, inp, kwargs):
            hidden = kwargs.get("hidden_states", inp[0] if inp else None)
            if hidden is not None:
                layer_context["attn_residual_ref"] = hidden.detach()
            return inp, kwargs

        def mlp_ref_hook(_mod, inp):
            if inp and torch.is_tensor(inp[0]):
                layer_context["mlp_residual_ref"] = inp[0].detach()
            return inp

        def make_value_hook(name: str):
            def hook(_mod, inp, out):
                if args.geometry_mode != "exact_projection" or (name not in exact_accum and name not in joint_accum):
                    return
                if not inp or not torch.is_tensor(inp[0]) or not torch.is_tensor(out):
                    return
                if name in exact_accum:
                    para, perp = component_coeff_pair(out.detach())
                    exact_accum[name].add_batch(inp[0], para, "para")
                    exact_accum[name].add_batch(inp[0], perp, "perp")
                else:
                    coeff = component_coeff(out.detach(), exact_component)
                    joint_accum[name].add_batch(inp[0], coeff)

            return hook

        handles = [layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True)]
        norm = getattr(layer, "post_attention_layernorm", None)
        if norm is not None:
            handles.append(norm.register_forward_pre_hook(mlp_ref_hook))
        for name, mod in subset.items():
            handles.append(mod.register_forward_hook(make_hook(name)))
            if exact_projection_kind(name, args.geometry_strategy, geometry_targets) == "value_self":
                handles.append(mod.register_forward_hook(make_value_hook(name)))
        try:
            for prompt in prompts:
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
                enc = {key: value.to(args.device) for key, value in enc.items()}
                with torch.no_grad():
                    model(**enc, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        for name, mod in subset.items():
            metric = mod.weight.detach().abs().float() * accum[name].scaler_row.sqrt().view(1, -1)
            base_metric = metric
            multiplier = None
            exact_used = False
            joint_used = False
            if args.geometry_mode == "exact_projection" and name in exact_accum:
                exact_metric = exact_metric_from_components(
                    exact_accum[name],
                    mod.weight,
                    exact_component,
                    args.geometry_exact_objective,
                    args.geometry_exact_margin_lambda,
                    args.geometry_exact_eps,
                )
                if args.geometry_alpha == 1.0:
                    metric = exact_metric
                else:
                    ratio = exact_metric / base_metric.clamp_min(1e-12)
                    metric = base_metric * ratio.clamp_min(1e-8).pow(args.geometry_alpha)
                exact_used = True
            elif args.geometry_mode == "exact_projection" and name in joint_accum:
                joint_used = True
            elif geometry_scores is not None:
                if args.geometry_mode == "contribution":
                    multiplier = _geometry_multiplier_for_linear(
                        scores=geometry_scores,
                        layer_idx=layer_idx,
                        linear_name=name,
                        linear=mod,
                        strategy=args.geometry_strategy,
                        targets=geometry_targets,
                    )
                    metric = _apply_metric_multiplier(metric, multiplier, alpha=args.geometry_alpha)
                elif args.geometry_mode in {"residual_error", "weight_error"}:
                    multiplier = _geometry_error_multiplier_for_linear(
                        scores=geometry_scores,
                        layer_idx=layer_idx,
                        linear_name=name,
                        linear=mod,
                        strategy=args.geometry_strategy,
                        targets=geometry_targets,
                    )
                    metric = _apply_error_component(metric, multiplier, alpha=args.geometry_alpha)
                else:
                    raise ValueError(f"Unsupported geometry_mode: {args.geometry_mode}")
            base_mask = build_mask(
                base_metric,
                args.sparsity_ratio,
                args.sparsity_type,
                args.threshold_scope,
                args.sparsity_axis,
            )
            if joint_used:
                prune_n, _ = parse_nm_sparsity(args.sparsity_type, args.sparsity_ratio)
                mask = joint_accum[name].build_mask(mod.weight, prune_n)
            else:
                mask = build_mask(
                    metric,
                    args.sparsity_ratio,
                    args.sparsity_type,
                    args.threshold_scope,
                    args.sparsity_axis,
                )
            mod.weight.data[mask.to(mod.weight.device)] = 0
            changed = int((mask != base_mask).sum().item())
            union = int((mask | base_mask).sum().item())
            records.append(
                {
                    "layer": layer_idx,
                    "linear": name,
                    "sparsity": float(mask.float().mean().item()),
                    "pruned": int(mask.sum().item()),
                    "total": int(mask.numel()),
                    "geometry_axis": "joint_block" if joint_used else ("matrix" if exact_used else (None if multiplier is None else multiplier[0])),
                    "geometry_mode": args.geometry_mode,
                    "geometry_strategy": args.geometry_strategy,
                    "geometry_alpha": args.geometry_alpha,
                    "geometry_exact_objective": args.geometry_exact_objective,
                    "geometry_applied": exact_used or joint_used or multiplier is not None,
                    "mask_changed_vs_base": changed,
                    "mask_jaccard_with_base": float(((mask & base_mask).sum().item()) / union) if union else 1.0,
                }
            )
        print(f"[LAYER] {layer_idx} pruned_modules={len(subset)}", flush=True)
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import accelerate.utils.other as accelerate_other

        accelerate_other.extract_model_from_parallel = lambda model, **_: model
        modeling_utils.extract_model_from_parallel = lambda model, **_: model
    except Exception:
        pass
    modeling_utils.unwrap_model = lambda model, *_, **__: model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "prune_records.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer",
            "linear",
            "sparsity",
            "pruned",
            "total",
            "geometry_axis",
            "geometry_mode",
            "geometry_strategy",
            "geometry_alpha",
            "geometry_exact_objective",
            "geometry_applied",
            "mask_changed_vs_base",
            "mask_jaccard_with_base",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "calib_file": args.calib_file,
        "nsamples": args.nsamples,
        "max_length": args.max_length,
        "sparsity_ratio": args.sparsity_ratio,
        "sparsity_type": args.sparsity_type,
        "sparsity_axis": args.sparsity_axis,
        "threshold_scope": args.threshold_scope,
        "targets": args.targets,
        "geometry_scores": args.geometry_scores,
        "geometry_strategy": args.geometry_strategy,
        "geometry_mode": args.geometry_mode,
        "geometry_alpha": args.geometry_alpha,
        "geometry_exact_objective": args.geometry_exact_objective,
        "geometry_exact_margin_lambda": args.geometry_exact_margin_lambda,
        "geometry_targets": args.geometry_targets,
        "modules": len(records),
        "mean_module_sparsity": sum(r["sparsity"] for r in records) / max(len(records), 1),
        "geometry_axis_counts": {
            axis: sum(1 for r in records if r.get("geometry_axis") == axis)
            for axis in sorted({r.get("geometry_axis") for r in records}, key=lambda x: str(x))
        },
        "mask_changed_modules": sum(1 for r in records if int(r.get("mask_changed_vs_base", 0)) > 0),
        "mask_changed_weights": sum(int(r.get("mask_changed_vs_base", 0)) for r in records),
    }
    (output_dir / "wanda_layerwise_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
