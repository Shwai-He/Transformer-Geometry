#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils as modeling_utils

from geometry_aware_pruning import (
    DEFAULT_PROMPTS,
    _build_prune_mask,
    _dtype_from_name,
    _find_decoder_layers,
    _flatten_tokens,
    _iter_linear_modules,
    _linear_is_target,
    _model_tag,
    _parse_targets,
    _project_norms,
    _read_prompts,
    _repeat_kv_to_query_width,
)


def _component_loss(update: torch.Tensor, ref: torch.Tensor, component: str) -> torch.Tensor:
    total, para, perp = _project_norms(update, ref)
    if component == "total":
        value = total
    elif component == "para":
        value = para
    elif component == "perp":
        value = perp
    else:
        raise ValueError(f"Unsupported component: {component}")
    return value.square().mean()


class GradientObjectiveCollector:
    def __init__(self, model: nn.Module, *, space: str, component: str, token_scope: str):
        self.model = model
        self.layers = _find_decoder_layers(model)
        self.space = space
        self.component = component
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.context: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        self.loss_terms: list[torch.Tensor] = []

    def install(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre_hook(layer_idx), with_kwargs=True))

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                self.handles.append(attn.register_forward_pre_hook(self._attn_pre_hook(layer_idx), with_kwargs=True))
                o_proj = getattr(attn, "o_proj", None)
                if o_proj is not None:
                    self.handles.append(o_proj.register_forward_pre_hook(self._o_proj_pre_hook(layer_idx)))
                    self.handles.append(o_proj.register_forward_hook(self._o_proj_forward_hook(layer_idx)))

            norm = getattr(layer, "post_attention_layernorm", None)
            if norm is not None:
                self.handles.append(norm.register_forward_pre_hook(self._mlp_ref_hook(layer_idx)))

            mlp = getattr(layer, "mlp", None)
            down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
            if down_proj is not None:
                self.handles.append(down_proj.register_forward_hook(self._down_proj_forward_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_terms(self) -> None:
        self.loss_terms.clear()
        self.context.clear()

    def _layer_pre_hook(self, layer_idx: int):
        def hook(_mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None:
                self.context[layer_idx]["attn_residual_ref"] = hidden.detach()
            return args, kwargs

        return hook

    def _attn_pre_hook(self, layer_idx: int):
        def hook(mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or not hasattr(mod, "v_proj"):
                return args, kwargs
            with torch.no_grad():
                raw_value = mod.v_proj(hidden.detach())
                value = _repeat_kv_to_query_width(mod, raw_value)
            self.context[layer_idx]["value_self_ref"] = value.detach()
            return args, kwargs

        return hook

    def _mlp_ref_hook(self, layer_idx: int):
        def hook(_mod, args):
            if args and torch.is_tensor(args[0]):
                self.context[layer_idx]["mlp_residual_ref"] = args[0].detach()
            return args

        return hook

    def _o_proj_pre_hook(self, layer_idx: int):
        def hook(_mod, args):
            if self.space != "value" or not args or not torch.is_tensor(args[0]):
                return args
            value_ref = self.context[layer_idx].get("value_self_ref")
            if value_ref is None or value_ref.shape[-1] != args[0].shape[-1]:
                return args
            y_flat = _flatten_tokens(args[0], self.token_scope)
            ref_flat = _flatten_tokens(value_ref, self.token_scope).to(y_flat.device)
            self.loss_terms.append(_component_loss(y_flat, ref_flat, self.component))
            return args

        return hook

    def _o_proj_forward_hook(self, layer_idx: int):
        def hook(_mod, _args, output):
            if self.space != "residual" or not torch.is_tensor(output):
                return output
            ref = self.context[layer_idx].get("attn_residual_ref")
            if ref is None:
                return output
            out_flat = _flatten_tokens(output, self.token_scope)
            ref_flat = _flatten_tokens(ref, self.token_scope).to(out_flat.device)
            self.loss_terms.append(_component_loss(out_flat, ref_flat, self.component))
            return output

        return hook

    def _down_proj_forward_hook(self, layer_idx: int):
        def hook(_mod, _args, output):
            if self.space != "residual" or not torch.is_tensor(output):
                return output
            ref = self.context[layer_idx].get("mlp_residual_ref")
            if ref is None:
                return output
            out_flat = _flatten_tokens(output, self.token_scope)
            ref_flat = _flatten_tokens(ref, self.token_scope).to(out_flat.device)
            self.loss_terms.append(_component_loss(out_flat, ref_flat, self.component))
            return output

        return hook


def compute_gradient_scores(model: nn.Module, tokenizer: Any, prompts: list[str], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    collector = GradientObjectiveCollector(
        model,
        space=args.grad_space,
        component=args.grad_component,
        token_scope=args.token_scope,
    )
    collector.install()
    try:
        for prompt in prompts:
            collector.clear_terms()
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {key: value.to(args.device) for key, value in enc.items()}
            model(**enc, use_cache=False)
            if not collector.loss_terms:
                continue
            loss = torch.stack([term.float() for term in collector.loss_terms]).mean()
            loss.backward()
    finally:
        collector.remove()

    scores: dict[str, torch.Tensor] = {}
    layers = _find_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        for linear_name, linear in _iter_linear_modules(layer):
            if linear.weight.grad is None:
                continue
            key = f"{layer_idx}.{linear_name}"
            grad = linear.weight.grad.detach().float().cpu()
            weight = linear.weight.detach().float().cpu()
            if args.grad_score == "snip":
                scores[key] = (weight * grad).abs()
            elif args.grad_score == "grad_abs":
                scores[key] = grad.abs()
            elif args.grad_score == "grad_square":
                scores[key] = grad.square()
            else:
                raise ValueError(f"Unsupported grad_score: {args.grad_score}")
    model.zero_grad(set_to_none=True)
    return scores


def apply_gradient_pruning(model: nn.Module, grad_scores: dict[str, torch.Tensor], args: argparse.Namespace) -> list[dict[str, Any]]:
    layers = _find_decoder_layers(model)
    prune_targets = _parse_targets(args.prune_targets)
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = [int(x) for x in args.sparsity_type.split(":")]
        if not math.isclose(args.sparsity_ratio, 0.5, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("N:M sparsity currently expects --sparsity_ratio 0.5.")

    records: list[dict[str, Any]] = []
    for layer_idx, layer in enumerate(layers):
        for linear_name, linear in _iter_linear_modules(layer):
            short = linear_name.split(".")[-1]
            if not _linear_is_target(short, prune_targets):
                continue
            key = f"{layer_idx}.{linear_name}"
            metric = grad_scores.get(key)
            if metric is None:
                continue
            metric = metric.to(linear.weight.device).float()
            mask = _build_prune_mask(metric, args.sparsity_ratio, prune_n, prune_m, args.threshold_scope)
            linear.weight.data[mask.to(linear.weight.device)] = 0
            records.append(
                {
                    "layer": layer_idx,
                    "linear": linear_name,
                    "pruned": int(mask.sum().item()),
                    "total": int(mask.numel()),
                    "sparsity": float(mask.float().mean().item()),
                    "grad_space": args.grad_space,
                    "grad_component": args.grad_component,
                    "grad_score": args.grad_score,
                }
            )
    return records


def save_grad_score_csv(scores: dict[str, torch.Tensor], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "row", "col", "value"])
        writer.writeheader()
        for module, tensor in sorted(scores.items()):
            rows, cols = tensor.shape
            for row in range(rows):
                values = tensor[row].tolist()
                for col, value in enumerate(values):
                    writer.writerow({"module": module, "row": row, "col": col, "value": float(value)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient-based geometry pruning for decoder-only LMs.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--max_prompts", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--token_scope", choices=["last", "all"], default="last")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_remote_files", action="store_false", dest="local_files_only")
    parser.add_argument("--grad_space", choices=["residual", "value"], default="residual")
    parser.add_argument("--grad_component", choices=["total", "para", "perp"], default="total")
    parser.add_argument("--grad_score", choices=["snip", "grad_abs", "grad_square"], default="snip")
    parser.add_argument("--prune_targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--sparsity_type", choices=["unstructured", "2:4", "4:8"], default="unstructured")
    parser.add_argument("--threshold_scope", choices=["row", "global"], default="global")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_score_csv", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        model_tag = _model_tag(args.model_name_or_path)
        tag = f"grad_{args.grad_space}_{args.grad_component}_{args.grad_score}_s{args.sparsity_ratio}_{args.sparsity_type.replace(':', 'to')}"
        args.output_dir = f"compression/outputs/grad_geo_prune/by_model/{model_tag}/{tag}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(args.device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    prompts = _read_prompts(args) or DEFAULT_PROMPTS[: args.max_prompts]
    grad_scores = compute_gradient_scores(model, tokenizer, prompts[: args.max_prompts], args)
    score_path = out_dir / "grad_scores.pt"
    torch.save(grad_scores, score_path)
    if args.save_score_csv:
        save_grad_score_csv(grad_scores, out_dir / "grad_scores.csv")

    prune_records = apply_gradient_pruning(model, grad_scores, args)
    with (out_dir / "prune_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "linear", "pruned", "total", "sparsity", "grad_space", "grad_component", "grad_score"],
        )
        writer.writeheader()
        writer.writerows(prune_records)

    if args.save_model:
        try:
            import accelerate.utils.other as accelerate_other

            accelerate_other.extract_model_from_parallel = lambda model, **_: model
            modeling_utils.extract_model_from_parallel = lambda model, **_: model
        except Exception:
            pass
        modeling_utils.unwrap_model = lambda model, *_, **__: model
        model.save_pretrained(out_dir / "model")
        tokenizer.save_pretrained(out_dir / "model")

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "num_prompts": len(prompts[: args.max_prompts]),
        "token_scope": args.token_scope,
        "grad_space": args.grad_space,
        "grad_component": args.grad_component,
        "grad_score": args.grad_score,
        "prune_targets": args.prune_targets,
        "sparsity_ratio": args.sparsity_ratio,
        "sparsity_type": args.sparsity_type,
        "threshold_scope": args.threshold_scope,
        "score_path": str(score_path),
        "pruned_modules": len(prune_records),
        "mean_module_sparsity": (
            float(sum(item["sparsity"] for item in prune_records) / len(prune_records)) if prune_records else 0.0
        ),
        "saved_model": bool(args.save_model),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
