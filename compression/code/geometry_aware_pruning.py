#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "Transformer compression should preserve behaviorally important representation directions.",
    "Geometry-aware pruning keeps updates that move hidden states in sensitive directions.",
]


def _dtype_from_name(name: str) -> torch.dtype | None:
    return {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _model_tag(path: str) -> str:
    return Path(path.rstrip("/")).name.replace("/", "__")


def _safe_mean_multiplier(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.detach().float().clamp_min(0.0)
    mean = values.mean().clamp_min(eps)
    return (values / mean).clamp_min(eps)


def _project_norms(update: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    update = update.float()
    ref = ref.float()
    ref_norm = ref.norm(dim=-1).clamp_min(eps)
    update_norm = update.norm(dim=-1)
    para = (update * ref).sum(dim=-1).abs() / ref_norm
    perp = (update_norm.square() - para.square()).clamp_min(0.0).sqrt()
    return update_norm, para, perp


def _token_slice(x: torch.Tensor, token_scope: str) -> torch.Tensor:
    if token_scope == "last":
        return x[:, -1:, :]
    if token_scope == "all":
        return x
    raise ValueError(f"Unsupported token_scope: {token_scope}")


def _flatten_tokens(x: torch.Tensor, token_scope: str) -> torch.Tensor:
    x = _token_slice(x, token_scope)
    return x.reshape(-1, x.shape[-1])


def _find_decoder_layers(model: nn.Module):
    candidates = (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise AttributeError("Could not locate decoder layers on this causal LM.")


def _iter_linear_modules(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, child
        else:
            yield from _iter_linear_modules(child, full)


def _repeat_kv_to_query_width(attn: nn.Module, value: torch.Tensor) -> torch.Tensor:
    num_heads = int(getattr(attn, "num_heads", getattr(attn, "num_attention_heads", 0)) or 0)
    num_kv_heads = int(getattr(attn, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    o_proj = getattr(attn, "o_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if head_dim > 0 and num_heads <= 0 and hasattr(o_proj, "in_features"):
        num_heads = int(o_proj.in_features) // head_dim
    if head_dim > 0 and num_kv_heads <= 0 and hasattr(v_proj, "out_features"):
        num_kv_heads = int(v_proj.out_features) // head_dim
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        return value
    expected = num_kv_heads * head_dim
    if value.shape[-1] != expected or num_heads == num_kv_heads:
        return value
    repeat = max(num_heads // num_kv_heads, 1)
    value = value.view(value.shape[0], value.shape[1], num_kv_heads, head_dim)
    value = value.repeat_interleave(repeat, dim=2)
    return value.reshape(value.shape[0], value.shape[1], num_heads * head_dim)


def _attn_shape(attn: nn.Module, width: int) -> tuple[int, int]:
    num_heads = int(getattr(attn, "num_heads", getattr(attn, "num_attention_heads", 0)) or 0)
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    if num_heads <= 0 and head_dim > 0:
        num_heads = width // head_dim
    if head_dim <= 0 and num_heads > 0:
        head_dim = width // num_heads
    if num_heads <= 0 or head_dim <= 0 or num_heads * head_dim != width:
        raise ValueError(f"Cannot infer attention head shape for width={width}.")
    return num_heads, head_dim


class SumCount:
    def __init__(self, width: int):
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.count = 0

    def add(self, values: torch.Tensor) -> None:
        values = values.detach().float().cpu()
        self.sum += values.double().sum(dim=0)
        self.count += int(values.shape[0])

    def mean(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.zeros_like(self.sum, dtype=torch.float32)
        return (self.sum / self.count).float()


class GeometryScoreCollector:
    def __init__(self, model: nn.Module, token_scope: str = "last"):
        self.model = model
        self.layers = _find_decoder_layers(model)
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.layer_context: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        self.wanda_inputs: dict[str, SumCount] = {}
        self.attn_scores: dict[str, list[SumCount | None]] = defaultdict(list)
        self.mlp_scores: dict[str, list[SumCount | None]] = defaultdict(list)
        self.output_scores: dict[str, list[SumCount | None]] = defaultdict(list)
        self.v_proj_scores: dict[str, list[SumCount | None]] = defaultdict(list)

        for _ in range(len(self.layers)):
            for key in (
                "value_total",
                "value_para",
                "value_perp",
                "residual_total",
                "residual_para",
                "residual_perp",
                "residual_col_total",
                "residual_col_para",
                "residual_col_perp",
            ):
                self.attn_scores[key].append(None)
            for key in ("residual_total", "residual_para", "residual_perp"):
                self.mlp_scores[key].append(None)
            for key in (
                "attn_para_unit",
                "attn_perp_unit",
                "mlp_para_unit",
                "mlp_perp_unit",
                "value_para_unit",
                "value_perp_unit",
                "attn_value_para_unit",
                "attn_value_perp_unit",
            ):
                self.output_scores[key].append(None)
            for key in ("value_col_total", "value_col_para", "value_col_perp"):
                self.v_proj_scores[key].append(None)

    def install(self) -> None:
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre_hook(li), with_kwargs=True))

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                self.handles.append(attn.register_forward_pre_hook(self._attn_pre_hook(li), with_kwargs=True))
                o_proj = getattr(attn, "o_proj", None) or getattr(attn, "dense", None)
                if o_proj is not None:
                    self.handles.append(o_proj.register_forward_pre_hook(self._o_proj_pre_hook(li, attn)))

            norm = getattr(layer, "post_attention_layernorm", None)
            if norm is not None:
                self.handles.append(norm.register_forward_pre_hook(self._mlp_ref_hook(li)))

            mlp = getattr(layer, "mlp", None)
            down_proj = None
            if mlp is not None:
                down_proj = getattr(mlp, "down_proj", None) or getattr(mlp, "fc2", None)
            if down_proj is not None:
                self.handles.append(down_proj.register_forward_pre_hook(self._down_proj_pre_hook(li, down_proj)))

            for linear_name, linear in _iter_linear_modules(layer):
                key = f"{li}.{linear_name}"
                self.wanda_inputs[key] = SumCount(linear.in_features)
                self.handles.append(linear.register_forward_pre_hook(self._wanda_input_hook(key)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _layer_pre_hook(self, layer_idx: int):
        def hook(_mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None:
                self.layer_context[layer_idx]["attn_residual_ref"] = hidden.detach()
            return args, kwargs

        return hook

    def _attn_pre_hook(self, layer_idx: int):
        def hook(mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or not hasattr(mod, "v_proj"):
                return args, kwargs
            with torch.no_grad():
                raw_value = mod.v_proj(hidden)
                value = _repeat_kv_to_query_width(mod, raw_value)
                weight = mod.v_proj.weight.detach().float()
                x_flat = _flatten_tokens(hidden, self.token_scope).float()
                v_flat = _flatten_tokens(raw_value, self.token_scope).float()
                if x_flat.numel() > 0 and v_flat.numel() > 0 and weight.shape[1] == x_flat.shape[-1]:
                    self._ensure_v_proj_width(layer_idx, weight.shape[1])
                    v_norm = v_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    dot = v_flat.matmul(weight).abs() / v_norm
                    total = x_flat.abs() * weight.norm(dim=0).view(1, -1)
                    para = x_flat.abs() * dot
                    perp = (total.square() - para.square()).clamp_min(0.0).sqrt()
                    self.v_proj_scores["value_col_total"][layer_idx].add(total)
                    self.v_proj_scores["value_col_para"][layer_idx].add(para)
                    self.v_proj_scores["value_col_perp"][layer_idx].add(perp)
            raw_value_flat = _flatten_tokens(raw_value, self.token_scope)
            self._ensure_output_width(layer_idx, raw_value_flat.shape[-1], "value")
            self._add_ref_unit_scores(self.output_scores, layer_idx, "value", raw_value_flat)
            value_flat = _flatten_tokens(value, self.token_scope)
            self._ensure_output_width(layer_idx, value_flat.shape[-1], "attn_value")
            self._add_ref_unit_scores(self.output_scores, layer_idx, "attn_value", value_flat)
            self.layer_context[layer_idx]["value_self"] = value.detach()
            return args, kwargs

        return hook

    def _mlp_ref_hook(self, layer_idx: int):
        def hook(_mod, args):
            if args and torch.is_tensor(args[0]):
                self.layer_context[layer_idx]["mlp_residual_ref"] = args[0].detach()
            return args

        return hook

    def _wanda_input_hook(self, key: str):
        def hook(_mod, args):
            if not args or not torch.is_tensor(args[0]):
                return args
            x = args[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            self.wanda_inputs[key].add(x.square())
            return args

        return hook

    def _ensure_attn_width(self, layer_idx: int, width: int) -> None:
        for key in ("value_total", "value_para", "value_perp", "residual_total", "residual_para", "residual_perp"):
            if self.attn_scores[key][layer_idx] is None:
                self.attn_scores[key][layer_idx] = SumCount(width)

    def _ensure_attn_col_width(self, layer_idx: int, width: int) -> None:
        for key in ("residual_col_total", "residual_col_para", "residual_col_perp"):
            if self.attn_scores[key][layer_idx] is None:
                self.attn_scores[key][layer_idx] = SumCount(width)

    def _ensure_mlp_width(self, layer_idx: int, width: int) -> None:
        for key in self.mlp_scores:
            if self.mlp_scores[key][layer_idx] is None:
                self.mlp_scores[key][layer_idx] = SumCount(width)

    def _ensure_output_width(self, layer_idx: int, width: int, prefix: str) -> None:
        for suffix in ("para_unit", "perp_unit"):
            key = f"{prefix}_{suffix}"
            if self.output_scores[key][layer_idx] is None:
                self.output_scores[key][layer_idx] = SumCount(width)

    def _ensure_v_proj_width(self, layer_idx: int, width: int) -> None:
        for key in self.v_proj_scores:
            if self.v_proj_scores[key][layer_idx] is None:
                self.v_proj_scores[key][layer_idx] = SumCount(width)

    @staticmethod
    def _add_ref_unit_scores(acc: dict[str, list[SumCount | None]], layer_idx: int, prefix: str, ref_flat: torch.Tensor) -> None:
        ref_flat = ref_flat.float()
        ref_norm = ref_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        para_unit = (ref_flat / ref_norm).abs()
        perp_unit = (1.0 - para_unit.square()).clamp_min(0.0).sqrt()
        acc[f"{prefix}_para_unit"][layer_idx].add(para_unit)
        acc[f"{prefix}_perp_unit"][layer_idx].add(perp_unit)

    def _o_proj_pre_hook(self, layer_idx: int, attn: nn.Module):
        def hook(mod, args):
            if not args or not torch.is_tensor(args[0]):
                return args
            y_pre = args[0].detach()
            width = y_pre.shape[-1]
            num_heads, head_dim = _attn_shape(attn, width)
            self._ensure_attn_width(layer_idx, num_heads)

            value_self = self.layer_context[layer_idx].get("value_self")
            if value_self is not None and value_self.shape[-1] == width:
                y = _flatten_tokens(y_pre, self.token_scope).view(-1, num_heads, head_dim)
                v = _flatten_tokens(value_self, self.token_scope).view(-1, num_heads, head_dim)
                total, para, perp = _project_norms(y, v)
                self.attn_scores["value_total"][layer_idx].add(total)
                self.attn_scores["value_para"][layer_idx].add(para)
                self.attn_scores["value_perp"][layer_idx].add(perp)

            ref = self.layer_context[layer_idx].get("attn_residual_ref")
            if ref is not None:
                ref_flat = _flatten_tokens(ref, self.token_scope).to(y_pre.device)
                self._ensure_output_width(layer_idx, ref_flat.shape[-1], "attn")
                self._add_ref_unit_scores(self.output_scores, layer_idx, "attn", ref_flat)
                y_flat = _flatten_tokens(y_pre, self.token_scope)
                weight = mod.weight.detach()
                self._ensure_attn_col_width(layer_idx, width)
                ref_norm = ref_flat.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
                dot = ref_flat.float().matmul(weight.float())
                col_para = y_flat.float().abs() * dot.abs() / ref_norm
                col_total = y_flat.float().abs() * weight.float().norm(dim=0).view(1, -1)
                col_perp = (col_total.square() - col_para.square()).clamp_min(0.0).sqrt()
                self.attn_scores["residual_col_total"][layer_idx].add(col_total)
                self.attn_scores["residual_col_para"][layer_idx].add(col_para)
                self.attn_scores["residual_col_perp"][layer_idx].add(col_perp)
                totals = []
                paras = []
                perps = []
                for head_idx in range(num_heads):
                    start = head_idx * head_dim
                    end = start + head_dim
                    contrib = y_flat[:, start:end].float().matmul(weight[:, start:end].float().t())
                    total, para, perp = _project_norms(contrib, ref_flat)
                    totals.append(total)
                    paras.append(para)
                    perps.append(perp)
                self.attn_scores["residual_total"][layer_idx].add(torch.stack(totals, dim=1))
                self.attn_scores["residual_para"][layer_idx].add(torch.stack(paras, dim=1))
                self.attn_scores["residual_perp"][layer_idx].add(torch.stack(perps, dim=1))
            return args

        return hook

    def _down_proj_pre_hook(self, layer_idx: int, down_proj: nn.Linear):
        def hook(_mod, args):
            if not args or not torch.is_tensor(args[0]):
                return args
            ref = self.layer_context[layer_idx].get("mlp_residual_ref")
            if ref is None:
                return args
            act = _flatten_tokens(args[0].detach(), self.token_scope).float()
            ref_flat = _flatten_tokens(ref, self.token_scope).to(act.device).float()
            self._ensure_output_width(layer_idx, ref_flat.shape[-1], "mlp")
            self._add_ref_unit_scores(self.output_scores, layer_idx, "mlp", ref_flat)
            weight = down_proj.weight.detach().float()
            width = act.shape[-1]
            self._ensure_mlp_width(layer_idx, width)

            ref_norm = ref_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            dot = ref_flat.matmul(weight)
            para = act.abs() * dot.abs() / ref_norm
            total = act.abs() * weight.norm(dim=0).view(1, -1)
            perp = (total.square() - para.square()).clamp_min(0.0).sqrt()
            self.mlp_scores["residual_total"][layer_idx].add(total)
            self.mlp_scores["residual_para"][layer_idx].add(para)
            self.mlp_scores["residual_perp"][layer_idx].add(perp)
            return args

        return hook

    def to_tensors(self) -> dict[str, Any]:
        wanda_inputs = {key: value.mean() for key, value in self.wanda_inputs.items()}
        attn = {}
        for key, entries in self.attn_scores.items():
            rows = [entry.mean() if entry is not None else torch.empty(0) for entry in entries]
            attn[key] = _pad_and_stack(rows)
        mlp = {}
        for key, entries in self.mlp_scores.items():
            rows = [entry.mean() if entry is not None else torch.empty(0) for entry in entries]
            mlp[key] = _pad_and_stack(rows)
        output = {}
        for key, entries in self.output_scores.items():
            rows = [entry.mean() if entry is not None else torch.empty(0) for entry in entries]
            output[key] = _pad_and_stack(rows)
        v_proj = {}
        for key, entries in self.v_proj_scores.items():
            rows = [entry.mean() if entry is not None else torch.empty(0) for entry in entries]
            v_proj[key] = _pad_and_stack(rows)
        return {
            "wanda_inputs": wanda_inputs,
            "attn": attn,
            "mlp": mlp,
            "output": output,
            "v_proj": v_proj,
            "token_scope": self.token_scope,
        }


def _pad_and_stack(rows: list[torch.Tensor]) -> torch.Tensor:
    width = max((int(row.numel()) for row in rows), default=0)
    if width == 0:
        return torch.empty(len(rows), 0)
    padded = []
    for row in rows:
        row = row.flatten().float()
        if row.numel() < width:
            row = torch.nn.functional.pad(row, (0, width - row.numel()))
        padded.append(row)
    return torch.stack(padded, dim=0)


def _score_from_strategy(scores: dict[str, Any], layer_idx: int, kind: str, strategy: str) -> torch.Tensor | None:
    if strategy == "none":
        return None
    if kind == "attn":
        table = scores["attn"]
        if strategy.startswith("value_"):
            prefix = "value"
        elif strategy == "hybrid_perp":
            value = table["value_perp"][layer_idx]
            residual = table["residual_perp"][layer_idx]
            return 0.5 * _safe_mean_multiplier(value) + 0.5 * _safe_mean_multiplier(residual)
        else:
            prefix = "residual"
    else:
        if strategy.startswith("value_"):
            return None
        table = scores["mlp"]
        prefix = "residual"

    if strategy.endswith("_perp_over_para"):
        return table[f"{prefix}_perp"][layer_idx] / table[f"{prefix}_para"][layer_idx].clamp_min(1e-8)
    if strategy.endswith("_perp_minus_para"):
        return table[f"{prefix}_perp"][layer_idx] - table[f"{prefix}_para"][layer_idx]
    if strategy.endswith("_perp"):
        return table[f"{prefix}_perp"][layer_idx]
    if strategy.endswith("_para"):
        return table[f"{prefix}_para"][layer_idx]
    if strategy == "hybrid_perp":
        return table[f"{prefix}_perp"][layer_idx]
    raise ValueError(f"Unsupported geometry_strategy: {strategy}")


def _geometry_multiplier_for_linear(
    *,
    scores: dict[str, Any],
    layer_idx: int,
    linear_name: str,
    linear: nn.Linear,
    strategy: str,
    targets: set[str],
) -> tuple[str, torch.Tensor] | None:
    if strategy == "none":
        return None

    short = linear_name.split(".")[-1]
    if short in {"o_proj", "dense"} and (short in targets or "o_proj" in targets or "attn" in targets):
        if strategy.startswith("value_"):
            output_scores = scores.get("output", {})
            para = output_scores.get("attn_value_para_unit")
            perp = output_scores.get("attn_value_perp_unit")
            if para is not None and perp is not None and layer_idx < para.shape[0] and layer_idx < perp.shape[0]:
                para_col = para[layer_idx]
                perp_col = perp[layer_idx]
                if para_col.numel() >= linear.in_features and perp_col.numel() >= linear.in_features:
                    if strategy.endswith("_perp_over_para"):
                        score = perp_col[: linear.in_features] / para_col[: linear.in_features].clamp_min(1e-8)
                    elif strategy.endswith("_perp_minus_para"):
                        score = perp_col[: linear.in_features] - para_col[: linear.in_features]
                    elif strategy.endswith("_perp"):
                        score = perp_col[: linear.in_features]
                    elif strategy.endswith("_para"):
                        score = para_col[: linear.in_features]
                    else:
                        score = None
                    if score is not None:
                        return "col", _safe_mean_multiplier(score)
        if not strategy.startswith("value_"):
            table = scores.get("attn", {})
            col_key = None
            if strategy.endswith("_perp_over_para"):
                perp = table.get("residual_col_perp")
                para = table.get("residual_col_para")
                if perp is not None and para is not None and layer_idx < perp.shape[0] and layer_idx < para.shape[0]:
                    score = perp[layer_idx] / para[layer_idx].clamp_min(1e-8)
                    if score.numel() >= linear.in_features:
                        return "col", _safe_mean_multiplier(score[: linear.in_features])
            elif strategy.endswith("_perp_minus_para"):
                perp = table.get("residual_col_perp")
                para = table.get("residual_col_para")
                if perp is not None and para is not None and layer_idx < perp.shape[0] and layer_idx < para.shape[0]:
                    score = perp[layer_idx] - para[layer_idx]
                    if score.numel() >= linear.in_features:
                        return "col", _safe_mean_multiplier(score[: linear.in_features])
            elif strategy.endswith("_perp"):
                col_key = "residual_col_perp"
            elif strategy.endswith("_para"):
                col_key = "residual_col_para"
            if col_key is not None:
                score_table = table.get(col_key)
                if score_table is not None and layer_idx < score_table.shape[0]:
                    score = score_table[layer_idx]
                    if score.numel() >= linear.in_features:
                        return "col", _safe_mean_multiplier(score[: linear.in_features])
        head_score = _score_from_strategy(scores, layer_idx, "attn", strategy)
        if head_score is None or head_score.numel() == 0:
            return None
        num_heads = int(head_score.numel())
        head_dim = linear.in_features // num_heads
        if head_dim * num_heads != linear.in_features:
            return None
        expanded = _safe_mean_multiplier(head_score).repeat_interleave(head_dim)
        return "col", expanded

    if short == "v_proj" and ("v_proj" in targets or "value" in targets) and strategy.startswith("value_"):
        table = scores.get("v_proj", {})
        if strategy.endswith("_perp_over_para"):
            perp = table.get("value_col_perp")
            para = table.get("value_col_para")
            if perp is None or para is None or layer_idx >= perp.shape[0] or layer_idx >= para.shape[0]:
                return None
            value_score = perp[layer_idx] / para[layer_idx].clamp_min(1e-8)
        elif strategy.endswith("_perp_minus_para"):
            perp = table.get("value_col_perp")
            para = table.get("value_col_para")
            if perp is None or para is None or layer_idx >= perp.shape[0] or layer_idx >= para.shape[0]:
                return None
            value_score = perp[layer_idx] - para[layer_idx]
        elif strategy.endswith("_perp"):
            value_col = table.get("value_col_perp")
            if value_col is None or layer_idx >= value_col.shape[0]:
                return None
            value_score = value_col[layer_idx]
        elif strategy.endswith("_para"):
            value_col = table.get("value_col_para")
            if value_col is None or layer_idx >= value_col.shape[0]:
                return None
            value_score = value_col[layer_idx]
        else:
            return None
        if value_score.numel() < linear.in_features:
            return None
        return "col", _safe_mean_multiplier(value_score[: linear.in_features])

    if short in {"down_proj", "fc2"} and (short in targets or "down_proj" in targets or "mlp" in targets):
        neuron_score = _score_from_strategy(scores, layer_idx, "mlp", strategy)
        if neuron_score is None or neuron_score.numel() == 0:
            return None
        if neuron_score.numel() < linear.in_features:
            return None
        return "col", _safe_mean_multiplier(neuron_score[: linear.in_features])

    if short in {"gate_proj", "up_proj", "fc1"} and short in targets:
        neuron_score = _score_from_strategy(scores, layer_idx, "mlp", strategy)
        if neuron_score is None or neuron_score.numel() < linear.out_features:
            return None
        return "row", _safe_mean_multiplier(neuron_score[: linear.out_features])

    return None


def _geometry_error_multiplier_for_linear(
    *,
    scores: dict[str, Any],
    layer_idx: int,
    linear_name: str,
    linear: nn.Linear,
    strategy: str,
    targets: set[str],
) -> tuple[str, torch.Tensor] | None:
    if strategy == "none":
        return None
    short = linear_name.split(".")[-1]
    output_scores = scores.get("output", {})

    if strategy.startswith("value_"):
        if short == "v_proj" and ("v_proj" in targets or "attn" in targets or "value" in targets):
            prefix = "value"
        else:
            return None
    elif short in {"o_proj", "dense"} and (short in targets or "o_proj" in targets or "attn" in targets):
        prefix = "attn"
    elif short in {"down_proj", "fc2"} and (short in targets or "down_proj" in targets or "mlp" in targets):
        prefix = "mlp"
    else:
        return None

    para = output_scores.get(f"{prefix}_para_unit")
    perp = output_scores.get(f"{prefix}_perp_unit")
    if para is None or perp is None or layer_idx >= para.shape[0] or layer_idx >= perp.shape[0]:
        return None
    para_row = para[layer_idx]
    perp_row = perp[layer_idx]
    if para_row.numel() < linear.out_features or perp_row.numel() < linear.out_features:
        return None
    para_row = para_row[: linear.out_features]
    perp_row = perp_row[: linear.out_features]
    if strategy.endswith("_perp_over_para"):
        row_score = perp_row / para_row.clamp_min(1e-8)
    elif strategy.endswith("_perp_minus_para"):
        row_score = perp_row - para_row
    elif strategy.endswith("_perp"):
        row_score = perp_row
    elif strategy.endswith("_para"):
        row_score = para_row
    else:
        return None
    return "row", row_score.float().clamp_min(1e-8)


def _apply_metric_multiplier(
    metric: torch.Tensor,
    axis_and_values: tuple[str, torch.Tensor] | None,
    alpha: float = 1.0,
) -> torch.Tensor:
    if axis_and_values is None:
        return metric
    axis, values = axis_and_values
    values = values.to(device=metric.device, dtype=metric.dtype)
    if alpha != 1.0:
        values = (1.0 + alpha * (values - 1.0)).clamp_min(1e-8)
    if axis == "col":
        if values.numel() != metric.shape[1]:
            return metric
        return metric * values.view(1, -1)
    if axis == "row":
        if values.numel() != metric.shape[0]:
            return metric
        return metric * values.view(-1, 1)
    raise ValueError(f"Unsupported multiplier axis: {axis}")


def _apply_error_component(
    metric: torch.Tensor,
    axis_and_values: tuple[str, torch.Tensor] | None,
    alpha: float = 1.0,
) -> torch.Tensor:
    if axis_and_values is None:
        return metric
    axis, values = axis_and_values
    values = values.to(device=metric.device, dtype=metric.dtype).clamp_min(1e-8)
    if alpha != 1.0:
        values = values.pow(alpha)
    if axis == "col":
        if values.numel() != metric.shape[1]:
            return metric
        return metric * values.view(1, -1)
    if axis == "row":
        if values.numel() != metric.shape[0]:
            return metric
        return metric * values.view(-1, 1)
    raise ValueError(f"Unsupported error component axis: {axis}")


def _build_prune_mask(metric: torch.Tensor, sparsity_ratio: float, prune_n: int, prune_m: int, threshold_scope: str) -> torch.Tensor:
    if prune_n > 0:
        mask = torch.zeros_like(metric, dtype=torch.bool)
        for start in range(0, metric.shape[1], prune_m):
            chunk = metric[:, start : start + prune_m].float()
            if chunk.shape[1] < prune_m:
                continue
            idx = torch.topk(chunk, prune_n, dim=1, largest=False).indices
            mask.scatter_(1, start + idx, True)
        return mask

    k = int(metric.numel() * sparsity_ratio)
    if k <= 0:
        return torch.zeros_like(metric, dtype=torch.bool)
    if threshold_scope == "global":
        flat_idx = torch.topk(metric.flatten().float(), k, largest=False).indices
        mask = torch.zeros(metric.numel(), device=metric.device, dtype=torch.bool)
        mask[flat_idx] = True
        return mask.view_as(metric)
    if threshold_scope == "row":
        per_row = max(1, int(metric.shape[1] * sparsity_ratio))
        idx = torch.topk(metric.float(), per_row, dim=1, largest=False).indices
        mask = torch.zeros_like(metric, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask
    raise ValueError(f"Unsupported threshold_scope: {threshold_scope}")


def _parse_targets(text: str) -> set[str]:
    text = str(text).replace("+", ",").replace(";", ",")
    return {item.strip() for item in text.split(",") if item.strip()}


def _linear_is_target(short: str, targets: set[str]) -> bool:
    if short in targets or "all" in targets:
        return True
    if short in {"q_proj", "k_proj", "v_proj", "o_proj", "dense"} and "attn" in targets:
        return True
    if short in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"} and "mlp" in targets:
        return True
    if short in {"gate_proj", "up_proj", "fc1"} and "mlp_in" in targets:
        return True
    return False


def apply_geometry_pruning(model: nn.Module, scores: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    layers = _find_decoder_layers(model)
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = [int(x) for x in args.sparsity_type.split(":")]
        if not math.isclose(args.sparsity_ratio, 0.5, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("N:M sparsity currently expects --sparsity_ratio 0.5.")

    prune_target_text = getattr(args, "prune_targets", None) or getattr(args, "geometry_targets", "o_proj,down_proj,v_proj")
    geometry_target_text = getattr(args, "geometry_targets", "o_proj,down_proj,v_proj")
    prune_targets = _parse_targets(prune_target_text)
    geometry_targets = _parse_targets(geometry_target_text)
    geometry_mode = getattr(args, "geometry_mode", "residual_error")
    geometry_strategy = getattr(args, "geometry_strategy", "none")
    threshold_scope = getattr(args, "threshold_scope", "row")
    sparsity_type = getattr(args, "sparsity_type", "unstructured")
    if geometry_mode in {"residual_error", "weight_error"} and geometry_strategy != "none" and (
        threshold_scope == "row" or sparsity_type != "unstructured"
    ):
        warnings.warn(
            "weight_error/residual_error geometry decomposes the per-weight WANDA pruning error along output coordinates; "
            "with row-wise unstructured or N:M pruning the mask is invariant to row-constant geometry factors. "
            "Use threshold_scope=global for unstructured pruning, or geometry_mode=contribution for N:M pruning.",
            RuntimeWarning,
        )
    records = []
    for li, layer in enumerate(layers):
        for linear_name, linear in _iter_linear_modules(layer):
            short = linear_name.split(".")[-1]
            if not _linear_is_target(short, prune_targets):
                continue
            key = f"{li}.{linear_name}"
            weight = linear.weight.data
            metric = weight.detach().abs().float()
            if args.prune_method == "wanda":
                scaler = scores["wanda_inputs"].get(key)
                if scaler is None:
                    scaler = torch.ones(linear.in_features)
                scaler = scaler.to(metric.device).float().clamp_min(0.0).sqrt()
                metric = metric * scaler.view(1, -1)
            elif args.prune_method != "magnitude":
                raise ValueError(f"Unsupported prune_method: {args.prune_method}")

            base_metric = metric
            geometry_alpha = float(getattr(args, "geometry_alpha", 1.0))
            if geometry_mode == "contribution":
                multiplier = _geometry_multiplier_for_linear(
                    scores=scores,
                    layer_idx=li,
                    linear_name=linear_name,
                    linear=linear,
                    strategy=geometry_strategy,
                    targets=geometry_targets,
                )
                metric = _apply_metric_multiplier(metric, multiplier, alpha=geometry_alpha)
            elif geometry_mode in {"residual_error", "weight_error"}:
                multiplier = _geometry_error_multiplier_for_linear(
                    scores=scores,
                    layer_idx=li,
                    linear_name=linear_name,
                    linear=linear,
                    strategy=geometry_strategy,
                    targets=geometry_targets,
                )
                metric = _apply_error_component(metric, multiplier, alpha=geometry_alpha)
            else:
                raise ValueError(f"Unsupported geometry_mode: {geometry_mode}")
            base_mask = _build_prune_mask(base_metric, args.sparsity_ratio, prune_n, prune_m, threshold_scope)
            mask = _build_prune_mask(metric, args.sparsity_ratio, prune_n, prune_m, threshold_scope)
            weight[mask.to(weight.device)] = 0
            changed = int((mask != base_mask).sum().item())
            union = int((mask | base_mask).sum().item())
            records.append(
                {
                    "layer": li,
                    "linear": linear_name,
                    "pruned": int(mask.sum().item()),
                    "total": int(mask.numel()),
                    "sparsity": float(mask.float().mean().item()),
                    "geometry_axis": None if multiplier is None else multiplier[0],
                    "geometry_mode": geometry_mode,
                    "geometry_strategy": geometry_strategy,
                    "geometry_alpha": geometry_alpha,
                    "geometry_applied": multiplier is not None,
                    "mask_changed_vs_base": changed,
                    "mask_jaccard_with_base": float(((mask & base_mask).sum().item()) / union) if union else 1.0,
                }
            )
    return records


def _read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [args.prompt]
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        prompts = [line.strip() for line in lines if line.strip()]
        if prompts:
            return prompts[: args.max_prompts]
    return DEFAULT_PROMPTS[: args.max_prompts]


def _save_score_csv(scores: dict[str, Any], path: Path) -> None:
    rows = []
    for space in ("attn", "mlp", "output", "v_proj"):
        for metric_name, tensor in scores[space].items():
            for layer_idx, row in enumerate(tensor):
                for index, value in enumerate(row.tolist()):
                    rows.append(
                        {
                            "space": space,
                            "metric": metric_name,
                            "layer": layer_idx,
                            "index": index,
                            "value": float(value),
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["space", "metric", "layer", "index", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _safe_save_pretrained(model, tokenizer, output_dir: Path) -> None:
    """Save without importing DeepSpeed through accelerate's unwrap helper."""
    try:
        from transformers import modeling_utils

        modeling_utils.unwrap_model = lambda model, *_, **__: model
        if hasattr(modeling_utils, "extract_model_from_parallel"):
            modeling_utils.extract_model_from_parallel = lambda model, **_: model
    except Exception:
        pass
    try:
        from accelerate.utils import other as accelerate_other

        accelerate_other.extract_model_from_parallel = lambda model, **_: model
    except Exception:
        pass
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-aware WANDA/magnitude pruning for decoder-only LMs.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--max_prompts", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--token_scope", choices=["last", "all"], default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_remote_files", action="store_false", dest="local_files_only")
    parser.add_argument("--prune_method", choices=["wanda", "magnitude"], default="wanda")
    parser.add_argument(
        "--geometry_strategy",
        choices=[
            "none",
            "residual_perp",
            "residual_para",
            "residual_perp_over_para",
            "residual_perp_minus_para",
            "value_perp",
            "value_para",
            "value_perp_over_para",
            "value_perp_minus_para",
            "hybrid_perp",
        ],
        default="residual_perp",
    )
    parser.add_argument("--prune_targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--geometry_mode",
        choices=["contribution", "residual_error", "weight_error"],
        default="residual_error",
    )
    parser.add_argument("--geometry_alpha", type=float, default=1.0)
    parser.add_argument("--geometry_targets", default="o_proj,down_proj,v_proj")
    parser.add_argument("--sparsity_ratio", type=float, default=0.0)
    parser.add_argument("--sparsity_type", choices=["unstructured", "2:4", "4:8"], default="unstructured")
    parser.add_argument("--threshold_scope", choices=["row", "global"], default="global")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        model_tag = _model_tag(args.model_name_or_path)
        args.output_dir = f"compression/outputs/geo_prune/by_model/{model_tag}/{args.prune_method}_{args.geometry_strategy}"
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

    prompts = _read_prompts(args)
    collector = GeometryScoreCollector(model, token_scope=args.token_scope)
    collector.install()
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {key: value.to(args.device) for key, value in enc.items()}
            with torch.no_grad():
                model(**enc, use_cache=False)
    finally:
        collector.remove()

    scores = collector.to_tensors()
    score_path = out_dir / "geometry_scores.pt"
    torch.save(scores, score_path)
    _save_score_csv(scores, out_dir / "geometry_scores.csv")

    prune_records = []
    if args.sparsity_ratio > 0.0:
        prune_records = apply_geometry_pruning(model, scores, args)
        with (out_dir / "prune_records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "layer",
                    "linear",
                    "pruned",
                    "total",
                    "sparsity",
                    "geometry_axis",
                    "geometry_mode",
                    "geometry_strategy",
                    "geometry_alpha",
                    "geometry_applied",
                    "mask_changed_vs_base",
                    "mask_jaccard_with_base",
                ],
            )
            writer.writeheader()
            writer.writerows(prune_records)
        if args.save_model:
            _safe_save_pretrained(model, tokenizer, out_dir / "model")

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "num_prompts": len(prompts),
        "token_scope": args.token_scope,
        "prune_method": args.prune_method,
        "geometry_strategy": args.geometry_strategy,
        "prune_targets": args.prune_targets,
        "geometry_mode": args.geometry_mode,
        "geometry_alpha": args.geometry_alpha,
        "geometry_targets": args.geometry_targets,
        "sparsity_ratio": args.sparsity_ratio,
        "sparsity_type": args.sparsity_type,
        "threshold_scope": args.threshold_scope,
        "score_path": str(score_path),
        "pruned_modules": len(prune_records),
        "mean_module_sparsity": (
            float(sum(item["sparsity"] for item in prune_records) / len(prune_records)) if prune_records else 0.0
        ),
        "mask_changed_modules": sum(1 for item in prune_records if int(item.get("mask_changed_vs_base", 0)) > 0),
        "mask_changed_weights": sum(int(item.get("mask_changed_vs_base", 0)) for item in prune_records),
        "saved_model": bool(args.save_model and args.sparsity_ratio > 0.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
