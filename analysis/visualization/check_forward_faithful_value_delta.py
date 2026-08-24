from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.forward_geometry.qwen_xsa_forward_ablation import (
    QwenXSAForwardHooks,
    _expand_attn_value_ref,
    _find_decoder_layers,
    _patch_transformers_tp_plan_check,
    _remove_parallel_attn_multihead,
    _resolve_dtype,
)
from analysis.visualization.plot_head_aware_attn_diag_compare import SHORT_TEXTS, _prepare_inputs


def _summarize_tensor(tensor: torch.Tensor) -> dict[str, float | int]:
    flat = tensor.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p90": math.nan, "p99": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _infer_heads(attn_module, width: int) -> tuple[int, int]:
    head_dim = int(getattr(attn_module, "head_dim"))
    num_heads = int(getattr(attn_module, "num_heads", getattr(attn_module, "num_attention_heads", 0)) or 0)
    if num_heads <= 0 and head_dim > 0 and width % head_dim == 0:
        num_heads = width // head_dim
    if num_heads <= 0 or width != num_heads * head_dim:
        raise RuntimeError(f"Cannot infer heads for width={width}, head_dim={head_dim}, num_heads={num_heads}")
    return num_heads, head_dim


def _project(y_pre: torch.Tensor, value: torch.Tensor, attn_module) -> tuple[torch.Tensor, torch.Tensor]:
    ref = _expand_attn_value_ref(value, y_pre, attn_module)
    if ref is None or ref.shape != y_pre.shape:
        raise RuntimeError(f"Bad ref expansion: value={tuple(value.shape)} y_pre={tuple(y_pre.shape)}")
    new_y_pre, _stats = _remove_parallel_attn_multihead(y_pre, ref, attn_module, alpha=1.0, perp_scale=1.0)
    proj = y_pre.float() - new_y_pre.float()
    num_heads, head_dim = _infer_heads(attn_module, y_pre.size(-1))
    r = ref.float().reshape(*ref.shape[:-1], num_heads, head_dim)
    z = proj.reshape(*proj.shape[:-1], num_heads, head_dim)
    coeff = (z * r).sum(dim=-1) / r.square().sum(dim=-1).clamp_min(1e-6)
    return proj, coeff


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate forward-faithful value delta against actual XSA hook output.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _patch_transformers_tp_plan_check()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_resolve_dtype(args.dtype),
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    input_ids, attention_mask = _prepare_inputs(tokenizer, SHORT_TEXTS[int(args.sample_idx)], args.max_length, args.device)
    layers = _find_decoder_layers(model)
    layer = layers[int(args.layer)]
    attn = layer.self_attn

    captured: dict[str, torch.Tensor] = {}

    def v_hook(_mod, _args, _kwargs, output):
        if isinstance(output, torch.Tensor):
            captured["value"] = output.detach()
        return output

    def before_hook(_mod, hook_args, hook_kwargs):
        y = hook_args[0] if hook_args and isinstance(hook_args[0], torch.Tensor) else hook_kwargs.get("input")
        if isinstance(y, torch.Tensor):
            captured["before"] = y.detach()
        return None

    def after_hook(_mod, hook_args, hook_kwargs):
        y = hook_args[0] if hook_args and isinstance(hook_args[0], torch.Tensor) else hook_kwargs.get("input")
        if isinstance(y, torch.Tensor):
            captured["after"] = y.detach()
        return None

    handles = [
        attn.v_proj.register_forward_hook(v_hook, with_kwargs=True),
        attn.o_proj.register_forward_pre_hook(before_hook, with_kwargs=True),
    ]
    hook_ctx = QwenXSAForwardHooks(
        model,
        target="attn",
        start_layer=int(args.layer),
        end_layer=int(args.layer) + 1,
        intervention_site="xsa_middle_multihead",
        xsa_alpha=1.0,
        xsa_perp_scale=1.0,
        track_stats=True,
        track_layerwise_stats=True,
    )
    hook_ctx.attach()
    handles.append(attn.o_proj.register_forward_pre_hook(after_hook, with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        hook_ctx.close()
        for handle in handles:
            handle.remove()

    before = captured["before"]
    after = captured["after"]
    value = captured["value"]
    actual_removed = before.float() - after.float()
    expected_removed, coeff = _project(before, value, attn)
    diff = actual_removed - expected_removed
    actual_sq = actual_removed.square().sum(dim=-1).clamp_min(1e-12)
    payload = {
        "mode": {
            "XSA_ATTN_ATTR_SOURCE": os.environ.get("XSA_ATTN_ATTR_SOURCE"),
            "XSA_VALUE_REF_EXPANSION": os.environ.get("XSA_VALUE_REF_EXPANSION"),
            "XSA_VALUE_PROJ_GROUP_SIZE": os.environ.get("XSA_VALUE_PROJ_GROUP_SIZE"),
        },
        "layer": int(args.layer),
        "shape": list(before.shape),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "relative_l2_diff": float(torch.sqrt(diff.square().sum() / actual_removed.square().sum().clamp_min(1e-12)).item()),
        "actual_removed_l2_mean": float(torch.sqrt(actual_sq).mean().item()),
        "expected_removed_l2_mean": float(torch.sqrt(expected_removed.square().sum(dim=-1).clamp_min(1e-12)).mean().item()),
        "coeff_mean": float(coeff.mean().item()),
        "coeff_min": float(coeff.min().item()),
        "coeff_max": float(coeff.max().item()),
        "coeff_abs": _summarize_tensor(coeff.abs()),
        "delta_full_removal_abs": _summarize_tensor(coeff.abs()),
        "xsa_stats_summary": hook_ctx.stats.summary(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
