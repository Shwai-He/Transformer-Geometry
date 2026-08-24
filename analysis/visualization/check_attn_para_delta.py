from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.forward_geometry.qwen_attn_x_parallel_removal_viz import (
    SHORT_TEXTS,
    collect_baseline_attention_value_and_residual,
    compute_x_reference_effective_diag_delta,
)
from analysis.forward_geometry.qwen_xsa_forward_ablation import (
    QwenXSAForwardHooks,
    _find_decoder_layers,
    _patch_transformers_tp_plan_check,
    _resolve_dtype,
)
from analysis.visualization.plot_head_aware_attn_diag_compare import _prepare_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Attn Para-Rem. effective delta against residual-output forward hook.")
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

    with torch.no_grad():
        baseline_bundle = collect_baseline_attention_value_and_residual(model, input_ids, attention_mask, int(args.layer))
        theory_delta, _baseline_mean, _ = compute_x_reference_effective_diag_delta(
            attn_probs=baseline_bundle["attention_layer"],
            value=baseline_bundle["value"],
            residual=baseline_bundle["residual"],
            attn_module=attn,
        )
    theory_diag = torch.diagonal(theory_delta[0].float(), dim1=-2, dim2=-1)

    captured: dict[str, torch.Tensor] = {}

    def layer_pre_hook(_mod, hook_args, hook_kwargs):
        hidden = hook_kwargs.get("hidden_states", hook_args[0] if hook_args else None)
        if isinstance(hidden, torch.Tensor):
            captured["residual"] = hidden.detach()

    def before_hook(_mod, _args, _kwargs, output):
        y = output[0] if isinstance(output, tuple) else output
        if isinstance(y, torch.Tensor):
            captured["before"] = y.detach()
        return output

    def after_hook(_mod, _args, _kwargs, output):
        y = output[0] if isinstance(output, tuple) else output
        if isinstance(y, torch.Tensor):
            captured["after"] = y.detach()
        return output

    handles = [
        layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True),
        attn.register_forward_hook(before_hook, with_kwargs=True),
    ]
    hook_ctx = QwenXSAForwardHooks(
        model,
        target="attn",
        start_layer=int(args.layer),
        end_layer=int(args.layer) + 1,
        intervention_site="residual_output",
        xsa_alpha=1.0,
        xsa_perp_scale=1.0,
        track_stats=True,
        track_layerwise_stats=True,
    )
    hook_ctx.attach()
    handles.append(attn.register_forward_hook(after_hook, with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        hook_ctx.close()
        for handle in handles:
            handle.remove()

    residual = captured["residual"].float()
    before = captured["before"].float()
    after = captured["after"].float()
    removed = before - after
    coeff = (removed * residual).sum(dim=-1) / residual.square().sum(dim=-1).clamp_min(1e-6)
    actual_delta = -coeff[0].detach().cpu()
    diff = theory_diag.cpu() - actual_delta
    payload = {
        "mode": {
            "XSA_ATTN_ATTR_SOURCE": os.environ.get("XSA_ATTN_ATTR_SOURCE"),
            "XSA_VALUE_REF_EXPANSION": os.environ.get("XSA_VALUE_REF_EXPANSION"),
        },
        "layer": int(args.layer),
        "shape": list(before.shape),
        "actual_forward_delta": {
            "mean": float(actual_delta.mean().item()),
            "min": float(actual_delta.min().item()),
            "max": float(actual_delta.max().item()),
            "abs_mean": float(actual_delta.abs().mean().item()),
            "abs_max": float(actual_delta.abs().max().item()),
        },
        "theory_effective_delta": {
            "mean": float(theory_diag.mean().item()),
            "min": float(theory_diag.min().item()),
            "max": float(theory_diag.max().item()),
            "abs_mean": float(theory_diag.abs().mean().item()),
            "abs_max": float(theory_diag.abs().max().item()),
        },
        "theory_minus_actual": {
            "mean_abs": float(diff.abs().mean().item()),
            "max_abs": float(diff.abs().max().item()),
            "relative_l2": float(torch.sqrt(diff.square().sum() / actual_delta.square().sum().clamp_min(1e-12)).item()),
        },
        "hook_stats": hook_ctx.stats.summary(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
