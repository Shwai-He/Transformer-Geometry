from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XSA_ATTN_ATTR_SOURCE", "head_aware")
os.environ.setdefault("XSA_VALUE_REF_EXPANSION", "head_aware")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.forward_geometry.qwen_attn_x_parallel_removal_viz import (
    SHORT_TEXTS,
    compute_x_reference_effective_diag_delta,
)
from analysis.forward_geometry.qwen_xsa_forward_ablation import (
    _expand_attn_value_ref,
    _find_decoder_layers,
    _get_token_mixer_kind_and_module,
    _patch_transformers_tp_plan_check,
    _remove_parallel_attn_multihead,
    _require_transformers_version,
    _resolve_dtype,
)


def _prepare_inputs(tokenizer, text: str, max_length: int, device: str):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return encoded["input_ids"].to(device), encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(device)


def _collect_baseline_artifacts(model, input_ids, attention_mask, layer_idx: int) -> dict[str, torch.Tensor]:
    layers = _find_decoder_layers(model)
    layer = layers[layer_idx]
    mixer_kind, _ = _get_token_mixer_kind_and_module(layer)
    attn = getattr(layer, "self_attn", None)
    if mixer_kind != "full_attention" or attn is None:
        raise ValueError(f"Layer {layer_idx} is not a supported full-attention layer.")
    if getattr(attn, "v_proj", None) is None or getattr(attn, "o_proj", None) is None:
        raise ValueError(f"Layer {layer_idx} lacks v_proj/o_proj hooks needed for this plot.")

    captured: dict[str, torch.Tensor] = {}

    def layer_pre_hook(_mod, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if isinstance(hidden, torch.Tensor):
            captured["residual"] = hidden.detach()

    def v_proj_hook(_mod, _args, _kwargs, output):
        if isinstance(output, torch.Tensor):
            captured["value"] = output.detach()
        return output

    def o_proj_pre_hook(_mod, args, kwargs):
        y_pre = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("input")
        if isinstance(y_pre, torch.Tensor):
            captured["y_pre"] = y_pre.detach()
        return None

    def attn_out_hook(_mod, _args, _kwargs, output):
        y = output[0] if isinstance(output, tuple) else output
        if isinstance(y, torch.Tensor):
            captured["attn_out"] = y.detach()
        return output

    handles = [
        layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True),
        attn.v_proj.register_forward_hook(v_proj_hook, with_kwargs=True),
        attn.o_proj.register_forward_pre_hook(o_proj_pre_hook, with_kwargs=True),
        attn.register_forward_hook(attn_out_hook, with_kwargs=True),
    ]
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    attention_layer = None
    if outputs.attentions:
        if len(outputs.attentions) == len(layers):
            candidate = outputs.attentions[layer_idx]
            if isinstance(candidate, torch.Tensor):
                attention_layer = candidate.detach().float().cpu()
        else:
            full_attention_layers = [
                idx
                for idx, candidate_layer in enumerate(layers)
                if _get_token_mixer_kind_and_module(candidate_layer)[0] == "full_attention"
            ]
            if len(outputs.attentions) == len(full_attention_layers) and layer_idx in full_attention_layers:
                candidate = outputs.attentions[full_attention_layers.index(layer_idx)]
                if isinstance(candidate, torch.Tensor):
                    attention_layer = candidate.detach().float().cpu()

    required = ("value", "y_pre", "residual", "attn_out")
    missing = [name for name in required if not isinstance(captured.get(name), torch.Tensor)]
    if attention_layer is None:
        missing.append("attention_layer")
    if missing:
        raise RuntimeError(f"Missing captured artifact(s) for layer {layer_idx}: {missing}")

    return {
        "attention_layer": attention_layer,
        "value": captured["value"].detach().float().cpu(),
        "y_pre": captured["y_pre"].detach().float().cpu(),
        "residual": captured["residual"].detach().float().cpu(),
        "attn_out": captured["attn_out"].detach().float().cpu(),
    }


def _compute_forward_value_diag_delta(
    y_pre: torch.Tensor,
    value: torch.Tensor,
    attn_module,
    *,
    value_head_mode: str = "head_specific",
    delta_scale: float = 1.0,
) -> torch.Tensor:
    if y_pre.dim() != 3 or value.dim() != 3:
        raise ValueError(f"Expected y_pre/value as [batch, seq, hidden], got {tuple(y_pre.shape)} and {tuple(value.shape)}")
    ref = _expand_attn_value_ref(value, y_pre, attn_module)
    if ref is None or ref.shape != y_pre.shape:
        raise RuntimeError(
            f"Failed to expand value reference for forward-faithful delta: value={tuple(value.shape)} y_pre={tuple(y_pre.shape)}"
        )
    head_dim = int(getattr(attn_module, "head_dim"))
    num_heads = int(getattr(attn_module, "num_heads", getattr(attn_module, "num_attention_heads", 0)) or 0)
    if num_heads <= 0 and head_dim > 0 and y_pre.size(-1) % head_dim == 0:
        num_heads = int(y_pre.size(-1) // head_dim)
    if num_heads <= 0 or y_pre.size(-1) != num_heads * head_dim:
        raise RuntimeError(f"Cannot reshape y_pre={tuple(y_pre.shape)} into heads={num_heads}, head_dim={head_dim}")

    value_head_mode = str(value_head_mode or "head_specific").lower().strip()
    if value_head_mode in {"head_concat", "concat", "merged", "legacy"}:
        coeff = (y_pre.float() * ref.float()).sum(dim=-1) / ref.float().square().sum(dim=-1).clamp_min(1e-6)
        return torch.diag_embed(-(float(delta_scale) * coeff)[:, None, :].expand(-1, num_heads, -1)).cpu()

    proj_group = {
        "head_specific": "",
        "head_aware": "",
        "per_head": "",
        "kv_group": "kv",
        "kv": "kv",
        "kv2_group": "kv2",
        "kv2": "kv2",
        "kv4_group": "kv4",
        "kv4": "kv4",
        "kv8_group": "kv8",
        "kv8": "kv8",
    }.get(value_head_mode, "")
    old_proj_group = os.environ.get("XSA_VALUE_PROJ_GROUP_SIZE")
    try:
        os.environ["XSA_VALUE_PROJ_GROUP_SIZE"] = proj_group
        new_y_pre, _stats = _remove_parallel_attn_multihead(
            y_pre,
            ref,
            attn_module,
            alpha=float(delta_scale),
            perp_scale=1.0,
        )
    finally:
        if old_proj_group is None:
            os.environ.pop("XSA_VALUE_PROJ_GROUP_SIZE", None)
        else:
            os.environ["XSA_VALUE_PROJ_GROUP_SIZE"] = old_proj_group
    removed = y_pre.float() - new_y_pre.float()
    r = ref.float().reshape(*ref.shape[:-1], num_heads, head_dim).transpose(1, 2)
    z = removed.reshape(*removed.shape[:-1], num_heads, head_dim).transpose(1, 2)
    coeff = (z * r).sum(dim=-1) / r.square().sum(dim=-1).clamp_min(1e-6)
    return torch.diag_embed(-coeff).cpu()


def _compute_forward_attn_diag_delta(attn_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if attn_out.shape != residual.shape:
        raise ValueError(f"attn_out/residual shape mismatch: {tuple(attn_out.shape)} vs {tuple(residual.shape)}")
    coeff = (attn_out.float() * residual.float()).sum(dim=-1) / residual.float().square().sum(dim=-1).clamp_min(1e-6)
    return torch.diag_embed(-coeff).cpu()


def _plot_matrix(ax, matrix: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, mark_diag: bool = True):
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if mark_diag:
        n = matrix.shape[0]
        for i in range(n):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=0.25, alpha=0.45))
    return im


def _linear_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    return [float(x) for x in np.linspace(vmin, vmax, n)]


def _save_three_panel(
    baseline: torch.Tensor,
    residual_delta: torch.Tensor,
    value_delta: torch.Tensor,
    out_path: Path,
    title_suffix: str,
) -> None:
    base = baseline.numpy()
    rdelta = residual_delta.numpy()
    vdelta = value_delta.numpy()
    delta_abs = float(max(np.max(np.abs(rdelta)), np.max(np.abs(vdelta)), 1e-6))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    im0 = _plot_matrix(axes[0], base, f"Baseline Attention\n{title_suffix}", "viridis", 0.0, max(float(base.max()), 1e-6))
    im1 = _plot_matrix(axes[1], rdelta, "Attn Para-Rem. Delta", "coolwarm", -delta_abs, delta_abs)
    im2 = _plot_matrix(axes[2], vdelta, "V-Para Rem. Delta", "coolwarm", -delta_abs, delta_abs)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1:], fraction=0.046, pad=0.04, ticks=_linear_ticks(-delta_abs, delta_abs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_three_panel_split_scale(
    baseline: torch.Tensor,
    residual_delta: torch.Tensor,
    value_delta: torch.Tensor,
    out_path: Path,
    title_suffix: str,
) -> None:
    base = baseline.numpy()
    rdelta = residual_delta.numpy()
    vdelta = value_delta.numpy()
    r_abs = float(max(np.max(np.abs(rdelta)), 1e-6))
    v_abs = float(max(np.max(np.abs(vdelta)), 1e-6))

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), constrained_layout=True)
    im0 = _plot_matrix(axes[0], base, f"Baseline Attention\n{title_suffix}", "viridis", 0.0, max(float(base.max()), 1e-6))
    im1 = _plot_matrix(axes[1], rdelta, "Attn Para-Rem. Delta", "coolwarm", -r_abs, r_abs)
    im2 = _plot_matrix(axes[2], vdelta, "V-Para Rem. Delta", "coolwarm", -v_abs, v_abs)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=_linear_ticks(-r_abs, r_abs))
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=_linear_ticks(-v_abs, v_abs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_head_panel(
    baseline_head: torch.Tensor,
    value_delta_head: torch.Tensor,
    out_path: Path,
    layer: int,
    head: int,
) -> None:
    base = baseline_head.numpy()
    delta = value_delta_head.numpy()
    compare = base + delta
    delta_abs = float(max(np.max(np.abs(delta)), 1e-6))
    vmax = float(max(base.max(), compare.max(), 1e-6))

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), constrained_layout=True)
    im0 = _plot_matrix(axes[0], base, f"L{layer} H{head} Baseline", "viridis", 0.0, vmax)
    im1 = _plot_matrix(axes[1], compare, "Effective V-Para Rem.", "viridis", 0.0, vmax)
    im2 = _plot_matrix(axes[2], delta, "V-Para Rem. Delta", "coolwarm", -delta_abs, delta_abs)
    diag = np.diag(compare)
    offdiag = compare.sum(axis=-1) - diag
    axes[3].plot(diag, label="diag", linewidth=1.8)
    axes[3].plot(offdiag, label="offdiag row-sum", linewidth=1.8)
    axes[3].set_title("Effective Row Sums", fontsize=11)
    axes[3].set_xlabel("Query token")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(fontsize=8)
    fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=_linear_ticks(-delta_abs, delta_abs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _diag_stats(matrix: torch.Tensor) -> dict[str, float]:
    diag = torch.diagonal(matrix.float(), dim1=-2, dim2=-1)
    return {
        "diag_mean": float(diag.mean().item()),
        "diag_min": float(diag.min().item()),
        "diag_max": float(diag.max().item()),
        "matrix_min": float(matrix.float().min().item()),
        "matrix_max": float(matrix.float().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot forward-faithful effective attention diagonal deltas.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--head", type=int, default=6)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--delta_scale", type=float, default=1.0, help="Scale applied to effective diagonal deltas.")
    parser.add_argument(
        "--value_head_mode",
        default="head_specific",
        choices=[
            "head_specific",
            "head_concat",
            "kv_group",
            "kv2_group",
            "kv4_group",
            "kv8_group",
            "legacy",
            "merged",
            "concat",
        ],
    )
    parser.add_argument("--output_dir", default="analysis/outputs/head_aware_attn_diag/qwen3_4b/layer19")
    args = parser.parse_args()

    _require_transformers_version(args.model_name)
    _patch_transformers_tp_plan_check()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_resolve_dtype(args.dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()

    text = SHORT_TEXTS[int(args.sample_idx)]
    input_ids, attention_mask = _prepare_inputs(tokenizer, text, args.max_length, args.device)
    layers = _find_decoder_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"layer={args.layer} is out of range for {len(layers)} layers")
    attn_module = getattr(layers[args.layer], "self_attn")

    with torch.no_grad():
        bundle = _collect_baseline_artifacts(model, input_ids, attention_mask, args.layer)
        baseline_layer = bundle["attention_layer"]
        value = bundle["value"]
        y_pre = bundle["y_pre"]
        residual = bundle["residual"]
        attn_out = bundle["attn_out"]
        residual_delta = float(args.delta_scale) * _compute_forward_attn_diag_delta(attn_out=attn_out, residual=residual)
        baseline_mean = baseline_layer.mean(dim=1)
        value_delta = _compute_forward_value_diag_delta(
            y_pre=y_pre,
            value=value,
            attn_module=attn_module,
            value_head_mode=args.value_head_mode,
            delta_scale=float(args.delta_scale),
        )

    head = int(args.head)
    if head < 0 or head >= baseline_layer.shape[1]:
        raise ValueError(f"head={head} is out of range for {baseline_layer.shape[1]} heads")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = str(args.value_head_mode).replace("_", "-")
    merged_path = output_dir / f"layer{args.layer:02d}_merged_three_map_{mode_tag}.pdf"
    merged_split_path = output_dir / f"layer{args.layer:02d}_merged_three_map_{mode_tag}_split_scale.pdf"
    head_path = output_dir / f"layer{args.layer:02d}_head{head:02d}_value_{mode_tag}.pdf"

    merged_value_delta = value_delta.mean(dim=1)[0]
    _save_three_panel(
        baseline=baseline_mean[0],
        residual_delta=residual_delta[0],
        value_delta=merged_value_delta,
        out_path=merged_path,
        title_suffix=f"L{args.layer} head-avg",
    )
    _save_three_panel_split_scale(
        baseline=baseline_mean[0],
        residual_delta=residual_delta[0],
        value_delta=merged_value_delta,
        out_path=merged_split_path,
        title_suffix=f"L{args.layer} head-avg",
    )
    _save_head_panel(
        baseline_head=baseline_layer[0, head],
        value_delta_head=value_delta[0, head],
        out_path=head_path,
        layer=args.layer,
        head=head,
    )

    summary: dict[str, Any] = {
        "model_name": args.model_name,
        "layer": int(args.layer),
        "head": int(head),
        "sample_idx": int(args.sample_idx),
        "num_tokens": int(input_ids.shape[-1]),
        "attn_attr_source": os.environ.get("XSA_ATTN_ATTR_SOURCE"),
        "value_ref_expansion": os.environ.get("XSA_VALUE_REF_EXPANSION"),
        "value_proj_group_size": os.environ.get("XSA_VALUE_PROJ_GROUP_SIZE"),
        "delta_scale": float(args.delta_scale),
        "merged_pdf": str(merged_path),
        "merged_split_scale_pdf": str(merged_split_path),
        "head_pdf": str(head_path),
        "baseline_mean": _diag_stats(baseline_mean[0]),
        "attn_para_rem_delta": _diag_stats(residual_delta[0]),
        "value_para_rem_delta_mean_heads": _diag_stats(merged_value_delta),
        "value_para_rem_delta_head": _diag_stats(value_delta[0, head]),
        "attn_delta_definition": "-<attn_out_i, residual_i>/||residual_i||^2, using the same post-o_proj branch output as residual_output forward hook",
        "value_delta_definition": "-<y_pre_i, ref_i>/||ref_i||^2, using the same ref expansion as the forward hook",
    }
    summary_path = output_dir / f"layer{args.layer:02d}_head{head:02d}_head_aware_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
