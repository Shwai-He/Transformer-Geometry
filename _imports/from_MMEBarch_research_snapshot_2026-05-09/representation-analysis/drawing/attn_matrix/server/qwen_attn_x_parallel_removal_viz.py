from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_xsa_forward_ablation import (
    QwenXSAForwardHooks,
    _expand_attn_value_ref,
    _find_decoder_layers,
    _get_token_mixer_kind_and_module,
    _patch_transformers_tp_plan_check,
    _require_transformers_version,
    _resolve_dtype,
)
from nanogpt_attention_utils import load_nanogpt_checkpoint, nanogpt_xsa_context


SHORT_TEXTS = [
    "Echo the final code exactly once: AX7Q-19",
    "Alice gave Bob the red key. Bob gave the red key to Clara. Who has the red key now?",
    "Finish the shortest valid continuation: The reviewer wrote, \"only the diagonal changed because",
    "Repeat the final item once: red, blue, red, blue, red,",
    "Update only the third item: A1 B2 C3 D4 ->",
    "Sarah told Mina that she would rerun the tests. If she means Sarah, who reruns the tests?",
]


SHARED_VALUE_CMAP = "viridis"


def _to_jsonable_plot_value(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    arr = np.asarray(value)
    return arr.tolist()


def _save_plot_bundle_json(out_path: Path, **arrays) -> Path:
    payload = {key: _to_jsonable_plot_value(value) for key, value in arrays.items() if value is not None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    return out_path


def _save_plot_data_tsv_bundle(base_dir: Path, **arrays) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"fields": {}}
    for key, value in arrays.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            arr = value.detach().float().cpu().numpy()
        else:
            arr = np.asarray(value)
        out_path = base_dir / f"{key}.tsv"
        if arr.ndim == 0:
            with out_path.open("w", encoding="utf-8") as f:
                f.write("value\n")
                f.write(f"{arr.item()}\n")
            columns = ["value"]
            kind = "scalar"
        elif arr.ndim == 1:
            with out_path.open("w", encoding="utf-8") as f:
                if arr.dtype.kind in {"U", "S", "O"}:
                    f.write("index	label\n")
                    for i, item in enumerate(arr.tolist()):
                        label = str(item).replace("\t", " ").replace("\n", "\\n")
                        f.write(f"{i}	{label}\n")
                    columns = ["index", "label"]
                    kind = "vector_text"
                else:
                    f.write("index	value\n")
                    for i, item in enumerate(arr.tolist()):
                        f.write(f"{i}	{float(item)}\n")
                    columns = ["index", "value"]
                    kind = "vector"
        elif arr.ndim == 2:
            with out_path.open("w", encoding="utf-8") as f:
                f.write("row	col	value\n")
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        f.write(f"{i}	{j}	{float(arr[i, j])}\n")
            columns = ["row", "col", "value"]
            kind = "matrix"
        elif arr.ndim == 3:
            with out_path.open("w", encoding="utf-8") as f:
                f.write("dim0	dim1	dim2	value\n")
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        for k in range(arr.shape[2]):
                            f.write(f"{i}	{j}	{k}	{float(arr[i, j, k])}\n")
            columns = ["dim0", "dim1", "dim2", "value"]
            kind = "tensor3"
        else:
            raise ValueError(f"Unsupported ndim for {key}: {arr.ndim}")
        manifest["fields"][key] = {
            "path": out_path.name,
            "kind": kind,
            "shape": list(arr.shape),
            "columns": columns,
        }
    manifest_path = base_dir / "plot_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _read_jsonl(path: str, text_key: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_key)
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def load_texts(args) -> List[str]:
    if args.jsonl_path:
        texts = _read_jsonl(args.jsonl_path, args.text_key)
    else:
        texts = list(SHORT_TEXTS)
    if args.max_samples > 0:
        texts = texts[: args.max_samples]
    if not texts:
        raise ValueError("No texts loaded for x-parallel-removal visualization.")
    return texts


def _format_prompt(tokenizer, text: str, args) -> str:
    if not args.use_chat_template:
        return text
    if not getattr(tokenizer, "chat_template", None):
        return text
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _prepare_inputs(tokenizer, text: str, args):
    prompt = _format_prompt(tokenizer, text, args)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    return prompt, input_ids, attention_mask


def _trim_token(tok: str, max_chars: int = 12) -> str:
    tok = tok.replace("\n", "\\n")
    if len(tok) <= max_chars:
        return tok
    return tok[: max_chars - 1] + "…"


def _overlay_diagonal_boxes(ax, n: int, color: str = "black", linewidth: float = 0.6, alpha: float = 0.55) -> None:
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=max(1.0, linewidth + 0.4),
                alpha=max(0.75, alpha),
            )
        )
        ax.add_patch(
            Rectangle(
                (i - 0.5 + 0.03, i - 0.5 + 0.03),
                0.94,
                0.94,
                fill=False,
                edgecolor="white",
                linewidth=max(0.8, linewidth),
                alpha=0.9,
            )
        )


def _plot_matrix(
    ax,
    matrix: torch.Tensor,
    title: str,
    labels: List[str],
    cmap: str,
    vmin=None,
    vmax=None,
    norm=None,
    diagonal_color: str = "black",
):
    arr = matrix.detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        upper_mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
        arr = np.ma.array(arr, mask=upper_mask)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, "copy") else cmap_obj
        cmap_obj.set_bad(color="#e6e6e6")
    else:
        cmap_obj = cmap
    im = ax.imshow(arr, cmap=cmap_obj, aspect="auto", vmin=vmin, vmax=vmax, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    _overlay_diagonal_boxes(ax, len(labels), color=diagonal_color)
    matrix_min = float(matrix.min().item())
    matrix_max = float(matrix.max().item())
    ax.text(
        0.99,
        0.01,
        f"min={matrix_min:.3g}\nmax={matrix_max:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.3),
    )
    return im


def _plot_shared_value_matrix(
    ax,
    matrix: torch.Tensor,
    title: str,
    labels: List[str],
    shared_vmin: float,
    shared_vmax: float,
    diagonal_color: str,
):
    matrix_min = float(matrix.min().item())
    matrix_max = float(matrix.max().item())
    if matrix_min < 0.0:
        signed_max = max(abs(matrix_min), abs(matrix_max))
        return _plot_matrix(
            ax,
            matrix,
            title,
            labels,
            'coolwarm',
            -signed_max,
            signed_max,
            diagonal_color=diagonal_color,
        )
    return _plot_matrix(
        ax,
        matrix,
        title,
        labels,
        SHARED_VALUE_CMAP,
        shared_vmin,
        shared_vmax,
        diagonal_color=diagonal_color,
    )


def _plot_diag_offdiag_curves(ax, matrix: torch.Tensor, title: str) -> None:
    diag = torch.diagonal(matrix)
    offdiag_sum = matrix.sum(dim=-1) - diag
    total_sum = matrix.sum(dim=-1)
    xs = list(range(diag.shape[0]))
    ax.plot(xs, diag.detach().cpu().tolist(), color="tab:orange", linewidth=1.5, label="diag")
    ax.plot(xs, offdiag_sum.detach().cpu().tolist(), color="tab:blue", linewidth=1.5, label="offdiag-sum")
    ax.plot(xs, total_sum.detach().cpu().tolist(), color="tab:green", linewidth=1.5, label="total-sum")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("token", fontsize=8)
    ax.set_ylabel("value", fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7, loc="best", frameon=True, facecolor="white", edgecolor="#cfcfcf")


@torch.no_grad()
def _coerce_attention_tensor(candidate):
    if isinstance(candidate, torch.Tensor):
        return candidate.detach().float().cpu()
    if isinstance(candidate, (tuple, list)):
        for item in candidate:
            if isinstance(item, torch.Tensor) and item.dim() >= 4:
                return item.detach().float().cpu()
    return None


def _extract_attention_tensor(outputs_attentions, layers, flip_layer: int):
    if not outputs_attentions:
        return None
    attention_items = list(outputs_attentions)

    if len(attention_items) == len(layers):
        candidate = _coerce_attention_tensor(attention_items[flip_layer])
        if candidate is not None:
            return candidate

    full_attention_layers = [
        idx
        for idx, candidate in enumerate(layers)
        if _get_token_mixer_kind_and_module(candidate)[0] == "full_attention"
    ]
    if flip_layer not in full_attention_layers:
        return None

    ordered_candidates = [_coerce_attention_tensor(item) for item in attention_items]
    non_null_candidates = [item for item in ordered_candidates if item is not None]
    full_idx = full_attention_layers.index(flip_layer)

    if len(attention_items) == len(full_attention_layers):
        candidate = ordered_candidates[full_idx]
        if candidate is not None:
            return candidate

    if len(non_null_candidates) == len(full_attention_layers):
        return non_null_candidates[full_idx]

    return None


def collect_attentions(model, input_ids, attention_mask, args, use_removal: bool):
    if args.nanogpt_ckpt:
        with nanogpt_xsa_context(
            model,
            enable=use_removal,
            ref="residual",
            space="post_o_proj",
            target="attn",
            start_layer=args.flip_layer,
            end_layer=args.flip_layer,
            single_layer=args.flip_layer,
        ):
            outputs = model(input_ids, use_cache=False, output_attentions=True, return_dict=True)
        return outputs.attentions[args.flip_layer].detach().float().cpu()
    hook_ctx = QwenXSAForwardHooks(
        model,
        target="attn",
        start_layer=args.flip_layer,
        end_layer=args.flip_layer + 1,
        skip_first_n=0,
        skip_last_n=0,
        intervention_site="residual_output",
        track_stats=False,
        track_layerwise_stats=False,
    ) if use_removal else None

    if hook_ctx is None:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    else:
        with hook_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    layers = _find_decoder_layers(model)
    return _extract_attention_tensor(outputs.attentions, layers, args.flip_layer)


@torch.no_grad()
def collect_baseline_attention_value_and_residual(model, input_ids, attention_mask, flip_layer: int):
    layers = _find_decoder_layers(model)
    layer = layers[flip_layer]
    mixer_kind, mixer = _get_token_mixer_kind_and_module(layer)
    attn = getattr(layer, "self_attn", None)
    if mixer_kind != "full_attention" or attn is None or getattr(attn, "v_proj", None) is None:
        return {
            "mixer_kind": mixer_kind,
            "supported": False,
            "attention_layer": None,
            "value": None,
            "residual": None,
        }

    captured: Dict[str, torch.Tensor] = {}

    def layer_pre_hook(_mod, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if isinstance(hidden, torch.Tensor):
            captured["residual"] = hidden.detach()

    def v_proj_hook(_mod, _args, _kwargs, output):
        if isinstance(output, torch.Tensor):
            captured["value"] = output.detach()
        return output

    layer_handle = layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True)
    v_handle = attn.v_proj.register_forward_hook(v_proj_hook, with_kwargs=True)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        v_handle.remove()
        layer_handle.remove()

    attention_layer = _extract_attention_tensor(outputs.attentions, layers, flip_layer)
    value = captured.get("value")
    residual = captured.get("residual")
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Failed to capture v_proj output for layer {flip_layer}.")
    if not isinstance(residual, torch.Tensor):
        raise RuntimeError(f"Failed to capture residual input for layer {flip_layer}.")
    return {
        "mixer_kind": mixer_kind,
        "supported": True,
        "attention_layer": attention_layer,
        "value": value.detach().float().cpu(),
        "residual": residual.detach().float().cpu(),
    }


def compute_x_reference_effective_diag_delta(
    attn_probs: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    attn_module,
) -> torch.Tensor:
    if attn_probs.dim() != 4:
        raise ValueError(f"attn_probs must be [batch, heads, q, k], got shape {tuple(attn_probs.shape)}")
    if value.dim() != 3:
        raise ValueError(f"value must be [batch, seq, hidden], got shape {tuple(value.shape)}")
    if residual.dim() != 3:
        raise ValueError(f"residual must be [batch, seq, hidden], got shape {tuple(residual.shape)}")

    batch_size, num_heads, seq_len, _ = attn_probs.shape
    head_dim = int(getattr(attn_module, "head_dim"))
    full_width = num_heads * head_dim
    dummy_y = torch.zeros(batch_size, seq_len, full_width, dtype=value.dtype, device=value.device)
    ref = _expand_attn_value_ref(value, dummy_y, attn_module)
    if ref is None:
        raise RuntimeError("Failed to expand value tensor to per-head reference.")

    ref_heads = ref.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    o_proj = getattr(attn_module, "o_proj", None)
    if o_proj is None:
        raise RuntimeError("Attention module has no o_proj; cannot compute post-W_O token basis.")

    y_basis = ref.reshape(batch_size, seq_len, num_heads, head_dim).reshape(batch_size, seq_len, full_width)
    u = o_proj(y_basis.to(dtype=o_proj.weight.dtype, device=o_proj.weight.device)).float().cpu()

    attn_mean = attn_probs.mean(dim=1)
    dot_j_i = torch.einsum("bjd,bid->bij", u, residual)
    denom_i = residual.square().sum(dim=-1).clamp_min(1e-6).unsqueeze(-1)
    alpha = dot_j_i / denom_i
    c = (attn_mean.float() * alpha).sum(dim=-1)
    diag_delta = -c
    merged_delta = torch.diag_embed(diag_delta).cpu()

    basis_heads = torch.zeros(batch_size, num_heads, seq_len, full_width, dtype=ref.dtype, device=ref.device)
    for head_idx in range(num_heads):
        start = head_idx * head_dim
        end = start + head_dim
        basis_heads[:, head_idx, :, start:end] = ref_heads[:, head_idx]
    basis_heads = basis_heads.reshape(batch_size * num_heads * seq_len, full_width)
    u_heads = o_proj(basis_heads.to(dtype=o_proj.weight.dtype, device=o_proj.weight.device))
    u_heads = u_heads.reshape(batch_size, num_heads, seq_len, -1).float().cpu()
    dot_h_j_i = torch.einsum("bhjd,bid->bhij", u_heads, residual)
    alpha_heads = dot_h_j_i / denom_i.unsqueeze(1)
    c_heads = (attn_probs.float() * alpha_heads).sum(dim=-1)
    diag_delta_heads = -c_heads
    head_delta = torch.diag_embed(diag_delta_heads).cpu()
    return merged_delta, attn_mean.cpu(), u.cpu(), head_delta


def save_merged_heatmap(
    baseline: torch.Tensor,
    compare: torch.Tensor,
    delta: torch.Tensor,
    out_path: Path,
    labels: List[str],
    baseline_title: str,
    compare_title: str,
    delta_title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    shared_vmax = float(max(baseline.max().item(), compare.max().item()))
    print(
        f"[INFO] merged baseline_range=({baseline.min().item():.6g}, {baseline.max().item():.6g}) "
        f"compare_range=({compare.min().item():.6g}, {compare.max().item():.6g}) "
        f"shared_positive_range=(0, {shared_vmax:.6g}) "
        f"delta_range=({delta.min().item():.6g}, {delta.max().item():.6g})",
        flush=True,
    )
    delta_abs = float(delta.abs().max().item())
    ims = [
        _plot_shared_value_matrix(
            axes[0],
            baseline,
            baseline_title,
            labels,
            0.0,
            shared_vmax,
            diagonal_color="white",
        ),
        _plot_shared_value_matrix(
            axes[1],
            compare,
            compare_title,
            labels,
            0.0,
            shared_vmax,
            diagonal_color="white",
        ),
        _plot_matrix(axes[2], delta, delta_title, labels, "coolwarm", -delta_abs, delta_abs, diagonal_color="black"),
    ]
    for ax, im in zip(axes[:3], ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    curve_ax = axes[3]
    _plot_diag_offdiag_curves(curve_ax, compare, f"{compare_title}: diag/offdiag")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_single_flip(
    model,
    raw_text: str,
    prompt: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_labels: List[str],
    args,
    flip_layer: int,
    sweep_dir: Path | None = None,
) -> Dict[str, object]:
    print(f"[INFO] Starting x-reference single-layer analysis for layer={flip_layer}", flush=True)
    baseline_bundle = collect_baseline_attention_value_and_residual(
        model,
        input_ids,
        attention_mask,
        flip_layer,
    )
    mixer_kind = str(baseline_bundle["mixer_kind"])
    if not bool(baseline_bundle["supported"]):
        note = (
            f"Layer {flip_layer} uses {mixer_kind}. This script computes x-reference effective deltas from "
            "self_attn.v_proj and only supports full-attention layers with explicit attention matrices."
        )
        output_prefix = Path(args.output_prefix)
        if sweep_dir is not None:
            json_path = sweep_dir / f"layer_{flip_layer:02d}_summary.json"
        else:
            run_dir = output_prefix.with_name(f"{output_prefix.name}-sample{args.sample_idx:02d}-flip{flip_layer:02d}")
            json_path = run_dir / "summary.json"
        payload = {
            "model_name": args.model_id,
            "raw_text": raw_text,
            "rendered_prompt": prompt,
            "sample_idx": args.sample_idx,
            "flip_layer": int(flip_layer),
            "intervention_site": "residual_output",
            "status": "skipped_no_self_attn_v_proj",
            "token_mixer_kind": mixer_kind,
            "note": note,
            "tokens": token_labels,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] flip_layer={flip_layer} skipped: {note}", flush=True)
        return payload

    baseline_layer = baseline_bundle["attention_layer"]
    baseline_value = baseline_bundle["value"]
    baseline_residual = baseline_bundle["residual"]
    xrm_layer = collect_attentions(model, input_ids, attention_mask, args, use_removal=True)
    if baseline_layer is None:
        note = (
            f"Layer {flip_layer} uses {mixer_kind}. This script needs an explicit baseline attention matrix, "
            "but the model did not surface one for this layer."
        )
        output_prefix = Path(args.output_prefix)
        if sweep_dir is not None:
            json_path = sweep_dir / f"layer_{flip_layer:02d}_summary.json"
        else:
            run_dir = output_prefix.with_name(f"{output_prefix.name}-sample{args.sample_idx:02d}-flip{flip_layer:02d}")
            json_path = run_dir / "summary.json"
        payload = {
            "model_name": args.model_id,
            "raw_text": raw_text,
            "rendered_prompt": prompt,
            "sample_idx": args.sample_idx,
            "flip_layer": int(flip_layer),
            "intervention_site": "residual_output",
            "status": "skipped_no_baseline_attention_matrix",
            "token_mixer_kind": mixer_kind,
            "note": note,
            "tokens": token_labels,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] flip_layer={flip_layer} skipped: {note}", flush=True)
        return payload
    if xrm_layer is None:
        print(
            f"[WARN] flip_layer={flip_layer} baseline attention is available but x-reference-removal attention "
            "was not surfaced; continuing with effective-diagonal visualization only.",
            flush=True,
        )

    layers = _find_decoder_layers(model)
    effective_delta, baseline_mean, post_wo_basis, effective_delta_heads = compute_x_reference_effective_diag_delta(
        attn_probs=baseline_layer,
        value=baseline_value,
        residual=baseline_residual,
        attn_module=getattr(layers[flip_layer], "self_attn"),
    )
    xrm_mean = xrm_layer.mean(dim=1) if xrm_layer is not None else None
    effective_compare = baseline_mean + effective_delta

    output_prefix = Path(args.output_prefix)
    if sweep_dir is not None:
        run_dir = sweep_dir
        if int(args.head) >= 0:
            image_path = sweep_dir / f"layer_{flip_layer:02d}_head{int(args.head):02d}_flip.png"
            plot_bundle_path = sweep_dir / f"layer_{flip_layer:02d}_head{int(args.head):02d}_plot_bundle.json"
        else:
            image_path = sweep_dir / f"layer_{flip_layer:02d}_flip.png"
            plot_bundle_path = sweep_dir / f"layer_{flip_layer:02d}_plot_bundle.json"
    else:
        if int(args.head) >= 0:
            stem = f"{output_prefix.name}-sample{args.sample_idx:02d}-flip{flip_layer:02d}-head{int(args.head):02d}"
        else:
            stem = f"{output_prefix.name}-sample{args.sample_idx:02d}-flip{flip_layer:02d}"
        run_dir = output_prefix.with_name(stem)
        image_path = run_dir / "layer_flip.png"
        plot_bundle_path = run_dir / "plot_bundle.json"

    if int(args.head) >= 0:
        flip_head = int(args.head)
        baseline_panel = baseline_layer[0, flip_head]
        compare_panel = baseline_layer[0, flip_head] + effective_delta_heads[0, flip_head]
        delta_panel = effective_delta_heads[0, flip_head]
        panel_title = f"L{flip_layer} H{flip_head}"
        raw_compare = xrm_layer[0, flip_head] if xrm_layer is not None else None
    else:
        flip_head = None
        baseline_panel = baseline_mean[0]
        compare_panel = effective_compare[0]
        delta_panel = effective_delta[0]
        panel_title = f"L{flip_layer} merged"
        raw_compare = xrm_mean[0] if xrm_mean is not None else None

    save_merged_heatmap(
        baseline=baseline_panel,
        compare=compare_panel,
        delta=delta_panel,
        out_path=image_path,
        labels=token_labels,
        baseline_title=f"{panel_title} baseline",
        compare_title=f"{panel_title} effective x-removal",
        delta_title="x-reference effective delta",
    )

    compare_diag = torch.diagonal(compare_panel)
    compare_offdiag_sum = compare_panel.sum(dim=-1) - compare_diag
    compare_total_sum = compare_panel.sum(dim=-1)

    plot_bundle_json_path = _save_plot_bundle_json(
        plot_bundle_path,
        panel1_baseline=baseline_panel,
        panel2_effective_compare=compare_panel,
        panel3_effective_delta=delta_panel,
        panel4_diag=compare_diag,
        panel4_offdiag_sum=compare_offdiag_sum,
        panel4_total_sum=compare_total_sum,
        raw_x_removal_mean=xrm_mean[0] if xrm_mean is not None else None,
        baseline_attention_layer=baseline_layer[0],
        raw_x_removal_attention_layer=xrm_layer[0] if xrm_layer is not None else None,
        raw_x_removal_selected=raw_compare,
        effective_delta_layer=effective_delta_heads[0] if int(args.head) >= 0 else effective_delta[0],
        token_labels=token_labels,
    )

    payload = {
        "model_name": args.model_id,
        "sample_idx": args.sample_idx,
        "flip_layer": int(flip_layer),
        "flip_head": flip_head,
        "intervention_site": "residual_output",
        "attention_view": "single_head_raw_attention" if flip_head is not None else "head_averaged_raw_attention",
        "effective_reference": "residual_x",
        "raw_x_removal_attention_available": xrm_layer is not None,
        "note": (
            "X-reference attention-parallel view. plot_bundle.json stores raw values before any plotting color remapping."
        ),
        "image_path": str(image_path),
        "plot_bundle_path": str(plot_bundle_json_path),
    }
    print(f"[INFO] flip_layer={flip_layer} saved merged heatmap to {image_path}", flush=True)
    print(f"[INFO] flip_layer={flip_layer} saved plot bundle to {plot_bundle_json_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize x-reference post-W_O attention parallel removal.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--nanogpt_ckpt", type=str, default=None)
    parser.add_argument("--nanogpt_repo_root", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--flip_layer", type=int, required=True, help="Single decoder layer to flip. Use -1 to sweep all layers.")
    parser.add_argument("--head", type=int, default=-1, help="Head to visualize for single-layer runs; -1 keeps merged/averaged view.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="representation-analysis/outputs/qwen_attn_x_parallel_removal",
    )
    args = parser.parse_args()

    if bool(args.model_name) == bool(args.nanogpt_ckpt):
        raise ValueError("Provide exactly one of --model_name or --nanogpt_ckpt")

    if args.nanogpt_ckpt:
        print(f"[INFO] Loading nanoGPT checkpoint from {args.nanogpt_ckpt}", flush=True)
        model, tokenizer, _, _ = load_nanogpt_checkpoint(
            ckpt_path=args.nanogpt_ckpt,
            device=args.device,
            dtype=args.dtype,
            nanogpt_repo_root=args.nanogpt_repo_root,
        )
        args.model_id = args.nanogpt_ckpt
    else:
        _require_transformers_version(args.model_name)
        if _patch_transformers_tp_plan_check():
            print("[INFO] Patched Transformers ALL_PARALLEL_STYLES for Qwen TP-plan init check.", flush=True)

        print(f"[INFO] Loading tokenizer from {args.model_name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[INFO] Loading model from {args.model_name} on device={args.device} dtype={args.dtype}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=_resolve_dtype(args.dtype),
            attn_implementation="eager",
            trust_remote_code=args.trust_remote_code,
        ).to(args.device)
        model.eval()
        args.model_id = args.model_name

    texts = load_texts(args)
    if not texts:
        raise ValueError("No texts available for visualization")

    n_layers = len(model.transformer.h) if args.nanogpt_ckpt else len(_find_decoder_layers(model))
    print(f"[INFO] Model has {n_layers} decoder layers", flush=True)
    output_prefix = Path(args.output_prefix)

    def _summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        layer_status_by_idx: Dict[str, str] = {}
        status_counts: Dict[str, int] = {}
        analyzed_layers: List[int] = []
        skipped_layers: List[int] = []
        for payload in runs:
            layer_idx = int(payload["flip_layer"])
            status = str(payload.get("status", "analyzed"))
            layer_status_by_idx[str(layer_idx)] = status
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "analyzed":
                analyzed_layers.append(layer_idx)
            else:
                skipped_layers.append(layer_idx)
        analyzed_layers.sort()
        skipped_layers.sort()
        return {
            "layer_status_by_idx": layer_status_by_idx,
            "status_counts": status_counts,
            "analyzed_layers": analyzed_layers,
            "skipped_layers": skipped_layers,
        }

    def run_one_sample(sample_idx: int, raw_text: str) -> Dict[str, Any]:
        prompt, input_ids, attention_mask = _prepare_inputs(tokenizer, raw_text, args)
        token_labels = [_trim_token(tok) for tok in tokenizer.convert_ids_to_tokens(input_ids[0])]
        print(f"[INFO] Prepared sample_idx={sample_idx} with {len(token_labels)} tokens", flush=True)

        if int(args.flip_layer) == -1:
            sweep_dir = output_prefix.with_name(f"{output_prefix.name}-sample{sample_idx:02d}-all_layers")
            sweep_dir.mkdir(parents=True, exist_ok=True)
            index_path = sweep_dir / "all_layers_index.json"
            payloads = []
            print(f"[INFO] Running x-reference removal sweep across all {n_layers} layers for sample_idx={sample_idx}", flush=True)
            for flip_layer in range(n_layers):
                payloads.append(
                    run_single_flip(
                        model=model,
                        raw_text=raw_text,
                        prompt=prompt,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_labels=token_labels,
                        args=args,
                        flip_layer=flip_layer,
                        sweep_dir=sweep_dir,
                    )
                )
            summary_payload = {
                "sample_idx": sample_idx,
                "runs": payloads,
                **_summarize_runs(payloads),
            }
            index_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                "[INFO] sample_idx="
                f"{sample_idx} analyzed_layers={summary_payload['analyzed_layers']} "
                f"skipped_layers={summary_payload['skipped_layers']}",
                flush=True,
            )
            print(f"[INFO] Saved all-layer x-reference index to {index_path}", flush=True)
            return {"sample_idx": sample_idx, "index_path": str(index_path), **_summarize_runs(payloads), "runs": payloads}

        if int(args.flip_layer) < 0 or int(args.flip_layer) >= n_layers:
            raise ValueError(f"flip_layer={args.flip_layer} out of range for {n_layers} layers")
        print(f"[INFO] Running x-reference removal for sample_idx={sample_idx} layer={int(args.flip_layer)}", flush=True)
        return run_single_flip(
            model=model,
            raw_text=raw_text,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_labels=token_labels,
            args=args,
            flip_layer=int(args.flip_layer),
            sweep_dir=None,
        )

    if args.sample_idx == -1:
        all_prompt_payloads = []
        print(f"[INFO] Running x-reference visualization for all {len(texts)} prompts", flush=True)
        for sample_idx, raw_text in enumerate(texts):
            all_prompt_payloads.append(run_one_sample(sample_idx, raw_text))
        index_path = output_prefix.with_name(f"{output_prefix.name}-all_prompts_index.json")
        index_path.write_text(json.dumps({"sample_idx": -1, "runs": all_prompt_payloads}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] Saved all-prompt x-reference index to {index_path}", flush=True)
    else:
        if args.sample_idx < 0 or args.sample_idx >= len(texts):
            raise ValueError(f"sample_idx={args.sample_idx} out of range for {len(texts)} texts")
        run_one_sample(args.sample_idx, texts[args.sample_idx])


if __name__ == "__main__":
    main()
