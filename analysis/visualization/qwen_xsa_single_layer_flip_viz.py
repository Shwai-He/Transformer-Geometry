from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.forward_geometry.qwen_xsa_forward_ablation import (
    QwenXSAForwardHooks,
    _expand_attn_value_ref,
    _extract_linear_value_ref,
    _find_decoder_layers,
    _get_token_mixer_kind_and_module,
    _patch_transformers_tp_plan_check,
    _require_transformers_version,
    _resolve_dtype,
)
from analysis.nanogpt.nanogpt_attention_utils import load_nanogpt_checkpoint, map_intervention_site, nanogpt_xsa_context


SHORT_TEXTS = [
    "Echo the final code exactly once: AX7Q-19",
    "Alice gave Bob the red key. Bob gave the red key to Clara. Who has the red key now?",
    "Finish the shortest valid continuation: The reviewer wrote, \"only the diagonal changed because",
    "Repeat the final item once: red, blue, red, blue, red,",
    "Update only the third item: A1 B2 C3 D4 ->",
    "Sarah told Mina that she would rerun the tests. If she means Sarah, who reruns the tests?",
]

LONG_TEXTS = [
    (
        "You are auditing a sequence labeling process over a long document. "
        "The document describes a pipeline with stages intake, parsing, normalization, validation, "
        "ranking, and export. Intake receives mixed records from three providers. Parsing extracts ids, "
        "timestamps, locale, and source confidence. Normalization maps aliases to canonical entities and "
        "converts all time fields to UTC. Validation applies schema checks, then semantic checks, then "
        "cross-record consistency checks. Ranking computes a final priority using recency, confidence, and "
        "conflict penalties. Export emits only records above threshold. "
        "Given this process, summarize in one sentence what validation does between normalization and ranking."
    ),
    (
        "Consider the following reasoning chain with references: Step A defines a set S of candidate actions. "
        "Step B filters S by hard constraints C1 and C2. Step C computes utility U for each remaining action. "
        "Step D applies a fairness adjustment F that reweights utility by group exposure. Step E selects top-k actions. "
        "Now suppose C2 is tightened, reducing feasible actions by 30 percent, while F is doubled. "
        "Explain how this can change the final top-k composition and whether average utility can increase or decrease."
    ),
    (
        "Read this multi-hop context carefully. Nina archived the latest design notes after Omar merged the patch. "
        "Before that, Priya had requested benchmark reruns because the latency variance was unstable across seeds. "
        "After reruns, Omar updated scheduler settings and disabled one cache optimization only for ablation. "
        "Nina then compared three reports: baseline, ablation, and patched-ablation. "
        "If the patched-ablation recovered throughput but not variance, what was Priya's original concern most likely about?"
    ),
    (
        "You are given a long instruction list for a synthetic planning task. "
        "1) Build an index over sections alpha through kappa. "
        "2) For each section, collect key terms and assign a confidence score in [0,1]. "
        "3) Merge adjacent sections when cosine similarity exceeds 0.82. "
        "4) Recompute section scores after merge. "
        "5) Keep only merged sections with score >= 0.6 and at least 4 key terms. "
        "6) Output the final section ids in descending score order. "
        "Given these steps, what happens first after a merge event?"
    ),
]


SHARED_VALUE_CMAP = "viridis"


def _load_existing_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
        prompt_set = str(getattr(args, "prompt_set", "short")).lower().strip()
        if prompt_set == "long":
            texts = list(LONG_TEXTS)
        elif prompt_set == "short":
            texts = list(SHORT_TEXTS)
        else:
            raise ValueError(f"Unsupported prompt_set={prompt_set}. Use short or long.")
    if args.max_samples > 0:
        texts = texts[: args.max_samples]
    if not texts:
        raise ValueError("No texts loaded for attention visualization.")
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


@torch.no_grad()
def collect_attentions(model, input_ids, attention_mask, args, use_xsa: bool):
    if args.nanogpt_ckpt:
        site_cfg = map_intervention_site(args.xsa_intervention_site)
        with nanogpt_xsa_context(
            model,
            enable=use_xsa,
            ref=site_cfg["ref"],
            space=site_cfg["space"],
            target=site_cfg["target"],
            start_layer=args.xsa_start_layer,
            end_layer=args.xsa_end_layer,
        ):
            outputs = model(input_ids, use_cache=False, output_attentions=True, return_dict=True)
        return [att.detach().float().cpu() for att in outputs.attentions]
    hook_ctx = QwenXSAForwardHooks(
        model,
        target="attn",
        start_layer=args.xsa_start_layer,
        end_layer=args.xsa_end_layer,
        skip_first_n=args.xsa_skip_first_n,
        skip_last_n=args.xsa_skip_last_n,
        intervention_site=args.xsa_intervention_site,
        track_stats=False,
        track_layerwise_stats=False,
    ) if use_xsa else None

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
    return [att.detach().float().cpu() for att in outputs.attentions]


def summarize_attention_deltas(
    baseline_atts: List[torch.Tensor],
    xsa_atts: List[torch.Tensor],
) -> Dict[str, List[Dict[str, float]]]:
    rows = []
    for layer_idx, (base, xsa) in enumerate(zip(baseline_atts, xsa_atts)):
        delta = xsa - base
        n_heads = int(delta.shape[1])
        for head_idx in range(n_heads):
            mat = delta[0, head_idx]
            diag = torch.diagonal(mat)
            rows.append(
                {
                    "layer": float(layer_idx),
                    "head": float(head_idx),
                    "mean_abs_delta": float(mat.abs().mean().item()),
                    "max_abs_delta": float(mat.abs().max().item()),
                    "mean_diag_delta": float(diag.mean().item()),
                    "mean_abs_diag_delta": float(diag.abs().mean().item()),
                }
            )
    rows.sort(key=lambda x: x["mean_abs_delta"], reverse=True)
    return {"layer_head_summary": rows}


def _reshape_token_mixer_heads(y_pre: torch.Tensor, mixer_kind: str, mixer_module) -> torch.Tensor:
    if y_pre.dim() != 3:
        raise ValueError(f"Expected [batch, seq, hidden] token mixer states, got {tuple(y_pre.shape)}")
    if mixer_kind == "full_attention":
        num_heads = int(getattr(mixer_module, "num_heads"))
        head_dim = int(getattr(mixer_module, "head_dim"))
    elif mixer_kind == "linear_attention":
        num_heads = int(getattr(mixer_module, "num_v_heads"))
        head_dim = int(getattr(mixer_module, "head_v_dim"))
    else:
        raise ValueError(f"Unsupported mixer_kind={mixer_kind}")
    if y_pre.size(-1) != num_heads * head_dim:
        raise ValueError(
            f"Cannot reshape token mixer states with hidden={y_pre.size(-1)} into {num_heads}x{head_dim} heads."
        )
    return y_pre.reshape(y_pre.shape[0], y_pre.shape[1], num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


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
    mark_diagonal: bool = True,
    diagonal_color: str = "black",
):
    arr = matrix.numpy()
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
    if mark_diagonal:
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
    ax.legend(fontsize=7, loc="best")


def _pick_head_for_layer(summary_rows: List[Dict[str, float]], layer_idx: int, preferred_head: int) -> int:
    if preferred_head >= 0:
        return preferred_head
    for row in summary_rows:
        if int(row["layer"]) == layer_idx:
            return int(row["head"])
    return 0


def _pick_layers_to_show(
    n_layers: int,
    flip_layer: int,
    focus_head: int,
    summary_rows: List[Dict[str, float]],
    num_extra_layers: int,
) -> List[Dict[str, int]]:
    picked = [{"layer": flip_layer, "head": focus_head, "tag": "flip"}]
    if flip_layer + 1 < n_layers:
        picked.append(
            {
                "layer": flip_layer + 1,
                "head": _pick_head_for_layer(summary_rows, flip_layer + 1, focus_head),
                "tag": "next",
            }
        )
    for row in summary_rows:
        layer_idx = int(row["layer"])
        if any(item["layer"] == layer_idx for item in picked):
            continue
        picked.append(
            {
                "layer": layer_idx,
                "head": int(row["head"]),
                "tag": "top_delta",
            }
        )
        if len(picked) >= 2 + max(0, int(num_extra_layers)):
            break
    return picked


@torch.no_grad()
def collect_flip_layer_artifacts(model, input_ids, attention_mask, flip_layer: int, args, use_xsa: bool):
    if args.nanogpt_ckpt:
        site_cfg = map_intervention_site(args.xsa_intervention_site)
        with nanogpt_xsa_context(
            model,
            enable=use_xsa,
            ref=site_cfg["ref"],
            space=site_cfg["space"],
            target=site_cfg["target"],
            start_layer=flip_layer,
            end_layer=flip_layer,
            single_layer=flip_layer,
        ):
            outputs = model(input_ids, use_cache=False, output_attentions=True, return_dict=True)
        attn_module = model.transformer.h[flip_layer].attn
        return {
            "mixer_kind": "nanogpt_attention",
            "mixer_module": attn_module,
            "value": attn_module._last_value.detach().float().cpu() if isinstance(attn_module._last_value, torch.Tensor) else None,
            "attention_layer": outputs.attentions[flip_layer].detach().float().cpu(),
        }
    layers = _find_decoder_layers(model)
    layer = layers[flip_layer]
    mixer_kind, mixer = _get_token_mixer_kind_and_module(layer)
    if mixer is None:
        raise ValueError(f"Layer {flip_layer} has no supported token mixer.")

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    if mixer_kind == "full_attention":
        attn = mixer
        if getattr(attn, "v_proj", None) is None:
            raise ValueError(f"Layer {flip_layer} has no self_attn.v_proj; cannot collect artifacts.")

        def value_hook(_mod, _args, _kwargs, output):
            if isinstance(output, torch.Tensor):
                captured["value"] = output.detach()
            return output

        def pre_out_proj_hook(_mod, args, kwargs):
            y_pre = args[0] if args else kwargs.get("input", None)
            if isinstance(y_pre, torch.Tensor):
                captured["y_pre"] = y_pre.detach()
            return None

        handles.append(attn.v_proj.register_forward_hook(value_hook, with_kwargs=True))
        handles.append(attn.o_proj.register_forward_pre_hook(pre_out_proj_hook, with_kwargs=True))
    hook_ctx = None
    if use_xsa:
        hook_ctx = QwenXSAForwardHooks(
            model,
            target="attn",
            start_layer=flip_layer,
            end_layer=flip_layer + 1,
            skip_first_n=0,
            skip_last_n=0,
            intervention_site=args.xsa_intervention_site,
            track_stats=False,
            track_layerwise_stats=False,
        )

    try:
        if hook_ctx is not None:
            hook_ctx.attach()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        if hook_ctx is not None:
            hook_ctx.close()
        for handle in handles:
            handle.remove()

    attention_layer = None
    if mixer_kind == "full_attention" and outputs.attentions:
        if len(outputs.attentions) == len(layers):
            candidate = outputs.attentions[flip_layer]
            if isinstance(candidate, torch.Tensor):
                attention_layer = candidate.detach().float().cpu()
        else:
            full_attention_layers = [
                idx
                for idx, candidate in enumerate(layers)
                if _get_token_mixer_kind_and_module(candidate)[0] == "full_attention"
            ]
            if len(outputs.attentions) == len(full_attention_layers) and flip_layer in full_attention_layers:
                attn_idx = full_attention_layers.index(flip_layer)
                candidate = outputs.attentions[attn_idx]
                if isinstance(candidate, torch.Tensor):
                    attention_layer = candidate.detach().float().cpu()

    value = captured.get("value")
    if mixer_kind == "full_attention" and not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Failed to capture token-mixer value reference for layer {flip_layer}.")

    return {
        "mixer_kind": mixer_kind,
        "mixer_module": mixer,
        "value": value.detach().float().cpu() if isinstance(value, torch.Tensor) else None,
        "attention_layer": attention_layer,
    }


def compute_effective_diag_delta(
    attn_probs: torch.Tensor,
    value: torch.Tensor,
    attn_module,
    *,
    value_head_mode: str = "head_specific",
) -> torch.Tensor:
    if attn_probs.dim() != 4:
        raise ValueError(f"attn_probs must be [batch, heads, q, k], got shape {tuple(attn_probs.shape)}")
    if value.dim() != 3:
        raise ValueError(f"value must be [batch, seq, hidden], got shape {tuple(value.shape)}")

    batch_size, num_heads, seq_len, _ = attn_probs.shape
    full_width = num_heads * getattr(attn_module, "head_dim", 0)
    dummy_y = torch.zeros(batch_size, seq_len, full_width, dtype=value.dtype, device=value.device)
    ref = _expand_attn_value_ref(value, dummy_y, attn_module)
    if ref is None:
        raise RuntimeError("Failed to expand value tensor to per-head reference.")

    head_dim = int(getattr(attn_module, "head_dim"))
    value_head_mode = str(value_head_mode or "head_specific").lower().strip()
    if value_head_mode in {"head_concat", "concat", "merged", "legacy"}:
        ref_heads = ref.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        y_heads = torch.einsum("bhij,bhjd->bhid", attn_probs.float(), ref_heads.float())
        y_concat = y_heads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, full_width)
        ref_concat = ref.float()
        coeff = (y_concat * ref_concat).sum(dim=-1) / ref_concat.square().sum(dim=-1).clamp_min(1e-6)
        coeff = coeff[:, None, :].expand(batch_size, num_heads, seq_len)
        return torch.diag_embed(-coeff).cpu()

    ref_heads = ref.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    dot_j_i = torch.einsum("bhjd,bhid->bhij", ref_heads, ref_heads)
    denom_i = ref_heads.square().sum(dim=-1).clamp_min(1e-6).unsqueeze(-1)
    alpha = dot_j_i / denom_i
    c = (attn_probs.float() * alpha).sum(dim=-1)
    diag_delta = -c
    return torch.diag_embed(diag_delta).cpu()


def save_layer_all_heads_heatmap_with_custom_delta(
    baseline_layer: torch.Tensor,
    compare_layer: torch.Tensor,
    layer_idx: int,
    out_path: Path,
    labels: List[str],
    custom_delta_layer: torch.Tensor | None = None,
    delta_title: str = "delta",
    compare_title: str = "xsa",
) -> None:
    n_heads = int(baseline_layer.shape[1])
    fig, axes = plt.subplots(n_heads, 4, figsize=(16, 3.8 * n_heads))
    if n_heads == 1:
        axes = axes.reshape(1, 4)
    for head in range(n_heads):
        baseline = baseline_layer[0, head]
        compare = compare_layer[0, head]
        raw_delta = compare - baseline
        shown_delta = custom_delta_layer[0, head] if custom_delta_layer is not None else raw_delta
        shared_vmax = float(max(baseline.max().item(), compare.max().item()))
        print(
            f"[INFO] layer={layer_idx} head={head} baseline_range=({baseline.min().item():.6g}, {baseline.max().item():.6g}) "
            f"compare_range=({compare.min().item():.6g}, {compare.max().item():.6g}) "
            f"shared_positive_range=(0, {shared_vmax:.6g}) "
            f"delta_range=({shown_delta.min().item():.6g}, {shown_delta.max().item():.6g})",
            flush=True,
        )
        delta_abs = float(shown_delta.abs().max().item())
        ims = [
            _plot_shared_value_matrix(
                axes[head, 0],
                baseline,
                f"L{layer_idx} H{head} baseline",
                labels,
                0.0,
                shared_vmax,
                diagonal_color="white",
            ),
            _plot_shared_value_matrix(
                axes[head, 1],
                compare,
                f"L{layer_idx} H{head} {compare_title}",
                labels,
                0.0,
                shared_vmax,
                diagonal_color="white",
            ),
            _plot_matrix(axes[head, 2], shown_delta, f"L{layer_idx} H{head} {delta_title}", labels, "coolwarm", -delta_abs, delta_abs, diagonal_color="black"),
        ]
        for col_idx, im in enumerate(ims):
            fig.colorbar(im, ax=axes[head, col_idx], fraction=0.046, pad=0.04)
        curve_ax = axes[head, 3]
        _plot_diag_offdiag_curves(curve_ax, compare, f"L{layer_idx} H{head} {compare_title}: diag/offdiag")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_selected_layer_images(
    baseline_atts: List[torch.Tensor],
    xsa_atts: List[torch.Tensor],
    layer_specs: List[Dict[str, int]],
    out_dir: Path,
    labels: List[str],
    flip_layer: int,
    effective_delta_layer: torch.Tensor | None,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for spec in layer_specs:
        layer_idx = int(spec["layer"])
        source_idx = int(spec.get("source_idx", layer_idx))
        tag = str(spec["tag"])
        out_path = out_dir / f"layer_{layer_idx:02d}_{tag}.png"
        if layer_idx == int(flip_layer) and effective_delta_layer is not None:
            save_layer_all_heads_heatmap_with_custom_delta(
                baseline_layer=baseline_atts[source_idx],
                compare_layer=baseline_atts[source_idx] + effective_delta_layer,
                layer_idx=layer_idx,
                out_path=out_path,
                labels=labels,
                custom_delta_layer=effective_delta_layer,
                delta_title="effective delta",
                compare_title="effective xsa",
            )
        else:
            save_layer_all_heads_heatmap_with_custom_delta(
                baseline_layer=baseline_atts[source_idx],
                compare_layer=xsa_atts[source_idx],
                layer_idx=layer_idx,
                out_path=out_path,
                labels=labels,
                custom_delta_layer=None,
                delta_title="raw delta",
                compare_title="raw xsa",
            )
        saved.append(str(out_path))
    return saved


def run_single_flip(
    model,
    tokenizer,
    raw_text: str,
    prompt: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_labels: List[str],
    args,
    flip_layer: int,
    only_save_flip_layer: bool = False,
    sweep_dir: Path | None = None,
) -> Dict[str, object]:
    print(f"[INFO] Starting single-layer flip analysis for layer={flip_layer}", flush=True)
    baseline_artifacts = collect_flip_layer_artifacts(
        model,
        input_ids,
        attention_mask,
        flip_layer,
        args,
        use_xsa=False,
    )
    xsa_artifacts = collect_flip_layer_artifacts(
        model,
        input_ids,
        attention_mask,
        flip_layer,
        args,
        use_xsa=True,
    )

    mixer_kind = str(baseline_artifacts["mixer_kind"])
    mixer_module = baseline_artifacts["mixer_module"]
    n_layers = len(_find_decoder_layers(model))
    if flip_layer < 0 or flip_layer >= n_layers:
        raise ValueError(f"flip_layer={flip_layer} out of range for {n_layers} layers")

    baseline_layer = baseline_artifacts["attention_layer"]
    xsa_layer = xsa_artifacts["attention_layer"]
    effective_delta_layer = None
    viz_mode = "attention_probs"

    if baseline_layer is None or xsa_layer is None:
        note = (
            f"Layer {flip_layer} uses {mixer_kind}. This script only visualizes layers with explicit "
            "attention matrices, so this layer is skipped for now."
        )
        payload = {
            "model_name": args.model_id,
            "raw_text": raw_text,
            "rendered_prompt": prompt,
            "sample_idx": args.sample_idx,
            "flip_layer": int(flip_layer),
            "xsa_intervention_site": args.xsa_intervention_site,
            "status": "skipped_no_attention_matrix",
            "token_mixer_kind": mixer_kind,
            "note": note,
            "tokens": token_labels,
        }
        output_prefix = Path(args.output_prefix)
        if sweep_dir is not None:
            json_path = sweep_dir / f"layer_{flip_layer:02d}_summary.json"
        else:
            run_dir = output_prefix.with_name(f"{output_prefix.name}-sample{args.sample_idx:02d}-flip{flip_layer:02d}")
            json_path = run_dir / "summary.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] flip_layer={flip_layer} skipped: {note}", flush=True)
        return payload
    else:
        effective_delta_layer = compute_effective_diag_delta(
            attn_probs=baseline_layer,
            value=baseline_artifacts["value"],
            attn_module=mixer_module,
            value_head_mode=args.value_head_mode,
        )

    summary = summarize_attention_deltas([baseline_layer], [xsa_layer])

    flip_head = int(args.head) if int(args.head) >= 0 else _pick_head_for_layer(summary["layer_head_summary"], 0, -1)
    layer_specs = [{"layer": int(flip_layer), "head": int(flip_head), "tag": "flip", "source_idx": 0}]

    output_prefix = Path(args.output_prefix)
    if sweep_dir is not None:
        run_dir = sweep_dir
        layers_dir = sweep_dir / "selected_layers"
        json_path = sweep_dir / f"layer_{flip_layer:02d}_summary.json"
    else:
        stem = f"{output_prefix.name}-sample{args.sample_idx:02d}-flip{flip_layer:02d}-head{flip_head:02d}"
        run_dir = output_prefix.with_name(stem)
        layers_dir = run_dir / "selected_layers"
        json_path = run_dir / "summary.json"

    selected_layer_paths = save_selected_layer_images(
        baseline_atts=[baseline_layer],
        xsa_atts=[xsa_layer],
        layer_specs=layer_specs,
        out_dir=layers_dir,
        labels=token_labels,
        flip_layer=int(flip_layer),
        effective_delta_layer=effective_delta_layer,
    )
    print(f"[INFO] layer={flip_layer} wrote {len(selected_layer_paths)} image(s)", flush=True)

    flip_baseline = baseline_layer[0, flip_head]
    flip_xsa = xsa_layer[0, flip_head]
    payload = {
        "model_name": args.model_id,
        "raw_text": raw_text,
        "rendered_prompt": prompt,
        "sample_idx": args.sample_idx,
        "flip_layer": int(flip_layer),
        "flip_head": int(flip_head),
        "xsa_intervention_site": args.xsa_intervention_site,
        "value_head_mode": args.value_head_mode,
        "viz_mode": viz_mode,
        "token_mixer_kind": mixer_kind,
        "xsa_layer_selection": {
            "xsa_start_layer": int(flip_layer),
            "xsa_end_layer": int(flip_layer) + 1,
        },
        "note": (
            "This run applies XSA to exactly one decoder layer. For the flipped layer, the second column uses "
            "baseline + effective delta instead of raw downstream attention, matching the closed-form effective matrix."
        ),
        "only_save_flip_layer": bool(only_save_flip_layer),
        "selected_layers": layer_specs,
        "top_layer_head_deltas": summary["layer_head_summary"][:20],
        "selected_layer_image_paths": selected_layer_paths,
        "tokens": token_labels,
        "flip_layer_baseline_diag": torch.diagonal(flip_baseline).tolist(),
        "flip_layer_raw_xsa_diag": torch.diagonal(flip_xsa).tolist(),
        "flip_layer_raw_diag_delta": torch.diagonal(flip_xsa - flip_baseline).tolist(),
    }
    if effective_delta_layer is not None:
        effective_compare = baseline_layer[0, flip_head] + effective_delta_layer[0, flip_head]
        effective_diag = torch.diagonal(effective_compare)
        payload["flip_layer_effective_diag_delta"] = torch.diagonal(effective_delta_layer[0, flip_head]).tolist()
        payload["flip_layer_effective_xsa_diag"] = effective_diag.tolist()
        payload["flip_layer_effective_offdiag_sum"] = (effective_compare.sum(dim=-1) - effective_diag).tolist()
        payload["flip_layer_effective_total_sum"] = effective_compare.sum(dim=-1).tolist()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] flip_layer={flip_layer} saved selected-layer all-head heatmaps to {layers_dir}")
    print(f"[INFO] flip_layer={flip_layer} saved summary to {json_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize single-layer XSA flip effects on attention matrices.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--nanogpt_ckpt", type=str, default=None)
    parser.add_argument("--nanogpt_repo_root", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--prompt_set", type=str, default="short", choices=["short", "long"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--flip_layer", type=int, required=True, help="Single decoder layer to flip to XSA. Use -1 to run one-layer flips for all layers.")
    parser.add_argument("--head", type=int, default=-1, help="Head to show for flip layer; -1 auto-picks largest delta head.")
    parser.add_argument("--value_head_mode", type=str, default="head_specific", choices=["head_specific", "head_concat", "legacy", "merged", "concat"])
    parser.add_argument("--num_extra_layers", type=int, default=2, help="Additional top-delta layers to include beyond flip and next.")
    parser.add_argument(
        "--xsa_intervention_site",
        type=str,
        default="xsa_middle_multihead",
        choices=["xsa_middle", "xsa_middle_multihead", "residual_output"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "flex_attention"],
        help="Attention backend to use when loading the model. Use eager for attention-matrix visualization.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="representation-analysis/outputs/qwen_xsa_single_layer_flip",
    )
    parser.add_argument("--overwrite_existing", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[INFO] seed={args.seed}", flush=True)

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
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        ).to(args.device)
        model.eval()
        print(
            f"[INFO] Loaded model with attn_implementation={getattr(model.config, '_attn_implementation', 'unknown')}",
            flush=True,
        )
        args.model_id = args.model_name

    texts = load_texts(args)
    if not texts:
        raise ValueError("No texts available for visualization")

    n_layers = len(model.transformer.h) if args.nanogpt_ckpt else len(_find_decoder_layers(model))
    print(f"[INFO] Model has {n_layers} decoder layers", flush=True)
    output_prefix = Path(args.output_prefix)
    print(f"[INFO] overwrite_existing={bool(args.overwrite_existing)}", flush=True)

    def run_one_sample(sample_idx: int, raw_text: str) -> Dict[str, Any]:
        prompt, input_ids, attention_mask = _prepare_inputs(tokenizer, raw_text, args)
        token_labels = [_trim_token(tok) for tok in tokenizer.convert_ids_to_tokens(input_ids[0])]
        print(f"[INFO] Prepared sample_idx={sample_idx} with {len(token_labels)} tokens", flush=True)

        if int(args.flip_layer) == -1:
            sweep_dir = output_prefix.with_name(f"{output_prefix.name}-sample{sample_idx:02d}-all_layers")
            index_path = sweep_dir / "all_layers_index.json"
            if index_path.exists() and not bool(args.overwrite_existing):
                print(f"[SKIP] sample_idx={sample_idx} already done: {index_path}", flush=True)
                return json.loads(index_path.read_text(encoding="utf-8"))
            sweep_dir.mkdir(parents=True, exist_ok=True)
            all_payloads = []
            print(f"[INFO] Running single-layer flip sweep across all {n_layers} layers for sample_idx={sample_idx}", flush=True)
            for flip_layer in range(n_layers):
                layer_summary_path = sweep_dir / f"layer_{flip_layer:02d}_summary.json"
                existing_payload = None if bool(args.overwrite_existing) else _load_existing_json(layer_summary_path)
                if existing_payload is not None:
                    print(
                        f"[SKIP] sample_idx={sample_idx} layer={flip_layer} reusing existing summary: {layer_summary_path}",
                        flush=True,
                    )
                    all_payloads.append(existing_payload)
                    continue
                all_payloads.append(
                    run_single_flip(
                        model=model,
                        tokenizer=tokenizer,
                        raw_text=raw_text,
                        prompt=prompt,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_labels=token_labels,
                        args=args,
                        flip_layer=flip_layer,
                        only_save_flip_layer=True,
                        sweep_dir=sweep_dir,
                    )
                )
            index_path.write_text(json.dumps({"sample_idx": sample_idx, "runs": all_payloads}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[INFO] Saved all-layer single-flip index to {index_path}", flush=True)
            return {"sample_idx": sample_idx, "index_path": str(index_path), "runs": all_payloads}

        print(f"[INFO] Running single-layer flip for sample_idx={sample_idx} layer={int(args.flip_layer)}", flush=True)
        payload = run_single_flip(
            model=model,
            tokenizer=tokenizer,
            raw_text=raw_text,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_labels=token_labels,
            args=args,
            flip_layer=int(args.flip_layer),
            only_save_flip_layer=False,
        )
        return payload

    if args.sample_idx == -1:
        index_path = output_prefix.with_name(f"{output_prefix.name}-all_prompts_index.json")
        if index_path.exists() and not bool(args.overwrite_existing):
            print(f"[SKIP] all_prompts_index already exists: {index_path}", flush=True)
        else:
            all_prompt_payloads = []
            print(f"[INFO] Running single-layer flip visualization for all {len(texts)} prompts", flush=True)
            for sample_idx, raw_text in enumerate(texts):
                all_prompt_payloads.append(run_one_sample(sample_idx, raw_text))
            index_path.write_text(json.dumps({"sample_idx": -1, "runs": all_prompt_payloads}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[INFO] Saved all-prompt single-flip index to {index_path}", flush=True)
    else:
        if args.sample_idx < 0 or args.sample_idx >= len(texts):
            raise ValueError(f"sample_idx={args.sample_idx} out of range for {len(texts)} texts")
        run_one_sample(args.sample_idx, texts[args.sample_idx])


if __name__ == "__main__":
    main()
