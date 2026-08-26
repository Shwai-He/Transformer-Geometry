#!/usr/bin/env python3
"""Recompute the paper attention-diagonal figure with explicit edit semantics.

The residual-space panel is a layer-level shared-diagonal equivalent after
W_O.  The value-space panel remains a per-head pre-W_O coefficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np

try:
    import torch
except ImportError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ANALYSIS = (
    REPO_ROOT
    / "_overleaf_/legacy/_EMNLP_2026_legacy/representation-analysis/analysis"
)
if str(LEGACY_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(LEGACY_ANALYSIS))

DEFAULT_MODEL = (
    REPO_ROOT.parent.parent
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B"
    / "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
DEFAULT_RESULT_DIR = (
    REPO_ROOT
    / "analysis/visualization/attn_matrix/source/effective_diagonal/"
    "qwen3-4b/layer19/h6"
)


def trim_token(token: str, max_chars: int = 12) -> str:
    token = token.replace("\n", r"\n")
    return token if len(token) <= max_chars else token[: max_chars - 1] + "…"


def expand_values(value: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"Expected [batch, seq, width] values, got {tuple(value.shape)}")
    if value.shape[-1] % head_dim:
        raise ValueError("Value width is not divisible by head_dim")
    num_kv_heads = value.shape[-1] // head_dim
    if num_heads % num_kv_heads:
        raise ValueError("Query-head count is not divisible by KV-head count")
    group = num_heads // num_kv_heads
    return (
        value.reshape(*value.shape[:-1], num_kv_heads, head_dim)
        .repeat_interleave(group, dim=-2)
        .reshape(*value.shape[:-1], num_heads * head_dim)
    )


def value_space_diag_delta(
    attention: torch.Tensor, expanded_value: torch.Tensor, head_dim: int
) -> torch.Tensor:
    batch, heads, seq_len, _ = attention.shape
    ref = expanded_value.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)
    dot_source_query = torch.einsum("bhjd,bhid->bhij", ref.float(), ref.float())
    query_norm_sq = ref.float().square().sum(dim=-1).clamp_min(1e-6).unsqueeze(-1)
    coefficient = dot_source_query / query_norm_sq
    removed_parallel_weight = (attention.float() * coefficient).sum(dim=-1)
    return -removed_parallel_weight


def residual_shared_diag_delta(
    branch_output: torch.Tensor,
    residual: torch.Tensor,
    self_message: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the shared diagonal coefficient matching the removed h-parallel scalar.

    For each token t, residual parallel removal changes the h_t projection by
    -<y_t,h_t>.  A shared diagonal change delta_t applied to every head changes
    that scalar by delta_t <W_O concat_h(v_t^h), h_t>.  Equating the two gives
    the coefficient below.  This is layer-level and must not be assigned to an
    individual head after W_O.
    """
    numerator = (branch_output.float() * residual.float()).sum(dim=-1)
    denominator = (self_message.float() * residual.float()).sum(dim=-1)
    eps = 1e-8
    unstable = denominator.abs() < eps
    safe_denominator = torch.where(
        unstable,
        torch.where(denominator >= 0, torch.full_like(denominator, eps), torch.full_like(denominator, -eps)),
        denominator,
    )
    return -numerator / safe_denominator, numerator, denominator


def norm_for_delta(values: np.ndarray):
    finite = np.abs(values[np.isfinite(values)])
    max_abs = float(max(finite.max(initial=0.0), 1e-8))
    nonzero = finite[finite > 1e-8]
    if nonzero.size and max_abs / float(nonzero.min()) > 100:
        return colors.SymLogNorm(linthresh=max(float(nonzero.min()), max_abs / 1000), vmin=-max_abs, vmax=max_abs)
    return colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def draw_diag_boxes(ax, n: int) -> None:
    for index in range(n):
        ax.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, edgecolor="#333333", linewidth=0.8))


def plot_bundle(bundle: dict, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    raw = np.asarray(bundle["raw_attention_head"], dtype=float)
    residual_delta = np.asarray(bundle["residual_layer_shared_diag_delta"], dtype=float)
    value_delta = np.asarray(bundle["value_head_diag_delta"], dtype=float)
    n = raw.shape[0]

    upper = np.triu(np.ones_like(raw, dtype=bool), k=1)
    raw_masked = np.ma.array(raw, mask=upper)
    residual_matrix = np.ma.masked_all_like(raw)
    value_matrix = np.ma.masked_all_like(raw)
    np.fill_diagonal(residual_matrix, residual_delta)
    np.fill_diagonal(value_matrix, value_delta)

    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.55))
    cmap_raw = plt.get_cmap("viridis").copy()
    cmap_delta = plt.get_cmap("coolwarm").copy()
    cmap_raw.set_bad("#eeeeee")
    cmap_delta.set_bad("#eeeeee")
    images = [
        axes[0].imshow(raw_masked, cmap=cmap_raw, vmin=0.0, vmax=max(float(raw_masked.max()), 1e-8)),
        axes[1].imshow(residual_matrix, cmap=cmap_delta, norm=norm_for_delta(residual_delta)),
        axes[2].imshow(value_matrix, cmap=cmap_delta, norm=norm_for_delta(value_delta)),
    ]
    titles = [
        "Raw attention",
        r"Residual-space $\Delta A_{tt}$",
        r"Value-space $\Delta A_{tt}$",
    ]
    arrays = [raw_masked.compressed(), residual_delta, value_delta]
    for ax, title, shown in zip(axes, titles, arrays):
        ax.set_title(title, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        draw_diag_boxes(ax, n)
        finite = np.asarray(shown)[np.isfinite(shown)]
        ax.text(
            0.97,
            0.97,
            f"max={finite.max():.3g}\nmin={finite.min():.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, linewidth=0.4),
        )
    for ax, image in zip(axes, images):
        figure.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
    figure.tight_layout(w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--head", type=int, default=6)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--replot-only", action="store_true")
    args = parser.parse_args()

    bundle_path = args.result_dir / "effective_diagonal_bundle.json"
    output = args.figure_output or args.result_dir / "h6_diag_delta_recomputed.pdf"
    if args.replot_only:
        plot_bundle(json.loads(bundle_path.read_text(encoding="utf-8")), output)
        print(output)
        return

    if torch is None:
        raise RuntimeError("Full recomputation requires PyTorch; use --replot-only otherwise.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from qwen_attn_x_parallel_removal_viz import (
            SHORT_TEXTS,
            _prepare_inputs,
            collect_baseline_attention_value_and_residual,
        )
        from qwen_xsa_forward_ablation import (
            _find_decoder_layers,
            _patch_transformers_tp_plan_check,
            _require_transformers_version,
            _resolve_dtype,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Full GPU regeneration requires the archived EMNLP runtime. "
            "Use --replot-only with the bundled effective-diagonal data "
            "for the paper figure."
        ) from exc

    _require_transformers_version(str(args.model_name))
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

    layers = _find_decoder_layers(model)
    layer = layers[args.layer]
    attention_module = layer.self_attn
    captured: dict[str, torch.Tensor] = {}

    def attention_output_hook(_module, _inputs, output):
        candidate = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(candidate, torch.Tensor):
            captured["branch_output"] = candidate.detach().float().cpu()

    handle = attention_module.register_forward_hook(attention_output_hook)
    raw_text = SHORT_TEXTS[args.sample_idx]
    input_args = SimpleNamespace(
        use_chat_template=False,
        system_prompt="",
        max_length=args.max_length,
        device=args.device,
    )
    prompt, input_ids, attention_mask = _prepare_inputs(tokenizer, raw_text, input_args)
    try:
        artifacts = collect_baseline_attention_value_and_residual(
            model, input_ids, attention_mask, args.layer
        )
    finally:
        handle.remove()
    if artifacts["attention_layer"] is None:
        raise RuntimeError("The eager forward did not return attention probabilities")
    if "branch_output" not in captured:
        raise RuntimeError("Failed to capture the post-W_O attention branch output")

    attention = artifacts["attention_layer"].float()
    residual = artifacts["residual"].float()
    value = artifacts["value"].float()
    branch_output = captured["branch_output"].float()
    batch, num_heads, _query, _key = attention.shape
    head_dim = int(attention_module.head_dim)
    expanded_value = expand_values(value, num_heads, head_dim)
    with torch.no_grad():
        self_message = attention_module.o_proj(
            expanded_value.to(device=args.device, dtype=attention_module.o_proj.weight.dtype)
        ).detach().float().cpu()
    residual_delta, residual_numerator, residual_denominator = residual_shared_diag_delta(
        branch_output, residual, self_message
    )
    value_delta = value_space_diag_delta(attention, expanded_value, head_dim)

    token_labels = [trim_token(token) for token in tokenizer.convert_ids_to_tokens(input_ids[0])]
    bundle = {
        "model_snapshot": str(args.model_name.resolve()),
        "layer": args.layer,
        "head": args.head,
        "sample_idx": args.sample_idx,
        "raw_text": raw_text,
        "rendered_prompt": prompt,
        "token_labels": token_labels,
        "raw_attention_head": attention[0, args.head].tolist(),
        "residual_layer_shared_diag_delta": residual_delta[0].tolist(),
        "residual_projection_numerator": residual_numerator[0].tolist(),
        "residual_self_message_denominator": residual_denominator[0].tolist(),
        "value_head_diag_delta": value_delta[0, args.head].tolist(),
        "semantics": {
            "raw_attention_head": "raw softmax attention for the selected head",
            "residual_layer_shared_diag_delta": "layer-level shared diagonal coefficient after W_O; delta_t = -<y_t,h_t>/<W_O concat_h(v_t^h),h_t>",
            "value_head_diag_delta": "head-level pre-W_O coefficient; delta_t^h = -sum_j A_tj^h <v_j^h,v_t^h>/||v_t^h||^2",
        },
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_bundle(bundle, output)
    with (args.result_dir / "sources.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "bundle", "model_snapshot", "layer", "head", "sample_idx", "semantics"])
        writer.writerow(
            [
                str(output),
                str(bundle_path),
                str(args.model_name.resolve()),
                args.layer,
                args.head,
                args.sample_idx,
                "raw attention; layer-level post-W_O residual shared-diagonal equivalent; head-level pre-W_O value coefficient",
            ]
        )
    print(bundle_path)
    print(output)


if __name__ == "__main__":
    main()
