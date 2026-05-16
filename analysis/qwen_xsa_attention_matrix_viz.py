from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_xsa_forward_ablation import (
    QwenXSAForwardHooks,
    _patch_transformers_tp_plan_check,
    _require_transformers_version,
    _resolve_dtype,
)
from nanogpt_attention_utils import load_nanogpt_checkpoint, map_intervention_site, nanogpt_xsa_context


SHORT_TEXTS = [
    "Echo the final code exactly once: AX7Q-19",
    "Alice gave Bob the red key. Bob gave the red key to Clara. Who has the red key now?",
    "Finish the shortest valid continuation: The reviewer wrote, \"only the diagonal changed because",
    "Repeat the final item once: red, blue, red, blue, red,",
    "Update only the third item: A1 B2 C3 D4 ->",
    "Sarah told Mina that she would rerun the tests. If she means Sarah, who reruns the tests?",
]


def _make_signed_cmap(neg_name: str = "Blues_r", pos_name: str = "viridis") -> colors.LinearSegmentedColormap:
    neg = plt.get_cmap(neg_name)(torch.linspace(0.0, 1.0, 128).tolist())
    pos = plt.get_cmap(pos_name)(torch.linspace(0.0, 1.0, 128).tolist())
    white = [[1.0, 1.0, 1.0, 1.0]]
    return colors.LinearSegmentedColormap.from_list(
        f"signed_{neg_name}_{pos_name}",
        list(neg) + white + list(pos),
    )


SIGNED_CMAP = _make_signed_cmap()


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
            outputs = model(
                input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
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
    attentions = [att.detach().float().cpu() for att in outputs.attentions]
    return attentions


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
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        )


def _plot_matrix(ax, matrix: torch.Tensor, title: str, labels: List[str], cmap: str, vmin=None, vmax=None, norm=None, mark_diagonal: bool = True, diagonal_color: str = "black"):
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


def _plot_signed_matrix(ax, matrix: torch.Tensor, title: str, labels: List[str], diagonal_color: str = "black"):
    matrix_min = float(matrix.min().item())
    matrix_max = float(matrix.max().item())
    if matrix_min < 0.0 < matrix_max:
        return _plot_matrix(
            ax,
            matrix,
            title,
            labels,
            SIGNED_CMAP,
            norm=colors.TwoSlopeNorm(vmin=matrix_min, vcenter=0.0, vmax=matrix_max),
            diagonal_color=diagonal_color,
        )
    if matrix_min < 0.0:
        return _plot_matrix(ax, matrix, title, labels, "Blues_r", matrix_min, matrix_max, diagonal_color=diagonal_color)
    return _plot_matrix(ax, matrix, title, labels, "viridis", 0.0, matrix_max, diagonal_color=diagonal_color)


def _select_layer_heads(summary_rows: List[Dict[str, float]], n: int) -> List[Dict[str, int]]:
    picked = []
    seen = set()
    for row in summary_rows:
        key = (int(row["layer"]), int(row["head"]))
        if key in seen:
            continue
        seen.add(key)
        picked.append({"layer": key[0], "head": key[1]})
        if len(picked) >= n:
            break
    return picked


def save_visualization(
    baseline: torch.Tensor,
    xsa: torch.Tensor,
    out_path: Path,
    labels: List[str],
    title_prefix: str,
) -> None:
    delta = xsa - baseline
    row_sum_xsa = xsa.sum(dim=-1, keepdim=True).repeat(1, xsa.shape[-1])
    print(
        f"[INFO] baseline_range=({baseline.min().item():.6g}, {baseline.max().item():.6g}) "
        f"xsa_range=({xsa.min().item():.6g}, {xsa.max().item():.6g}) "
        f"delta_range=({delta.min().item():.6g}, {delta.max().item():.6g})",
        flush=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    attn_vmax = float(max(baseline.max().item(), xsa.max().item()))
    delta_abs = float(delta.abs().max().item())
    row_sum_abs = float(row_sum_xsa.abs().max().item())

    im0 = _plot_matrix(axes[0, 0], baseline, f"{title_prefix} baseline", labels, cmap="viridis", vmin=0.0, vmax=attn_vmax, diagonal_color="white")
    im1 = _plot_matrix(axes[0, 1], xsa, f"{title_prefix} xsa", labels, cmap="viridis", vmin=0.0, vmax=attn_vmax, diagonal_color="white")
    im2 = _plot_signed_matrix(axes[1, 0], delta, f"{title_prefix} delta (xsa - baseline)", labels, diagonal_color="black")
    im3 = _plot_signed_matrix(axes[1, 1], row_sum_xsa, f"{title_prefix} row-sum(xsa)", labels, diagonal_color="black")

    for ax, im in zip(axes.flatten(), [im0, im1, im2, im3]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_multi_layer_heatmaps(
    baseline_atts: List[torch.Tensor],
    xsa_atts: List[torch.Tensor],
    layer_heads: List[Dict[str, int]],
    out_path: Path,
    labels: List[str],
) -> None:
    n = len(layer_heads)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.0 * n))
    if n == 1:
        axes = axes.reshape(1, 4)
    for row_idx, item in enumerate(layer_heads):
        layer = int(item["layer"])
        head = int(item["head"])
        baseline = baseline_atts[layer][0, head]
        xsa = xsa_atts[layer][0, head]
        delta = xsa - baseline
        row_sum_xsa = xsa.sum(dim=-1, keepdim=True).repeat(1, xsa.shape[-1])
        attn_vmax = float(max(baseline.max().item(), xsa.max().item()))
        delta_abs = float(delta.abs().max().item())
        row_sum_abs = float(row_sum_xsa.abs().max().item())
        ims = [
            _plot_matrix(axes[row_idx, 0], baseline, f"L{layer} H{head} baseline", labels, "viridis", 0.0, attn_vmax, diagonal_color="white"),
            _plot_matrix(axes[row_idx, 1], xsa, f"L{layer} H{head} xsa", labels, "viridis", 0.0, attn_vmax, diagonal_color="white"),
            _plot_matrix(axes[row_idx, 2], delta, f"L{layer} H{head} delta", labels, "coolwarm", -delta_abs, delta_abs, diagonal_color="black"),
            _plot_matrix(axes[row_idx, 3], row_sum_xsa, f"L{layer} H{head} row-sum(xsa)", labels, "coolwarm", -row_sum_abs, row_sum_abs, diagonal_color="black"),
        ]
        for col_idx, im in enumerate(ims):
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_layer_all_heads_heatmap(
    baseline_layer: torch.Tensor,
    xsa_layer: torch.Tensor,
    layer_idx: int,
    out_path: Path,
    labels: List[str],
) -> None:
    n_heads = int(baseline_layer.shape[1])
    fig, axes = plt.subplots(n_heads, 4, figsize=(16, 3.8 * n_heads))
    if n_heads == 1:
        axes = axes.reshape(1, 4)
    for head in range(n_heads):
        baseline = baseline_layer[0, head]
        xsa = xsa_layer[0, head]
        delta = xsa - baseline
        row_sum_xsa = xsa.sum(dim=-1, keepdim=True).repeat(1, xsa.shape[-1])
        attn_vmax = float(max(baseline.max().item(), xsa.max().item()))
        delta_abs = float(delta.abs().max().item())
        row_sum_abs = float(row_sum_xsa.abs().max().item())
        ims = [
            _plot_matrix(axes[head, 0], baseline, f"L{layer_idx} H{head} baseline", labels, "viridis", 0.0, attn_vmax, diagonal_color="white"),
            _plot_matrix(axes[head, 1], xsa, f"L{layer_idx} H{head} xsa", labels, "viridis", 0.0, attn_vmax, diagonal_color="white"),
            _plot_matrix(axes[head, 2], delta, f"L{layer_idx} H{head} delta", labels, "coolwarm", -delta_abs, delta_abs, diagonal_color="black"),
            _plot_matrix(axes[head, 3], row_sum_xsa, f"L{layer_idx} H{head} row-sum(xsa)", labels, "coolwarm", -row_sum_abs, row_sum_abs, diagonal_color="black"),
        ]
        for col_idx, im in enumerate(ims):
            fig.colorbar(im, ax=axes[head, col_idx], fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_layers_all_heads(
    baseline_atts: List[torch.Tensor],
    xsa_atts: List[torch.Tensor],
    out_dir: Path,
    labels: List[str],
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for layer_idx, (baseline_layer, xsa_layer) in enumerate(zip(baseline_atts, xsa_atts)):
        out_path = out_dir / f"layer_{layer_idx:02d}.png"
        save_layer_all_heads_heatmap(
            baseline_layer=baseline_layer,
            xsa_layer=xsa_layer,
            layer_idx=layer_idx,
            out_path=out_path,
            labels=labels,
        )
        saved.append(str(out_path))
    return saved


def save_multi_layer_diag_plot(
    baseline_atts: List[torch.Tensor],
    xsa_atts: List[torch.Tensor],
    layer_heads: List[Dict[str, int]],
    out_path: Path,
) -> None:
    n = len(layer_heads)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, item in zip(axes, layer_heads):
        layer = int(item["layer"])
        head = int(item["head"])
        baseline = baseline_atts[layer][0, head]
        xsa = xsa_atts[layer][0, head]
        base_diag = torch.diagonal(baseline).numpy()
        xsa_diag = torch.diagonal(xsa).numpy()
        delta_diag = (torch.diagonal(xsa) - torch.diagonal(baseline)).numpy()
        xs = list(range(len(base_diag)))
        ax.plot(xs, base_diag, label="baseline_diag", linewidth=1.5)
        ax.plot(xs, xsa_diag, label="xsa_diag", linewidth=1.5)
        ax.plot(xs, delta_diag, label="diag_delta", linewidth=1.2, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(f"L{layer} H{head}")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("token position")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize baseline/XSA attention matrices on short prompts.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--nanogpt_ckpt", type=str, default=None)
    parser.add_argument("--nanogpt_repo_root", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to visualize for the single focus plot; -1 uses layer with largest mean_abs_delta.")
    parser.add_argument("--head", type=int, default=-1, help="Head index. Use -1 to auto-pick the largest-delta head per layer.")
    parser.add_argument("--num_layers_to_plot", type=int, default=4, help="Number of top-changing layer-head pairs to visualize.")
    parser.add_argument("--xsa_start_layer", type=int, default=0)
    parser.add_argument("--xsa_end_layer", type=int, default=-1)
    parser.add_argument("--xsa_skip_first_n", type=int, default=0)
    parser.add_argument("--xsa_skip_last_n", type=int, default=0)
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
        "--output_prefix",
        type=str,
        default="representation-analysis/outputs/qwen_xsa_attention_matrix",
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
        model_id = args.nanogpt_ckpt
    else:
        _require_transformers_version()
        if _patch_transformers_tp_plan_check():
            print("[INFO] Patched Transformers ALL_PARALLEL_STYLES for Qwen TP-plan init check.", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=_resolve_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
        ).to(args.device)
        model.eval()
        model_id = args.model_name

    texts = load_texts(args)
    if not texts:
        raise ValueError("No texts available for visualization")

    output_prefix = Path(args.output_prefix)

    def run_one_sample(sample_idx: int, raw_text: str) -> Dict[str, Any]:
        print(f"[INFO] Running attention visualization for sample_idx={sample_idx}", flush=True)
        prompt, input_ids, attention_mask = _prepare_inputs(tokenizer, raw_text, args)
        token_labels = [_trim_token(tok) for tok in tokenizer.convert_ids_to_tokens(input_ids[0])]

        baseline_atts = collect_attentions(model, input_ids, attention_mask, args, use_xsa=False)
        xsa_atts = collect_attentions(model, input_ids, attention_mask, args, use_xsa=True)
        summary = summarize_attention_deltas(baseline_atts, xsa_atts)

        if args.layer < 0:
            top = summary["layer_head_summary"][0]
            chosen_layer = int(top["layer"])
            chosen_head = int(top["head"]) if args.head < 0 else int(args.head)
        else:
            chosen_layer = int(args.layer)
            if args.head < 0:
                chosen_head = next(
                    (int(row["head"]) for row in summary["layer_head_summary"] if int(row["layer"]) == chosen_layer),
                    0,
                )
            else:
                chosen_head = int(args.head)

        if chosen_layer < 0 or chosen_layer >= len(baseline_atts):
            raise ValueError(f"Chosen layer {chosen_layer} out of range for {len(baseline_atts)} layers")
        n_heads = int(baseline_atts[chosen_layer].shape[1])
        if chosen_head < 0 or chosen_head >= n_heads:
            raise ValueError(f"Chosen head {chosen_head} out of range for layer {chosen_layer} with {n_heads} heads")

        baseline = baseline_atts[chosen_layer][0, chosen_head]
        xsa = xsa_atts[chosen_layer][0, chosen_head]
        layer_heads_to_plot = _select_layer_heads(summary["layer_head_summary"], max(1, int(args.num_layers_to_plot)))

        stem = f"{output_prefix.name}-sample{sample_idx:02d}-layer{chosen_layer:02d}-head{chosen_head:02d}"
        run_dir = output_prefix.with_name(stem)
        layers_dir = run_dir / "layers"
        image_path = run_dir / "focus.png"
        multi_image_path = run_dir / "top_multilayer.png"
        diag_plot_path = run_dir / "top_diaglines.png"
        json_path = run_dir / "summary.json"

        save_visualization(
            baseline=baseline,
            xsa=xsa,
            out_path=image_path,
            labels=token_labels,
            title_prefix=f"layer {chosen_layer} head {chosen_head}",
        )
        save_multi_layer_heatmaps(
            baseline_atts=baseline_atts,
            xsa_atts=xsa_atts,
            layer_heads=layer_heads_to_plot,
            out_path=multi_image_path,
            labels=token_labels,
        )
        save_multi_layer_diag_plot(
            baseline_atts=baseline_atts,
            xsa_atts=xsa_atts,
            layer_heads=layer_heads_to_plot,
            out_path=diag_plot_path,
        )
        per_layer_paths = save_all_layers_all_heads(
            baseline_atts=baseline_atts,
            xsa_atts=xsa_atts,
            out_dir=layers_dir,
            labels=token_labels,
        )

        payload = {
            "model_name": model_id,
            "raw_text": raw_text,
            "rendered_prompt": prompt,
            "sample_idx": sample_idx,
            "chosen_layer": chosen_layer,
            "chosen_head": chosen_head,
            "xsa_intervention_site": args.xsa_intervention_site,
            "xsa_layer_selection": {
                "xsa_start_layer": args.xsa_start_layer,
                "xsa_end_layer": args.xsa_end_layer,
                "xsa_skip_first_n": args.xsa_skip_first_n,
                "xsa_skip_last_n": args.xsa_skip_last_n,
            },
            "note": (
                "XSA modifies attention outputs after weights are formed, so the intervened layer's own attention "
                "matrix is usually unchanged; observed differences mainly appear in downstream layers."
            ),
            "top_layer_head_deltas": summary["layer_head_summary"][:20],
            "layer_heads_plotted": layer_heads_to_plot,
            "image_path": str(image_path),
            "multilayer_image_path": str(multi_image_path),
            "diag_plot_path": str(diag_plot_path),
            "per_layer_all_heads_image_paths": per_layer_paths,
            "tokens": token_labels,
            "baseline_diag": torch.diagonal(baseline).tolist(),
            "xsa_diag": torch.diagonal(xsa).tolist(),
            "diag_delta": torch.diagonal(xsa - baseline).tolist(),
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"[INFO] sample_idx={sample_idx} saved visualization to {image_path}", flush=True)
        print(f"[INFO] sample_idx={sample_idx} saved multi-layer visualization to {multi_image_path}", flush=True)
        print(f"[INFO] sample_idx={sample_idx} saved diag plot to {diag_plot_path}", flush=True)
        print(f"[INFO] sample_idx={sample_idx} saved per-layer all-head heatmaps to {layers_dir}", flush=True)
        print(f"[INFO] sample_idx={sample_idx} saved summary to {json_path}", flush=True)
        return payload

    if args.sample_idx == -1:
        all_payloads = []
        print(f"[INFO] Running attention visualization for all {len(texts)} prompts", flush=True)
        for sample_idx, raw_text in enumerate(texts):
            all_payloads.append(run_one_sample(sample_idx, raw_text))
        index_path = output_prefix.with_name(f"{output_prefix.name}-all_prompts_index.json")
        index_path.write_text(json.dumps({"sample_idx": -1, "runs": all_payloads}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] Saved all-prompt index to {index_path}", flush=True)
    else:
        if args.sample_idx < 0 or args.sample_idx >= len(texts):
            raise ValueError(f"sample_idx={args.sample_idx} out of range for {len(texts)} texts")
        run_one_sample(args.sample_idx, texts[args.sample_idx])


if __name__ == "__main__":
    main()
