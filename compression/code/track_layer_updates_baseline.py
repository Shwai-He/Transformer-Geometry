#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import List

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompts(prompts_file: str, max_prompts: int) -> List[str]:
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def init_agg(num_layers: int, device: torch.device):
    return {
        "alpha": torch.zeros(num_layers, device=device),
        "para_abs": torch.zeros(num_layers, device=device),
        "perp_abs": torch.zeros(num_layers, device=device),
        "x_norm": torch.zeros(num_layers, device=device),
        "delta_norm": torch.zeros(num_layers, device=device),
        "delta_over_x": torch.zeros(num_layers, device=device),
        "para_over_x": torch.zeros(num_layers, device=device),
        "perp_over_x": torch.zeros(num_layers, device=device),
        "perp_ratio": torch.zeros(num_layers, device=device),
        "alpha_sq": torch.zeros(num_layers, device=device),
        "para_abs_sq": torch.zeros(num_layers, device=device),
        "perp_abs_sq": torch.zeros(num_layers, device=device),
        "x_norm_sq": torch.zeros(num_layers, device=device),
        "delta_norm_sq": torch.zeros(num_layers, device=device),
        "delta_over_x_sq": torch.zeros(num_layers, device=device),
        "para_over_x_sq": torch.zeros(num_layers, device=device),
        "perp_over_x_sq": torch.zeros(num_layers, device=device),
        "perp_ratio_sq": torch.zeros(num_layers, device=device),
        "count": torch.zeros(num_layers, device=device),
    }


def decompose(delta: torch.Tensor, base: torch.Tensor, eps: float = 1e-12):
    # delta/base: [B, H]
    if delta.device != base.device:
        delta = delta.to(base.device)
    base_norm_sq = (base * base).sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = (delta * base).sum(dim=-1, keepdim=True) / base_norm_sq
    delta_para = alpha * base
    delta_perp = delta - delta_para
    para_abs = delta_para.norm(dim=-1)
    perp_abs = delta_perp.norm(dim=-1)
    x_norm = base.norm(dim=-1).clamp_min(eps)
    delta_norm = delta.norm(dim=-1)
    delta_over_x = delta_norm / x_norm
    para_over_x = para_abs / x_norm
    perp_over_x = perp_abs / x_norm
    delta_norm_sq = (delta * delta).sum(dim=-1).clamp_min(eps)
    perp_ratio = (delta_perp * delta_perp).sum(dim=-1) / delta_norm_sq
    return alpha.squeeze(-1), para_abs, perp_abs, x_norm, delta_norm, delta_over_x, para_over_x, perp_over_x, perp_ratio


def save_plot(layer_ids, ys_by_name, ylabel, title, save_path):
    plt.figure(figsize=(10, 4.5))
    for name, ys in ys_by_name.items():
        plt.plot(layer_ids, ys, marker="o", linewidth=1.3, markersize=3, label=name)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Track baseline layer updates from dense model only.")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--prompts_file", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--method_name", type=str, default="dense_baseline_track")
    ap.add_argument(
        "--baseline_mode",
        type=str,
        default="aligned_drop",
        choices=["aligned_drop", "natural_update"],
        help=(
            "aligned_drop: compare keep vs dense drop/intervene path using shared x_ref. "
            "natural_update: report the dense model's natural update geometry."
        ),
    )
    ap.add_argument(
        "--ref_mode",
        type=str,
        default="baseline_update",
        choices=["baseline_update", "component_input", "component_output"],
        help=(
            "Reference used for para/perp decomposition and *_over_x normalization. "
            "baseline_update: use the dense baseline update for each component. "
            "component_input: use x_block/x_attn/x_mlp. "
            "component_output: use dense component outputs."
        ),
    )
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, args.method_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    num_layers = len(model.model.layers)
    layer_ids = list(range(num_layers))

    agg_block = init_agg(num_layers, device)
    agg_attn = init_agg(num_layers, device)
    agg_mlp = init_agg(num_layers, device)
    n = 0

    prompts = read_prompts(args.prompts_file, args.max_prompts)
    if not prompts:
        raise RuntimeError("No prompts loaded.")

    for prompt in prompts:
        residual_in = [None] * num_layers
        mlp_in = [None] * num_layers
        attn_out = [None] * num_layers
        mlp_out = [None] * num_layers
        block_out = [None] * num_layers
        handles = []

        for li, layer in enumerate(model.model.layers):
            def _pre_hook(_mod, args, kwargs, _li=li):
                h = kwargs.get("hidden_states", args[0] if args else None)
                if h is not None:
                    residual_in[_li] = h[:, -1, :].detach().float()
                return args, kwargs

            handles.append(layer.register_forward_pre_hook(_pre_hook, with_kwargs=True))

            if hasattr(layer, "self_attn") and layer.self_attn is not None:
                def _attn_hook(_mod, _args, out, _li=li):
                    y = out[0] if isinstance(out, tuple) else out
                    if y is not None:
                        attn_out[_li] = y[:, -1, :].detach().float()

                handles.append(layer.self_attn.register_forward_hook(_attn_hook))

            if hasattr(layer, "post_attention_layernorm") and layer.post_attention_layernorm is not None:
                def _mlp_pre_hook(_mod, args, _li=li):
                    h = args[0] if args else None
                    if h is not None:
                        mlp_in[_li] = h[:, -1, :].detach().float()

                handles.append(layer.post_attention_layernorm.register_forward_pre_hook(_mlp_pre_hook))

            if hasattr(layer, "mlp") and layer.mlp is not None:
                def _mlp_hook(_mod, _args, out, _li=li):
                    y = out[0] if isinstance(out, tuple) else out
                    if y is not None:
                        mlp_out[_li] = y[:, -1, :].detach().float()

                handles.append(layer.mlp.register_forward_hook(_mlp_hook))

            def _block_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                if y is not None:
                    block_out[_li] = y[:, -1, :].detach().float()

            handles.append(layer.register_forward_hook(_block_hook))

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": False,
                "use_cache": False,
                "return_dict": True,
            }
            model(**kwargs)

        for h in handles:
            h.remove()

        for li in range(num_layers):
            x_block = residual_in[li]
            x_attn = residual_in[li]
            x_mlp = mlp_in[li]
            a = attn_out[li]
            m = mlp_out[li]
            b = block_out[li]
            if x_block is None or x_attn is None or x_mlp is None or a is None or m is None or b is None:
                continue

            if args.baseline_mode == "aligned_drop":
                attn_delta = torch.zeros_like(a) - a
            else:
                attn_delta = a
            if args.ref_mode == "baseline_update":
                attn_ref = a
            elif args.ref_mode == "component_input":
                attn_ref = x_attn
            else:
                attn_ref = a
            alpha, para_abs, perp_abs, x_norm, delta_norm, delta_over_x, para_over_x, perp_over_x, perp_ratio = decompose(attn_delta, attn_ref)
            a_mean = alpha.mean()
            p_mean = para_abs.mean()
            q_mean = perp_abs.mean()
            xn_mean = x_norm.mean()
            dn_mean = delta_norm.mean()
            dx_mean = delta_over_x.mean()
            px_mean = para_over_x.mean()
            qx_mean = perp_over_x.mean()
            r_mean = perp_ratio.mean()
            agg_attn["alpha"][li] += a_mean
            agg_attn["para_abs"][li] += p_mean
            agg_attn["perp_abs"][li] += q_mean
            agg_attn["x_norm"][li] += xn_mean
            agg_attn["delta_norm"][li] += dn_mean
            agg_attn["delta_over_x"][li] += dx_mean
            agg_attn["para_over_x"][li] += px_mean
            agg_attn["perp_over_x"][li] += qx_mean
            agg_attn["perp_ratio"][li] += r_mean
            agg_attn["alpha_sq"][li] += a_mean * a_mean
            agg_attn["para_abs_sq"][li] += p_mean * p_mean
            agg_attn["perp_abs_sq"][li] += q_mean * q_mean
            agg_attn["x_norm_sq"][li] += xn_mean * xn_mean
            agg_attn["delta_norm_sq"][li] += dn_mean * dn_mean
            agg_attn["delta_over_x_sq"][li] += dx_mean * dx_mean
            agg_attn["para_over_x_sq"][li] += px_mean * px_mean
            agg_attn["perp_over_x_sq"][li] += qx_mean * qx_mean
            agg_attn["perp_ratio_sq"][li] += r_mean * r_mean
            agg_attn["count"][li] += 1.0

            if args.baseline_mode == "aligned_drop":
                mlp_delta = torch.zeros_like(m) - m
            else:
                mlp_delta = m
            if args.ref_mode == "baseline_update":
                mlp_ref = m
            elif args.ref_mode == "component_input":
                mlp_ref = x_mlp
            else:
                mlp_ref = m
            alpha, para_abs, perp_abs, x_norm, delta_norm, delta_over_x, para_over_x, perp_over_x, perp_ratio = decompose(mlp_delta, mlp_ref)
            a_mean = alpha.mean()
            p_mean = para_abs.mean()
            q_mean = perp_abs.mean()
            xn_mean = x_norm.mean()
            dn_mean = delta_norm.mean()
            dx_mean = delta_over_x.mean()
            px_mean = para_over_x.mean()
            qx_mean = perp_over_x.mean()
            r_mean = perp_ratio.mean()
            agg_mlp["alpha"][li] += a_mean
            agg_mlp["para_abs"][li] += p_mean
            agg_mlp["perp_abs"][li] += q_mean
            agg_mlp["x_norm"][li] += xn_mean
            agg_mlp["delta_norm"][li] += dn_mean
            agg_mlp["delta_over_x"][li] += dx_mean
            agg_mlp["para_over_x"][li] += px_mean
            agg_mlp["perp_over_x"][li] += qx_mean
            agg_mlp["perp_ratio"][li] += r_mean
            agg_mlp["alpha_sq"][li] += a_mean * a_mean
            agg_mlp["para_abs_sq"][li] += p_mean * p_mean
            agg_mlp["perp_abs_sq"][li] += q_mean * q_mean
            agg_mlp["x_norm_sq"][li] += xn_mean * xn_mean
            agg_mlp["delta_norm_sq"][li] += dn_mean * dn_mean
            agg_mlp["delta_over_x_sq"][li] += dx_mean * dx_mean
            agg_mlp["para_over_x_sq"][li] += px_mean * px_mean
            agg_mlp["perp_over_x_sq"][li] += qx_mean * qx_mean
            agg_mlp["perp_ratio_sq"][li] += r_mean * r_mean
            agg_mlp["count"][li] += 1.0

            if args.baseline_mode == "aligned_drop":
                # Full block deletion path uses y_drop = x_ref, aligned with local flip semantics.
                delta_block = x_block - b
            else:
                delta_block = b - x_block
            if args.ref_mode == "baseline_update":
                block_ref = b - x_block
            elif args.ref_mode == "component_input":
                block_ref = x_block
            else:
                block_ref = b
            alpha, para_abs, perp_abs, x_norm, delta_norm, delta_over_x, para_over_x, perp_over_x, perp_ratio = decompose(delta_block, block_ref)
            a_mean = alpha.mean()
            p_mean = para_abs.mean()
            q_mean = perp_abs.mean()
            xn_mean = x_norm.mean()
            dn_mean = delta_norm.mean()
            dx_mean = delta_over_x.mean()
            px_mean = para_over_x.mean()
            qx_mean = perp_over_x.mean()
            r_mean = perp_ratio.mean()
            agg_block["alpha"][li] += a_mean
            agg_block["para_abs"][li] += p_mean
            agg_block["perp_abs"][li] += q_mean
            agg_block["x_norm"][li] += xn_mean
            agg_block["delta_norm"][li] += dn_mean
            agg_block["delta_over_x"][li] += dx_mean
            agg_block["para_over_x"][li] += px_mean
            agg_block["perp_over_x"][li] += qx_mean
            agg_block["perp_ratio"][li] += r_mean
            agg_block["alpha_sq"][li] += a_mean * a_mean
            agg_block["para_abs_sq"][li] += p_mean * p_mean
            agg_block["perp_abs_sq"][li] += q_mean * q_mean
            agg_block["x_norm_sq"][li] += xn_mean * xn_mean
            agg_block["delta_norm_sq"][li] += dn_mean * dn_mean
            agg_block["delta_over_x_sq"][li] += dx_mean * dx_mean
            agg_block["para_over_x_sq"][li] += px_mean * px_mean
            agg_block["perp_over_x_sq"][li] += qx_mean * qx_mean
            agg_block["perp_ratio_sq"][li] += r_mean * r_mean
            agg_block["count"][li] += 1.0

        n += 1

    if n == 0:
        raise RuntimeError("No valid prompt processed.")

    for agg in (agg_block, agg_attn, agg_mlp):
        cnt = agg["count"].clamp_min(1.0)
        for k in ("alpha", "para_abs", "perp_abs", "x_norm", "delta_norm", "delta_over_x", "para_over_x", "perp_over_x", "perp_ratio"):
            mean = agg[k] / cnt
            sq_mean = agg[f"{k}_sq"] / cnt
            var = (sq_mean - mean * mean).clamp_min(0.0)
            std = torch.sqrt(var)
            agg[k] = mean.detach().cpu().tolist()
            agg[f"{k}_std"] = std.detach().cpu().tolist()
        agg["count"] = agg["count"].detach().cpu().tolist()

    csv_path = os.path.join(out_dir, "layerwise_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "component", "layer",
            "alpha", "alpha_std",
            "para_abs", "para_abs_std",
            "perp_abs", "perp_abs_std",
            "x_norm", "x_norm_std",
            "delta_norm", "delta_norm_std",
            "delta_over_x", "delta_over_x_std",
            "para_over_x", "para_over_x_std",
            "perp_over_x", "perp_over_x_std",
            "perp_ratio", "perp_ratio_std",
            "count",
        ])
        comps = [("block_out", agg_block), ("attn_out", agg_attn), ("mlp_out", agg_mlp)]
        for cname, comp in comps:
            for li in layer_ids:
                w.writerow([
                    args.method_name, cname, li,
                    comp["alpha"][li], comp["alpha_std"][li],
                    comp["para_abs"][li], comp["para_abs_std"][li],
                    comp["perp_abs"][li], comp["perp_abs_std"][li],
                    comp["x_norm"][li], comp["x_norm_std"][li],
                    comp["delta_norm"][li], comp["delta_norm_std"][li],
                    comp["delta_over_x"][li], comp["delta_over_x_std"][li],
                    comp["para_over_x"][li], comp["para_over_x_std"][li],
                    comp["perp_over_x"][li], comp["perp_over_x_std"][li],
                    comp["perp_ratio"][li], comp["perp_ratio_std"][li],
                    comp["count"][li],
                ])

    summary = {
        "method": args.method_name,
        "model_name": args.model_name,
        "baseline_mode": args.baseline_mode,
        "ref_mode": args.ref_mode,
        "num_prompts": n,
        "block_out": {"mean_perp_ratio": float(sum(agg_block["perp_ratio"]) / len(agg_block["perp_ratio"]))},
        "attn_out": {"mean_perp_ratio": float(sum(agg_attn["perp_ratio"]) / len(agg_attn["perp_ratio"]))},
        "mlp_out": {"mean_perp_ratio": float(sum(agg_mlp["perp_ratio"]) / len(agg_mlp["perp_ratio"]))},
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plot(
        layer_ids,
        {
            f"{args.method_name}:block": agg_block["perp_ratio"],
            f"{args.method_name}:attn": agg_attn["perp_ratio"],
            f"{args.method_name}:mlp": agg_mlp["perp_ratio"],
        },
        ylabel="perp_ratio",
        title="Baseline Layer Update Perpendicular Ratio",
        save_path=os.path.join(out_dir, "fig_a_perp_ratio.png"),
    )
    save_plot(
        layer_ids,
        {
            f"{args.method_name}:block": agg_block["alpha"],
            f"{args.method_name}:attn": agg_attn["alpha"],
            f"{args.method_name}:mlp": agg_mlp["alpha"],
        },
        ylabel="alpha",
        title="Baseline Layer Update Alpha",
        save_path=os.path.join(out_dir, "fig_b_alpha.png"),
    )

    print(f"[INFO] done, output={os.path.abspath(out_dir)}")
    print(f"[INFO] csv={csv_path}")


if __name__ == "__main__":
    main()
