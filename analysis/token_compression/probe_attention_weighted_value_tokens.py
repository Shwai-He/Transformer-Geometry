#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_layers(model):
    for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise ValueError("Could not find decoder layers.")


def find_attn(layer):
    for name in ("self_attn", "attn", "attention"):
        mod = getattr(layer, name, None)
        if mod is not None and hasattr(mod, "v_proj"):
            return mod
    raise ValueError(f"Could not find attention v_proj in {type(layer).__name__}.")


def read_prompts(path: Path, max_prompts: int, min_chars: int) -> list[str]:
    prompts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                text = obj.get("text") or obj.get("prompt") or obj.get("document") or ""
            else:
                text = str(obj)
        except json.JSONDecodeError:
            text = line
        if len(text) < min_chars:
            continue
        prompts.append(text)
        if len(prompts) >= max_prompts:
            break
    return prompts


def component_stats(values: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    values = values.float()
    ref = ref.float()
    dot = (values * ref[None, :]).sum(dim=-1).abs()
    value_norm = values.norm(dim=-1).clamp_min(eps)
    ref_norm = ref.norm().clamp_min(eps)
    para = dot / ref_norm
    perp = (value_norm.square() - para.square()).clamp_min(0.0).sqrt()
    return {
        "value_norm": value_norm,
        "para": para,
        "perp": perp,
        "para_ratio": para / value_norm,
        "perp_ratio": perp / value_norm,
        "para_over_perp": para / perp.clamp_min(eps),
    }


def summarize(score: torch.Tensor, metrics: dict[str, torch.Tensor], attn_mass: torch.Tensor, frac: float, largest: bool):
    k = max(1, int(round(score.numel() * frac)))
    idx = torch.topk(score, k=k, largest=largest).indices
    return {
        "k": k,
        "mean_contrib_norm": float(metrics["value_norm"][idx].mean().item()),
        "mean_para_ratio": float(metrics["para_ratio"][idx].mean().item()),
        "mean_perp_ratio": float(metrics["perp_ratio"][idx].mean().item()),
        "mean_para_over_perp": float(metrics["para_over_perp"][idx].mean().item()),
        "mean_attention_mass": float(attn_mass[idx].mean().item()),
        "sum_attention_mass": float(attn_mass[idx].sum().item()),
    }


def capture_attention_inputs(model, layer_attns: dict[int, torch.nn.Module], input_ids: torch.Tensor):
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states") if kwargs else None
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                raise ValueError(f"Could not capture hidden_states for layer {layer_idx}.")
            captured[layer_idx] = hidden.detach()

        return hook

    for layer_idx, attn in layer_attns.items():
        handles.append(attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(layer_attns) - set(captured))
    if missing:
        raise ValueError(f"Missing captured attention inputs for layers: {missing}")
    if outputs.attentions is None:
        raise ValueError("Model did not return attentions. Use an eager attention implementation.")
    return captured, outputs.attentions


def num_heads(attn, config):
    h = getattr(attn, "num_heads", None) or getattr(config, "num_attention_heads", None)
    kv = (
        getattr(attn, "num_key_value_heads", None)
        or getattr(config, "num_key_value_heads", None)
        or h
    )
    if h is None or kv is None:
        raise ValueError("Could not infer attention head counts.")
    return int(h), int(kv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--prompts_file", default="compression/calibration/c4_wanda_ns128_seq2048_seed0.txt")
    p.add_argument("--output_dir", default="results/analysis/token_compression/attention_weighted_value_probe")
    p.add_argument("--layers", default="0,mid,last")
    p.add_argument("--max_prompts", type=int, default=4)
    p.add_argument("--min_chars", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--top_frac", type=float, default=0.10)
    p.add_argument("--preserve_first", type=int, default=32)
    p.add_argument("--preserve_last", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--local_files_only", action="store_true", default=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device).eval()

    layers = find_layers(model)
    layer_specs = []
    for raw in args.layers.split(","):
        raw = raw.strip()
        if raw == "mid":
            layer_specs.append(len(layers) // 2)
        elif raw == "last":
            layer_specs.append(len(layers) - 1)
        elif raw:
            layer_specs.append(int(raw))
    layer_specs = sorted(set(max(0, min(len(layers) - 1, x)) for x in layer_specs))
    layer_attns = {layer_idx: find_attn(layers[layer_idx]) for layer_idx in layer_specs}

    prompts = read_prompts(Path(args.prompts_file), args.max_prompts, args.min_chars)
    special_ids = set(tok.all_special_ids or [])
    rows = []
    token_rows = []

    for prompt_idx, text in enumerate(prompts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(args.device)
        seq_len = input_ids.shape[1]
        if seq_len <= args.preserve_first + args.preserve_last + 4:
            continue

        candidate = torch.ones(seq_len, device=args.device, dtype=torch.bool)
        candidate[: args.preserve_first] = False
        candidate[max(0, seq_len - args.preserve_last) :] = False
        if special_ids:
            ids = input_ids[0]
            special = torch.zeros(seq_len, device=args.device, dtype=torch.bool)
            for sid in special_ids:
                special |= ids == int(sid)
            candidate &= ~special
        cand_idx = torch.nonzero(candidate, as_tuple=False).flatten()

        attention_inputs, attentions = capture_attention_inputs(model, layer_attns, input_ids)
        for layer_idx in layer_specs:
            attn = layer_attns[layer_idx]
            n_heads, n_kv_heads = num_heads(attn, model.config)
            hidden = attention_inputs[layer_idx]
            values = attn.v_proj(hidden).squeeze(0).float()
            head_dim = values.shape[-1] // n_kv_heads
            values = values.view(seq_len, n_kv_heads, head_dim)
            repeat = n_heads // n_kv_heads
            values = values.repeat_interleave(repeat, dim=1)
            attn_last = attentions[layer_idx][0, :, -1, :].float()
            contrib = (attn_last.transpose(0, 1).unsqueeze(-1) * values).reshape(seq_len, n_heads * head_dim)
            attn_mass = attn_last.mean(dim=0)[cand_idx]
            ref = contrib[cand_idx].sum(dim=0)
            metrics = component_stats(contrib[cand_idx], ref)
            buckets = {
                "high_parallel_contribution": summarize(metrics["para_ratio"], metrics, attn_mass, args.top_frac, True),
                "low_perp_contribution": summarize(metrics["perp_ratio"], metrics, attn_mass, args.top_frac, False),
                "high_perp_contribution": summarize(metrics["perp_ratio"], metrics, attn_mass, args.top_frac, True),
                "low_attention": summarize(attn_mass, metrics, attn_mass, args.top_frac, False),
            }
            base = {
                "prompt_idx": prompt_idx,
                "seq_len": seq_len,
                "layer": layer_idx,
                "candidate_tokens": int(cand_idx.numel()),
                "top_frac": args.top_frac,
                "all_mean_para_ratio": float(metrics["para_ratio"].mean().item()),
                "all_mean_perp_ratio": float(metrics["perp_ratio"].mean().item()),
                "all_sum_attention_mass": float(attn_mass.sum().item()),
            }
            for bucket, stats in buckets.items():
                row = dict(base)
                row["bucket"] = bucket
                row.update({f"{bucket}_{k}": v for k, v in stats.items()})
                rows.append(row)

            local_by_pos = {int(pos): local for local, pos in enumerate(cand_idx.tolist())}
            top_pos = cand_idx[torch.topk(metrics["para_ratio"], k=min(20, cand_idx.numel()), largest=True).indices]
            for rank, pos in enumerate(top_pos.tolist()):
                local = local_by_pos[int(pos)]
                token_rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "layer": layer_idx,
                        "rank": rank,
                        "position": pos,
                        "token": tok.decode([int(input_ids[0, pos].item())], skip_special_tokens=False).replace("\n", "\\n"),
                        "attention_mass": float(attn_mass[local].item()),
                        "para_ratio": float(metrics["para_ratio"][local].item()),
                        "perp_ratio": float(metrics["perp_ratio"][local].item()),
                        "para_over_perp": float(metrics["para_over_perp"][local].item()),
                    }
                )

    summary_path = out_dir / "summary.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    token_path = out_dir / "top_parallel_contribution_tokens.csv"
    token_fields = ["prompt_idx", "layer", "rank", "position", "token", "attention_mass", "para_ratio", "perp_ratio", "para_over_perp"]
    with token_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=token_fields)
        writer.writeheader()
        writer.writerows(token_rows)

    grouped = {}
    for row in rows:
        key = (row["layer"], row["bucket"])
        grouped.setdefault(key, []).append(row)
    readme_path = out_dir / "README.md"
    with readme_path.open("w") as f:
        f.write("# Attention-Weighted Value-Space Token Probe\n\n")
        f.write(
            "This probe estimates token-drop impact at the last query position. "
            "For each selected layer, it decomposes each candidate token contribution `alpha_{q,t} v_t` "
            "into components parallel/perpendicular to the aggregate candidate-token contribution.\n\n"
        )
        f.write("| Layer | Bucket | Para Ratio | Perp Ratio | Mean Attn | Sum Attn |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for (layer, bucket), vals in sorted(grouped.items()):
            def mean_metric(name: str) -> float:
                metric_name = f"{bucket}_{name}"
                return sum(float(v[metric_name]) for v in vals) / len(vals)

            f.write(
                f"| {layer} | {bucket} | {mean_metric('mean_para_ratio'):.3f} | "
                f"{mean_metric('mean_perp_ratio'):.3f} | {mean_metric('mean_attention_mass'):.6f} | "
                f"{mean_metric('sum_attention_mass'):.3f} |\n"
            )
        f.write(f"\nFull tables: `{summary_path.name}`, `{token_path.name}`.\n")

    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {token_path}")
    print(f"[OK] wrote {readme_path}")


if __name__ == "__main__":
    main()
