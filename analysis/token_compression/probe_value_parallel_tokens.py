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
    para_ratio = para / value_norm
    perp_ratio = perp / value_norm
    para_over_perp = para / perp.clamp_min(eps)
    return {
        "value_norm": value_norm,
        "para": para,
        "perp": perp,
        "para_ratio": para_ratio,
        "perp_ratio": perp_ratio,
        "para_over_perp": para_over_perp,
    }


def summarize_top(values: torch.Tensor, score: torch.Tensor, metrics: dict[str, torch.Tensor], frac: float, largest: bool):
    n = score.numel()
    k = max(1, int(round(n * frac)))
    idx = torch.topk(score, k=k, largest=largest).indices
    return {
        "k": k,
        "mean_value_norm": float(metrics["value_norm"][idx].mean().item()),
        "mean_para_ratio": float(metrics["para_ratio"][idx].mean().item()),
        "mean_perp_ratio": float(metrics["perp_ratio"][idx].mean().item()),
        "mean_para_over_perp": float(metrics["para_over_perp"][idx].mean().item()),
    }


def capture_attention_inputs(model, layer_attns: dict[int, torch.nn.Module], input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
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
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(layer_attns) - set(captured))
    if missing:
        raise ValueError(f"Missing captured attention inputs for layers: {missing}")
    return captured


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--prompts_file", default="compression/calibration/c4_wanda_ns128_seq2048_seed0.txt")
    p.add_argument("--output_dir", default="results/analysis/token_compression/value_parallel_probe")
    p.add_argument("--layers", default="0,mid,last")
    p.add_argument("--max_prompts", type=int, default=8)
    p.add_argument("--min_chars", type=int, default=2000)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--top_frac", type=float, default=0.10)
    p.add_argument("--preserve_first", type=int, default=32)
    p.add_argument("--preserve_last", type=int, default=256)
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

    prompts = read_prompts(Path(args.prompts_file), args.max_prompts, args.min_chars)
    rows = []
    token_rows = []
    special_ids = set(tok.all_special_ids or [])

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

        layer_attns = {layer_idx: find_attn(layers[layer_idx]) for layer_idx in layer_specs}
        attention_inputs = capture_attention_inputs(model, layer_attns, input_ids)

        for layer_idx in layer_specs:
            attn = layer_attns[layer_idx]
            with torch.no_grad():
                value = attn.v_proj(attention_inputs[layer_idx]).squeeze(0).float()
            ref_last = value[-1]
            ref_mean = value[cand_idx].mean(dim=0)
            for ref_name, ref in (("last_value", ref_last), ("mean_value", ref_mean)):
                metrics = component_stats(value[cand_idx], ref)
                para_score = metrics["para_ratio"]
                perp_score = metrics["perp_ratio"]
                high_para = summarize_top(value[cand_idx], para_score, metrics, args.top_frac, largest=True)
                low_perp = summarize_top(value[cand_idx], perp_score, metrics, args.top_frac, largest=False)
                high_perp = summarize_top(value[cand_idx], perp_score, metrics, args.top_frac, largest=True)
                row_base = {
                    "prompt_idx": prompt_idx,
                    "seq_len": seq_len,
                    "layer": layer_idx,
                    "ref": ref_name,
                    "candidate_tokens": int(cand_idx.numel()),
                    "top_frac": args.top_frac,
                    "all_mean_para_ratio": float(metrics["para_ratio"].mean().item()),
                    "all_mean_perp_ratio": float(metrics["perp_ratio"].mean().item()),
                    "all_mean_para_over_perp": float(metrics["para_over_perp"].mean().item()),
                }
                for prefix, stats in (("high_para", high_para), ("low_perp", low_perp), ("high_perp", high_perp)):
                    row = dict(row_base)
                    row["bucket"] = prefix
                    row.update({f"{prefix}_{k}": v for k, v in stats.items()})
                    rows.append(row)

                pos_to_local = {int(pos): local for local, pos in enumerate(cand_idx.tolist())}
                high_idx = cand_idx[torch.topk(para_score, k=min(20, para_score.numel()), largest=True).indices]
                for rank, pos in enumerate(high_idx.tolist()):
                    local = pos_to_local[int(pos)]
                    token_rows.append(
                        {
                            "prompt_idx": prompt_idx,
                            "layer": layer_idx,
                            "ref": ref_name,
                            "rank": rank,
                            "position": pos,
                            "token": tok.decode([int(input_ids[0, pos].item())], skip_special_tokens=False).replace("\n", "\\n"),
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
    examples_path = out_dir / "top_parallel_tokens.csv"
    ex_fields = ["prompt_idx", "layer", "ref", "rank", "position", "token", "para_ratio", "perp_ratio", "para_over_perp"]
    with examples_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ex_fields)
        writer.writeheader()
        writer.writerows(token_rows)

    # Paper-facing compact table: average by layer/ref/bucket.
    grouped = {}
    for row in rows:
        key = (row["layer"], row["ref"], row["bucket"])
        grouped.setdefault(key, []).append(row)
    md_path = out_dir / "README.md"
    with md_path.open("w") as f:
        f.write("# Value-Space Parallelness of Droppable Tokens\n\n")
        f.write(
            "This diagnostic asks whether some prompt tokens mainly contribute value-space parallel components. "
            "For each prompt and selected layer, candidate tokens exclude preserved prefix/suffix tokens. "
            "Value vectors are computed from the actual hidden states entering each layer's attention module, "
            "captured by forward hooks. We compare high-parallel, low-perpendicular, and high-perpendicular token buckets.\n\n"
        )
        f.write("| Layer | Ref | Bucket | Para Ratio | Perp Ratio | Para/Perp |\n")
        f.write("|---:|---|---|---:|---:|---:|\n")
        for (layer, ref, bucket), vals in sorted(grouped.items()):
            def mean_metric(name: str) -> float:
                metric_name = f"{bucket}_{name}"
                return sum(float(v[metric_name]) for v in vals) / len(vals)
            f.write(
                f"| {layer} | {ref} | {bucket} | {mean_metric('mean_para_ratio'):.3f} | {mean_metric('mean_perp_ratio'):.3f} | {mean_metric('mean_para_over_perp'):.3f} |\n"
            )
        f.write(f"\nFull tables: `{summary_path.name}`, `{examples_path.name}`.\n")

    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {examples_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
