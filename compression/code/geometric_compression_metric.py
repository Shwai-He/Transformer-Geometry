#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "Transformer compression should preserve the directions that matter for model behavior.",
    "A geometric diagnostic can separate rescaling-like error from direction-changing error.",
]


def _project_parallel(delta: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = (ref * ref).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = (delta * ref).sum(dim=-1, keepdim=True) / denom
    return coeff * ref


def _metric(error: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    error = error.float()
    ref = ref.float()
    para = _project_parallel(error, ref, eps=eps)
    perp = error - para

    error_norm = error.norm(dim=-1)
    ref_norm = ref.norm(dim=-1).clamp_min(eps)
    para_norm = para.norm(dim=-1)
    perp_norm = perp.norm(dim=-1)
    error_sq = (error * error).sum(dim=-1).clamp_min(eps)
    signed_alpha = ((error * ref).sum(dim=-1) / (ref * ref).sum(dim=-1).clamp_min(eps))

    return {
        "error_norm": float(error_norm.mean().item()),
        "ref_norm": float(ref_norm.mean().item()),
        "error_over_ref": float((error_norm / ref_norm).mean().item()),
        "parallel_over_ref": float((para_norm / ref_norm).mean().item()),
        "perpendicular_over_ref": float((perp_norm / ref_norm).mean().item()),
        "perp_energy_ratio": float(((perp * perp).sum(dim=-1) / error_sq).mean().item()),
        "signed_alpha": float(signed_alpha.mean().item()),
    }


def _append_metric(
    rows: list[dict[str, Any]],
    *,
    space: str,
    component: str,
    layer: int,
    error: torch.Tensor | None,
    ref: torch.Tensor | None,
) -> None:
    if error is None or ref is None:
        return
    stats = _metric(error, ref)
    rows.append({"space": space, "component": component, "layer": layer, **stats})


def _read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [args.prompt]
    if args.prompts_file:
        path = Path(args.prompts_file)
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if prompts:
            return prompts[: args.max_prompts]
    return DEFAULT_PROMPTS[: args.max_prompts]


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _last_token(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :].detach().float().cpu()


def _repeat_kv_value_to_query_width(mod: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    num_heads = int(getattr(mod, "num_heads", getattr(mod, "num_attention_heads", 0)) or 0)
    num_kv_heads = int(getattr(mod, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(mod, "head_dim", 0) or 0)
    o_proj = getattr(mod, "o_proj", None)
    v_proj = getattr(mod, "v_proj", None)
    if head_dim > 0 and num_heads <= 0 and hasattr(o_proj, "in_features"):
        num_heads = int(o_proj.in_features) // head_dim
    if head_dim > 0 and num_kv_heads <= 0 and hasattr(v_proj, "out_features"):
        num_kv_heads = int(v_proj.out_features) // head_dim
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        return value
    expected = num_kv_heads * head_dim
    if value.shape[-1] != expected or num_heads == num_kv_heads:
        return value
    repeat = max(num_heads // num_kv_heads, 1)
    value = value.view(value.shape[0], value.shape[1], num_kv_heads, head_dim)
    value = value.repeat_interleave(repeat, dim=2)
    return value.reshape(value.shape[0], value.shape[1], num_heads * head_dim)


def _capture_model(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, list[Any]]:
    layers = model.model.layers
    n_layers = len(layers)
    out: dict[str, list[Any]] = {
        "hidden": [None] * (n_layers + 1),
        "block_ref": [None] * n_layers,
        "attn_out": [None] * n_layers,
        "attn_ref": [None] * n_layers,
        "mlp_out": [None] * n_layers,
        "mlp_ref": [None] * n_layers,
        "value_agg": [None] * n_layers,
        "value_self": [None] * n_layers,
    }
    handles = []

    for li, layer in enumerate(layers):
        def layer_pre_hook(_mod, args, kwargs, _li=li):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None:
                out["block_ref"][_li] = _last_token(hidden)
                out["attn_ref"][_li] = _last_token(hidden)
            return args, kwargs

        handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            def attn_pre_hook(mod, args, kwargs, _li=li):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is None or not hasattr(mod, "v_proj"):
                    return args, kwargs
                with torch.no_grad():
                    value = mod.v_proj(hidden.to(_module_device(mod)))
                    value = _repeat_kv_value_to_query_width(mod, value)
                out["value_self"][_li] = _last_token(value)
                return args, kwargs

            def attn_hook(_mod, _args, output, _li=li):
                y = output[0] if isinstance(output, tuple) else output
                if y is not None:
                    out["attn_out"][_li] = _last_token(y)

            handles.append(attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True))
            handles.append(attn.register_forward_hook(attn_hook))

            o_proj = getattr(attn, "o_proj", None)
            if o_proj is not None:
                def o_proj_pre_hook(_mod, args, _li=li):
                    if args and torch.is_tensor(args[0]):
                        out["value_agg"][_li] = _last_token(args[0])
                    return args

                handles.append(o_proj.register_forward_pre_hook(o_proj_pre_hook))

        norm = getattr(layer, "post_attention_layernorm", None)
        if norm is not None:
            def mlp_pre_hook(_mod, args, _li=li):
                hidden = args[0] if args else None
                if hidden is not None:
                    out["mlp_ref"][_li] = _last_token(hidden)
                return args

            handles.append(norm.register_forward_pre_hook(mlp_pre_hook))

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            def mlp_hook(_mod, _args, output, _li=li):
                y = output[0] if isinstance(output, tuple) else output
                if y is not None:
                    out["mlp_out"][_li] = _last_token(y)

            handles.append(mlp.register_forward_hook(mlp_hook))

    with torch.no_grad():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    for h in handles:
        h.remove()

    for i, hidden in enumerate(result.hidden_states):
        out["hidden"][i] = _last_token(hidden)
    return out


def _load_model(path: str, device: str, dtype: str, local_files_only: bool) -> torch.nn.Module:
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if dtype_map[dtype] is not None:
        kwargs["torch_dtype"] = dtype_map[dtype]
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual/value-space geometric compression metric.")
    parser.add_argument("--dense_model", required=True)
    parser.add_argument("--compressed_model", default="", help="Defaults to dense_model for a zero-error sanity check.")
    parser.add_argument("--method_name", default="sanity_dense_vs_dense")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--max_prompts", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="compression/outputs/geometric_compression_metric")
    args = parser.parse_args()

    dense_path = args.dense_model
    comp_path = args.compressed_model or args.dense_model
    prompts = _read_prompts(args)

    tokenizer = AutoTokenizer.from_pretrained(
        dense_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dense = _load_model(dense_path, args.device, args.dtype, args.local_files_only)
    comp = _load_model(comp_path, args.device, args.dtype, args.local_files_only)

    n_layers = len(dense.model.layers)
    sum_rows: dict[tuple[str, str, int], dict[str, float]] = {}
    counts: dict[tuple[str, str, int], int] = {}

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)

        dense_cap = _capture_model(dense, input_ids, attention_mask)
        comp_cap = _capture_model(comp, input_ids, attention_mask)
        rows: list[dict[str, Any]] = []

        for li in range(n_layers):
            h0 = dense_cap["hidden"][li]
            h1 = dense_cap["hidden"][li + 1]
            h1c = comp_cap["hidden"][li + 1]
            if h0 is not None and h1 is not None and h1c is not None:
                _append_metric(
                    rows,
                    space="residual",
                    component="block_out",
                    layer=li,
                    error=h1c - h1,
                    ref=h1 - h0,
                )

            attn_dense = dense_cap["attn_out"][li]
            attn_comp = comp_cap["attn_out"][li]
            if attn_dense is not None and attn_comp is not None:
                _append_metric(
                    rows,
                    space="residual",
                    component="attn_out",
                    layer=li,
                    error=attn_comp - attn_dense,
                    ref=attn_dense,
                )

            mlp_dense = dense_cap["mlp_out"][li]
            mlp_comp = comp_cap["mlp_out"][li]
            if mlp_dense is not None and mlp_comp is not None:
                _append_metric(
                    rows,
                    space="residual",
                    component="mlp_out",
                    layer=li,
                    error=mlp_comp - mlp_dense,
                    ref=mlp_dense,
                )

            value_dense = dense_cap["value_agg"][li]
            value_comp = comp_cap["value_agg"][li]
            if value_dense is not None and value_comp is not None:
                value_error = value_comp - value_dense
                _append_metric(
                    rows,
                    space="value",
                    component="attn_value_agg_ref",
                    layer=li,
                    error=value_error,
                    ref=value_dense,
                )
                _append_metric(
                    rows,
                    space="value",
                    component="attn_value_self_ref",
                    layer=li,
                    error=value_error,
                    ref=dense_cap["value_self"][li],
                )

        for row in rows:
            key = (row["space"], row["component"], int(row["layer"]))
            if key not in sum_rows:
                sum_rows[key] = {k: 0.0 for k in row if k not in {"space", "component", "layer"}}
                counts[key] = 0
            counts[key] += 1
            for k, v in row.items():
                if k not in {"space", "component", "layer"}:
                    sum_rows[key][k] += float(v)

    out_dir = Path(args.output_dir) / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "layerwise_geometric_metrics.csv"
    fieldnames = [
        "method",
        "space",
        "component",
        "layer",
        "count",
        "error_norm",
        "ref_norm",
        "error_over_ref",
        "parallel_over_ref",
        "perpendicular_over_ref",
        "perp_energy_ratio",
        "signed_alpha",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(sum_rows):
            space, component, layer = key
            count = counts[key]
            row = {
                "method": args.method_name,
                "space": space,
                "component": component,
                "layer": layer,
                "count": count,
            }
            row.update({k: v / count for k, v in sum_rows[key].items()})
            writer.writerow(row)

    summary: dict[str, Any] = {
        "method": args.method_name,
        "dense_model": dense_path,
        "compressed_model": comp_path,
        "num_prompts": len(prompts),
        "num_layers": n_layers,
        "csv_path": str(csv_path),
        "means": {},
    }
    for component in sorted({k[1] for k in sum_rows}):
        vals = []
        perp_vals = []
        for key, metrics in sum_rows.items():
            if key[1] != component:
                continue
            count = counts[key]
            vals.append(metrics["error_over_ref"] / count)
            perp_vals.append(metrics["perpendicular_over_ref"] / count)
        if vals:
            summary["means"][component] = {
                "error_over_ref": sum(vals) / len(vals),
                "perpendicular_over_ref": sum(perp_vals) / len(perp_vals),
            }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Finished geometric compression metric: {out_dir}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
