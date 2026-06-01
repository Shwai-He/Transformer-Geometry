#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from layerwise_para_perp_compare import derive_model_tag, read_prompts
from select_layer_drop_by_geometry import _dtype_from_name, _parse_ints, _select_token_scope


EPS = 1e-12


@dataclass
class JointAccumulator:
    error_gram: dict[str, torch.Tensor]
    para_gram: dict[str, torch.Tensor]
    ref_sq_sum: dict[str, float]
    counts: dict[str, list[int]]


def _empty_accumulator(num_layers: int) -> JointAccumulator:
    return JointAccumulator(
        error_gram={component: torch.zeros((num_layers, num_layers), dtype=torch.float64) for component in ("attn", "mlp")},
        para_gram={component: torch.zeros((num_layers, num_layers), dtype=torch.float64) for component in ("attn", "mlp")},
        ref_sq_sum={component: 0.0 for component in ("attn", "mlp")},
        counts={component: [0 for _ in range(num_layers)] for component in ("attn", "mlp")},
    )


def collect_updates_and_final_ref(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_scope: str,
    metric_device: str,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], torch.Tensor]:
    num_layers = len(model.model.layers)
    attn_out: list[torch.Tensor | None] = [None] * num_layers
    mlp_out: list[torch.Tensor | None] = [None] * num_layers
    handles = []

    def capture(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        scoped = _select_token_scope(x, token_scope, attention_mask)
        return scoped.detach().to(device=metric_device, dtype=torch.float32)

    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and layer.self_attn is not None:
            def _attn_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                attn_out[_li] = capture(y)

            handles.append(layer.self_attn.register_forward_hook(_attn_hook))

        if hasattr(layer, "mlp") and layer.mlp is not None:
            def _mlp_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                mlp_out[_li] = capture(y)

            handles.append(layer.mlp.register_forward_hook(_mlp_hook))

    forward_params = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }
    if "attention_mask" in forward_params:
        kwargs["attention_mask"] = attention_mask

    try:
        with torch.no_grad():
            outputs = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    final_hidden = outputs.hidden_states[-1]
    final_ref = capture(final_hidden)
    if final_ref is None:
        raise RuntimeError("Failed to capture final hidden state.")
    return attn_out, mlp_out, final_ref


def _accumulate_component(
    acc: JointAccumulator,
    component: str,
    updates: list[torch.Tensor | None],
    final_ref: torch.Tensor,
) -> None:
    active_layers: list[int] = []
    errors: list[torch.Tensor] = []
    num_positions = int(final_ref.shape[0])
    for layer, update in enumerate(updates):
        if update is None:
            continue
        if update.shape != final_ref.shape:
            raise ValueError(
                f"Shape mismatch for {component} layer {layer}: update={tuple(update.shape)} "
                f"final_ref={tuple(final_ref.shape)}"
            )
        active_layers.append(layer)
        errors.append(-update)
        acc.counts[component][layer] += num_positions
    if not errors:
        return

    stacked = torch.stack(errors, dim=0)
    ref = final_ref.float()
    ref_sq = (ref * ref).sum(dim=-1).clamp_min(EPS)
    acc.ref_sq_sum[component] += float(ref_sq.sum().item())

    error_gram = torch.einsum("lnd,mnd->lm", stacked, stacked).detach().cpu().double()
    dot_ref = torch.einsum("lnd,nd->ln", stacked, ref)
    weighted_dot = dot_ref / torch.sqrt(ref_sq).unsqueeze(0)
    para_gram = (weighted_dot @ weighted_dot.t()).detach().cpu().double()

    for i, layer_i in enumerate(active_layers):
        for j, layer_j in enumerate(active_layers):
            acc.error_gram[component][layer_i, layer_j] += error_gram[i, j]
            acc.para_gram[component][layer_i, layer_j] += para_gram[i, j]


def _set_stats(
    error_gram: torch.Tensor,
    para_gram: torch.Tensor,
    ref_sq_sum: float,
    layers: list[int],
) -> dict[str, float]:
    if not layers:
        return {
            "error_sq": 0.0,
            "para_sq": 0.0,
            "perp_sq": 0.0,
            "error_over_final": 0.0,
            "perp_over_final": 0.0,
            "joint_perp_ratio_final": 0.0,
        }
    idx = torch.tensor(layers, dtype=torch.long)
    error_sq = float(error_gram[idx][:, idx].sum().item())
    para_sq = float(para_gram[idx][:, idx].sum().item())
    perp_sq = max(error_sq - para_sq, 0.0)
    ref_sq = max(ref_sq_sum, EPS)
    return {
        "error_sq": error_sq,
        "para_sq": para_sq,
        "perp_sq": perp_sq,
        "error_over_final": math.sqrt(max(error_sq, 0.0) / ref_sq),
        "perp_over_final": math.sqrt(perp_sq / ref_sq),
        "joint_perp_ratio_final": perp_sq / max(error_sq, EPS),
    }


def _score_from_stats(stats: dict[str, float], rank_metric: str) -> float:
    if rank_metric == "joint_perp_over_final":
        return stats["perp_over_final"]
    if rank_metric == "joint_perp_ratio_final":
        return stats["joint_perp_ratio_final"]
    if rank_metric == "joint_error_over_final":
        return stats["error_over_final"]
    raise ValueError(f"Unsupported rank_metric={rank_metric}")


def _greedy_select(
    error_gram: torch.Tensor,
    para_gram: torch.Tensor,
    ref_sq_sum: float,
    counts: list[int],
    drop_counts: list[int],
    rank_metric: str,
    min_gap: int,
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    max_count = max(drop_counts) if drop_counts else 0
    candidates = [layer for layer, count in enumerate(counts) if count > 0]
    selected: list[int] = []
    step_rows: list[dict[str, Any]] = []

    for step in range(1, min(max_count, len(candidates)) + 1):
        best_layer: int | None = None
        best_stats: dict[str, float] | None = None
        best_score = float("inf")
        for layer in candidates:
            if layer in selected:
                continue
            if min_gap > 0 and any(abs(layer - prev) <= min_gap for prev in selected):
                continue
            trial_layers = selected + [layer]
            stats = _set_stats(error_gram, para_gram, ref_sq_sum, trial_layers)
            score = _score_from_stats(stats, rank_metric)
            if (score, layer) < (best_score, best_layer if best_layer is not None else 10**9):
                best_score = score
                best_layer = layer
                best_stats = stats
        if best_layer is None:
            for layer in candidates:
                if layer in selected:
                    continue
                trial_layers = selected + [layer]
                stats = _set_stats(error_gram, para_gram, ref_sq_sum, trial_layers)
                score = _score_from_stats(stats, rank_metric)
                if (score, layer) < (best_score, best_layer if best_layer is not None else 10**9):
                    best_score = score
                    best_layer = layer
                    best_stats = stats
        if best_layer is None or best_stats is None:
            break
        selected.append(best_layer)
        step_rows.append({"step": step, "layer": best_layer, "layers": ",".join(str(x) for x in selected), **best_stats})

    recommendations = {}
    for count in drop_counts:
        recommendations[f"drop_{count}"] = selected[:count]
    return recommendations, step_rows


def _load_calibration_meta(prompts_file: str) -> dict[str, Any]:
    if not prompts_file:
        return {}
    meta_path = Path(f"{prompts_file}.meta.json")
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {f"calibration_{key}": value for key, value in meta.items()}


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select layer-drop sets by joint residual error against the final hidden state."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_tag", default="")
    parser.add_argument("--output_tag", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_prompts", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--token_scope", choices=["all", "last"], default="all")
    parser.add_argument("--drop_counts", default="4,8")
    parser.add_argument(
        "--rank_metric",
        choices=["joint_perp_over_final", "joint_perp_ratio_final", "joint_error_over_final"],
        default="joint_perp_over_final",
    )
    parser.add_argument("--min_gap", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric_device", default="")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="compression/outputs/layer_drop_geometry")
    args = parser.parse_args()

    metric_device = args.metric_device or args.device
    model_tag = args.model_tag or derive_model_tag(args.model_name)
    tag = args.output_tag or f"{args.rank_metric}_{args.token_scope}"
    if args.min_gap:
        tag = f"{tag}_gap{args.min_gap}"
    out_dir = Path(args.output_dir) / model_tag / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    dtype = _dtype_from_name(args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(args.device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    prompts = read_prompts(args)
    num_layers = len(model.model.layers)
    acc = _empty_accumulator(num_layers)

    for idx, prompt in enumerate(prompts, start=1):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        attn_out, mlp_out, final_ref = collect_updates_and_final_ref(
            model, input_ids, attention_mask, args.token_scope, metric_device
        )
        _accumulate_component(acc, "attn", attn_out, final_ref)
        _accumulate_component(acc, "mlp", mlp_out, final_ref)
        print(f"[PROMPT] {idx}/{len(prompts)} positions={final_ref.shape[0]}", flush=True)

    drop_counts = _parse_ints(args.drop_counts)
    recommendations: dict[str, dict[str, list[int]]] = {}
    step_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for component in ("attn", "mlp"):
        recs, steps = _greedy_select(
            acc.error_gram[component],
            acc.para_gram[component],
            acc.ref_sq_sum[component],
            acc.counts[component],
            drop_counts,
            args.rank_metric,
            args.min_gap,
        )
        recommendations[component] = recs
        for row in steps:
            step_rows.append({"component": component, **row})
        for layer in range(num_layers):
            stats = _set_stats(
                acc.error_gram[component],
                acc.para_gram[component],
                acc.ref_sq_sum[component],
                [layer],
            )
            metric_rows.append(
                {
                    "component": component,
                    "layer": layer,
                    "count": acc.counts[component][layer],
                    **stats,
                }
            )

    metric_csv = out_dir / "joint_residual_layer_metrics.csv"
    _write_rows(
        metric_csv,
        metric_rows,
        [
            "component",
            "layer",
            "count",
            "error_sq",
            "para_sq",
            "perp_sq",
            "error_over_final",
            "perp_over_final",
            "joint_perp_ratio_final",
        ],
    )
    step_csv = out_dir / "joint_residual_greedy_steps.csv"
    _write_rows(
        step_csv,
        step_rows,
        [
            "component",
            "step",
            "layer",
            "layers",
            "error_sq",
            "para_sq",
            "perp_sq",
            "error_over_final",
            "perp_over_final",
            "joint_perp_ratio_final",
        ],
    )

    summary = {
        "model_name": args.model_name,
        "model_tag": model_tag,
        "output_tag": tag,
        "prompts_file": args.prompts_file,
        "num_prompts": len(prompts),
        "max_prompts": args.max_prompts,
        "max_length": args.max_length,
        "token_scope": args.token_scope,
        "num_layers": num_layers,
        "rank_metric": tag,
        "score_name": args.rank_metric,
        "score_formula": (
            "Greedy set selection minimizing the chosen metric of E_S = -sum_{l in S} update_l "
            "after decomposing E_S into parallel/perpendicular components relative to the final hidden state."
        ),
        "min_gap": args.min_gap,
        "drop_counts": drop_counts,
        "recommendations": recommendations,
        "metric_csv": str(metric_csv),
        "step_csv": str(step_csv),
    }
    summary.update(_load_calibration_meta(args.prompts_file))
    json_path = out_dir / "drop_selection.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
