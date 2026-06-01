from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from analysis.forward_geometry.qwen_xsa_forward_ablation import (
    QwenXSAForwardHooks,
    iter_batches,
    _patch_transformers_tp_plan_check,
    _require_transformers_version,
    _resolve_dtype,
)


DEFAULT_PROMPTS = [
    "Explain in two paragraphs why removing the attention branch component parallel to the residual stream might improve a pretrained decoder model.",
    "Solve the problem step by step: A factory produces 480 parts per hour. It runs for 7.5 hours, but 8% of the parts fail inspection. How many usable parts remain?",
    "Write a concise code review comment for a PyTorch forward hook that modifies attention outputs but accidentally drops cache outputs.",
    "请用英文解释：为什么降低 branch parallel ratio 不一定会降低最终 hidden state 和 residual stream 之间的 cosine similarity?",
    "A researcher observes that attention-only projection removal improves validation loss, but MLP-only projection removal hurts. Give three plausible explanations.",
]


def _read_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompts = []
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("prompt") or obj.get("text")
                if isinstance(text, str) and text.strip():
                    prompts.append(text)
    else:
        with p.open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def _format_generation_prompts(tokenizer, prompts: List[str], args) -> List[str]:
    if not args.use_chat_template:
        return prompts
    if not getattr(tokenizer, "chat_template", None):
        print("[WARN] Tokenizer has no chat_template; using raw prompts.", flush=True)
        return prompts

    rendered = []
    for prompt in prompts:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


def _decode_new_text(tokenizer, sequences: torch.Tensor, prompt_length: int) -> List[str]:
    out = []
    for seq in sequences:
        out.append(tokenizer.decode(seq[prompt_length:], skip_special_tokens=True))
    return out


def _iter_prompt_batches(prompts: List[str], batch_size: int, desc: str):
    total_batches = (len(prompts) + batch_size - 1) // batch_size if batch_size > 0 else 0
    batches = iter_batches(prompts, batch_size)
    if tqdm is None:
        for batch_idx, batch in enumerate(batches, start=1):
            print(f"[INFO] {desc}: batch {batch_idx}/{total_batches}", flush=True)
            yield batch
        return

    progress = tqdm(
        batches,
        total=total_batches,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    try:
        for batch in progress:
            yield batch
    finally:
        progress.close()


@torch.no_grad()
def _generate_batch(model, tokenizer, prompts: List[str], target: str, args) -> List[str]:
    model_inputs = _format_generation_prompts(tokenizer, prompts, args)
    enc = tokenizer(
        model_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    prompt_length = input_ids.size(1)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "use_cache": args.gen_use_cache,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    sequences = model.generate(**gen_kwargs)
    return _decode_new_text(tokenizer, sequences, prompt_length)


@torch.no_grad()
def generate_for_target(model, tokenizer, prompts: List[str], target: str, args) -> Dict:
    generations = []
    stats = {}
    layer_window = {}
    progress_prefix = f"{args.progress_label}: " if getattr(args, "progress_label", "") else ""
    desc = f"{progress_prefix}target={target}"

    if target == "none":
        for batch in _iter_prompt_batches(prompts, args.batch_size, desc):
            generations.extend(_generate_batch(model, tokenizer, batch, target, args))
    else:
        with QwenXSAForwardHooks(
            model,
            target=target,
            start_layer=args.xsa_start_layer,
            end_layer=args.xsa_end_layer,
            skip_first_n=args.xsa_skip_first_n,
            skip_last_n=args.xsa_skip_last_n,
            intervention_site=args.xsa_intervention_site,
            xsa_alpha=args.xsa_alpha,
            xsa_perp_scale=args.xsa_perp_scale,
            track_stats=args.xsa_track_stats,
            track_layerwise_stats=args.xsa_layerwise_stats,
        ) as hooks:
            for batch in _iter_prompt_batches(prompts, args.batch_size, desc):
                generations.extend(_generate_batch(model, tokenizer, batch, target, args))
            stats = hooks.stats.summary()
            layer_window = hooks.layer_window

    return {
        "target": target,
        "generations": generations,
        "xsa_stats": stats,
        "layer_window": layer_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen outputs with forward-only XSA ablations.")
    parser.add_argument("--model_name", type=str, required=True, help="HF model id or local model path.")
    parser.add_argument("--targets", type=str, default="none,attn,mlp,both", help="Comma-separated: none,attn,mlp,both.")
    parser.add_argument("--prompts_path", type=str, default=None, help="Optional txt lines or jsonl with prompt/text field.")
    parser.add_argument("--max_prompts", type=int, default=5)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--gen_use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether model.generate should use KV cache. Default matches normal generation.",
    )
    parser.add_argument("--xsa_start_layer", type=int, default=0, help="Inclusive first layer to modify.")
    parser.add_argument("--xsa_end_layer", type=int, default=-1, help="Exclusive end layer to modify; -1 means all layers.")
    parser.add_argument("--xsa_skip_first_n", type=int, default=0, help="Do not modify the first N decoder layers.")
    parser.add_argument("--xsa_skip_last_n", type=int, default=0, help="Do not modify the last N decoder layers.")
    parser.add_argument("--xsa_alpha", type=float, default=1.0, help="Scale for removed parallel component.")
    parser.add_argument("--xsa_perp_scale", type=float, default=1.0, help="Scale for perpendicular component.")
    parser.add_argument(
        "--xsa_track_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compute and save XSA geometric stats. Default is off to reduce overhead.",
    )
    parser.add_argument(
        "--xsa_layerwise_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to keep per-layer XSA stats. Default is off to reduce output size and overhead.",
    )
    parser.add_argument(
        "--xsa_intervention_site",
        type=str,
        default="xsa_middle_multihead",
        choices=["xsa_middle", "xsa_middle_multihead", "residual_output"],
        help=(
            "xsa_middle is attention-only and modifies the XSA middle pair on merged heads: "
            "attention v_proj -> o_proj input. xsa_middle_multihead applies the same idea per head. "
            "residual_output is the older approximation: "
            "residual stream input -> final sublayer output for attn/mlp/both."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render each prompt as a single-turn chat before generation. Recommended for Qwen Instruct models.",
    )
    parser.add_argument("--system_prompt", type=str, default="", help="Optional system message when using chat template.")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_jsonl", type=str, default="representation-analysis/outputs/qwen_xsa_forward_generations.jsonl")
    parser.add_argument("--prompt_idx_offset", type=int, default=0)
    parser.add_argument("--progress_label", type=str, default="")
    args = parser.parse_args()

    target_list = [x.strip() for x in args.targets.split(",") if x.strip()]
    bad_targets = sorted(set(target_list) - {"none", "attn", "mlp", "both"})
    if bad_targets:
        raise ValueError(f"Unknown targets: {bad_targets}")

    _require_transformers_version(args.model_name)

    if _patch_transformers_tp_plan_check():
        print(
            "[INFO] Patched Transformers ALL_PARALLEL_STYLES for Qwen TP-plan init check "
            "(generation does not use HF tensor parallelism).",
            flush=True,
        )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    prompts = _read_prompts(args.prompts_path)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    print(f"[INFO] Loaded {len(prompts)} prompts for generation", flush=True)

    all_results = {}
    for target in target_list:
        print(f"[INFO] Generating target={target}", flush=True)
        all_results[target] = generate_for_target(model, tokenizer, prompts, target, args)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pretty_records = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            rec = {
                "prompt_idx": args.prompt_idx_offset + i,
                "prompt": prompt,
                "generations": {
                    target: all_results[target]["generations"][i]
                    for target in target_list
                },
            }
            pretty_records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pretty_path = out_path.with_suffix(".pretty.json")
    pretty_path.write_text(json.dumps(pretty_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats_path = out_path.with_suffix(".stats.json")
    stats = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "gen_use_cache": bool(args.gen_use_cache),
        "use_chat_template": bool(args.use_chat_template),
        "system_prompt": args.system_prompt,
        "xsa_layer_selection": {
            "xsa_start_layer": args.xsa_start_layer,
            "xsa_end_layer": args.xsa_end_layer,
            "xsa_skip_first_n": args.xsa_skip_first_n,
            "xsa_skip_last_n": args.xsa_skip_last_n,
        },
        "xsa_track_stats": bool(args.xsa_track_stats),
        "xsa_layerwise_stats": bool(args.xsa_layerwise_stats),
        "xsa_alpha": float(args.xsa_alpha),
        "xsa_perp_scale": float(args.xsa_perp_scale),
        "xsa_intervention_site": args.xsa_intervention_site,
        "xsa_intervention_pair": (
            (
                "attn_v_proj_to_o_proj_input_merged_heads"
                if args.xsa_intervention_site == "xsa_middle"
                else (
                    "attn_v_proj_to_o_proj_input_multihead"
                    if args.xsa_intervention_site == "xsa_middle_multihead"
                    else "x_to_y_post"
                )
            )
        ),
        "targets": {
            target: {
                "xsa_stats": all_results[target]["xsa_stats"],
                "layer_window": all_results[target]["layer_window"],
            }
            for target in target_list
            if target != "none"
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for i, prompt in enumerate(prompts):
        print("=" * 100)
        print(f"[PROMPT {i}] {prompt}")
        for target in target_list:
            print("-" * 100)
            print(f"[{target}]")
            print(all_results[target]["generations"][i])

    print(f"[INFO] Wrote generations to {out_path}")
    print(f"[INFO] Wrote pretty generations to {pretty_path}")
    print(f"[INFO] Wrote XSA stats to {stats_path}")


if __name__ == "__main__":
    main()
