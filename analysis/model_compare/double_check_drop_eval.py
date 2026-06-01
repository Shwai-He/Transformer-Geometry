import argparse
import json
import os
import re
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.utils.generation_forward_utils import apply_drop_masks, generate_with_custom_forward


MATH_PROMPTS = [
    {"difficulty": "easy", "prompt": "14 + 9 - 6 = ?"},
    {"difficulty": "easy", "prompt": "18 - 7 + 5 = ?"},
    {"difficulty": "easy", "prompt": "6 * 3 - 8 = ?"},
    {"difficulty": "medium", "prompt": "(16 + 8) / 4 + 3 = ?"},
    {"difficulty": "medium", "prompt": "7 * (5 - 2) + 4 = ?"},
    {"difficulty": "medium", "prompt": "45 / 9 + 6 * 2 = ?"},
    {"difficulty": "hard", "prompt": "3 * (12 - 5) + 18 / 3 = ?"},
    {"difficulty": "hard", "prompt": "(48 / 6) * (9 - 4) - 7 = ?"},
    {"difficulty": "hard", "prompt": "64 / (4 + 4) + 7 * 3 = ?"},
    {"difficulty": "hard", "prompt": "(8 * 7) - (36 / 6) + 5 = ?"},
]


def parse_int_list(values: str) -> List[int]:
    out: List[int] = []
    for part in values.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("--max_new_tokens_values cannot be empty")
    return out


def build_prompt(raw_prompt: str, brief_answer: bool) -> str:
    if not brief_answer:
        return raw_prompt
    return (
        "Solve this arithmetic problem. "
        "Output only the final numeric answer (no explanation, no steps).\n"
        f"Problem: {raw_prompt}\n"
        "Answer:"
    )


def extract_first_number(text: str) -> str:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return m.group(0) if m else text.strip()


def dropped_config_path(dropped_root: str, model_tag: str, target_layer: str, drop_n: int) -> str:
    if target_layer in ["attn", "mlp"]:
        return os.path.join(
            dropped_root,
            f"{model_tag}-layer_drop_{target_layer}-discrete-drop{drop_n}",
            "checkpoint",
            "config.json",
        )
    return os.path.join(
        dropped_root,
        "block_drop",
        f"{model_tag}-block_drop-{target_layer}-discrete-drop{drop_n}",
        "checkpoint",
        "config.json",
    )


def run_generation(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool,
) -> str:
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    texts, _, _, _, _, _ = generate_with_custom_forward(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=[prompt],
        max_length=prompt_len + max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=use_cache,
        collect_sublayer=False,
    )
    return texts[0]


def main():
    parser = argparse.ArgumentParser(description="Compare dense vs dropped outputs on math prompts.")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--dropped_root_path", type=str, required=False)
    parser.add_argument("--drop_n", type=int, default=8)
    parser.add_argument("--target_layer", type=str, default="attn", choices=["attn", "mlp", "all"])
    parser.add_argument("--max_new_tokens_values", type=str, default="8,16,32")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=159)
    parser.add_argument("--prompt_idx", type=int, default=None)
    # Use explicit 0/1 flags instead of store_true for easier in-file editing.
    parser.add_argument("--brief_answer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--answer_only_extract", type=int, choices=[0, 1], default=1)
    parser.add_argument("--output_json", type=str, default="./drop_logs_clean/math_outputs.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    max_new_tokens_values = parse_int_list(args.max_new_tokens_values)

    # Keep exactly your original path logic.
    model_postfix = "Qwen/Qwen2.5-7B-Instruct"
    args.model_name = f"/mnt/bn/seed-aws-va/shwai.he/models/{model_postfix}-copy"
    args.drop_n = 8
    args.dropped_root_path = "/mnt/bn/seed-aws-va/shwai.he/LLM-Drop/results_prune"
    args.target_layer = "mlp"
    # In-file switches (0/1) to match your editing style.
    args.brief_answer = 1
    args.answer_only_extract = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_tag = model_postfix
    args.output_json = f"/mnt/bn/seed-aws-va/shwai.he/gen-collapse/drop_logs_math/{model_tag}/{args.target_layer}-drop-{args.drop_n:02d}.json"
    args.brief_answer = bool(args.brief_answer)
    args.answer_only_extract = bool(args.answer_only_extract)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    drop_attn_list: List[int] = []
    drop_mlp_list: List[int] = []
    config_path = None

    if args.drop_n > 0:
        config_path = dropped_config_path(args.dropped_root_path, model_tag, args.target_layer, args.drop_n)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Dropped config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        drop_attn_list = cfg.get("drop_attn_list", [])
        drop_mlp_list = cfg.get("drop_mlp_list", [])

    results = {
        "model_name": args.model_name,
        "model_tag": model_tag,
        "target_layer": args.target_layer,
        "drop_n": args.drop_n,
        "config_path": config_path,
        "brief_answer": args.brief_answer,
        "max_new_tokens_values": max_new_tokens_values,
        "rows": [],
    }

    for i, item in enumerate(MATH_PROMPTS):
        if args.prompt_idx is not None and i != args.prompt_idx:
            continue

        raw_prompt = item["prompt"]
        difficulty = item["difficulty"]
        prompt = build_prompt(raw_prompt, args.brief_answer)

        per_len_outputs = []
        print(f"[{i}] ({difficulty}) {raw_prompt}")

        for max_new_tokens in max_new_tokens_values:
            apply_drop_masks(model, target_layer="all", drop_attn_list=[], drop_mlp_list=[], drop_n=args.drop_n)
            dense_full = run_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
            )

            dropped_full = None
            if args.drop_n > 0:
                apply_drop_masks(
                    model=model,
                    target_layer=args.target_layer,
                    drop_attn_list=drop_attn_list,
                    drop_mlp_list=drop_mlp_list,
                    drop_n=args.drop_n,
                )
                dropped_full = run_generation(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    use_cache=True,
                )

            dense_raw = dense_full.replace(prompt, "").strip()
            dropped_raw = dropped_full.replace(prompt, "").strip() if dropped_full is not None else None

            if args.answer_only_extract:
                dense_answer = extract_first_number(dense_raw)
                dropped_answer = extract_first_number(dropped_raw) if dropped_raw is not None else None
            else:
                dense_answer = dense_raw
                dropped_answer = dropped_raw

            print(f"  [max_new_tokens={max_new_tokens}] dense  : {dense_answer}")
            if dropped_answer is not None:
                print(f"  [max_new_tokens={max_new_tokens}] dropped: {dropped_answer}")

            per_len_outputs.append(
                {
                    "max_new_tokens": max_new_tokens,
                    "dense_raw": dense_raw,
                    "dropped_raw": dropped_raw,
                    "dense_output": dense_answer,
                    "dropped_output": dropped_answer,
                }
            )

        row = {
            "idx": i,
            "difficulty": difficulty,
            "prompt": raw_prompt,
            "formatted_prompt": prompt,
            "outputs": per_len_outputs,
        }
        results["rows"].append(row)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {args.output_json}")


if __name__ == "__main__":
    main()
