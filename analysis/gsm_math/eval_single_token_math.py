import argparse
import ast
import json
import os
import re
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.utils.generation_forward_utils import apply_drop_masks, generate_with_custom_forward


MATH_PROMPTS = [
    {"difficulty": "ultra_easy", "prompt": "7 + 5 = ?"},
    {"difficulty": "ultra_easy", "prompt": "13 - 4 = ?"},
    {"difficulty": "ultra_easy", "prompt": "6 * 8 = ?"},
    {"difficulty": "ultra_easy", "prompt": "36 / 9 = ?"},
    {"difficulty": "ultra_easy", "prompt": "15 + 12 = ?"},
    {"difficulty": "ultra_easy", "prompt": "20 - 11 = ?"},
    {"difficulty": "ultra_easy", "prompt": "9 * 7 = ?"},
    {"difficulty": "ultra_easy", "prompt": "42 / 6 = ?"},
    {"difficulty": "ultra_easy", "prompt": "18 + 17 = ?"},
    {"difficulty": "ultra_easy", "prompt": "25 - 8 = ?"},
    {"difficulty": "ultra_easy", "prompt": "11 * 4 = ?"},
    {"difficulty": "ultra_easy", "prompt": "56 / 8 = ?"},
    {"difficulty": "ultra_easy", "prompt": "8 + 9 = ?"},
    {"difficulty": "ultra_easy", "prompt": "19 - 7 = ?"},
    {"difficulty": "ultra_easy", "prompt": "12 * 3 = ?"},
    {"difficulty": "ultra_easy", "prompt": "49 / 7 = ?"},
    {"difficulty": "ultra_easy", "prompt": "21 + 14 = ?"},
    {"difficulty": "ultra_easy", "prompt": "30 - 18 = ?"},
    {"difficulty": "ultra_easy", "prompt": "7 * 6 = ?"},
    {"difficulty": "ultra_easy", "prompt": "54 / 6 = ?"},
    {"difficulty": "ultra_easy", "prompt": "16 + 13 = ?"},
    {"difficulty": "ultra_easy", "prompt": "27 - 9 = ?"},
    {"difficulty": "ultra_easy", "prompt": "5 * 9 = ?"},
    {"difficulty": "ultra_easy", "prompt": "63 / 9 = ?"},
    {"difficulty": "ultra_easy", "prompt": "24 + 15 = ?"},
    {"difficulty": "ultra_easy", "prompt": "41 - 22 = ?"},
    {"difficulty": "ultra_easy", "prompt": "8 * 4 = ?"},
    {"difficulty": "ultra_easy", "prompt": "72 / 8 = ?"},
    {"difficulty": "ultra_easy", "prompt": "17 + 18 = ?"},
    {"difficulty": "ultra_easy", "prompt": "34 - 16 = ?"},
    {"difficulty": "ultra_easy", "prompt": "9 * 5 = ?"},
    {"difficulty": "ultra_easy", "prompt": "81 / 9 = ?"},
    {"difficulty": "ultra_easy", "prompt": "26 + 19 = ?"},
    {"difficulty": "ultra_easy", "prompt": "50 - 23 = ?"},
    {"difficulty": "ultra_easy", "prompt": "12 * 4 = ?"},
    {"difficulty": "ultra_easy", "prompt": "96 / 12 = ?"},
    {"difficulty": "ultra_easy", "prompt": "29 + 21 = ?"},
    {"difficulty": "ultra_easy", "prompt": "44 - 25 = ?"},
    {"difficulty": "ultra_easy", "prompt": "7 * 8 = ?"},
    {"difficulty": "ultra_easy", "prompt": "64 / 8 = ?"},
    {"difficulty": "ultra_easy", "prompt": "18 + 24 = ?"},
    {"difficulty": "ultra_easy", "prompt": "39 - 17 = ?"},
    {"difficulty": "ultra_easy", "prompt": "11 * 6 = ?"},
    {"difficulty": "ultra_easy", "prompt": "84 / 7 = ?"},
    {"difficulty": "ultra_easy", "prompt": "33 + 12 = ?"},
    {"difficulty": "ultra_easy", "prompt": "46 - 28 = ?"},
    {"difficulty": "ultra_easy", "prompt": "10 * 7 = ?"},
    {"difficulty": "ultra_easy", "prompt": "90 / 9 = ?"},
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
    expr = raw_prompt.strip()
    expr = re.sub(r"\s*=\s*\?\s*$", "", expr)
    expr = re.sub(r"\s*\?\s*$", "", expr)
    expr = re.sub(r"\s*=\s*$", "", expr)
    expr = f"{expr} ="

    if not brief_answer:
        return expr
    return (
        "Solve this arithmetic problem. "
        "Output only the final numeric answer (no explanation, no steps).\n"
        "Complete the expression after '=' with only the number.\n"
        f"{expr}"
    )


def extract_first_number(text: str) -> str:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return m.group(0) if m else text.strip()


def safe_eval_expr(expr: str) -> str:
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary = (ast.UAdd, ast.USub)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
            l = _eval(node.left)
            r = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unary):
            v = _eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError("unsupported expression")

    expr = expr.strip()
    expr = re.sub(r"\s*=\s*\?\s*$", "", expr)
    expr = re.sub(r"\s*\?\s*$", "", expr)
    expr = re.sub(r"\s*=\s*$", "", expr)
    val = _eval(ast.parse(expr, mode="eval"))
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.6f}".rstrip("0").rstrip(".")


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


def build_difficulty_summary(rows, max_new_tokens_values: List[int]):
    summary = {}
    for t in max_new_tokens_values:
        bucket = {}
        for row in rows:
            diff = row["difficulty"]
            out = None
            for x in row["outputs"]:
                if x["max_new_tokens"] == t:
                    out = x
                    break
            if out is None:
                continue
            if diff not in bucket:
                bucket[diff] = {
                    "count": 0,
                    "dense_correct": 0,
                    "dropped_correct": 0,
                    "same_answer": 0,
                    "dropped_count": 0,
                }
            b = bucket[diff]
            b["count"] += 1
            b["dense_correct"] += int(bool(out.get("dense_correct")))
            if out.get("dropped_correct") is not None:
                b["dropped_count"] += 1
                b["dropped_correct"] += int(bool(out.get("dropped_correct")))
            if out.get("dropped_output") is not None:
                b["same_answer"] += int(out.get("dense_output") == out.get("dropped_output"))

        all_count = sum(v["count"] for v in bucket.values())
        all_dense = sum(v["dense_correct"] for v in bucket.values())
        all_drop = sum(v["dropped_correct"] for v in bucket.values())
        all_same = sum(v["same_answer"] for v in bucket.values())
        all_drop_cnt = sum(v["dropped_count"] for v in bucket.values())

        merged = {}
        for diff, v in bucket.items():
            merged[diff] = {
                "count": v["count"],
                "dense_acc": (v["dense_correct"] / v["count"]) if v["count"] else None,
                "dropped_acc": (v["dropped_correct"] / v["dropped_count"]) if v["dropped_count"] else None,
                "same_rate": (v["same_answer"] / v["dropped_count"]) if v["dropped_count"] else None,
            }
        merged["all"] = {
            "count": all_count,
            "dense_acc": (all_dense / all_count) if all_count else None,
            "dropped_acc": (all_drop / all_drop_cnt) if all_drop_cnt else None,
            "same_rate": (all_same / all_drop_cnt) if all_drop_cnt else None,
        }
        summary[str(t)] = merged
    return summary


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
    parser.add_argument("--brief_answer", type=bool, default=True)
    # action="store_true")
    # parser.add_argument("--answer_only_extract", action="store_true")
    parser.add_argument("--answer_only_extract", type=bool, default=True)
    # action="store_true")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_tag = model_postfix
    args.output_json = f"/mnt/bn/seed-aws-va/shwai.he/gen-collapse/drop_logs_math/{model_tag}/{args.target_layer}-drop-{args.drop_n:02d}.json"

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

        raw_prompt = item.get("prompt") or item.get("problem")
        if raw_prompt is None:
            raise KeyError("Each MATH_PROMPTS item must contain `prompt` or `problem`.")
        difficulty = item["difficulty"]
        ground_truth = safe_eval_expr(raw_prompt)
        prompt = build_prompt(raw_prompt, args.brief_answer)

        per_len_outputs = []
        print(f"[{i}] ({difficulty}) {raw_prompt}")
        print(f"  gt: {ground_truth}")

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
                    "ground_truth": ground_truth,
                    "dense_raw": dense_raw,
                    "dropped_raw": dropped_raw,
                    "dense_output": dense_answer,
                    "dropped_output": dropped_answer,
                    "dense_correct": (dense_answer == ground_truth) if dense_answer else False,
                    "dropped_correct": (dropped_answer == ground_truth) if dropped_answer is not None else None,
                }
            )

        row = {
            "idx": i,
            "difficulty": difficulty,
            "prompt": raw_prompt,
            "ground_truth": ground_truth,
            "formatted_prompt": prompt,
            "outputs": per_len_outputs,
        }
        results["rows"].append(row)

    results["summary_by_max_new_tokens"] = build_difficulty_summary(
        rows=results["rows"], max_new_tokens_values=max_new_tokens_values
    )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=== Summary By Difficulty ===")
    for t in max_new_tokens_values:
        print(f"[max_new_tokens={t}]")
        for diff, m in results["summary_by_max_new_tokens"][str(t)].items():
            print(
                f"  {diff:>10} | n={m['count']:>3} | "
                f"dense_acc={m['dense_acc']} | dropped_acc={m['dropped_acc']} | same_rate={m['same_rate']}"
            )

    print(f"Saved outputs to: {args.output_json}")


if __name__ == "__main__":
    main()
