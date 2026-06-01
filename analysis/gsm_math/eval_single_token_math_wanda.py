import argparse
import ast
import json
import os
import re
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.utils.generation_forward_utils import generate_with_custom_forward


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
    {"difficulty": "easy", "prompt": "14 + 9 - 6 = ?"},
    {"difficulty": "easy", "prompt": "18 - 7 + 5 = ?"},
    {"difficulty": "easy", "prompt": "6 * 3 - 8 = ?"},
    {"difficulty": "easy", "prompt": "22 - 9 + 4 = ?"},
    {"difficulty": "easy", "prompt": "9 + 8 - 3 = ?"},
    {"difficulty": "easy", "prompt": "7 * 4 - 11 = ?"},
    {"difficulty": "easy", "prompt": "15 + 6 / 3 = ?"},
    {"difficulty": "easy", "prompt": "28 / 7 + 9 = ?"},
    {"difficulty": "easy", "prompt": "30 - 12 / 3 = ?"},
    {"difficulty": "easy", "prompt": "5 * 6 + 4 = ?"},
    {"difficulty": "easy", "prompt": "40 / 5 + 7 = ?"},
    {"difficulty": "easy", "prompt": "27 - 8 + 2 = ?"},
    {"difficulty": "easy", "prompt": "3 * 9 - 10 = ?"},
    {"difficulty": "easy", "prompt": "12 + 18 / 6 = ?"},
    {"difficulty": "easy", "prompt": "50 / 10 + 13 = ?"},
    {"difficulty": "easy", "prompt": "16 + 5 * 2 = ?"},
    {"difficulty": "easy", "prompt": "33 - 14 + 6 = ?"},
    {"difficulty": "easy", "prompt": "8 * 5 - 17 = ?"},
    {"difficulty": "easy", "prompt": "24 / 6 + 15 = ?"},
    {"difficulty": "easy", "prompt": "11 + 7 * 3 = ?"},
    {"difficulty": "easy", "prompt": "42 / 7 + 8 = ?"},
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
                }
            b = bucket[diff]
            b["count"] += 1
            b["dense_correct"] += int(bool(out.get("dense_correct")))
            b["dropped_correct"] += int(bool(out.get("dropped_correct")))
            b["same_answer"] += int(out.get("dense_output") == out.get("dropped_output"))

        all_count = sum(v["count"] for v in bucket.values())
        all_dense = sum(v["dense_correct"] for v in bucket.values())
        all_drop = sum(v["dropped_correct"] for v in bucket.values())
        all_same = sum(v["same_answer"] for v in bucket.values())

        merged = {}
        for diff, v in bucket.items():
            merged[diff] = {
                "count": v["count"],
                "dense_acc": (v["dense_correct"] / v["count"]) if v["count"] else None,
                "dropped_acc": (v["dropped_correct"] / v["count"]) if v["count"] else None,
                "same_rate": (v["same_answer"] / v["count"]) if v["count"] else None,
            }
        merged["all"] = {
            "count": all_count,
            "dense_acc": (all_dense / all_count) if all_count else None,
            "dropped_acc": (all_drop / all_count) if all_count else None,
            "same_rate": (all_same / all_count) if all_count else None,
        }
        summary[str(t)] = merged
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare dense vs Wanda-pruned outputs on single-token math prompts.")
    parser.add_argument("--dense_model_name", type=str, required=False)
    parser.add_argument("--dropped_model_name", type=str, required=False)
    parser.add_argument("--max_new_tokens_values", type=str, default="8,16,32")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=159)
    parser.add_argument("--prompt_idx", type=int, default=None)
    parser.add_argument("--brief_answer", type=bool, default=True)
    parser.add_argument("--answer_only_extract", type=bool, default=True)
    parser.add_argument("--output_json", type=str, default="./drop_logs_clean/math_outputs_wanda.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    max_new_tokens_values = parse_int_list(args.max_new_tokens_values)

    # ===== Editable config block (file-first) =====
    model_postfix = "Qwen/Qwen2.5-7B-Instruct"
    args.dense_model_name = f"/mnt/bn/seed-aws-va/shwai.he/models/{model_postfix}-copy"
    wanda_root = f"/mnt/bn/seed-aws-va/shwai.he/wanda/out/{model_postfix}"
    wanda_sparsity = "4-8"
    wanda_method = "wanda"
    args.dropped_model_name = f"{wanda_root}/{wanda_sparsity}/{wanda_method}"
    args.output_json = (
        f"/mnt/bn/seed-aws-va/shwai.he/gen-collapse/drop_logs_math/{model_postfix}/"
        f"wanda-single-token-{wanda_sparsity}-{wanda_method}.json"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(args.dropped_model_name):
        raise FileNotFoundError(f"Wanda model path not found: {args.dropped_model_name}")

    dense_tokenizer = AutoTokenizer.from_pretrained(args.dense_model_name, trust_remote_code=True)
    if dense_tokenizer.pad_token is None:
        dense_tokenizer.pad_token = dense_tokenizer.eos_token
    dense_model = AutoModelForCausalLM.from_pretrained(args.dense_model_name, trust_remote_code=True).to(device).eval()

    dropped_tokenizer = AutoTokenizer.from_pretrained(args.dropped_model_name, trust_remote_code=True)
    if dropped_tokenizer.pad_token is None:
        dropped_tokenizer.pad_token = dropped_tokenizer.eos_token
    dropped_model = AutoModelForCausalLM.from_pretrained(args.dropped_model_name, trust_remote_code=True).to(device).eval()

    results = {
        "dense_model_name": args.dense_model_name,
        "dropped_model_name": args.dropped_model_name,
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
            dense_full = run_generation(
                model=dense_model,
                tokenizer=dense_tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
            )
            dropped_full = run_generation(
                model=dropped_model,
                tokenizer=dropped_tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
            )

            dense_raw = dense_full.replace(prompt, "").strip()
            dropped_raw = dropped_full.replace(prompt, "").strip()

            if args.answer_only_extract:
                dense_answer = extract_first_number(dense_raw)
                dropped_answer = extract_first_number(dropped_raw)
            else:
                dense_answer = dense_raw
                dropped_answer = dropped_raw

            print(f"  [max_new_tokens={max_new_tokens}] dense  : {dense_answer}")
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
                    "dropped_correct": (dropped_answer == ground_truth) if dropped_answer else False,
                }
            )

        results["rows"].append(
            {
                "idx": i,
                "difficulty": difficulty,
                "prompt": raw_prompt,
                "ground_truth": ground_truth,
                "formatted_prompt": prompt,
                "outputs": per_len_outputs,
            }
        )

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
