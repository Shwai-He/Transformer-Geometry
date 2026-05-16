import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_forward_utils import apply_drop_masks, generate_with_custom_forward


MATH_PROMPTS = [
    {"difficulty": "ultra_easy", "problem": "7 + 5"},
    {"difficulty": "ultra_easy", "problem": "13 - 4"},
    {"difficulty": "ultra_easy", "problem": "6 * 8"},
    {"difficulty": "ultra_easy", "problem": "36 / 9"},
    {"difficulty": "ultra_easy", "problem": "15 + 12"},
    {"difficulty": "ultra_easy", "problem": "20 - 11"},
    {"difficulty": "ultra_easy", "problem": "9 * 7"},
    {"difficulty": "ultra_easy", "problem": "42 / 6"},
    {"difficulty": "ultra_easy", "problem": "18 + 17"},
    {"difficulty": "ultra_easy", "problem": "25 - 8"},
    {"difficulty": "ultra_easy", "problem": "11 * 4"},
    {"difficulty": "ultra_easy", "problem": "56 / 8"},
    {"difficulty": "easy", "problem": "14 + 9 - 6"},
    {"difficulty": "easy", "problem": "18 - 7 + 5"},
    {"difficulty": "easy", "problem": "6 * 3 - 8"},
    {"difficulty": "easy", "problem": "22 - 9 + 4"},
    {"difficulty": "easy", "problem": "9 + 8 - 3"},
    {"difficulty": "easy", "problem": "7 * 4 - 11"},
    {"difficulty": "easy", "problem": "15 + 6 / 3"},
    {"difficulty": "easy", "problem": "28 / 7 + 9"},
    {"difficulty": "easy", "problem": "30 - 12 / 3"},
    {"difficulty": "easy", "problem": "5 * 6 + 4"},
    {"difficulty": "easy", "problem": "40 / 5 + 7"},
    {"difficulty": "easy", "problem": "27 - 8 + 2"},
    {"difficulty": "easy", "problem": "3 * 9 - 10"},
    {"difficulty": "easy", "problem": "12 + 18 / 6"},
    {"difficulty": "easy", "problem": "50 / 10 + 13"},
    {"difficulty": "easy", "problem": "16 + 5 * 2"},
    {"difficulty": "easy", "problem": "33 - 14 + 6"},
    {"difficulty": "easy", "problem": "8 * 5 - 17"},
    {"difficulty": "easy", "problem": "24 / 6 + 15"},
    {"difficulty": "easy", "problem": "11 + 7 * 3"},
    {"difficulty": "easy", "problem": "42 / 7 + 8"},
    {"difficulty": "medium", "problem": "(16 + 8) / 4 + 3"},
    {"difficulty": "medium", "problem": "7 * (5 - 2) + 4"},
    {"difficulty": "medium", "problem": "45 / 9 + 6 * 2"},
    {"difficulty": "hard", "problem": "3 * (12 - 5) + 18 / 3"},
    {"difficulty": "hard", "problem": "(48 / 6) * (9 - 4) - 7"},
    {"difficulty": "hard", "problem": "64 / (4 + 4) + 7 * 3"},
    {"difficulty": "hard", "problem": "(8 * 7) - (36 / 6) + 5"},
]


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


def parse_first_number(text: str) -> Optional[str]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return m.group(0) if m else None


def parse_first_number_span(text: str) -> Optional[re.Match]:
    return re.search(r"[-+]?\d+(?:\.\d+)?", text)


def build_dense_prompt(problem: str) -> str:
    return (
        "Solve this arithmetic expression and you may reason briefly.\n"
        "But final output must strictly follow this format:\n"
        "Final answer: <number>\n"
        f"Problem: {problem}\n"
        "Now answer."
    )


def build_dropped_prompt(dense_prompt: str, masked_dense_completion: str) -> str:
    """
    User-required protocol:
    dropped_prompt = dense_input + dense_output(without final answer)
    and dropped model predicts only the final answer.
    """
    return f"{dense_prompt}{masked_dense_completion}"


def model_generate(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
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
        use_cache=True,
        collect_sublayer=False,
    )
    return texts[0]


def extract_completion(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    return full_text


def parse_final_answer_matches(text: str) -> List[re.Match]:
    """
    Find answer markers in order of appearance.
    Supports:
    - Final answer: 55
    - Final answer = 55
    - final answer is 55
    """
    pattern = r"(?:Final answer|Answer)\s*(?::|=|is)\s*([-+]?\d+(?:\.\d+)?)"
    return list(re.finditer(pattern, text, flags=re.IGNORECASE))


def parse_first_final_answer_marker(text: str) -> Optional[str]:
    matches = parse_final_answer_matches(text)
    if not matches:
        return None
    return matches[0].group(1)


def extract_answer_value(text: str) -> Optional[str]:
    marker_val = parse_first_final_answer_marker(text)
    if marker_val is not None:
        return marker_val
    return parse_first_number(text)


def mask_final_answer_and_suffix(dense_completion: str, ground_truth: Optional[str]) -> Dict[str, Any]:
    """
    Default: use the first final-answer marker.
    Override: if any marker value matches AST ground truth, use that marker.
    """
    marker_matches = parse_final_answer_matches(dense_completion)
    if not marker_matches:
        # Fallback: strip only the first numeric token (best-effort when marker is absent).
        m_num = parse_first_number_span(dense_completion)
        dense_answer = m_num.group(0) if m_num is not None else None
        masked_reasoning = dense_completion if m_num is None else dense_completion[: m_num.start()]
        return {
            "dense_answer": dense_answer,
            "masked_reasoning": masked_reasoning,
            "mask_start": None if m_num is None else m_num.start(),
            "mask_end": None if m_num is None else m_num.end(),
            "parse_mode": "fallback_first_number",
        }

    selected_idx = 0  # default: first final answer
    if ground_truth is not None:
        for i, m in enumerate(marker_matches):
            cand = normalize_num_str(m.group(1))
            if cand == ground_truth:
                selected_idx = i
                break

    selected_marker = marker_matches[selected_idx]
    dense_answer = normalize_num_str(selected_marker.group(1))
    # Keep prompt prefix + answer marker text (e.g., "Final answer: "),
    # and mask only the numeric answer and everything after it.
    masked_reasoning = dense_completion[: selected_marker.start(1)]

    parse_mode = "first_marker"
    if selected_idx != 0:
        parse_mode = "gt_override"
    return {
        "dense_answer": dense_answer,
        "masked_reasoning": masked_reasoning,
        "mask_start": selected_marker.start(),
        "mask_end": len(dense_completion),
        "parse_mode": parse_mode,
    }


def normalize_num_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return x.strip()
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def safe_eval_expr(expr: str) -> Optional[str]:
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

    try:
        tree = ast.parse(expr, mode="eval")
        val = _eval(tree)
    except Exception:
        return None

    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.6f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser(description="Dense CoT teacher -> mask final answer -> dropped predicts numeric answer.")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--dropped_root_path", type=str, required=False)
    parser.add_argument("--drop_n", type=int, default=8)
    parser.add_argument("--target_layer", type=str, default="mlp", choices=["attn", "mlp", "all"])

    parser.add_argument("--dense_max_new_tokens", type=int, default=96)
    parser.add_argument("--dropped_max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=159)
    parser.add_argument("--prompt_idx", type=int, default=None)

    parser.add_argument("--output_json", type=str, default="./drop_logs_clean/masked_teacher_eval.json")
    args = parser.parse_args()

    # ===== Editable config block (file-first) =====
    model_postfix = "Qwen/Qwen2.5-7B-Instruct"
    args.model_name = f"/mnt/bn/seed-aws-va/shwai.he/models/{model_postfix}-copy"
    args.dropped_root_path = "/mnt/bn/seed-aws-va/shwai.he/LLM-Drop/results_prune"
    args.drop_n = 8
    args.target_layer = "mlp"
    args.output_json = (
        f"/mnt/bn/seed-aws-va/shwai.he/gen-collapse/drop_logs_math/{model_postfix}/"
        f"masked-teacher-{args.target_layer}-drop-{args.drop_n:02d}.json"
    )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_tag = args.model_tag or model_postfix

    dense_path = args.model_name
    config_path = dropped_config_path(args.dropped_root_path, model_tag, args.target_layer, args.drop_n)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dropped config not found: {config_path}")

    tokenizer = AutoTokenizer.from_pretrained(dense_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(dense_path, trust_remote_code=True).to(device).eval()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    drop_attn_list = cfg.get("drop_attn_list", [])
    drop_mlp_list = cfg.get("drop_mlp_list", [])
    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(MATH_PROMPTS):
        if args.prompt_idx is not None and i != args.prompt_idx:
            continue

        problem = item["problem"]
        difficulty = item["difficulty"]
        ground_truth = normalize_num_str(safe_eval_expr(problem))

        dense_prompt = build_dense_prompt(problem)
        apply_drop_masks(
            model=model,
            target_layer="all",
            drop_attn_list=[],
            drop_mlp_list=[],
            drop_n=args.drop_n,
        )
        dense_full_text = model_generate(
            model,
            tokenizer,
            device,
            dense_prompt,
            max_new_tokens=args.dense_max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        dense_completion = extract_completion(dense_full_text, dense_prompt)

        mask_info = mask_final_answer_and_suffix(dense_completion, ground_truth)
        dense_answer = normalize_num_str(mask_info["dense_answer"] or extract_answer_value(dense_completion))
        masked_reasoning = mask_info["masked_reasoning"]

        dropped_prompt = build_dropped_prompt(dense_prompt, masked_reasoning)
        apply_drop_masks(
            model=model,
            target_layer=args.target_layer,
            drop_attn_list=drop_attn_list,
            drop_mlp_list=drop_mlp_list,
            drop_n=args.drop_n,
        )
        dropped_full_text = model_generate(
            model,
            tokenizer,
            device,
            dropped_prompt,
            max_new_tokens=args.dropped_max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        dropped_completion = extract_completion(dropped_full_text, dropped_prompt)
        dropped_answer = normalize_num_str(extract_answer_value(dropped_completion))

        same_as_dense = (
            dense_answer is not None and dropped_answer is not None and dropped_answer == dense_answer
        )
        dense_correct_vs_gt = (
            dense_answer is not None and ground_truth is not None and dense_answer == ground_truth
        )
        dropped_correct_vs_gt = (
            dropped_answer is not None and ground_truth is not None and dropped_answer == ground_truth
        )

        row = {
            "idx": i,
            "difficulty": difficulty,
            "problem": problem,
            "ground_truth": ground_truth,
            "dense_prompt": dense_prompt,
            "dense_out_full": dense_full_text,
            "dense_completion": dense_completion,
            "dense_answer": dense_answer,
            "dense_parse_mode": mask_info.get("parse_mode"),
            "masked_reasoning": masked_reasoning,
            "dropped_prompt": dropped_prompt,
            "dropped_out_full": dropped_full_text,
            "dropped_completion": dropped_completion,
            "dropped_answer": dropped_answer,
            "same_as_dense": same_as_dense,
            "dense_correct_vs_gt": dense_correct_vs_gt,
            "dropped_correct_vs_gt": dropped_correct_vs_gt,
        }
        rows.append(row)

        print(f"[{i}] ({difficulty}) {problem}")
        print(f"  gt     : {ground_truth}")
        print(f"  dense_prompt:\n{dense_prompt}")
        print(f"  dropped_prompt:\n{dropped_prompt}")
        print(f"  dense_out_full:\n{dense_full_text}")
        print(f"  dropped_out_full:\n{dropped_full_text}")
        print(f"  dense  : {dense_answer}")
        print(f"  dropped: {dropped_answer}")
        print(f"  same   : {same_as_dense}")

    comparable_count = sum(1 for r in rows if r["dense_answer"] is not None and r["dropped_answer"] is not None)
    same_count = sum(1 for r in rows if r["same_as_dense"] is True)
    gt_count = sum(1 for r in rows if r["ground_truth"] is not None)
    dense_gt_count = sum(1 for r in rows if r["dense_correct_vs_gt"] is True)
    dropped_gt_count = sum(1 for r in rows if r["dropped_correct_vs_gt"] is True)

    results = {
        "dense_path": dense_path,
        "dropped_path": dense_path,
        "dropped_config_path": config_path,
        "drop_n": args.drop_n,
        "target_layer": args.target_layer,
        "dense_max_new_tokens": args.dense_max_new_tokens,
        "dropped_max_new_tokens": args.dropped_max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "summary": {
            "compared": comparable_count,
            "same_as_dense": same_count,
            "same_rate": (same_count / comparable_count) if comparable_count > 0 else None,
            "gt_count": gt_count,
            "dense_correct_vs_gt": dense_gt_count,
            "dropped_correct_vs_gt": dropped_gt_count,
            "dense_acc_vs_gt": (dense_gt_count / gt_count) if gt_count > 0 else None,
            "dropped_acc_vs_gt": (dropped_gt_count / gt_count) if gt_count > 0 else None,
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()
