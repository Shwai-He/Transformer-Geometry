import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_forward_utils import apply_drop_masks


def normalize_num(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.replace(",", "").strip()
    try:
        v = float(s)
    except Exception:
        return s
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def extract_number(text: str) -> Optional[str]:
    patterns = [
        r"(?:Final answer|Answer)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"####\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        ms = list(re.finditer(p, text, flags=re.IGNORECASE))
        if ms:
            return ms[-1].group(1)
    nums = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
    return nums[-1].group(0) if nums else None


def parse_gsm8k_gt(answer: str) -> Optional[str]:
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if m:
        return normalize_num(m.group(1))
    return normalize_num(extract_number(answer))


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


def default_fewshot_examples() -> List[Dict[str, str]]:
    return [
        {
            "q": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?",
            "a": "50 minutes is 50/60 hour. 12 * (50/60) = 10. Final answer: 10",
        },
        {
            "q": "A pack has 8 pencils. Emma buys 7 packs and gives away 9 pencils. How many pencils remain?",
            "a": "7 * 8 = 56. 56 - 9 = 47. Final answer: 47",
        },
        {
            "q": "A store sold 35 apples on Monday and twice as many on Tuesday. How many apples in total?",
            "a": "Tuesday: 2 * 35 = 70. Total: 35 + 70 = 105. Final answer: 105",
        },
    ]


def build_prompt(question: str, use_few_shot: bool, few_shot_k: int) -> str:
    header = (
        "Solve the following GSM8K question.\n"
        "You may reason briefly and end with this exact format:\n"
        "Final answer: <number>\n"
    )
    if not use_few_shot:
        return f"{header}\nQuestion: {question}\nAnswer:"

    demos = default_fewshot_examples()[: max(0, few_shot_k)]
    chunks = [header, "Examples:\n"]
    for ex in demos:
        chunks.append(f"Question: {ex['q']}\nAnswer: {ex['a']}\n")
    chunks.append(f"\nQuestion: {question}\nAnswer:")
    return "\n".join(chunks)


def generate_batch_full_texts(
    model,
    tokenizer,
    device: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[str]:
    if not prompts:
        return []
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    do_sample = temperature != 0.0
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            return_dict_in_generate=True,
            use_cache=True,
        )
    return tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)


def extract_completion(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    if prompt in full_text:
        return full_text.split(prompt, 1)[1]
    return full_text


def load_gsm8k(max_questions: int) -> List[Dict[str, str]]:
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        out = []
        for row in ds:
            out.append({"question": row["question"], "answer": row["answer"]})
            if len(out) >= max_questions:
                break
        if out:
            return out
    except Exception:
        pass

    return [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "In April she sold 48 clips. In May she sold 24 clips. Total 72. #### 72",
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="GSM8K free-budget few-shot dense vs dropped.")
    parser.add_argument("--max_questions", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="./gsm8k_free_budget_fewshot_eval.json")
    args = parser.parse_args()

    # ===== Editable config (file-first) =====
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model_postfix = "Qwen/Qwen2.5-7B-Instruct"
    model_name = f"/mnt/bn/seed-aws-va/shwai.he/models/{model_postfix}-copy"
    dropped_root = "/mnt/bn/seed-aws-va/shwai.he/LLM-Drop/results_prune"
    target_layer = "mlp"
    drop_n = 8
    batch_size = 8
    cfg_use_few_shot = 1
    cfg_few_shot_k = 3
    args.output_json = (
        f"/mnt/bn/seed-aws-va/shwai.he/drop_logs_easy/"
        f"gsm8k_free_budget_fewshot_{target_layer}_drop{drop_n}.json"
    )

    use_few_shot = bool(cfg_use_few_shot)
    few_shot_k = int(cfg_few_shot_k)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    cfg_path = dropped_config_path(dropped_root, model_postfix, target_layer, drop_n)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    drop_attn_list = cfg.get("drop_attn_list", [])
    drop_mlp_list = cfg.get("drop_mlp_list", [])

    data = load_gsm8k(args.max_questions)
    prompts = [build_prompt(qa["question"], use_few_shot=use_few_shot, few_shot_k=few_shot_k) for qa in data]
    gts = [parse_gsm8k_gt(qa["answer"]) for qa in data]

    dense_full_all: List[str] = []
    apply_drop_masks(model, target_layer="all", drop_attn_list=[], drop_mlp_list=[], drop_n=drop_n)
    for st in range(0, len(prompts), batch_size):
        ed = min(st + batch_size, len(prompts))
        print(f"[dense] {st}/{len(prompts)}")
        dense_full_all.extend(
            generate_batch_full_texts(
                model, tokenizer, device, prompts[st:ed], args.max_new_tokens, args.temperature, args.top_k, args.top_p
            )
        )

    dropped_full_all: List[str] = []
    apply_drop_masks(
        model,
        target_layer=target_layer,
        drop_attn_list=drop_attn_list,
        drop_mlp_list=drop_mlp_list,
        drop_n=drop_n,
    )
    for st in range(0, len(prompts), batch_size):
        ed = min(st + batch_size, len(prompts))
        print(f"[dropped] {st}/{len(prompts)}")
        dropped_full_all.extend(
            generate_batch_full_texts(
                model, tokenizer, device, prompts[st:ed], args.max_new_tokens, args.temperature, args.top_k, args.top_p
            )
        )

    rows = []
    for i, qa in enumerate(data):
        prompt = prompts[i]
        gt = gts[i]
        dense_full = dense_full_all[i]
        dropped_full = dropped_full_all[i]
        dense_comp = extract_completion(dense_full, prompt)
        dropped_comp = extract_completion(dropped_full, prompt)
        dense_ans = normalize_num(extract_number(dense_comp))
        dropped_ans = normalize_num(extract_number(dropped_comp))

        rows.append(
            {
                "idx": i,
                "question": qa["question"],
                "ground_truth": gt,
                "prompt": prompt,
                "dense_out_full": dense_full,
                "dense_completion": dense_comp,
                "dense_answer": dense_ans,
                "dropped_out_full": dropped_full,
                "dropped_completion": dropped_comp,
                "dropped_answer": dropped_ans,
                "same_as_dense": (dense_ans == dropped_ans) if (dense_ans and dropped_ans) else False,
                "dense_correct_vs_gt": (dense_ans == gt) if (dense_ans and gt) else False,
                "dropped_correct_vs_gt": (dropped_ans == gt) if (dropped_ans and gt) else False,
            }
        )

    n = len(rows)
    same = sum(1 for r in rows if r["same_as_dense"])
    dense_ok = sum(1 for r in rows if r["dense_correct_vs_gt"])
    drop_ok = sum(1 for r in rows if r["dropped_correct_vs_gt"])

    out = {
        "task": "gsm8k_free_budget_fewshot_single_model_dropmask",
        "model_name": model_name,
        "dropped_config": cfg_path,
        "target_layer": target_layer,
        "drop_n": drop_n,
        "use_few_shot": use_few_shot,
        "few_shot_k": few_shot_k,
        "summary": {
            "n": n,
            "same_as_dense": same,
            "same_rate": (same / n) if n else None,
            "dense_acc_vs_gt": (dense_ok / n) if n else None,
            "dropped_acc_vs_gt": (drop_ok / n) if n else None,
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
