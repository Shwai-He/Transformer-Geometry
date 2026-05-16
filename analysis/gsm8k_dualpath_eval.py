import argparse
import gc
import json
import os
import re
from typing import Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_number(text: str) -> Optional[str]:
    markers = [
        r"Final answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"####\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for p in markers:
        ms = list(re.finditer(p, text, flags=re.IGNORECASE))
        if ms:
            return ms[-1].group(1)

    nums = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
    if not nums:
        return None
    return nums[-1].group(0)


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


def parse_gsm8k_gt(answer_text: str) -> Optional[str]:
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text)
    if m:
        return normalize_num(m.group(1))
    return normalize_num(extract_number(answer_text))


def generate_text(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    input_ids = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    do_sample = temperature != 0.0

    with torch.no_grad():
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

    return tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]


def extract_completion(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    if prompt in full_text:
        return full_text.split(prompt, 1)[1]
    return full_text


def find_answer_marker_value_span(text: str) -> Optional[re.Match]:
    # value span only, after marker
    ms = list(re.finditer(r"(?:Final answer|Answer)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE))
    if ms:
        return ms[-1]
    ms2 = list(re.finditer(r"####\s*([-+]?\d+(?:\.\d+)?)", text))
    if ms2:
        return ms2[-1]
    return None


def mask_after_dense_answer(dense_completion: str) -> Dict[str, Optional[str]]:
    m = find_answer_marker_value_span(dense_completion)
    if m is None:
        # fallback: no marker, keep all and let sparse continue naturally
        return {
            "dense_answer": normalize_num(extract_number(dense_completion)),
            "masked_prefix": dense_completion,
            "mask_mode": "no_marker",
        }

    dense_ans = normalize_num(m.group(1))
    masked_prefix = dense_completion[: m.start(1)]  # keep marker, remove value+suffix
    return {
        "dense_answer": dense_ans,
        "masked_prefix": masked_prefix,
        "mask_mode": "marker_value_cut",
    }


def build_dense_prompt(question: str) -> str:
    return (
        "Solve the following GSM8K math question step by step.\n"
        "End your response with: Final answer: <number>\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_free_prompt(question: str) -> str:
    return (
        "Solve the following GSM8K math question.\n"
        "Respond within the token budget.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_gsm8k_questions(max_questions: int) -> List[Dict[str, str]]:
    # First try HF datasets, fallback to hardcoded samples
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        out = []
        for row in ds:
            out.append({
                "question": row["question"],
                "answer": row["answer"],
            })
            if len(out) >= max_questions:
                break
        if out:
            return out
    except Exception:
        pass

    fallback = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "In April she sold 48 clips. In May she sold 48/2 = 24 clips. Altogether she sold 48+24 = 72 clips. #### 72",
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "Weng earns 12/60 = 0.2 dollars per minute. For 50 minutes she earns 50*0.2 = 10 dollars. #### 10",
        },
    ]
    return fallback[:max_questions]


def main():
    parser = argparse.ArgumentParser(description="GSM8K dense vs sparse dual-path experiments.")
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dense_max_new_tokens", type=int, default=128)
    parser.add_argument("--controlled_sparse_max_new_tokens", type=int, default=16)
    parser.add_argument("--free_max_new_tokens", type=int, default=64)

    parser.add_argument("--output_json", type=str, default="./gsm8k_dualpath_results.json")
    args = parser.parse_args()

    # ===== Editable config block (file-first) =====
    model_postfix = "Qwen/Qwen2.5-7B-Instruct"
    dense_model_path = f"/mnt/bn/seed-aws-va/shwai.he/models/{model_postfix}-copy"

    sparse_root = f"/mnt/bn/seed-aws-va/shwai.he/wanda/out/{model_postfix}"
    sparsity_list = ["2-4", "4-8", "unstructured"]
    methods = ["wanda", "sparsegpt"]

    log_dir = "/mnt/bn/seed-aws-va/shwai.he/drop_logs_easy/gsm8k_dualpath"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")
    args.output_json = os.path.join(log_dir, "result.json")

    def write_log(s: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== GSM8K Dual-Path Dense vs Sparse ===\n")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    questions = load_gsm8k_questions(args.max_questions)
    write_log(f"Loaded questions: {len(questions)}")

    # Load dense once
    dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_path)
    if dense_tokenizer.pad_token is None:
        dense_tokenizer.pad_token = dense_tokenizer.eos_token

    dense_model = AutoModelForCausalLM.from_pretrained(
        dense_model_path,
        trust_remote_code=True,
    ).to(device)
    dense_model.eval()

    # Precompute dense outputs for controlled-prefix experiment
    dense_cache: List[Dict[str, str]] = []
    write_log("\n--- Precomputing dense outputs ---")
    for i, qa in enumerate(questions):
        q = qa["question"]
        gt = parse_gsm8k_gt(qa["answer"])

        dense_prompt = build_dense_prompt(q)
        dense_full = generate_text(
            model=dense_model,
            tokenizer=dense_tokenizer,
            device=device,
            prompt=dense_prompt,
            max_new_tokens=args.dense_max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        dense_comp = extract_completion(dense_full, dense_prompt)
        mask_info = mask_after_dense_answer(dense_comp)

        dense_cache.append(
            {
                "idx": i,
                "question": q,
                "gt": gt,
                "dense_prompt": dense_prompt,
                "dense_out_full": dense_full,
                "dense_completion": dense_comp,
                "dense_answer": mask_info["dense_answer"],
                "masked_prefix": mask_info["masked_prefix"],
                "mask_mode": mask_info["mask_mode"],
            }
        )

    all_results: Dict[str, object] = {
        "dense_model_path": dense_model_path,
        "sparse_root": sparse_root,
        "settings": {
            "max_questions": len(questions),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "dense_max_new_tokens": args.dense_max_new_tokens,
            "controlled_sparse_max_new_tokens": args.controlled_sparse_max_new_tokens,
            "free_max_new_tokens": args.free_max_new_tokens,
        },
        "variants": [],
    }

    # Sparse ckpt loop
    for sparsity in sparsity_list:
        for method in methods:
            variant_name = f"{sparsity}/{method}"
            sparse_model_path = f"{sparse_root}/{sparsity}/{method}"

            write_log("\n" + "=" * 100)
            write_log(f"Variant: {variant_name}")
            write_log(f"Path: {sparse_model_path}")

            if not os.path.isdir(sparse_model_path):
                write_log("[WARN] missing path, skip")
                all_results["variants"].append(
                    {
                        "variant": variant_name,
                        "path": sparse_model_path,
                        "status": "missing",
                    }
                )
                continue

            sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_model_path)
            if sparse_tokenizer.pad_token is None:
                sparse_tokenizer.pad_token = sparse_tokenizer.eos_token

            sparse_model = AutoModelForCausalLM.from_pretrained(
                sparse_model_path,
                trust_remote_code=True,
            ).to(device)
            sparse_model.eval()

            rows = []
            controlled_same = 0
            dense_gt_ok = 0
            sparse_controlled_gt_ok = 0
            dense_free_gt_ok = 0
            sparse_free_gt_ok = 0
            valid_gt = 0

            for item in dense_cache:
                q = item["question"]
                gt = item["gt"]

                # (1) Controlled-prefix experiment:
                # dropped prompt = dense input + dense output (no final answer)
                controlled_prompt = f"{item['dense_prompt']}{item['masked_prefix']}"
                sparse_controlled_full = generate_text(
                    model=sparse_model,
                    tokenizer=sparse_tokenizer,
                    device=device,
                    prompt=controlled_prompt,
                    max_new_tokens=args.controlled_sparse_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                sparse_controlled_comp = extract_completion(sparse_controlled_full, controlled_prompt)
                sparse_controlled_ans = normalize_num(extract_number(sparse_controlled_comp))

                dense_ans = normalize_num(item["dense_answer"])
                same_flag = (
                    dense_ans is not None
                    and sparse_controlled_ans is not None
                    and dense_ans == sparse_controlled_ans
                )

                # (2) Free-generation experiment under token budget
                free_prompt = build_free_prompt(q)
                dense_free_full = generate_text(
                    model=dense_model,
                    tokenizer=dense_tokenizer,
                    device=device,
                    prompt=free_prompt,
                    max_new_tokens=args.free_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                sparse_free_full = generate_text(
                    model=sparse_model,
                    tokenizer=sparse_tokenizer,
                    device=device,
                    prompt=free_prompt,
                    max_new_tokens=args.free_max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                dense_free_comp = extract_completion(dense_free_full, free_prompt)
                sparse_free_comp = extract_completion(sparse_free_full, free_prompt)
                dense_free_ans = normalize_num(extract_number(dense_free_comp))
                sparse_free_ans = normalize_num(extract_number(sparse_free_comp))

                dense_gt = None
                sparse_ctrl_gt = None
                dense_free_gt = None
                sparse_free_gt = None
                if gt is not None:
                    valid_gt += 1
                    dense_gt = (dense_ans == gt) if dense_ans is not None else False
                    sparse_ctrl_gt = (sparse_controlled_ans == gt) if sparse_controlled_ans is not None else False
                    dense_free_gt = (dense_free_ans == gt) if dense_free_ans is not None else False
                    sparse_free_gt = (sparse_free_ans == gt) if sparse_free_ans is not None else False

                    dense_gt_ok += int(bool(dense_gt))
                    sparse_controlled_gt_ok += int(bool(sparse_ctrl_gt))
                    dense_free_gt_ok += int(bool(dense_free_gt))
                    sparse_free_gt_ok += int(bool(sparse_free_gt))

                controlled_same += int(bool(same_flag))

                rows.append(
                    {
                        "idx": item["idx"],
                        "question": q,
                        "ground_truth": gt,
                        "dense_cached_prompt": item["dense_prompt"],
                        "dense_cached_out_full": item["dense_out_full"],
                        "dense_cached_answer": dense_ans,
                        "masked_prefix": item["masked_prefix"],
                        "controlled_prompt": controlled_prompt,
                        "sparse_controlled_out_full": sparse_controlled_full,
                        "sparse_controlled_completion": sparse_controlled_comp,
                        "sparse_controlled_answer": sparse_controlled_ans,
                        "same_as_dense_controlled": same_flag,
                        "dense_correct_vs_gt_controlled": dense_gt,
                        "sparse_correct_vs_gt_controlled": sparse_ctrl_gt,
                        "free_prompt": free_prompt,
                        "dense_free_out_full": dense_free_full,
                        "dense_free_answer": dense_free_ans,
                        "sparse_free_out_full": sparse_free_full,
                        "sparse_free_answer": sparse_free_ans,
                        "dense_correct_vs_gt_free": dense_free_gt,
                        "sparse_correct_vs_gt_free": sparse_free_gt,
                    }
                )

            n = len(rows)
            summary = {
                "n": n,
                "gt_count": valid_gt,
                "controlled_same_as_dense": controlled_same,
                "controlled_same_rate": (controlled_same / n) if n > 0 else None,
                "dense_acc_vs_gt_controlled": (dense_gt_ok / valid_gt) if valid_gt > 0 else None,
                "sparse_acc_vs_gt_controlled": (sparse_controlled_gt_ok / valid_gt) if valid_gt > 0 else None,
                "dense_acc_vs_gt_free": (dense_free_gt_ok / valid_gt) if valid_gt > 0 else None,
                "sparse_acc_vs_gt_free": (sparse_free_gt_ok / valid_gt) if valid_gt > 0 else None,
            }

            write_log(f"Summary: {json.dumps(summary, ensure_ascii=False)}")

            all_results["variants"].append(
                {
                    "variant": variant_name,
                    "path": sparse_model_path,
                    "status": "ok",
                    "summary": summary,
                    "rows": rows,
                }
            )

            # cleanup per sparse model
            del sparse_model
            del sparse_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # cleanup dense
    del dense_model
    del dense_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    write_log("\n=== All experiments finished ===")
    write_log(f"Saved JSON: {args.output_json}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
