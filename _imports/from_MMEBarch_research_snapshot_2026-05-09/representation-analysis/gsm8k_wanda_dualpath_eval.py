import argparse
import gc
import json
import os
import re
from typing import Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate_batch_texts(
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


def build_prompt(question: str) -> str:
    return (
        "Solve the following GSM8K question within the token budget.\n"
        "You may reason briefly and end with 'Final answer: <number>'.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser(description="GSM8K Wanda dual-path eval (dense path + sparse path).")
    parser.add_argument("--max_questions", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="./gsm8k_wanda_dualpath_eval.json")
    args = parser.parse_args()

    # ===== Editable config (file-first) =====
    model_postfix = "Qwen/Qwen2.5-7B-Instruct"
    dense_model_path = f"/mnt/bn/seed-aws-va/shwai.he/models/{model_postfix}-copy"
    sparse_root = f"/mnt/bn/seed-aws-va/shwai.he/wanda/out/{model_postfix}"
    sparsity_list = ["2-4", "4-8", "unstructured"]
    methods = ["wanda", "sparsegpt"]
    batch_size = 8
    args.output_json = "/mnt/bn/seed-aws-va/shwai.he/drop_logs_easy/gsm8k_wanda_dualpath/result.json"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_gsm8k(args.max_questions)
    prompts = [build_prompt(qa["question"]) for qa in data]
    gts = [parse_gsm8k_gt(qa["answer"]) for qa in data]

    dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_path)
    if dense_tokenizer.pad_token is None:
        dense_tokenizer.pad_token = dense_tokenizer.eos_token
    dense_tokenizer.padding_side = "left"
    dense_model = AutoModelForCausalLM.from_pretrained(dense_model_path, trust_remote_code=True).to(device).eval()

    dense_full_all: List[str] = []
    for st in range(0, len(prompts), batch_size):
        ed = min(st + batch_size, len(prompts))
        print(f"[dense] {st}/{len(prompts)}")
        dense_full_all.extend(
            generate_batch_texts(
                dense_model,
                dense_tokenizer,
                device,
                prompts[st:ed],
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
            )
        )

    dense_comp_all = [extract_completion(full, p) for full, p in zip(dense_full_all, prompts)]
    dense_ans_all = [normalize_num(extract_number(c)) for c in dense_comp_all]

    all_results = {
        "task": "gsm8k_wanda_dualpath",
        "dense_model_path": dense_model_path,
        "sparse_root": sparse_root,
        "settings": {
            "max_questions": len(data),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "variants": [],
    }

    for sparsity in sparsity_list:
        for method in methods:
            sparse_model_path = f"{sparse_root}/{sparsity}/{method}"
            variant_name = f"{sparsity}/{method}"
            if not os.path.isdir(sparse_model_path):
                all_results["variants"].append(
                    {"variant": variant_name, "path": sparse_model_path, "status": "missing"}
                )
                continue

            sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_model_path)
            if sparse_tokenizer.pad_token is None:
                sparse_tokenizer.pad_token = sparse_tokenizer.eos_token
            sparse_tokenizer.padding_side = "left"
            sparse_model = AutoModelForCausalLM.from_pretrained(
                sparse_model_path, trust_remote_code=True
            ).to(device).eval()

            rows = []
            same = 0
            dense_ok = 0
            sparse_ok = 0
            gt_cnt = 0

            sparse_full_all: List[str] = []
            for st in range(0, len(prompts), batch_size):
                ed = min(st + batch_size, len(prompts))
                print(f"[{variant_name}] {st}/{len(prompts)}")
                sparse_full_all.extend(
                    generate_batch_texts(
                        sparse_model,
                        sparse_tokenizer,
                        device,
                        prompts[st:ed],
                        args.max_new_tokens,
                        args.temperature,
                        args.top_k,
                        args.top_p,
                    )
                )

            for i, qa in enumerate(data):
                question = qa["question"]
                gt = gts[i]
                prompt = prompts[i]
                dense_full = dense_full_all[i]
                dense_comp = dense_comp_all[i]
                dense_ans = dense_ans_all[i]
                sparse_full = sparse_full_all[i]
                sparse_comp = extract_completion(sparse_full, prompt)
                sparse_ans = normalize_num(extract_number(sparse_comp))

                same_flag = (dense_ans == sparse_ans) if (dense_ans and sparse_ans) else False
                dense_gt = None
                sparse_gt = None
                if gt is not None:
                    gt_cnt += 1
                    dense_gt = (dense_ans == gt) if dense_ans else False
                    sparse_gt = (sparse_ans == gt) if sparse_ans else False
                    dense_ok += int(bool(dense_gt))
                    sparse_ok += int(bool(sparse_gt))
                same += int(bool(same_flag))

                rows.append(
                    {
                        "idx": i,
                        "question": question,
                        "ground_truth": gt,
                        "prompt": prompt,
                        "dense_out_full": dense_full,
                        "dense_completion": dense_comp,
                        "dense_answer": dense_ans,
                        "sparse_out_full": sparse_full,
                        "sparse_completion": sparse_comp,
                        "sparse_answer": sparse_ans,
                        "same_as_dense": same_flag,
                        "dense_correct_vs_gt": dense_gt,
                        "sparse_correct_vs_gt": sparse_gt,
                    }
                )

            n = len(rows)
            summary = {
                "n": n,
                "gt_count": gt_cnt,
                "same_as_dense": same,
                "same_rate": (same / n) if n else None,
                "dense_acc_vs_gt": (dense_ok / gt_cnt) if gt_cnt else None,
                "sparse_acc_vs_gt": (sparse_ok / gt_cnt) if gt_cnt else None,
            }

            all_results["variants"].append(
                {
                    "variant": variant_name,
                    "path": sparse_model_path,
                    "status": "ok",
                    "summary": summary,
                    "rows": rows,
                }
            )

            del sparse_model
            del sparse_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    del dense_model
    del dense_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
