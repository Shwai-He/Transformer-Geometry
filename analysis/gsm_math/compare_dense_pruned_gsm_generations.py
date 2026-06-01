from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_gsm_answer(text: str) -> Optional[str]:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).replace(",", "")
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def normalize_answer(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p is None or g is None:
        return False
    return abs(p - g) < 1e-6


def format_prompt(question: str, use_cot_prompt: bool) -> str:
    if use_cot_prompt:
        return f"Question: {question}\nLet's think step by step.\n"
    return f"Question: {question}\nAnswer:"


def load_examples(args) -> List[Dict[str, Any]]:
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    indices: Iterable[int]
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        start = max(args.start, 0)
        stop = min(start + args.max_examples, len(ds))
        indices = range(start, stop)

    examples = []
    for idx in indices:
        row = ds[int(idx)]
        gold = extract_gsm_answer(row["answer"])
        examples.append(
            {
                "idx": int(idx),
                "question": row["question"],
                "gold_answer": gold,
                "gold_solution": row["answer"],
                "prompt": format_prompt(row["question"], args.use_cot_prompt),
            }
        )
    return examples


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(model_path: str, args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, args) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    )
    enc = {k: v.to(args.device) for k, v in enc.items()}
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "use_cache": args.use_cache,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    out = model.generate(**enc, **gen_kwargs)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def run_model(model_path: str, examples: List[Dict[str, Any]], label: str, args) -> Dict[int, Dict[str, Any]]:
    print(f"[INFO] Loading {label}: {model_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_path, args)
    outputs = {}
    for pos, ex in enumerate(examples, start=1):
        print(f"[INFO] {label}: example {pos}/{len(examples)} idx={ex['idx']}", flush=True)
        text = generate_one(model, tokenizer, ex["prompt"], args)
        pred = extract_gsm_answer(text)
        outputs[ex["idx"]] = {
            "generation": text,
            "pred_answer": pred,
            "correct": answers_match(pred, ex["gold_answer"]),
        }
    del model
    torch.cuda.empty_cache()
    return outputs


def write_outputs(records: List[Dict[str, Any]], args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "gsm_dense_pruned_generations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "dense_correct": sum(1 for r in records if r["dense"]["correct"]),
        "pruned_correct": sum(1 for r in records if r["pruned"]["correct"]),
        "num_examples": len(records),
        "dense_model": args.dense_model,
        "pruned_model": args.pruned_model,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    summary["dense_accuracy"] = summary["dense_correct"] / max(summary["num_examples"], 1)
    summary["pruned_accuracy"] = summary["pruned_correct"] / max(summary["num_examples"], 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# GSM Dense vs Pruned Generation Check",
        "",
        f"- Dense model: `{args.dense_model}`",
        f"- Pruned model: `{args.pruned_model}`",
        f"- Examples: {summary['num_examples']}",
        f"- Dense correct: {summary['dense_correct']} / {summary['num_examples']}",
        f"- Pruned correct: {summary['pruned_correct']} / {summary['num_examples']}",
        "",
        "| idx | gold | dense pred | dense ok | pruned pred | pruned ok |",
        "|---:|---:|---:|:---:|---:|:---:|",
    ]
    for rec in records:
        md_lines.append(
            f"| {rec['idx']} | {rec['gold_answer']} | {rec['dense']['pred_answer']} | "
            f"{'yes' if rec['dense']['correct'] else 'no'} | {rec['pruned']['pred_answer']} | "
            f"{'yes' if rec['pruned']['correct'] else 'no'} |"
        )
    md_lines.extend(["", "## Examples", ""])
    for rec in records:
        md_lines.extend(
            [
                f"### GSM8K idx {rec['idx']}",
                "",
                rec["question"],
                "",
                f"Gold: `{rec['gold_answer']}`",
                "",
                "**Dense output**",
                "",
                "```text",
                rec["dense"]["generation"],
                "```",
                "",
                "**Pruned output**",
                "",
                "```text",
                rec["pruned"]["generation"],
                "```",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[INFO] Wrote {jsonl_path}", flush=True)
    print(f"[INFO] Wrote {out_dir / 'README.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense and pruned model generations on GSM8K examples.")
    parser.add_argument("--dense_model", required=True)
    parser.add_argument("--pruned_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=16)
    parser.add_argument("--indices", default="", help="Optional comma-separated GSM8K indices.")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_cot_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    examples = load_examples(args)
    dense = run_model(args.dense_model, examples, "dense", args)
    pruned = run_model(args.pruned_model, examples, "pruned", args)

    records = []
    for ex in examples:
        records.append(
            {
                "idx": ex["idx"],
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "gold_solution": ex["gold_solution"],
                "prompt": ex["prompt"],
                "dense": dense[ex["idx"]],
                "pruned": pruned[ex["idx"]],
            }
        )
    write_outputs(records, args)


if __name__ == "__main__":
    main()
