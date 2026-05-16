#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import List

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompts(prompts_file: str, max_prompts: int) -> List[str]:
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def get_hidden_last_token(model, input_ids, attention_mask):
    kwargs = {
        "input_ids": input_ids,
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }
    sig = model.forward.__code__.co_varnames
    if "attention_mask" in sig:
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        out = model(**kwargs)
    # out.hidden_states: embedding + layer outputs
    return [h[:, -1, :].detach().float() for h in out.hidden_states]


def sanitize_tag(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def detect_awq(model_path: str) -> bool:
    cfg = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            j = json.load(f)
        qcfg = j.get("quantization_config", {})
        return isinstance(qcfg, dict) and str(qcfg.get("quant_method", "")).lower() == "awq"
    except Exception:
        return False


def load_model(model_name: str, dtype: str):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if detect_awq(model_name):
        print("[INFO] AWQ detected, loading quant model via native path.")
        return AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto", torch_dtype=dtype_map[dtype]
        ).eval()
    return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto").eval()


def main():
    p = argparse.ArgumentParser("Compare dense vs quant hidden-state differences.")
    p.add_argument("--dense_model", type=str, required=True)
    p.add_argument("--quant_model", type=str, required=True)
    p.add_argument("--prompts_file", type=str, required=True)
    p.add_argument("--max_prompts", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    tag = args.tag if args.tag else f"{sanitize_tag(os.path.basename(args.dense_model))}__vs__{sanitize_tag(os.path.basename(args.quant_model))}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading dense model...")
    model_d = AutoModelForCausalLM.from_pretrained(args.dense_model, trust_remote_code=True, device_map="auto").eval()
    print("[INFO] Loading quant model...")
    model_q = load_model(args.quant_model, args.dtype)

    prompts = read_prompts(args.prompts_file, args.max_prompts)
    if not prompts:
        raise RuntimeError("No prompts loaded.")

    num_layers = len(model_d.model.layers)
    agg_delta = torch.zeros(num_layers)
    agg_rel = torch.zeros(num_layers)
    agg_cos = torch.zeros(num_layers)
    n = 0

    for text in prompts:
        tok = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        # send to first device of dense lm_head
        dev = next(model_d.parameters()).device
        input_ids = tok["input_ids"].to(dev)
        attention_mask = tok["attention_mask"].to(dev)

        hs_d = get_hidden_last_token(model_d, input_ids, attention_mask)[1:]
        hs_q = get_hidden_last_token(model_q, input_ids, attention_mask)[1:]

        for li in range(num_layers):
            x = hs_d[li]
            y = hs_q[li]
            if y.device != x.device:
                y = y.to(x.device)
            delta = (y - x).norm(dim=-1).mean()
            xnorm = x.norm(dim=-1).mean().clamp_min(1e-8)
            rel = delta / xnorm
            cos = torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()
            agg_delta[li] += delta.detach().cpu()
            agg_rel[li] += rel.detach().cpu()
            agg_cos[li] += cos.detach().cpu()
        n += 1

    agg_delta /= max(n, 1)
    agg_rel /= max(n, 1)
    agg_cos /= max(n, 1)

    tsv_path = os.path.join(out_dir, "dense_vs_quant_layerwise.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["layer", "delta_norm", "rel_delta_norm", "cosine_dense_quant"])
        for li in range(num_layers):
            w.writerow([li, float(agg_delta[li]), float(agg_rel[li]), float(agg_cos[li])])

    xs = list(range(num_layers))
    for name, ys, ylabel in [
        ("fig_delta_norm.png", agg_delta.tolist(), "||h_q - h_d||"),
        ("fig_rel_delta_norm.png", agg_rel.tolist(), "||h_q-h_d|| / ||h_d||"),
        ("fig_cosine_dense_quant.png", agg_cos.tolist(), "cos(h_d, h_q)"),
    ]:
        plt.figure(figsize=(10, 4.5))
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.3)
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), dpi=180)
        plt.close()

    print(f"[INFO] Done. Output: {os.path.abspath(out_dir)}")
    print(f"[INFO] TSV: {tsv_path}")


if __name__ == "__main__":
    main()
