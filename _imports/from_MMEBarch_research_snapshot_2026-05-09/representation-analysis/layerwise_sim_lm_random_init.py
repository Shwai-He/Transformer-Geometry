import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MATH_PROMPTS = [
    "7 + 5 =",
    "13 - 4 =",
    "6 * 8 =",
    "36 / 9 =",
    "14 + 9 - 6 =",
    "(16 + 8) / 4 + 3 =",
]


def cosine_last_token(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a[:, -1, :].float()
    y = b[:, -1, :].float()
    return F.cosine_similarity(x, y, dim=-1).mean().item()


def generate_text_with_probabilities(model, tokenizer, device: str, prompts: List[str]) -> Dict[str, torch.Tensor]:
    toks = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        out = model(**toks, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = [h.detach().cpu() for h in out.hidden_states]  # [emb, layer1, ..., layerN]
    logits = out.logits.detach().cpu()
    return {"hidden_states": hidden_states, "logits": logits}


def untie_and_randomize_lm_head(model, seed: int = 0):
    torch.manual_seed(seed)
    old_head = model.lm_head
    in_features = old_head.in_features
    out_features = old_head.out_features
    bias = old_head.bias is not None
    new_head = nn.Linear(in_features, out_features, bias=bias, device=old_head.weight.device, dtype=old_head.weight.dtype)
    nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
    if bias:
        nn.init.zeros_(new_head.bias)
    model.lm_head = new_head


def inplace_randomize_lm_head(model, seed: int = 0):
    torch.manual_seed(seed)
    with torch.no_grad():
        model.lm_head.weight.normal_(mean=0.0, std=0.02)
        if model.lm_head.bias is not None:
            model.lm_head.bias.zero_()


def main():
    parser = argparse.ArgumentParser(description="Layerwise cosine after LM-head random init (aligned to dropping_cosine style).")
    parser.add_argument("--model_root_path", type=str, required=False)
    parser.add_argument("--model_postfix", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="./layerwise_sim_lm_random_init.json")
    args = parser.parse_args()

    # ===== Editable config (file-first) =====
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args.model_root_path = "/mnt/bn/seed-aws-va/shwai.he/models"
    model_name = f"{args.model_root_path}/{args.model_postfix}-copy"
    randomize_mode = "untie_then_random"  # choices: ["untie_then_random", "inplace"]
    prompts = MATH_PROMPTS
    log_dir = f"./cosine_logs/layerwise_lm_random_init/temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log-cosine.txt")
    args.output_json = f"/mnt/bn/seed-aws-va/shwai.he/gen-collapse/drop_logs_math/{args.model_postfix}/lm_random_layerwise_sim.json"

    def write_log(s: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== LM Random Init Layerwise Cosine Log ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    write_log(f"model_name: {model_name}")
    write_log(f"randomize_mode: {randomize_mode}")
    write_log(f"seed: {args.seed}")
    write_log(f"prompts: {prompts}")

    base = generate_text_with_probabilities(model, tokenizer, device, prompts)

    if randomize_mode == "untie_then_random":
        untie_and_randomize_lm_head(model, seed=args.seed)
    elif randomize_mode == "inplace":
        inplace_randomize_lm_head(model, seed=args.seed)
    else:
        raise ValueError(f"Unsupported randomize_mode: {randomize_mode}")

    after = generate_text_with_probabilities(model, tokenizer, device, prompts)

    # hidden_states[0] is embedding output
    emb_cos_sim = cosine_last_token(base["hidden_states"][0], after["hidden_states"][0])
    hidden_cos_sim = []
    for i in range(1, len(base["hidden_states"])):
        hidden_cos_sim.append(cosine_last_token(base["hidden_states"][i], after["hidden_states"][i]))

    logits_cos_sim = F.cosine_similarity(
        base["logits"][:, -1, :].float(), after["logits"][:, -1, :].float(), dim=-1
    ).mean().item()

    higher_than_emb = [c > emb_cos_sim for c in hidden_cos_sim]
    num_higher_than_emb = int(sum(1 for x in higher_than_emb if x))

    result = {
        "model_name": model_name,
        "model_postfix": args.model_postfix,
        "randomize_mode": randomize_mode,
        "seed": args.seed,
        "prompts": prompts,
        "emb_cos_sim": emb_cos_sim,
        "hidden_cos_sim": hidden_cos_sim,
        "logits_cos_sim": logits_cos_sim,
        "num_layers_higher_than_emb": num_higher_than_emb,
        "ratio_layers_higher_than_emb": (num_higher_than_emb / len(hidden_cos_sim)) if hidden_cos_sim else None,
        "log_path": log_path,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    write_log(f"emb_cos_sim: {emb_cos_sim}")
    write_log(f"hidden_cos_sim: {hidden_cos_sim}")
    write_log(f"logits_cos_sim: {logits_cos_sim}")
    write_log(f"num_layers_higher_than_emb: {num_higher_than_emb}/{len(hidden_cos_sim)}")

    print(f"emb_cos_sim: {emb_cos_sim:.6f}")
    print(f"logits_cos_sim: {logits_cos_sim:.6f}")
    print(f"num_layers_higher_than_emb: {num_higher_than_emb}/{len(hidden_cos_sim)}")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
