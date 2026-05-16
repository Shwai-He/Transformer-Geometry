import os
import sys

import torch
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from forward_utils import generate_with_custom_forward

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transition_metrics_logging import compute_and_log_transition_metrics


def generate_text_with_probabilities(prompts=None, input_ids=None, max_length=512, temperature=0.0, top_k=50, top_p=0.9, use_cache=False):
    texts, probs, ids, hidden, logits, sublayer_steps = generate_with_custom_forward(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=use_cache,
        collect_sublayer=True,
    )
    return texts, probs, ids, hidden, logits, sublayer_steps


def randomize_lm_head(model, mode="untie_then_random", seed=42):
    torch.manual_seed(seed)
    old_head = model.lm_head
    if mode == "untie_then_random":
        new_head = nn.Linear(
            old_head.in_features,
            old_head.out_features,
            bias=(old_head.bias is not None),
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
        if new_head.bias is not None:
            nn.init.zeros_(new_head.bias)
        model.lm_head = new_head
        return

    if mode == "inplace":
        with torch.no_grad():
            model.lm_head.weight.normal_(mean=0.0, std=0.02)
            if model.lm_head.bias is not None:
                model.lm_head.bias.zero_()
        return

    raise ValueError(f"Unsupported lm random mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with different hyperparameters.")
    parser.add_argument("--model_root_path", type=str, default="your_model_root_path")
    parser.add_argument("--model_postfix", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--lm_random_init", type=int, choices=[0, 1], default=0)
    parser.add_argument("--lm_random_mode", type=str, default="untie_then_random", choices=["untie_then_random", "inplace"])
    parser.add_argument("--lm_random_seed", type=int, default=42)

    args = parser.parse_args()

    # File-first defaults (override CLI for your workflow)
    args.lm_random_init = 1
    args.lm_random_mode = "untie_then_random"
    args.lm_random_seed = 42

    mode_tag = "lm_random" if args.lm_random_init == 1 else "dense"
    log_dir = f"./cosine_logs/layerwise_{mode_tag}/temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log-cosine-{mode_tag}-{args.lm_random_mode}-seed{args.lm_random_seed}.txt",
    )

    def write_log(s):
        print(f"log_path: {log_path}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Layerwise Sublayer Metrics Log ===\n")

    model_root_path = args.model_root_path
    model_root_path = "/mnt/bn/seed-aws-va/shwai.he/Pruning-Hierarchies/models"
    model_postfix = args.model_postfix
    model_name = f"{model_root_path}/{model_postfix}"

    max_length = args.max_length
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    use_cache = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    category_chars = ["A", "B", "C", "D"]

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    if args.lm_random_init == 1:
        randomize_lm_head(model, mode=args.lm_random_mode, seed=args.lm_random_seed)
        write_log(f"[LM_RANDOM_INIT] enabled | mode={args.lm_random_mode} | seed={args.lm_random_seed}")
    else:
        write_log("[LM_RANDOM_INIT] disabled")

    prompts = [
        "John has twice as many books as Mary. Together they have 18 books. How many books does John have?"
    ]

    # attach lm_head to each layer
    for layer in model.model.layers:
        layer.lm_head = model.lm_head

    for i, prompt in enumerate(prompts):
        print(f"prompt: {prompt}")
        write_log(f"\n=== Prompt {i+1} ===\n{prompt}\n")

        output, probabilities, _, _, _, sublayer_traces = generate_text_with_probabilities(
            prompts=[prompt],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache
        )

        dense_text = output[0].replace(prompt, "")
        print("\nGenerated Output from Dense Model:")
        print(dense_text)

        write_log("--- Dense Model Output ---")
        write_log(dense_text + "\n")

        metric_topk = top_k if top_k and top_k > 0 else 5
        for step_idx, step_rec in enumerate(sublayer_traces):
            all_layer_idxs = sorted(
                set(step_rec.get("attn", {}).keys()) | set(step_rec.get("mlp", {}).keys())
            )
            for layer_idx in all_layer_idxs:
                for sublabel in ("attn", "mlp"):
                    rec = step_rec.get(sublabel, {}).get(layer_idx)
                    if rec is None or "residual" not in rec or "hidden_states" not in rec:
                        continue
                    compute_and_log_transition_metrics(
                        residual=rec["residual"],
                        hidden_states=rec["hidden_states"],
                        lm_head=model.lm_head,
                        layer_idx=layer_idx,
                        drop_n=0,
                        temperature=temperature,
                        prompt_idx=i,
                        label=f"{sublabel}_step{step_idx}",
                        log_path=log_path,
                        topk=metric_topk,
                        decode_topk=False,
                        tokenizer=tokenizer,
                        category_chars=category_chars,
                    )
