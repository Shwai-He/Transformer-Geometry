import os

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import json
from generation_forward_utils import apply_drop_masks, forward_last_token

def log_subword_probs(prob_orig, prob_drop, sub_token_ids, tokenizer):
    # last step
    p1 = prob_orig[-1].squeeze(0)   # [V]
    p2 = prob_drop[-1].squeeze(0)   # [V]

    write_log("\n===== SUBWORD PROBABILITIES (Original vs Dropped) =====")
    print("\n===== SUBWORD PROBABILITIES (Original vs Dropped) =====")

    for tid in sub_token_ids:
        token = tokenizer.decode([tid])
        po = p1[tid].log().item()
        pd = p2[tid].log().item()

        s = f"{token:<10} | orig={po:.6f} | drop={pd:.6f}"
        print(s)
        write_log(s)


def generate_text_with_probabilities(prompts=None, input_ids=None,
                                     temperature=0.0, use_cache=False):
    return forward_last_token(
        model=model_obj,
        tokenizer=tokenizer_obj,
        device=device_obj,
        prompts=prompts,
        input_ids=input_ids,
        use_cache=use_cache,
        temperature=temperature,
    )


def log_top_tokens_with_dropped(prob_orig, prob_drop, tokenizer, top_k=10):
    # last step probabilities
    p1 = prob_orig[-1].squeeze(0)      # original probs [V]
    p2 = prob_drop[-1].squeeze(0)      # dropped probs  [V]

    values, indices = torch.topk(p1, k=top_k, dim=-1)
    tokens = []

    write_log("\n===== TOP TOKENS (Original vs Dropped) =====")
    for v, idx in zip(values, indices):
        idx = idx.item()
        token = repr(tokenizer.decode([idx]))
        p_orig = v.item()
        p_drop = p2[idx].item()

        s = f"{token:<15} | orig={p_orig:.6f} | drop={p_drop:.6f}"
        print(s)
        write_log(s)
        tokens.append(token)

    write_log(f"\n===== TOP TOKENS {tokens} =====")



def get_top_tokens(prob_tensor, tokenizer, top_k=10):
    probs = prob_tensor[-1].squeeze(0)   # last step, shape: [Vocab]
    values, indices = torch.topk(probs, k=top_k, dim=-1)

    tokens = [tokenizer.decode([i]) for i in indices.tolist()]
    probs = values.tolist()
    return tokens, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with different hyperparameters.")
    parser.add_argument("--analysis_mode", type=str, default="dropped", choices=["dropped", "pruned"])
    parser.add_argument("--model_name", type=str, required=True, help="Dense model name/path (HF id or local path).")
    parser.add_argument("--model_tag", type=str, default=None, help="Tag used to locate dropped checkpoints under dropped_root_path (defaults to a sanitized model_name).")
    parser.add_argument("--pruned_model_name", type=str, default=None, help="Pruned model path when analysis_mode=pruned.")
    parser.add_argument("--dropped_root_path", type=str, default="you_dropped_root_path")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--drop_n", type=int, default=8)
    parser.add_argument("--target_layer", type=str, default="attn")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--mode", type=str, default="default", help="# FIX: mode tag used in log filename")

    args = parser.parse_args()

    log_dir = f"./cosine_logs/{args.target_layer}_{args.analysis_mode}_temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_drop{args.drop_n}-{args.mode}-subspace.txt")  # FIX: args.mode is now explicitly defined

    def write_log(s):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Drop Experiment Log ===\n")

    model_name = args.model_name
    model_tag = args.model_tag or args.model_name
    dropped_root_path = args.dropped_root_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dense = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    model_target = None
    if args.analysis_mode == "pruned":
        if not args.pruned_model_name:
            raise ValueError("--pruned_model_name is required when --analysis_mode=pruned")
        model_target = AutoModelForCausalLM.from_pretrained(args.pruned_model_name, trust_remote_code=True).to(device).eval()
    else:
        model_target = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()


    prompts = [
        """Which animal is a mammal?
        Choose the correct answer:
        a) Snake
        b) Frog
        c) Dog
        d) Lizard""",
    ]

    for j, layer in enumerate(model_dense.model.layers):
        layer.lm_head = model_dense.lm_head
        layer.drop_n = args.drop_n

    for prompt in prompts:

        write_log(prompt)

        for j, layer in enumerate(model_dense.model.layers):
            layer.drop_attn = False
            layer.drop_mlp = False

        torch.manual_seed(42)
        model_obj = model_dense
        tokenizer_obj = tokenizer
        device_obj = device
        output, probabilities, _, hidden_states, logits = generate_text_with_probabilities(
            prompts=[prompt],
            temperature=args.temperature,
            use_cache=True, 
        )

        dense_text = output[0].replace(prompt, "")
        write_log(dense_text)

        if args.analysis_mode == "dropped" and args.drop_n > 0:
            if args.target_layer in ["attn", "mlp"]:
                dropped_model_path = f"{dropped_root_path}/{model_tag}-layer_drop_{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
            else:
                dropped_model_path = f"{dropped_root_path}/block_drop/{model_tag}-block_drop-{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"

            with open(os.path.join(dropped_model_path, "config.json"), "r") as f:
                config = json.load(f)

            drop_attn_list = config.get("drop_attn_list", [])
            drop_mlp_list = config.get("drop_mlp_list", [])

            write_log(f"drop_attn_list: {list(drop_attn_list)} | drop_mlp_list: {list(drop_mlp_list)}")

            apply_drop_masks(
                model=model_target,
                target_layer=args.target_layer,
                drop_attn_list=drop_attn_list,
                drop_mlp_list=drop_mlp_list,
                drop_n=args.drop_n,
            )

            torch.manual_seed(42)
            model_obj = model_target
            tokenizer_obj = tokenizer
            device_obj = device
            output_dropped, probabilities_dropped, _, hidden_states_dropped, logits_dropped = generate_text_with_probabilities(
                prompts=[prompt],
                temperature=args.temperature,
                use_cache=True
            )

        elif args.analysis_mode == "pruned":
            torch.manual_seed(42)
            model_obj = model_target
            tokenizer_obj = tokenizer
            device_obj = device
            output_dropped, probabilities_dropped, _, hidden_states_dropped, logits_dropped = generate_text_with_probabilities(
                prompts=[prompt],
                temperature=args.temperature,
                use_cache=True
            )
            write_log("compare_mode: pruned")
        else:
            continue

        option_tokens = [" a", " b", " c", " d"]
        sub_token_ids = []

        for t in option_tokens:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 0:
                raise ValueError(f"Tokenizer failed for {t}")
            sub_token_ids.append(ids[-1])

        print("Subspace token ids:", sub_token_ids)
        write_log(f"Subspace token ids: {sub_token_ids}")

        sub_cos_logits = []
        sub_cos_prob = []
        sub_kl = []

        global_cos_logits = []
        global_cos_prob = []
        global_kl = []

        z  = logits[:, -1, :].squeeze(0)          # [V]
        z2 = logits_dropped[:, -1, :].squeeze(0)  # [V]

        p  = probabilities.squeeze(0)             # [V]
        p2 = probabilities_dropped.squeeze(0)     # [V]

        global_cos_logits = [
            F.cosine_similarity(z.unsqueeze(0), z2.unsqueeze(0), dim=-1).item()
        ]

        global_cos_prob = [
            F.cosine_similarity(p.unsqueeze(0), p2.unsqueeze(0), dim=-1).item()
        ]

        global_kl = [
            F.kl_div((p2 + 1e-12).log(), p, reduction="batchmean").item()
        ]

        # ---------------- SUBSPACE ----------------
        sub_z  = z[sub_token_ids]
        sub_z2 = z2[sub_token_ids]

        sub_p  = p[sub_token_ids]
        sub_p2 = p2[sub_token_ids]

        # normalize inside subspace
        sub_p  = sub_p  / sub_p.sum()
        sub_p2 = sub_p2 / sub_p2.sum()

        sub_cos_logits = [
            F.cosine_similarity(sub_z.unsqueeze(0), sub_z2.unsqueeze(0), dim=-1).item()
        ]

        sub_cos_prob = [
            F.cosine_similarity(sub_p.unsqueeze(0), sub_p2.unsqueeze(0), dim=-1).item()
        ]

        sub_kl = [
            F.kl_div((sub_p2 + 1e-12).log(), sub_p, reduction="batchmean").item()
        ]


        print("\n===== GLOBAL =====")
        print("Global logits cosine:", global_cos_logits)
        print("Global prob cosine:", global_cos_prob)
        print("Global KL:", global_kl)

        print("\n===== SUBSPACE (a/b/c/d) =====")
        print("Subspace logits cosine:", sub_cos_logits)
        print("Subspace prob cosine:", sub_cos_prob)
        print("Subspace KL:", sub_kl)

        write_log("\n===== GLOBAL =====")
        write_log(f"Global logits cosine: {global_cos_logits}")
        write_log(f"Global prob cosine: {global_cos_prob}")
        write_log(f"Global KL: {global_kl}")

        log_top_tokens_with_dropped(probabilities, probabilities_dropped, tokenizer, top_k=30)

        write_log("\n===== SUBSPACE (a/b/c/d) =====")
        write_log(f"Subspace logits cosine: {sub_cos_logits}")
        write_log(f"Subspace prob cosine: {sub_cos_prob}")
        write_log(f"Subspace KL: {sub_kl}")
        write_log("=" * 80)

        log_subword_probs(probabilities, probabilities_dropped, sub_token_ids, tokenizer)
