import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_forward_utils import apply_drop_masks, generate_with_custom_forward


def generate_text_with_probabilities(
    model_obj,
    tokenizer_obj,
    device_obj,
    prompts=None,
    input_ids=None,
    max_length=512,
    temperature=0.0,
    top_k=50,
    top_p=0.9,
    use_cache=False,
):
    texts, probs, ids, hidden, logits, _ = generate_with_custom_forward(
        model=model_obj,
        tokenizer=tokenizer_obj,
        device=device_obj,
        prompts=prompts,
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=use_cache,
    )
    return texts, probs, ids, hidden, logits


def cosine_hidden_states(h1, h2):
    sims = []
    for step1, step2 in zip(h1, h2):
        sim = F.cosine_similarity(step1, step2, dim=-1).mean()
        sims.append(sim)
    return torch.stack(sims)


def flatten_tensor_list(x):
    if isinstance(x, list):
        x = torch.cat([t.flatten() for t in x], dim=0)
    return x.detach().cpu().flatten()


def weighted_variance(x, weights, dim=-1):
    mean = (weights * x).sum(dim=dim, keepdim=True)
    var = (weights * (x - mean) ** 2).sum(dim=dim)
    return var


def topk_variance(delta, p, k):
    topk_p, topk_idx = torch.topk(p, k=k, dim=-1)
    delta_topk = torch.gather(delta, dim=-1, index=topk_idx)
    topk_p = topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-12)
    return weighted_variance(delta_topk, topk_p)


def compare_and_log(
    write_log,
    probs_a,
    probs_b,
    hidden_a,
    hidden_b,
    logits_a,
    logits_b,
    name_a="dense",
    name_b="target",
    analysis_temperature=1.0,
):
    emb_cos_sim = cosine_hidden_states(hidden_a, hidden_b)
    logits_cos_sim = cosine_hidden_states(logits_a, logits_b)
    prob_cos_sim = cosine_hidden_states(probs_a, probs_b)

    kl_list = []
    for p, q in zip(probs_a, probs_b):
        kl = F.kl_div((q + 1e-12).log(), p, reduction="batchmean")
        kl_list.append(kl)
    kl_divergence = torch.stack(kl_list)

    write_log(f"=== Cosine Similarity ({name_a} vs {name_b}) ===")
    write_log(f"emb_cos_sim_flat: {flatten_tensor_list(emb_cos_sim).tolist()}")
    write_log(f"logits_cos_sim_flat: {flatten_tensor_list(logits_cos_sim).tolist()}")
    write_log(f"prob_cos_sim_flat: {flatten_tensor_list(prob_cos_sim).tolist()}")
    write_log(f"=== KL Divergence ({name_a} vs {name_b}) ===")
    write_log(f"kl_divergence_flat: {flatten_tensor_list(kl_divergence).tolist()}")

    kl_estimates = []
    cos_estimates = []
    uniform_vars = []
    top5_vars = []
    top10_vars = []
    deltas = []
    l1_topk = []
    l2_topk = []

    for z, z2, p in zip(logits_a, logits_b, probs_a):
        z = z.float()
        z2 = z2.float()
        p = p.float()
        delta = z2 - z
        deltas.append(delta)

        var_p = weighted_variance(delta, p)
        kl_estimates.append((var_p / (2 * analysis_temperature * analysis_temperature)).mean().item())

        r = p**2
        r = r / (r.sum(dim=-1, keepdim=True) + 1e-12)
        var_r = weighted_variance(delta, r)
        cos_estimates.append((var_r / (2 * analysis_temperature * analysis_temperature)).mean().item())

        V = delta.size(-1)
        uniform_w = torch.full_like(delta, 1.0 / V)
        uniform_vars.append((weighted_variance(delta, uniform_w) / (2 * analysis_temperature * analysis_temperature)).mean().item())
        top5_vars.append((topk_variance(delta, p, k=5) / (2 * analysis_temperature * analysis_temperature)).mean().item())
        top10_vars.append((topk_variance(delta, p, k=10) / (2 * analysis_temperature * analysis_temperature)).mean().item())

        k_top = 5
        topk_idx = p.topk(k_top, dim=-1).indices
        delta_topk = delta.gather(-1, topk_idx)
        l1_topk.append(delta_topk.abs().sum(dim=-1).mean().item())
        l2_topk.append(delta_topk.pow(2).mean(dim=-1).sqrt().mean().item())

    deltas = torch.stack(deltas, dim=0)  # (T, B, V)
    l1_global = deltas.abs().mean(dim=-1)            # (T, B)
    l2_global = deltas.pow(2).mean(dim=-1).sqrt()    # (T, B)

    write_log(f"analysis_temperature: {analysis_temperature}")
    write_log(f"kl_estimates: {kl_estimates}")
    write_log(f"cos_estimates: {cos_estimates}")
    write_log(f"uniform_var: {uniform_vars}")
    write_log(f"top5_var: {top5_vars}")
    write_log(f"top10_var: {top10_vars}")
    write_log(f"global_l1: {l1_global.detach().cpu().tolist()}")
    write_log(f"global_l2: {l2_global.detach().cpu().tolist()}")
    write_log(f"top5_l1: {l1_topk}")
    write_log(f"top5_l2: {l2_topk}")
    write_log(f"summary_global_l1_mean: {l1_global.mean().item():.6f}")
    write_log(f"summary_global_l2_mean: {l2_global.mean().item():.6f}")
    write_log(f"summary_top5_l1_mean: {sum(l1_topk) / max(len(l1_topk), 1):.6f}")
    write_log(f"summary_top5_l2_mean: {sum(l2_topk) / max(len(l2_topk), 1):.6f}")
    write_log("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final-step comparison for dropped/pruned settings.")
    parser.add_argument("--analysis_mode", type=str, default="dropped", choices=["dropped", "pruned"])
    parser.add_argument("--model_name", type=str, required=True, help="Dense model name/path (HF id or local path).")
    parser.add_argument("--model_tag", type=str, default=None, help="Tag used to locate dropped checkpoints under dropped_root_path (defaults to a sanitized model_name).")
    parser.add_argument("--dropped_root_path", type=str, default="you_dropped_root_path")
    parser.add_argument("--pruned_model_name", type=str, default=None, help="Pruned model path when analysis_mode=pruned.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--drop_n", type=int, default=8)
    parser.add_argument("--target_layer", type=str, default="attn")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_dir = f"./cosine_logs/{args.target_layer}/temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{args.analysis_mode}_drop{args.drop_n}-final.txt")

    def write_log(s):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Final Comparison Log ({args.analysis_mode}) ===\n")

    model_name = args.model_name
    model_tag = args.model_tag or args.model_name
    analysis_temperature = args.temperature if args.temperature > 0 else 1.0
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
        "John has twice as many books as Mary. Together they have 18 books. How many books does John have?"
    ]

    for i, prompt in enumerate(prompts):
        write_log(f"\n=== Prompt {i+1} ===\n{prompt}\n")

        torch.manual_seed(42)
        out_d, probs_d, input_ids, hidden_d, logits_d = generate_text_with_probabilities(
            model_obj=model_dense,
            tokenizer_obj=tokenizer,
            device_obj=device,
            prompts=[prompt],
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=True,
        )
        dense_text = out_d[0].replace(prompt, "")
        write_log("--- Dense Model Output ---")
        write_log(dense_text + "\n")

        if args.analysis_mode == "dropped":
            if args.drop_n <= 0:
                write_log("drop_n <= 0, skip dropped comparison.")
                continue

            if args.target_layer in ["attn", "mlp"]:
                dropped_model_path = (
                    f"{args.dropped_root_path}/{model_tag}-layer_drop_{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
                )
            else:
                dropped_model_path = (
                    f"{args.dropped_root_path}/block_drop/{model_tag}-block_drop-{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
                )
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
            out_t, probs_t, _, hidden_t, logits_t = generate_text_with_probabilities(
                model_obj=model_target,
                tokenizer_obj=tokenizer,
                device_obj=device,
                prompts=[prompt],
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
            )
            text_t = out_t[0].replace(prompt, "")
            write_log("--- Dropped Model Output ---")
            write_log(text_t + "\n")
            compare_and_log(
                write_log,
                probs_d,
                probs_t,
                hidden_d,
                hidden_t,
                logits_d,
                logits_t,
                "dense",
                "dropped",
                analysis_temperature=analysis_temperature,
            )

        else:
            torch.manual_seed(42)
            out_t, probs_t, _, hidden_t, logits_t = generate_text_with_probabilities(
                model_obj=model_target,
                tokenizer_obj=tokenizer,
                device_obj=device,
                prompts=[prompt],
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
            )
            text_t = out_t[0].replace(prompt, "")
            write_log("--- Pruned Model Output ---")
            write_log(text_t + "\n")
            compare_and_log(
                write_log,
                probs_d,
                probs_t,
                hidden_d,
                hidden_t,
                logits_d,
                logits_t,
                "dense",
                "pruned",
                analysis_temperature=analysis_temperature,
            )
