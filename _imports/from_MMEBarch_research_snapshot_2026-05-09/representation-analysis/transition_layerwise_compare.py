import os
import sys
import json

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_forward_utils import apply_drop_masks, generate_with_custom_forward

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transition_metrics_logging import compute_and_log_transition_metrics


def _last_token(x: torch.Tensor) -> torch.Tensor:
    if x.dim() >= 3:
        return x[:, -1:, :]
    if x.dim() == 2:
        return x.unsqueeze(1)
    raise ValueError(f"Unexpected tensor rank: {x.dim()}")


def align_pruned_sublayers(model_dense, model_pruned):
    for dense_layer, pruned_layer in zip(model_dense.model.layers, model_pruned.model.layers):
        if hasattr(dense_layer, "self_attn") and hasattr(pruned_layer, "self_attn") and pruned_layer.self_attn is not None:
            ref = dense_layer.self_attn.q_proj.weight
            pruned_layer.self_attn.to(device=ref.device, dtype=ref.dtype)
        if hasattr(dense_layer, "mlp") and hasattr(pruned_layer, "mlp") and pruned_layer.mlp is not None:
            ref = dense_layer.mlp.gate_proj.weight
            pruned_layer.mlp.to(device=ref.device, dtype=ref.dtype)


def attach_counterfactual_hooks(model_dense, model_pruned, log_path, temperature, prompt_idx, drop_n):
    state = {"step": -1, "prompt_idx": prompt_idx, "attn_residual": {}, "mlp_residual": {}}
    handles = []

    dense_layers = model_dense.model.layers
    pruned_layers = model_pruned.model.layers

    for layer_idx, dense_layer in enumerate(dense_layers):
        pruned_layer = pruned_layers[layer_idx]

        def layer_pre_hook(_mod, args, kwargs, _idx=layer_idx):
            if _idx == 0:
                state["step"] += 1
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return
            state["attn_residual"][_idx] = _last_token(hidden.detach())

        handles.append(dense_layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

        if getattr(dense_layer, "self_attn", None) is not None and getattr(pruned_layer, "self_attn", None) is not None:
            def attn_hook(_mod, args, kwargs, output, _idx=layer_idx, _pruned=pruned_layer):
                residual = state["attn_residual"].get(_idx)
                if residual is None:
                    return

                dense_attn_out = output[0] if isinstance(output, tuple) else output
                dense_attn_out = _last_token(dense_attn_out.detach())
                h_full = residual + dense_attn_out
                state["mlp_residual"][_idx] = h_full

                hidden_in = kwargs.get("hidden_states", args[0] if args else None)
                if hidden_in is None:
                    return

                with torch.no_grad():
                    pruned_out = _pruned.self_attn(
                        hidden_states=hidden_in,
                        attention_mask=kwargs.get("attention_mask", None),
                        position_ids=kwargs.get("position_ids", None),
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=None,
                        position_embeddings=kwargs.get("position_embeddings", None),
                    )

                pruned_attn_out = pruned_out[0] if isinstance(pruned_out, tuple) else pruned_out
                pruned_attn_out = _last_token(pruned_attn_out.detach())
                h_pruned = residual + pruned_attn_out

                compute_and_log_transition_metrics(
                    residual=h_full,
                    hidden_states=h_pruned,
                    lm_head=model_dense.lm_head,
                    layer_idx=_idx,
                    drop_n=drop_n,
                    temperature=temperature,
                    prompt_idx=state["prompt_idx"],
                    label=f"attn_step{state['step']}",
                    log_path=log_path,
                    topk=5,
                    decode_topk=False,
                )

            handles.append(dense_layer.self_attn.register_forward_hook(attn_hook, with_kwargs=True))

        if getattr(dense_layer, "mlp", None) is not None and getattr(pruned_layer, "mlp", None) is not None:
            def mlp_hook(_mod, args, kwargs, output, _idx=layer_idx, _pruned=pruned_layer):
                residual = state["mlp_residual"].get(_idx)
                if residual is None:
                    return

                dense_mlp_out = output[0] if isinstance(output, tuple) else output
                dense_mlp_out = _last_token(dense_mlp_out.detach())
                h_full = residual + dense_mlp_out

                hidden_in = args[0] if args else kwargs.get("hidden_states", None)
                if hidden_in is None:
                    return

                with torch.no_grad():
                    pruned_mlp_out = _pruned.mlp(hidden_in)

                pruned_mlp_out = _last_token(pruned_mlp_out.detach())
                h_pruned = residual + pruned_mlp_out

                compute_and_log_transition_metrics(
                    residual=h_full,
                    hidden_states=h_pruned,
                    lm_head=model_dense.lm_head,
                    layer_idx=_idx,
                    drop_n=drop_n,
                    temperature=temperature,
                    prompt_idx=state["prompt_idx"],
                    label=f"mlp_step{state['step']}",
                    log_path=log_path,
                    topk=5,
                    decode_topk=False,
                )

            handles.append(dense_layer.mlp.register_forward_hook(mlp_hook, with_kwargs=True))

    return handles, state


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
    texts, probs, ids, hidden, logits, sublayer_steps = generate_with_custom_forward(
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
        collect_sublayer=True,
    )
    return texts, probs, ids, hidden, logits, sublayer_steps


def log_sublayer_metrics(model_obj, sublayer_traces, prompt_idx, temperature, top_k, log_path, label_prefix):
    metric_topk = top_k if top_k and top_k > 0 else 5
    for step_idx, step_rec in enumerate(sublayer_traces):
        for sublabel in ("attn", "mlp"):
            layer_recs = step_rec.get(sublabel, {})
            for layer_idx in sorted(layer_recs.keys()):
                rec = layer_recs[layer_idx]
                if "residual" not in rec or "hidden_states" not in rec:
                    continue
                compute_and_log_transition_metrics(
                    residual=rec["residual"],
                    hidden_states=rec["hidden_states"],
                    lm_head=model_obj.lm_head,
                    layer_idx=layer_idx,
                    drop_n=0,
                    temperature=temperature,
                    prompt_idx=prompt_idx,
                    label=f"{label_prefix}_{sublabel}_step{step_idx}",
                    log_path=log_path,
                    topk=metric_topk,
                    decode_topk=False,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with different hyperparameters.")
    parser.add_argument("--analysis_mode", type=str, default="dropped", choices=["dropped", "pruned"])
    parser.add_argument("--model_name", type=str, required=True, help="Dense model name/path (HF id or local path).")
    parser.add_argument("--model_tag", type=str, default=None, help="Tag used to locate dropped checkpoints under dropped_root_path (defaults to a sanitized model_name).")
    parser.add_argument("--pruned_model_name", type=str, default=None, help="Pruned model path when analysis_mode=pruned.")
    parser.add_argument("--dropped_root_path", type=str, default="you_dropped_root_path")
    parser.add_argument("--target_layer", type=str, default="attn", choices=["attn", "mlp", "all"])
    parser.add_argument("--drop_n", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    parser.add_argument("--prompt_idx", type=int, default=None, help="Only run one prompt index if set.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    log_dir = f"./cosine_logs/layerwise_{args.analysis_mode}/temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log-cosine.txt")

    def write_log(s):
        print(f"log_path: {log_path}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Layerwise Sublayer Metrics Log ({args.analysis_mode}) ===\n")

    model_tag = args.model_tag or args.model_name
    model_name = args.model_name

    max_length = args.max_length
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    use_cache = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dense = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    model_pruned = None
    if args.analysis_mode == "pruned":
        if not args.pruned_model_name:
            raise ValueError("--pruned_model_name is required when --analysis_mode=pruned")
        model_pruned = AutoModelForCausalLM.from_pretrained(args.pruned_model_name, trust_remote_code=True).to(device).eval()
        align_pruned_sublayers(model_dense, model_pruned)

    prompts = [
        "John has twice as many books as Mary. Together they have 18 books. How many books does John have?"
    ]

    for i, prompt in enumerate(prompts):
        if args.prompt_idx is not None and i != args.prompt_idx:
            continue

        print(f"prompt: {prompt}")
        write_log(f"\n=== Prompt {i+1} ===\n{prompt}\n")

        if args.analysis_mode == "pruned":
            handles, hook_state = attach_counterfactual_hooks(
                model_dense=model_dense,
                model_pruned=model_pruned,
                log_path=log_path,
                temperature=temperature,
                prompt_idx=i,
                drop_n=args.drop_n,
            )
            try:
                hook_state["prompt_idx"] = i
                output, _, _, _, _, _ = generate_with_custom_forward(
                    model=model_dense,
                    tokenizer=tokenizer,
                    device=device,
                    prompts=[prompt],
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=use_cache,
                    collect_sublayer=False,
                )
            finally:
                for h in handles:
                    h.remove()

            dense_text = output[0].replace(prompt, "")
            print("\nGenerated Output from Dense Model:")
            print(dense_text)

            write_log("--- Dense Model Output ---")
            write_log(dense_text + "\n")
            continue

        # dropped mode
        model_target = model_dense
        if args.drop_n > 0:
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

        output, probabilities, input_ids, _, _, sublayer_traces = generate_text_with_probabilities(
            model_obj=model_target,
            tokenizer_obj=tokenizer,
            device_obj=device,
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

        log_sublayer_metrics(
            model_obj=model_target,
            sublayer_traces=sublayer_traces,
            prompt_idx=i,
            temperature=temperature,
            top_k=top_k,
            log_path=log_path,
            label_prefix="dropped",
        )
