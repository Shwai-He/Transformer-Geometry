import argparse
import csv
import inspect
import json
import os
from typing import List

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generation_forward_utils import apply_drop_masks


def sanitize_tag(name: str) -> str:
    return name.replace('/', '__').replace(' ', '_')


def read_prompts(args) -> List[str]:
    if args.prompt:
        return [args.prompt]
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
        if prompts:
            return prompts[: args.max_prompts] if args.max_prompts > 0 else prompts
    return [
        'John has twice as many books as Mary. Together they have 18 books. How many books does John have?',
        'Which animal is a mammal? Choose the correct answer: a) Snake b) Frog c) Dog d) Lizard',
    ]


def get_hidden_last_token(model, input_ids, attention_mask):
    forward_params = inspect.signature(model.forward).parameters
    kwargs = {
        'input_ids': input_ids,
        'output_hidden_states': True,
        'use_cache': False,
        'return_dict': True,
    }
    if 'attention_mask' in forward_params:
        kwargs['attention_mask'] = attention_mask
    with torch.no_grad():
        out = model(**kwargs)
    # hidden_states: embedding + each layer output
    return [h[:, -1, :].detach().float() for h in out.hidden_states]


def collect_sublayer_last_token(model, input_ids, attention_mask):
    num_layers = len(model.model.layers)
    attn_last = [None] * num_layers
    mlp_last = [None] * num_layers
    handles = []

    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn') and layer.self_attn is not None:
            def _attn_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                if y is not None:
                    attn_last[_li] = y[:, -1, :].detach().float()

            handles.append(layer.self_attn.register_forward_hook(_attn_hook))

        if hasattr(layer, 'mlp') and layer.mlp is not None:
            def _mlp_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                if y is not None:
                    mlp_last[_li] = y[:, -1, :].detach().float()

            handles.append(layer.mlp.register_forward_hook(_mlp_hook))

    forward_params = inspect.signature(model.forward).parameters
    kwargs = {
        'input_ids': input_ids,
        'output_hidden_states': False,
        'use_cache': False,
        'return_dict': True,
    }
    if 'attention_mask' in forward_params:
        kwargs['attention_mask'] = attention_mask
    with torch.no_grad():
        model(**kwargs)

    for h in handles:
        h.remove()

    return attn_last, mlp_last


def collect_local_counterfactual_last_token(model_dense, model_comp, input_ids, attention_mask, focus_layer):
    num_layers = len(model_dense.model.layers)
    attn_last = [None] * num_layers
    mlp_last = [None] * num_layers
    block_last = [None] * num_layers
    handles = []

    dense_layer = model_dense.model.layers[focus_layer]
    comp_layer = model_comp.model.layers[focus_layer]
    state = {'run': False}

    def layer_pre_hook(_mod, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden is None:
            return args, kwargs
        with torch.no_grad():
            comp_kwargs = {'hidden_states': hidden}
            if 'attention_mask' in comp_layer_forward_params:
                comp_kwargs['attention_mask'] = kwargs.get('attention_mask', None)
            if 'position_ids' in comp_layer_forward_params:
                comp_kwargs['position_ids'] = kwargs.get('position_ids', None)
            if 'past_key_value' in comp_layer_forward_params:
                comp_kwargs['past_key_value'] = None
            if 'output_attentions' in comp_layer_forward_params:
                comp_kwargs['output_attentions'] = False
            if 'use_cache' in comp_layer_forward_params:
                comp_kwargs['use_cache'] = False
            if 'cache_position' in comp_layer_forward_params:
                comp_kwargs['cache_position'] = kwargs.get('cache_position', None)
            if 'position_embeddings' in comp_layer_forward_params:
                comp_kwargs['position_embeddings'] = kwargs.get('position_embeddings', None)
            comp_out = comp_layer(**comp_kwargs)
        dense_out = comp_out[0] if isinstance(comp_out, tuple) else comp_out
        kwargs['hidden_states'] = dense_out
        state['run'] = True
        return args, kwargs

    handles.append(dense_layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

    if hasattr(comp_layer, 'self_attn') and comp_layer.self_attn is not None:
        def attn_hook(_mod, _args, out):
            if not state['run']:
                return
            y = out[0] if isinstance(out, tuple) else out
            if y is not None:
                attn_last[focus_layer] = y[:, -1, :].detach().float()

        handles.append(comp_layer.self_attn.register_forward_hook(attn_hook))

    if hasattr(comp_layer, 'mlp') and comp_layer.mlp is not None:
        def mlp_hook(_mod, _args, out):
            if not state['run']:
                return
            y = out[0] if isinstance(out, tuple) else out
            if y is not None:
                mlp_last[focus_layer] = y[:, -1, :].detach().float()

        handles.append(comp_layer.mlp.register_forward_hook(mlp_hook))

    def block_hook(_mod, _args, out):
        y = out[0] if isinstance(out, tuple) else out
        if y is not None:
            block_last[focus_layer] = y[:, -1, :].detach().float()

    handles.append(dense_layer.register_forward_hook(block_hook))

    with torch.no_grad():
        dense_kwargs = {
            'input_ids': input_ids,
            'output_hidden_states': False,
            'use_cache': False,
            'return_dict': True,
        }
        if 'attention_mask' in dense_model_forward_params:
            dense_kwargs['attention_mask'] = attention_mask
        model_dense(**dense_kwargs)

    for h in handles:
        h.remove()

    return block_last, attn_last, mlp_last


def compute_layer_metrics(h_dense, h_comp, eps=1e-12):
    # h_* is layer output for same layer index (skip embedding slot outside)
    delta = h_comp - h_dense

    x = h_dense
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = ((delta * x).sum(dim=-1, keepdim=True) / x_norm_sq)
    delta_para = alpha * x
    delta_perp = delta - delta_para

    para_abs = delta_para.norm(dim=-1)
    perp_abs = delta_perp.norm(dim=-1)
    delta_norm_sq = (delta * delta).sum(dim=-1).clamp_min(eps)
    perp_ratio = (delta_perp * delta_perp).sum(dim=-1) / delta_norm_sq

    return {
        'alpha': alpha.squeeze(-1),
        'para_abs': para_abs,
        'perp_abs': perp_abs,
        'perp_ratio': perp_ratio,
    }


def init_agg(num_layers, device):
    return {
        'alpha': torch.zeros(num_layers, device=device),
        'para_abs': torch.zeros(num_layers, device=device),
        'perp_abs': torch.zeros(num_layers, device=device),
        'perp_ratio': torch.zeros(num_layers, device=device),
    }


def accumulate_component(agg, dense_list, comp_list, n_layers):
    for li in range(n_layers):
        if dense_list[li] is None or comp_list[li] is None:
            continue
        m = compute_layer_metrics(dense_list[li], comp_list[li])
        agg['alpha'][li] += m['alpha'].mean()
        agg['para_abs'][li] += m['para_abs'].mean()
        agg['perp_abs'][li] += m['perp_abs'].mean()
        agg['perp_ratio'][li] += m['perp_ratio'].mean()


def finalize_agg(agg, n):
    out = {}
    for k in agg:
        out[k] = (agg[k] / max(n, 1)).detach().cpu().tolist()
    return out


def load_quantized_model(model_name, args):
    compute_dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype_map[args.bnb_4bit_compute_dtype],
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map='auto',
        quantization_config=quant_cfg,
    ).eval()


def save_plot(layer_ids, ys_by_method, ylabel, title, save_path):
    plt.figure(figsize=(9, 4.5))
    for method_name, ys in ys_by_method.items():
        plt.plot(layer_ids, ys, marker='o', linewidth=1.3, markersize=3, label=method_name)
    plt.xlabel('Layer')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_heatmap(method_names, layer_ids, values_2d, title, save_path):
    # values_2d shape: [num_methods, num_layers]
    plt.figure(figsize=(max(8, len(layer_ids) * 0.25), max(2.5, len(method_names) * 0.5)))
    plt.imshow(values_2d, aspect='auto')
    plt.colorbar(label='perp_ratio')
    plt.yticks(range(len(method_names)), method_names)
    plt.xticks(range(len(layer_ids)), layer_ids, rotation=0)
    plt.xlabel('Layer')
    plt.ylabel('Method')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Layer-wise parallel/perpendicular decomposition across compression methods.')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--analysis_mode', type=str, choices=['dropped', 'pruned'], required=True)
    parser.add_argument('--compression_type', type=str, default='prune', choices=['prune', 'quant'])
    parser.add_argument('--effect_scope', type=str, default='global', choices=['global', 'local'])
    parser.add_argument('--focus_layer', type=int, default=-1, help='Required when effect_scope=local')
    parser.add_argument('--method_name', type=str, required=True, help='Label used in output plots/csv, e.g. wanda_50 or attn_drop_8')
    parser.add_argument('--model_tag', type=str, default=None)
    parser.add_argument('--pruned_model_name', type=str, default=None)
    parser.add_argument('--dropped_root_path', type=str, default='you_dropped_root_path')
    parser.add_argument('--target_layer', type=str, default='attn', choices=['attn', 'mlp', 'all'])
    parser.add_argument('--drop_n', type=int, default=0)
    parser.add_argument('--prompts_file', type=str, default='')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--max_prompts', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='representation-analysis/outputs/layerwise_para_perp')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', choices=['fp4', 'nf4'])
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    args = parser.parse_args()
    if args.effect_scope == 'local' and args.focus_layer < 0:
        raise ValueError('--focus_layer is required when --effect_scope=local')

    model_tag = args.model_tag or sanitize_tag(args.model_name)
    run_tag = f"{model_tag}__{args.method_name}"
    out_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dense = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    if args.analysis_mode == 'pruned':
        if not args.pruned_model_name:
            raise ValueError('--pruned_model_name is required when --analysis_mode=pruned')
        if args.compression_type == 'quant':
            if args.load_in_4bit and args.load_in_8bit:
                raise ValueError('Use only one of --load_in_4bit / --load_in_8bit')
            if not args.load_in_4bit and not args.load_in_8bit:
                raise ValueError('For quant mode, specify --load_in_4bit or --load_in_8bit')
            model_comp = load_quantized_model(args.pruned_model_name, args)
        else:
            model_comp = AutoModelForCausalLM.from_pretrained(args.pruned_model_name, trust_remote_code=True).to(device).eval()
    else:
        model_comp = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
        if args.drop_n > 0:
            if args.target_layer in ['attn', 'mlp']:
                dropped_model_path = (
                    f"{args.dropped_root_path}/{model_tag}-layer_drop_{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
                )
            else:
                dropped_model_path = (
                    f"{args.dropped_root_path}/block_drop/{model_tag}-block_drop-{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
                )
            cfg_path = os.path.join(dropped_model_path, 'config.json')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            apply_drop_masks(
                model=model_comp,
                target_layer=args.target_layer,
                drop_attn_list=cfg.get('drop_attn_list', []),
                drop_mlp_list=cfg.get('drop_mlp_list', []),
                drop_n=args.drop_n,
            )

    prompts = read_prompts(args)

    num_layers = len(model_dense.model.layers)
    if args.effect_scope == 'local' and args.focus_layer >= num_layers:
        raise ValueError(f'focus_layer out of range: {args.focus_layer} >= {num_layers}')
    agg_block = init_agg(num_layers, device)
    agg_attn = init_agg(num_layers, device)
    agg_mlp = init_agg(num_layers, device)
    n = 0

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=args.max_length)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        hs_dense = get_hidden_last_token(model_dense, input_ids, attention_mask)
        attn_dense, mlp_dense = collect_sublayer_last_token(model_dense, input_ids, attention_mask)

        if args.effect_scope == 'global':
            hs_comp = get_hidden_last_token(model_comp, input_ids, attention_mask)
            attn_comp, mlp_comp = collect_sublayer_last_token(model_comp, input_ids, attention_mask)
            # block output starts at hidden_states[1], [0] is embedding output
            accumulate_component(agg_block, hs_dense[1:], hs_comp[1:], num_layers)
            accumulate_component(agg_attn, attn_dense, attn_comp, num_layers)
            accumulate_component(agg_mlp, mlp_dense, mlp_comp, num_layers)
        else:
            block_cf, attn_cf, mlp_cf = collect_local_counterfactual_last_token(
                model_dense=model_dense,
                model_comp=model_comp,
                input_ids=input_ids,
                attention_mask=attention_mask,
                focus_layer=args.focus_layer,
            )
            accumulate_component(agg_block, hs_dense[1:], block_cf, num_layers)
            accumulate_component(agg_attn, attn_dense, attn_cf, num_layers)
            accumulate_component(agg_mlp, mlp_dense, mlp_cf, num_layers)

        n += 1

    if n == 0:
        raise RuntimeError('No prompts available.')

    block = finalize_agg(agg_block, n)
    attn = finalize_agg(agg_attn, n)
    mlp = finalize_agg(agg_mlp, n)

    layer_ids = list(range(num_layers))

    csv_path = os.path.join(out_dir, 'layerwise_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'component', 'layer', 'alpha', 'para_abs', 'perp_abs', 'perp_ratio'])
        components = [('block_out', block), ('attn_out', attn), ('mlp_out', mlp)]
        for comp_name, comp in components:
            for li in layer_ids:
                writer.writerow([
                    args.method_name,
                    comp_name,
                    li,
                    comp['alpha'][li],
                    comp['para_abs'][li],
                    comp['perp_abs'][li],
                    comp['perp_ratio'][li],
                ])

    json_path = os.path.join(out_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'method': args.method_name,
                'analysis_mode': args.analysis_mode,
                'compression_type': args.compression_type,
                'effect_scope': args.effect_scope,
                'focus_layer': args.focus_layer if args.effect_scope == 'local' else None,
                'model_name': args.model_name,
                'num_prompts': n,
                'block_out': {
                    'mean_perp_ratio': float(sum(block['perp_ratio']) / len(block['perp_ratio'])),
                    'std_perp_ratio': float(torch.tensor(block['perp_ratio']).std(unbiased=False).item()),
                },
                'attn_out': {
                    'mean_perp_ratio': float(sum(attn['perp_ratio']) / len(attn['perp_ratio'])),
                    'std_perp_ratio': float(torch.tensor(attn['perp_ratio']).std(unbiased=False).item()),
                },
                'mlp_out': {
                    'mean_perp_ratio': float(sum(mlp['perp_ratio']) / len(mlp['perp_ratio'])),
                    'std_perp_ratio': float(torch.tensor(mlp['perp_ratio']).std(unbiased=False).item()),
                },
            },
            f,
            indent=2,
        )

    save_plot(
        layer_ids,
        {
            f'{args.method_name}:block': block['perp_ratio'],
            f'{args.method_name}:attn': attn['perp_ratio'],
            f'{args.method_name}:mlp': mlp['perp_ratio'],
        },
        ylabel='perp_ratio',
        title='Layer-wise Perpendicular Ratio (block/attn/mlp)',
        save_path=os.path.join(out_dir, 'fig_a_perp_ratio.png'),
    )

    save_plot(
        layer_ids,
        {
            f'{args.method_name}:block': block['alpha'],
            f'{args.method_name}:attn': attn['alpha'],
            f'{args.method_name}:mlp': mlp['alpha'],
        },
        ylabel='alpha',
        title='Layer-wise Signed Parallel Coefficient (block/attn/mlp)',
        save_path=os.path.join(out_dir, 'fig_b_alpha.png'),
    )

    save_heatmap(
        [f'{args.method_name}:block', f'{args.method_name}:attn', f'{args.method_name}:mlp'],
        layer_ids,
        [block['perp_ratio'], attn['perp_ratio'], mlp['perp_ratio']],
        title='Component x Layer Perpendicular Ratio',
        save_path=os.path.join(out_dir, 'fig_c_heatmap.png'),
    )

    print('Finished.')
    print(f'Output dir: {os.path.abspath(out_dir)}')
    print(f'CSV: {os.path.abspath(csv_path)}')


if __name__ == '__main__':
    main()
    dense_forward_params = inspect.signature(model_dense.model.layers[focus_layer].forward).parameters
    dense_model_forward_params = inspect.signature(model_dense.forward).parameters
    comp_layer_forward_params = inspect.signature(comp_layer.forward).parameters
