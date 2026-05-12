import argparse
import csv
import inspect
import json
import os
import sys
import tempfile
from typing import List

import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure local helper modules can be imported when launched from sibling scripts/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generation_forward_utils import apply_drop_masks


METRIC_KEYS = (
    'alpha',
    'para_abs',
    'perp_abs',
    'x_norm',
    'delta_norm',
    'delta_over_x',
    'para_over_x',
    'perp_over_x',
    'perp_ratio',
    'base_update_norm',
    'error_norm',
    'error_over_base_update',
    'para_over_base_update',
    'perp_over_base_update',
    'perp_ratio_to_base_update',
    'hidden_state_norm',
    'base_update_over_hidden_state',
)


def sanitize_tag(name: str) -> str:
    return name.replace('/', '__').replace(' ', '_')


def derive_model_tag(model_name: str) -> str:
    s = (model_name or '').rstrip('/').strip()
    if not s:
        return 'model'
    base = os.path.basename(s)
    # Avoid generic leaf names; use parent directory when possible.
    if base.lower() in {'checkpoint', 'checkpoints', 'model', 'models'}:
        parent = os.path.basename(os.path.dirname(s))
        if parent:
            base = parent
    return sanitize_tag(base or s)


def get_runtime_meta(args, quant_method):
    meta = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "model_name": args.model_name,
        "pruned_model_name": args.pruned_model_name,
        "analysis_mode": args.analysis_mode,
        "compare_mode": getattr(args, "compare_mode", "auto"),
        "compression_type": args.compression_type,
        "effect_scope": args.effect_scope,
        "focus_layer": args.focus_layer if args.effect_scope == "local" else None,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "quant_method_from_config": quant_method,
        "strict_quant_loading": bool(getattr(args, "strict_quant_loading", False)),
    }
    return meta


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
    attn_ref = [None] * num_layers
    mlp_ref = [None] * num_layers
    handles = []

    for li, layer in enumerate(model.model.layers):
        def _layer_pre_hook(_mod, args, kwargs, _li=li):
            h = kwargs.get('hidden_states', args[0] if args else None)
            if h is not None:
                attn_ref[_li] = h[:, -1, :].detach().float()
            return args, kwargs

        handles.append(layer.register_forward_pre_hook(_layer_pre_hook, with_kwargs=True))

        if hasattr(layer, 'self_attn') and layer.self_attn is not None:
            def _attn_hook(_mod, _args, out, _li=li):
                y = out[0] if isinstance(out, tuple) else out
                if y is not None:
                    attn_last[_li] = y[:, -1, :].detach().float()

            handles.append(layer.self_attn.register_forward_hook(_attn_hook))

        if hasattr(layer, 'post_attention_layernorm') and layer.post_attention_layernorm is not None:
            def _mlp_pre_hook(_mod, args, _li=li):
                h = args[0] if args else None
                if h is not None:
                    mlp_ref[_li] = h[:, -1, :].detach().float()

            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(_mlp_pre_hook))

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

    return attn_last, mlp_last, attn_ref, mlp_ref


def collect_local_counterfactual_last_token(model_dense, model_comp, input_ids, attention_mask, focus_layer):
    num_layers = len(model_dense.model.layers)
    attn_last = [None] * num_layers
    mlp_last = [None] * num_layers
    block_last = [None] * num_layers
    block_ref = [None] * num_layers
    attn_ref = [None] * num_layers
    mlp_ref = [None] * num_layers
    handles = []

    dense_layer = model_dense.model.layers[focus_layer]
    comp_layer = model_comp.model.layers[focus_layer]
    dense_model_forward_params = inspect.signature(model_dense.forward).parameters
    comp_layer_forward_params = inspect.signature(comp_layer.forward).parameters
    state = {'run': False, 'comp_out': None}
    comp_device = next(comp_layer.parameters()).device

    def layer_pre_hook(_mod, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden is None:
            return args, kwargs
        state['run'] = True
        block_ref[focus_layer] = hidden[:, -1, :].detach().float()
        attn_ref[focus_layer] = hidden[:, -1, :].detach().float()
        hidden_for_comp = hidden.to(comp_device) if hidden.device != comp_device else hidden

        def _to_comp_device(x):
            if x is None or not torch.is_tensor(x):
                return x
            return x.to(comp_device) if x.device != comp_device else x

        with torch.no_grad():
            comp_kwargs = {'hidden_states': hidden_for_comp}
            if 'attention_mask' in comp_layer_forward_params:
                comp_kwargs['attention_mask'] = _to_comp_device(kwargs.get('attention_mask', None))
            if 'position_ids' in comp_layer_forward_params:
                comp_kwargs['position_ids'] = _to_comp_device(kwargs.get('position_ids', None))
            if 'past_key_value' in comp_layer_forward_params:
                comp_kwargs['past_key_value'] = None
            if 'output_attentions' in comp_layer_forward_params:
                comp_kwargs['output_attentions'] = False
            if 'use_cache' in comp_layer_forward_params:
                comp_kwargs['use_cache'] = False
            if 'cache_position' in comp_layer_forward_params:
                comp_kwargs['cache_position'] = _to_comp_device(kwargs.get('cache_position', None))
            if 'position_embeddings' in comp_layer_forward_params:
                pe = kwargs.get('position_embeddings', None)
                if isinstance(pe, tuple):
                    comp_kwargs['position_embeddings'] = tuple(_to_comp_device(t) for t in pe)
                else:
                    comp_kwargs['position_embeddings'] = _to_comp_device(pe)
            comp_out = comp_layer(**comp_kwargs)
        state['comp_out'] = comp_out
        return args, kwargs

    handles.append(dense_layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

    if hasattr(comp_layer, 'self_attn') and comp_layer.self_attn is not None:
        def attn_hook(_mod, _args, out):
            y = out[0] if isinstance(out, tuple) else out
            if y is not None:
                attn_last[focus_layer] = y[:, -1, :].detach().float()

        handles.append(comp_layer.self_attn.register_forward_hook(attn_hook))

    if hasattr(comp_layer, 'mlp') and comp_layer.mlp is not None:
        def mlp_hook(_mod, _args, out):
            y = out[0] if isinstance(out, tuple) else out
            if y is not None:
                mlp_last[focus_layer] = y[:, -1, :].detach().float()

        handles.append(comp_layer.mlp.register_forward_hook(mlp_hook))

    if hasattr(dense_layer, 'post_attention_layernorm') and dense_layer.post_attention_layernorm is not None:
        def dense_mlp_pre_hook(_mod, args):
            h = args[0] if args else None
            if h is not None:
                mlp_ref[focus_layer] = h[:, -1, :].detach().float()

        handles.append(dense_layer.post_attention_layernorm.register_forward_pre_hook(dense_mlp_pre_hook))

    def block_hook_and_replace(_mod, _args, out):
        comp_out = state.get('comp_out', None)
        if comp_out is None:
            y = out[0] if isinstance(out, tuple) else out
            if y is not None:
                block_last[focus_layer] = y[:, -1, :].detach().float()
            return out

        # Replace dense layer output with comp layer output.
        if isinstance(out, tuple):
            base_y = out[0]
        else:
            base_y = out

        if isinstance(comp_out, tuple):
            comp_y = comp_out[0]
        else:
            comp_y = comp_out

        if base_y is not None and comp_y is not None and comp_y.device != base_y.device:
            comp_y = comp_y.to(base_y.device)

        if comp_y is not None:
            block_last[focus_layer] = comp_y[:, -1, :].detach().float()

        if isinstance(out, tuple):
            out_list = list(out)
            out_list[0] = comp_y
            return tuple(out_list)
        return comp_y

    handles.append(dense_layer.register_forward_hook(block_hook_and_replace))

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

    return block_last, attn_last, mlp_last, block_ref, attn_ref, mlp_ref


def compute_layer_metrics(h_base, h_comp, ref, state_ref=None, eps=1e-12):
    """Decompose comp-vs-baseline error with respect to the baseline update.

    ``ref`` is intentionally the dense/base update vector:
    - block_out: h_base[l + 1] - h_base[l]
    - attn_out/mlp_out: dense sublayer update

    ``state_ref`` is the layer/sublayer input hidden state used only for
    measuring ||base update|| / ||hidden state||.

    Older output columns keep the historical ``*_over_x`` names for backward
    compatibility. The new ``*_over_base_update`` aliases expose the update
    reference explicitly for paper plots.
    """
    if h_comp.device != h_base.device:
        h_comp = h_comp.to(h_base.device)
    if ref.device != h_base.device:
        ref = ref.to(h_base.device)
    if state_ref is not None and state_ref.device != h_base.device:
        state_ref = state_ref.to(h_base.device)
    delta = h_comp - h_base

    ref_norm_sq = (ref * ref).sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = ((delta * ref).sum(dim=-1, keepdim=True) / ref_norm_sq)
    delta_para = alpha * ref
    delta_perp = delta - delta_para

    para_abs = delta_para.norm(dim=-1)
    perp_abs = delta_perp.norm(dim=-1)
    x_norm = ref.norm(dim=-1).clamp_min(eps)
    hidden_state_norm = state_ref.norm(dim=-1).clamp_min(eps) if state_ref is not None else torch.full_like(x_norm, float('nan'))
    delta_norm = delta.norm(dim=-1)
    delta_over_x = delta_norm / x_norm
    para_over_x = para_abs / x_norm
    perp_over_x = perp_abs / x_norm
    delta_norm_sq = (delta * delta).sum(dim=-1).clamp_min(eps)
    perp_ratio = (delta_perp * delta_perp).sum(dim=-1) / delta_norm_sq

    return {
        'alpha': alpha.squeeze(-1),
        'para_abs': para_abs,
        'perp_abs': perp_abs,
        'x_norm': x_norm,
        'delta_norm': delta_norm,
        'delta_over_x': delta_over_x,
        'para_over_x': para_over_x,
        'perp_over_x': perp_over_x,
        'perp_ratio': perp_ratio,
        'base_update_norm': x_norm,
        'error_norm': delta_norm,
        'error_over_base_update': delta_over_x,
        'para_over_base_update': para_over_x,
        'perp_over_base_update': perp_over_x,
        'perp_ratio_to_base_update': perp_ratio,
        'hidden_state_norm': hidden_state_norm,
        'base_update_over_hidden_state': x_norm / hidden_state_norm,
    }


def init_agg(num_layers, device):
    agg = {'count': torch.zeros(num_layers, device=device)}
    for k in METRIC_KEYS:
        agg[k] = torch.zeros(num_layers, device=device)
        agg[f'{k}_sq'] = torch.zeros(num_layers, device=device)
    return agg


def accumulate_component(agg, base_list, comp_list, ref_list, state_ref_list, n_layers):
    for li in range(n_layers):
        if base_list[li] is None or comp_list[li] is None or ref_list[li] is None:
            continue
        state_ref = None if state_ref_list is None else state_ref_list[li]
        m = compute_layer_metrics(base_list[li], comp_list[li], ref_list[li], state_ref)
        for k in METRIC_KEYS:
            v = m[k].mean()
            agg[k][li] += v
            agg[f'{k}_sq'][li] += v * v
        agg['count'][li] += 1.0


def finalize_agg(agg, n):
    # n kept for backward compatibility; per-layer count is tracked in agg['count'].
    out = {}
    cnt = agg['count'].clamp_min(1.0)
    for k in METRIC_KEYS:
        mean = agg[k] / cnt
        sq_mean = agg[f'{k}_sq'] / cnt
        var = (sq_mean - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var)
        out[k] = mean.detach().cpu().tolist()
        out[f'{k}_std'] = std.detach().cpu().tolist()
    out['count'] = agg['count'].detach().cpu().tolist()
    return out


def load_quantized_model(model_name, args):
    compute_dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    # Detect quantization method declared in model config (if present).
    quant_method = None
    if os.path.isdir(model_name):
        cfg_path = os.path.join(model_name, 'config.json')
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg_json = json.load(f)
                qcfg = cfg_json.get('quantization_config', {})
                if isinstance(qcfg, dict):
                    quant_method = qcfg.get('quant_method')
            except Exception:
                pass

    # AWQ models should be loaded via AWQ path, not BitsAndBytes 4bit args.
    if isinstance(quant_method, str) and quant_method.lower() == 'awq':
        print('[INFO] Detected AWQ model config; loading via AWQ/native path (skip BitsAndBytes args).')
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=compute_dtype_map[args.bnb_4bit_compute_dtype],
        ).eval()
        return m, "awq_native", quant_method

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype_map[args.bnb_4bit_compute_dtype],
    )
    try:
        # Newer Transformers + BitsAndBytes path.
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto',
            quantization_config=quant_cfg,
        ).eval()
        return m, "bnb_quant_config", quant_method
    except (AttributeError, TypeError, ValueError) as e:
        # Older Transformers fallback: pass legacy kwargs directly.
        if 'get_loading_attributes' not in str(e):
            raise
        print('[WARN] Falling back to legacy 4bit/8bit loading args due to Transformers/BnB version mismatch.')
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map='auto',
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype_map[args.bnb_4bit_compute_dtype],
            ).eval()
            return m, "bnb_legacy_args", quant_method
        except (AttributeError, TypeError, ValueError) as e2:
            if 'get_loading_attributes' not in str(e2):
                raise
            print('[WARN] Legacy quantization args also failed; loading model without quantization config.')
            if getattr(args, "strict_quant_loading", False):
                raise RuntimeError(
                    "Strict quant loading enabled: both BitsAndBytes quant loading paths failed. "
                    "Refusing fallback to non-quantized loading."
                ) from e2
            # Last-resort compatibility path: strip quantization_config from config.json
            # and load from a temporary local model copy.
            patched_model_name = model_name
            if os.path.isdir(model_name):
                cfg_path = os.path.join(model_name, 'config.json')
                if os.path.isfile(cfg_path):
                    with open(cfg_path, 'r', encoding='utf-8') as f:
                        cfg_json = json.load(f)
                    if 'quantization_config' in cfg_json:
                        cfg_json.pop('quantization_config', None)
                        tmp_dir = tempfile.mkdtemp(prefix='hf_model_no_quant_cfg_')
                        for fn in os.listdir(model_name):
                            src = os.path.join(model_name, fn)
                            dst = os.path.join(tmp_dir, fn)
                            if fn == 'config.json':
                                continue
                            try:
                                os.symlink(src, dst)
                            except Exception:
                                if os.path.isdir(src):
                                    # Keep fallback simple: only symlink files; skip dirs.
                                    continue
                                import shutil
                                shutil.copy2(src, dst)
                        with open(os.path.join(tmp_dir, 'config.json'), 'w', encoding='utf-8') as f:
                            json.dump(cfg_json, f, ensure_ascii=False, indent=2)
                        patched_model_name = tmp_dir
                        print(f'[WARN] Using patched temp model dir without quantization_config: {patched_model_name}')

            cfg = AutoConfig.from_pretrained(patched_model_name, trust_remote_code=True)
            if hasattr(cfg, 'quantization_config'):
                try:
                    delattr(cfg, 'quantization_config')
                except Exception:
                    pass
            cfg.__dict__.pop('quantization_config', None)
            m = AutoModelForCausalLM.from_pretrained(
                patched_model_name,
                trust_remote_code=True,
                device_map='auto',
                config=cfg,
            ).eval()
            return m, "fallback_non_quantized", quant_method


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
    parser.add_argument(
        '--compare_mode',
        type=str,
        default='auto',
        choices=['auto', 'dual_model', 'single_model_intervene'],
        help='dual_model: dense vs comp model; single_model_intervene: dense-only intervention (drop/ablation). auto infers from analysis_mode.',
    )
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
    parser.add_argument('--output_dir', type=str, default='compression/outputs/layerwise_para_perp')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', choices=['fp4', 'nf4'])
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--strict_quant_loading', action='store_true', help='Fail if quant model cannot be loaded via quant path.')
    args = parser.parse_args()
    if args.effect_scope == 'local' and args.focus_layer < 0:
        raise ValueError('--focus_layer is required when --effect_scope=local')

    # Backward-compatible mode normalization:
    # - old "pruned" => dual_model
    # - old "dropped" => single_model_intervene
    inferred_compare_mode = 'dual_model' if args.analysis_mode == 'pruned' else 'single_model_intervene'
    if args.compare_mode == 'auto':
        args.compare_mode = inferred_compare_mode

    if args.compare_mode == 'dual_model':
        if not args.pruned_model_name:
            raise ValueError('--pruned_model_name is required when --compare_mode=dual_model')
    elif args.compare_mode == 'single_model_intervene':
        if args.pruned_model_name:
            raise ValueError('--pruned_model_name must be empty when --compare_mode=single_model_intervene')

    model_tag = args.model_tag or derive_model_tag(args.model_name)
    run_tag = f"{model_tag}__{args.method_name}"
    out_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dense = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    quant_load_path = None
    quant_method = None
    if args.compare_mode == 'dual_model':
        if args.compression_type == 'quant':
            is_awq = False
            if os.path.isdir(args.pruned_model_name):
                cfg_path = os.path.join(args.pruned_model_name, 'config.json')
                if os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            cfg_json = json.load(f)
                        qcfg = cfg_json.get('quantization_config', {})
                        is_awq = isinstance(qcfg, dict) and str(qcfg.get('quant_method', '')).lower() == 'awq'
                    except Exception:
                        is_awq = False

            if not is_awq:
                if args.load_in_4bit and args.load_in_8bit:
                    raise ValueError('Use only one of --load_in_4bit / --load_in_8bit')
                if not args.load_in_4bit and not args.load_in_8bit:
                    raise ValueError('For quant mode, specify --load_in_4bit or --load_in_8bit')
            model_comp, quant_load_path, quant_method = load_quantized_model(args.pruned_model_name, args)
        else:
            model_comp = AutoModelForCausalLM.from_pretrained(args.pruned_model_name, trust_remote_code=True).to(device).eval()
    else:
        model_comp = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
        if args.drop_n > 0:
            # Default: use dense model directly for layer-drop baseline.
            # Optional: if dropped_root_path points to a prepared checkpoint/config,
            # load drop lists from config.json.
            cfg = {}
            dropped_root = (args.dropped_root_path or "").strip()
            if dropped_root and dropped_root != "-":
                if args.target_layer in ['attn', 'mlp']:
                    dropped_model_path = (
                        f"{dropped_root}/{model_tag}-layer_drop_{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
                    )
                else:
                    dropped_model_path = (
                        f"{dropped_root}/block_drop/{model_tag}-block_drop-{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
                    )
                cfg_path = os.path.join(dropped_model_path, 'config.json')
                if os.path.isfile(cfg_path):
                    with open(cfg_path, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                else:
                    print(f"[WARN] dropped config not found at {cfg_path}; fallback to dense-only drop_n mode.")
            drop_attn_list = cfg.get('drop_attn_list', [])
            drop_mlp_list = cfg.get('drop_mlp_list', [])
            if not drop_attn_list and not drop_mlp_list and args.drop_n <= 0:
                raise ValueError(
                    "single_model_intervene requested, but no effective drop configuration was found. "
                    "Provide non-empty drop_attn_list/drop_mlp_list or set --drop_n > 0."
                )
            print(
                f"[INFO] single_model_intervene masks: "
                f"target_layer={args.target_layer} drop_n={args.drop_n} "
                f"drop_attn_list={drop_attn_list} drop_mlp_list={drop_mlp_list}"
            )
            apply_drop_masks(
                model=model_comp,
                target_layer=args.target_layer,
                drop_attn_list=drop_attn_list,
                drop_mlp_list=drop_mlp_list,
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
        attn_dense, mlp_dense, attn_ref_dense, mlp_ref_dense = collect_sublayer_last_token(model_dense, input_ids, attention_mask)
        block_ref_dense = hs_dense[:-1]
        block_update_ref_dense = []
        for li in range(num_layers):
            if hs_dense[li] is None or hs_dense[li + 1] is None:
                block_update_ref_dense.append(None)
            else:
                block_update_ref_dense.append(hs_dense[li + 1] - hs_dense[li])

        if args.effect_scope == 'global':
            hs_comp = get_hidden_last_token(model_comp, input_ids, attention_mask)
            attn_comp, mlp_comp, _, _ = collect_sublayer_last_token(model_comp, input_ids, attention_mask)
            # block output starts at hidden_states[1], [0] is embedding output
            accumulate_component(agg_block, hs_dense[1:], hs_comp[1:], block_update_ref_dense, block_ref_dense, num_layers)
            accumulate_component(agg_attn, attn_dense, attn_comp, attn_dense, attn_ref_dense, num_layers)
            accumulate_component(agg_mlp, mlp_dense, mlp_comp, mlp_dense, mlp_ref_dense, num_layers)
        else:
            block_cf, attn_cf, mlp_cf, block_ref_cf, attn_ref_cf, mlp_ref_cf = collect_local_counterfactual_last_token(
                model_dense=model_dense,
                model_comp=model_comp,
                input_ids=input_ids,
                attention_mask=attention_mask,
                focus_layer=args.focus_layer,
            )
            block_update_ref_local = [None] * num_layers
            attn_update_ref_local = [None] * num_layers
            mlp_update_ref_local = [None] * num_layers
            li = args.focus_layer
            if block_ref_cf[li] is not None and hs_dense[li + 1] is not None:
                block_update_ref_local[li] = hs_dense[li + 1] - block_ref_cf[li]
            if attn_dense[li] is not None:
                attn_update_ref_local[li] = attn_dense[li]
            if mlp_dense[li] is not None:
                mlp_update_ref_local[li] = mlp_dense[li]
            accumulate_component(agg_block, hs_dense[1:], block_cf, block_update_ref_local, block_ref_cf, num_layers)
            accumulate_component(agg_attn, attn_dense, attn_cf, attn_update_ref_local, attn_ref_cf, num_layers)
            accumulate_component(agg_mlp, mlp_dense, mlp_cf, mlp_update_ref_local, mlp_ref_cf, num_layers)

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
        writer.writerow([
            'method', 'component', 'layer',
            'alpha', 'alpha_std',
            'para_abs', 'para_abs_std',
            'perp_abs', 'perp_abs_std',
            'x_norm', 'x_norm_std',
            'delta_norm', 'delta_norm_std',
            'delta_over_x', 'delta_over_x_std',
            'para_over_x', 'para_over_x_std',
            'perp_over_x', 'perp_over_x_std',
            'perp_ratio', 'perp_ratio_std',
            'base_update_norm', 'base_update_norm_std',
            'error_norm', 'error_norm_std',
            'error_over_base_update', 'error_over_base_update_std',
            'para_over_base_update', 'para_over_base_update_std',
            'perp_over_base_update', 'perp_over_base_update_std',
            'perp_ratio_to_base_update', 'perp_ratio_to_base_update_std',
            'hidden_state_norm', 'hidden_state_norm_std',
            'base_update_over_hidden_state', 'base_update_over_hidden_state_std',
            'count',
        ])
        components = [('block_out', block), ('attn_out', attn), ('mlp_out', mlp)]
        csv_layers = layer_ids
        if args.effect_scope == 'local':
            csv_layers = [args.focus_layer]
        for comp_name, comp in components:
            for li in csv_layers:
                writer.writerow([
                    args.method_name,
                    comp_name,
                    li,
                    comp['alpha'][li],
                    comp['alpha_std'][li],
                    comp['para_abs'][li],
                    comp['para_abs_std'][li],
                    comp['perp_abs'][li],
                    comp['perp_abs_std'][li],
                    comp['x_norm'][li],
                    comp['x_norm_std'][li],
                    comp['delta_norm'][li],
                    comp['delta_norm_std'][li],
                    comp['delta_over_x'][li],
                    comp['delta_over_x_std'][li],
                    comp['para_over_x'][li],
                    comp['para_over_x_std'][li],
                    comp['perp_over_x'][li],
                    comp['perp_over_x_std'][li],
                    comp['perp_ratio'][li],
                    comp['perp_ratio_std'][li],
                    comp['base_update_norm'][li],
                    comp['base_update_norm_std'][li],
                    comp['error_norm'][li],
                    comp['error_norm_std'][li],
                    comp['error_over_base_update'][li],
                    comp['error_over_base_update_std'][li],
                    comp['para_over_base_update'][li],
                    comp['para_over_base_update_std'][li],
                    comp['perp_over_base_update'][li],
                    comp['perp_over_base_update_std'][li],
                    comp['perp_ratio_to_base_update'][li],
                    comp['perp_ratio_to_base_update_std'][li],
                    comp['hidden_state_norm'][li],
                    comp['hidden_state_norm_std'][li],
                    comp['base_update_over_hidden_state'][li],
                    comp['base_update_over_hidden_state_std'][li],
                    comp['count'][li],
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
                    'mean_std_perp_ratio': float(sum(block['perp_ratio_std']) / len(block['perp_ratio_std'])),
                },
                'attn_out': {
                    'mean_perp_ratio': float(sum(attn['perp_ratio']) / len(attn['perp_ratio'])),
                    'std_perp_ratio': float(torch.tensor(attn['perp_ratio']).std(unbiased=False).item()),
                    'mean_std_perp_ratio': float(sum(attn['perp_ratio_std']) / len(attn['perp_ratio_std'])),
                },
                'mlp_out': {
                    'mean_perp_ratio': float(sum(mlp['perp_ratio']) / len(mlp['perp_ratio'])),
                    'std_perp_ratio': float(torch.tensor(mlp['perp_ratio']).std(unbiased=False).item()),
                    'mean_std_perp_ratio': float(sum(mlp['perp_ratio_std']) / len(mlp['perp_ratio_std'])),
                },
            },
            f,
            indent=2,
        )

    meta = get_runtime_meta(args, quant_method)
    meta["quant_load_path"] = quant_load_path
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Local flip runs are per-layer sparse signals (mostly zeros elsewhere),
    # so per-run full-layer plots are not informative. Keep CSV/JSON only.
    if args.effect_scope == 'global':
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
