from __future__ import annotations

import importlib
import logging
import math
import types
from typing import List, Optional

import torch


eval_logger = logging.getLogger(__name__)


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return list(layers)
    raise ValueError(
        "Could not find decoder layers. Expected model.layers, transformer.h, or gpt_neox.layers."
    )


def _repeat_kv_fallback(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _resolve_repeat_kv(module):
    mod = importlib.import_module(type(module).__module__)
    return getattr(mod, "repeat_kv", _repeat_kv_fallback)


def _resolve_apply_rotary_pos_emb(module):
    mod = importlib.import_module(type(module).__module__)
    fn = getattr(mod, "apply_rotary_pos_emb", None)
    if fn is None:
        raise ValueError(
            f"Could not resolve apply_rotary_pos_emb from module {type(module).__module__}"
        )
    return fn


def _diag_indices_from_cache_position(
    q_len: int, k_len: int, cache_position: Optional[torch.Tensor]
):
    if cache_position is None:
        n = min(q_len, k_len)
        row_idx = torch.arange(n)
        col_idx = torch.arange(n)
        return row_idx, col_idx

    pos = cache_position
    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)
    pos = pos.detach().to(dtype=torch.long).reshape(-1)
    if pos.numel() == 1 and q_len > 1:
        pos = pos.expand(q_len)
    if pos.numel() != q_len:
        return None, None
    row_idx = torch.arange(q_len, device=pos.device, dtype=torch.long)
    valid = (pos >= 0) & (pos < k_len)
    if not bool(valid.any()):
        return None, None
    return row_idx[valid], pos[valid]


def _apply_diag_mode(
    attn_weights: torch.Tensor,
    *,
    mode: str,
    keep_first: bool,
    cache_position: Optional[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    mode_norm = str(mode).lower().strip()
    if mode_norm == "none":
        return attn_weights

    if attn_weights.dim() != 4:
        raise ValueError(
            f"Expected attn_weights to be rank-4 [batch, heads, q, k], got {tuple(attn_weights.shape)}"
        )

    q_len = attn_weights.size(-2)
    k_len = attn_weights.size(-1)
    row_idx, col_idx = _diag_indices_from_cache_position(q_len, k_len, cache_position)
    if row_idx is None or col_idx is None or row_idx.numel() == 0:
        return attn_weights

    row_idx = row_idx.to(device=attn_weights.device)
    col_idx = col_idx.to(device=attn_weights.device)

    if keep_first:
        keep_mask = col_idx == 0
        row_idx = row_idx[~keep_mask]
        col_idx = col_idx[~keep_mask]
        if row_idx.numel() == 0:
            return attn_weights

    out = attn_weights.clone()
    out[..., row_idx, col_idx] = 0.0

    if mode_norm == "zero_no_renorm":
        return out
    if mode_norm == "zero_renorm":
        denom = out.sum(dim=-1, keepdim=True).clamp_min(eps)
        return out / denom
    raise ValueError(
        f"attn_diag_mode must be one of none/zero_no_renorm/zero_renorm, got {mode}"
    )


class AttentionDiagonalHooks:
    def __init__(
        self,
        model,
        *,
        mode: str = "zero_renorm",
        keep_first: bool = True,
        start_layer: int = 0,
        end_layer: int = -1,
        skip_first_n: int = 0,
        skip_last_n: int = 0,
    ):
        self.model = model
        self.mode = str(mode).lower().strip()
        self.keep_first = bool(keep_first)
        self.layers = _find_decoder_layers(model)
        n_layers = len(self.layers)
        lo = max(0, int(start_layer), int(skip_first_n))
        hi = n_layers if int(end_layer) < 0 else min(n_layers, int(end_layer))
        hi = min(hi, max(0, n_layers - int(skip_last_n)))
        if hi < lo:
            hi = lo
        self.active_layer_indices = set(range(lo, hi))
        self.patched_modules = []

    def _patch_qwen_style_attention(self, module):
        repeat_kv = _resolve_repeat_kv(module)
        apply_rotary_pos_emb = _resolve_apply_rotary_pos_emb(module)
        orig_forward = module.forward

        def patched_forward(
            attn_self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_value=None,
            cache_position: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_self.head_dim)

            q_proj = attn_self.q_proj(hidden_states).view(hidden_shape)
            k_proj = attn_self.k_proj(hidden_states).view(hidden_shape)
            if hasattr(attn_self, "q_norm"):
                q_proj = attn_self.q_norm(q_proj)
            if hasattr(attn_self, "k_norm"):
                k_proj = attn_self.k_norm(k_proj)

            query_states = q_proj.transpose(1, 2)
            key_states = k_proj.transpose(1, 2)
            value_states = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            if position_embeddings is None:
                if not hasattr(attn_self, "rotary_emb"):
                    return orig_forward(
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_value=past_key_value,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        **kwargs,
                    )
                position_embeddings = attn_self.rotary_emb(hidden_states, position_ids)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn_self.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, attn_self.num_key_value_groups)
            value_states = repeat_kv(value_states, attn_self.num_key_value_groups)

            scaling = getattr(
                attn_self, "scaling", float(attn_self.head_dim) ** -0.5
            )
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) * scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = _apply_diag_mode(
                attn_weights,
                mode=self.mode,
                keep_first=self.keep_first,
                cache_position=cache_position,
            )
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=0.0 if not attn_self.training else attn_self.attention_dropout,
                training=attn_self.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_self.o_proj(attn_output)
            return attn_output, attn_weights

        module.forward = types.MethodType(patched_forward, module)
        self.patched_modules.append((module, orig_forward))

    def _patch_llama_style_attention(self, module):
        repeat_kv = _resolve_repeat_kv(module)
        apply_rotary_pos_emb = _resolve_apply_rotary_pos_emb(module)
        orig_forward = module.forward

        def patched_forward(
            attn_self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            if getattr(attn_self.config, "pretraining_tp", 1) > 1:
                return orig_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            query_states = attn_self.q_proj(hidden_states)
            key_states = attn_self.k_proj(hidden_states)
            value_states = attn_self.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, attn_self.num_heads, attn_self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, attn_self.num_key_value_heads, attn_self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, attn_self.num_key_value_heads, attn_self.head_dim
            ).transpose(1, 2)

            past_key_value = getattr(attn_self, "past_key_value", past_key_value)
            cos, sin = attn_self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    getattr(attn_self, "kv_cache_idx", attn_self.layer_idx),
                    cache_kwargs,
                )

            key_states = repeat_kv(key_states, attn_self.num_key_value_groups)
            value_states = repeat_kv(value_states, attn_self.num_key_value_groups)

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(attn_self.head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                if cache_position is not None and attention_mask.size(-2) != q_len:
                    causal_mask = attention_mask[
                        :, :, cache_position, : key_states.shape[-2]
                    ]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = _apply_diag_mode(
                attn_weights,
                mode=self.mode,
                keep_first=self.keep_first,
                cache_position=cache_position,
            )
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=attn_self.attention_dropout,
                training=attn_self.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, attn_self.hidden_size)
            attn_output = attn_self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        module.forward = types.MethodType(patched_forward, module)
        self.patched_modules.append((module, orig_forward))

    def attach(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in self.active_layer_indices:
                continue
            module = getattr(layer, "self_attn", None)
            if module is None:
                continue
            class_name = type(module).__name__.lower()
            if "qwen" in class_name:
                self._patch_qwen_style_attention(module)
            elif "llama" in class_name:
                self._patch_llama_style_attention(module)
            else:
                eval_logger.warning(
                    "Skipping attn-diag patch for layer %s unsupported attention class %s",
                    layer_idx,
                    type(module).__name__,
                )

    def close(self) -> None:
        for module, orig_forward in reversed(self.patched_modules):
            module.forward = orig_forward
        self.patched_modules = []

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
