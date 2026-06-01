from __future__ import annotations

import logging
import math
import types
from typing import Optional

import torch

from lm_eval.models.attn_diag_hooks import (
    _find_decoder_layers,
    _repeat_kv_fallback,
    _resolve_apply_rotary_pos_emb,
    _resolve_repeat_kv,
)


eval_logger = logging.getLogger(__name__)


def _topk_sparse_attention(
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
    *,
    mode: str,
    keep_ratio: float,
    keep_tokens: int,
    sink_tokens: int,
    local_tokens: int,
    prefill_only: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Sparsify normalized attention weights at query-key edge level.

    Shapes:
      attn_weights: [batch, heads, q_len, k_len]
      value_states: [batch, heads, k_len, head_dim]
    """

    if keep_ratio >= 1.0 and keep_tokens <= 0:
        return attn_weights
    if prefill_only and attn_weights.size(-2) <= 1:
        return attn_weights

    mode = str(mode).lower().strip()
    q_len = int(attn_weights.size(-2))
    k_len = int(attn_weights.size(-1))
    if q_len <= 0 or k_len <= 0:
        return attn_weights

    if keep_tokens > 0:
        keep = min(k_len, max(1, int(keep_tokens)))
    else:
        keep = min(k_len, max(1, int(round(k_len * float(keep_ratio)))))
    keep = max(keep, min(k_len, int(sink_tokens) + int(local_tokens)))
    if keep >= k_len:
        return attn_weights

    value_norm = value_states.float().norm(dim=-1).clamp_min(eps)
    attn_float = attn_weights.float()
    valid = attn_float > 0
    rows = torch.arange(q_len, device=attn_weights.device)[:, None]
    cols = torch.arange(k_len, device=attn_weights.device)[None, :]

    forced = torch.zeros((q_len, k_len), device=attn_weights.device, dtype=torch.bool)
    if sink_tokens > 0:
        forced[:, : min(k_len, int(sink_tokens))] = True
    if local_tokens > 0:
        forced |= (cols <= rows) & (cols > rows - int(local_tokens))
    forced = forced[None, None, :, :] & valid

    if mode in {"local_window", "sliding_window"}:
        local_mask = (cols <= rows) & (cols > rows - keep)
        if sink_tokens > 0:
            local_mask |= cols < min(k_len, int(sink_tokens))
        keep_mask = local_mask[None, None, :, :] & valid
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"longformer", "longformer_global", "global_local"}:
        global_n = min(k_len, max(1, int(sink_tokens)))
        local_n = max(1, keep - global_n)
        keep_mask_2d = (cols < global_n).expand(q_len, k_len).clone()
        keep_mask_2d |= (cols <= rows) & (cols > rows - local_n)
        keep_mask = keep_mask_2d[None, None, :, :] & valid
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"bigbird", "bigbird_block", "global_local_random"}:
        global_n = min(k_len, max(1, int(sink_tokens)))
        local_n = max(1, min(keep, int(local_tokens) if int(local_tokens) > 0 else keep // 2))
        keep_mask_2d = (cols < global_n).expand(q_len, k_len).clone()
        keep_mask_2d |= (cols <= rows) & (cols > rows - local_n)
        forced_bb = keep_mask_2d[None, None, :, :] & valid
        forced_count = forced_bb.long().sum(dim=-1, keepdim=True)
        remaining = max(1, keep - int(min(keep, global_n + local_n)))
        rand = torch.frac(torch.sin((rows.float() + 1.0) * 12.9898 + (cols.float() + 1.0) * 78.233) * 43758.5453)
        rand_score = rand[None, None, :, :].expand_as(attn_float).masked_fill(~valid | forced_bb, float("-inf"))
        top = torch.topk(rand_score, k=min(remaining, k_len), dim=-1, largest=True).indices
        keep_mask = forced_bb.clone()
        keep_mask.scatter_(-1, top, True)
        # If local/global already exceed budget for early rows, keep them; this
        # mirrors block-sparse baselines where global/local patterns are fixed.
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"sparse_transformer", "strided", "local_strided", "dilated"}:
        local_n = max(1, min(keep, int(local_tokens) if int(local_tokens) > 0 else keep // 2))
        keep_mask_2d = (cols <= rows) & (cols > rows - local_n)
        remaining = max(1, keep - local_n)
        stride = max(1, k_len // remaining)
        rel = (rows - cols).clamp_min(0)
        keep_mask_2d |= (cols <= rows) & (rel.remainder(stride) == 0)
        keep_mask = keep_mask_2d[None, None, :, :] & valid
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"vertical_slash", "vertical_slash_attention", "slash"}:
        local_n = max(1, min(keep, int(local_tokens) if int(local_tokens) > 0 else keep // 2))
        vertical_n = max(1, keep - local_n)
        local_mask = ((cols <= rows) & (cols > rows - local_n))[None, None, :, :]
        key_score = attn_float.masked_fill(~valid, 0.0).sum(dim=-2)
        top_cols = torch.topk(key_score, k=min(vertical_n, k_len), dim=-1, largest=True).indices
        vertical_mask = torch.zeros((attn_float.shape[0], attn_float.shape[1], k_len), device=attn_weights.device, dtype=torch.bool)
        vertical_mask.scatter_(-1, top_cols, True)
        keep_mask = (local_mask | vertical_mask[:, :, None, :]) & valid
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"block_sparse", "block_topk", "block_sparse_oracle"}:
        block_size = max(16, int(local_tokens) if int(local_tokens) > 0 else 64)
        n_blocks = math.ceil(k_len / block_size)
        pad = n_blocks * block_size - k_len
        block_keep = max(1, min(n_blocks, math.ceil(keep / block_size)))
        score_for_blocks = attn_float.masked_fill(~valid, 0.0)
        if pad:
            score_for_blocks = torch.nn.functional.pad(score_for_blocks, (0, pad))
        block_score = score_for_blocks.view(*score_for_blocks.shape[:-1], n_blocks, block_size).sum(dim=-1)
        top_blocks = torch.topk(block_score, k=block_keep, dim=-1, largest=True).indices
        block_mask = torch.zeros_like(block_score, dtype=torch.bool)
        block_mask.scatter_(-1, top_blocks, True)
        keep_mask = block_mask.repeat_interleave(block_size, dim=-1)[..., :k_len] & valid
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"minference_mix", "minference", "a_shape_vertical_block"}:
        local_n = max(1, min(keep, int(local_tokens) if int(local_tokens) > 0 else keep // 2))
        vertical_n = max(1, keep - local_n)
        local_2d = (cols <= rows) & (cols > rows - keep)
        slash_local = (cols <= rows) & (cols > rows - local_n)

        key_score = attn_float.masked_fill(~valid, 0.0).sum(dim=-2)
        top_cols = torch.topk(key_score, k=min(vertical_n, k_len), dim=-1, largest=True).indices
        vertical_mask = torch.zeros((attn_float.shape[0], attn_float.shape[1], k_len), device=attn_weights.device, dtype=torch.bool)
        vertical_mask.scatter_(-1, top_cols, True)
        slash_mask = (slash_local[None, None, :, :] | vertical_mask[:, :, None, :]) & valid

        block_size = max(16, int(local_tokens) if int(local_tokens) > 0 else 64)
        n_blocks = math.ceil(k_len / block_size)
        pad = n_blocks * block_size - k_len
        block_keep = max(1, min(n_blocks, math.ceil(keep / block_size)))
        score_for_blocks = attn_float.masked_fill(~valid, 0.0)
        if pad:
            score_for_blocks = torch.nn.functional.pad(score_for_blocks, (0, pad))
        block_score = score_for_blocks.view(*score_for_blocks.shape[:-1], n_blocks, block_size).sum(dim=-1)
        top_blocks = torch.topk(block_score, k=block_keep, dim=-1, largest=True).indices
        block_mask = torch.zeros_like(block_score, dtype=torch.bool)
        block_mask.scatter_(-1, top_blocks, True)
        block_mask = block_mask.repeat_interleave(block_size, dim=-1)[..., :k_len] & valid

        local_mask = local_2d[None, None, :, :] & valid
        head_ids = torch.arange(attn_float.shape[1], device=attn_weights.device)
        local_heads = (head_ids % 3 == 0)[None, :, None, None]
        slash_heads = (head_ids % 3 == 1)[None, :, None, None]
        block_heads = (head_ids % 3 == 2)[None, :, None, None]
        keep_mask = (local_heads & local_mask) | (slash_heads & slash_mask) | (block_heads & block_mask)
        sparse = attn_weights.masked_fill(~keep_mask, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
        return sparse / denom

    if mode in {"sink_local", "streamingllm", "streaming_llm", "local_sink"}:
        sparse = attn_weights.masked_fill(~forced, 0.0)
        denom = sparse.sum(dim=-1, keepdim=True)
        fallback = denom <= eps
        if bool(fallback.any()):
            recent_score = torch.arange(k_len, device=attn_weights.device, dtype=torch.float32)
            recent_score = recent_score[None, None, None, :].expand_as(attn_float).masked_fill(~valid, float("-inf"))
            top = torch.topk(recent_score, k=1, dim=-1, largest=True).indices
            keep_mask = forced.clone()
            keep_mask.scatter_(-1, top, True)
            sparse = attn_weights.masked_fill(~keep_mask, 0.0)
            denom = sparse.sum(dim=-1, keepdim=True)
        return sparse / denom.clamp_min(eps)

    if mode in {"h2o", "heavy_hitter", "heavy_hitter_oracle"}:
        # H2O-style oracle: select heavy hitters by accumulated attention mass
        # over the current prefill block, plus the recent tokens forced above.
        key_score = attn_float.sum(dim=-2, keepdim=True)
        score = key_score.expand_as(attn_float)
    elif mode in {"snapkv", "snapkv_observation", "observation_attention"}:
        # SnapKV-style oracle: estimate key importance from an observation
        # window near the end of the prompt, plus the recent tokens.
        obs = max(1, min(q_len, int(local_tokens) if int(local_tokens) > 0 else q_len))
        key_score = attn_float[:, :, -obs:, :].sum(dim=-2, keepdim=True)
        score = key_score.expand_as(attn_float)
    elif mode in {"attention", "attention_keep", "attn", "attn_keep", "attention_topk", "attn_topk"}:
        score = attn_float
    elif mode in {
        "contribution_norm",
        "contribution_norm_keep",
        "value_norm",
        "target_norm",
        "contribution_norm_topk",
    }:
        score = attn_float * value_norm[:, :, None, :]
    elif mode in {
        "contribution_perp",
        "contribution_perp_keep",
        "perp",
        "perp_keep",
        "geometry_perp",
        "contribution_perp_topk",
        "geometry_perp_topk",
    }:
        dense_ref = torch.matmul(attn_float.to(value_states.dtype), value_states).float()
        ref_norm = dense_ref.norm(dim=-1).clamp_min(eps)
        dot = torch.matmul(dense_ref, value_states.float().transpose(-1, -2)).abs()
        value_norm_q = value_norm[:, :, None, :].clamp_min(eps)
        cos = (dot / (value_norm_q * ref_norm[:, :, :, None])).clamp(0.0, 1.0)
        perp_ratio = (1.0 - cos.square()).clamp_min(0.0).sqrt()
        score = attn_float * value_norm_q * perp_ratio
    elif mode in {"uniform", "recent"}:
        pos = torch.arange(k_len, device=attn_weights.device, dtype=torch.float32)
        score = pos[None, None, None, :].expand_as(attn_float)
    elif mode in {"random", "random_keep"}:
        pos = torch.arange(k_len, device=attn_weights.device, dtype=torch.float32)
        score = torch.frac(torch.sin((pos + 1.0) * 12.9898) * 43758.5453)
        score = score[None, None, None, :].expand_as(attn_float)
    else:
        raise ValueError(
            "sparse_attention_mode must be one of local_window/longformer/bigbird/sparse_transformer/vertical_slash/block_sparse/minference_mix/attention_topk/contribution_norm_topk/contribution_perp_topk, "
            f"got {mode}"
        )

    score = score.masked_fill(~valid, float("-inf"))

    candidate_score = score.masked_fill(forced, float("inf"))
    # Avoid asking topk for more elements than a row can contain; invalid keys
    # are -inf and therefore remain unselected whenever enough valid keys exist.
    top = torch.topk(candidate_score, k=keep, dim=-1, largest=True).indices
    keep_mask = torch.zeros_like(valid)
    keep_mask.scatter_(-1, top, True)
    keep_mask |= forced

    sparse = attn_weights.masked_fill(~keep_mask, 0.0)
    denom = sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
    return sparse / denom


class SparseAttentionHooks:
    def __init__(
        self,
        model,
        *,
        mode: str = "contribution_perp",
        keep_ratio: float = 0.5,
        keep_tokens: int = 0,
        sink_tokens: int = 32,
        local_tokens: int = 128,
        prefill_only: bool = True,
        start_layer: int = 0,
        end_layer: int = -1,
        skip_first_n: int = 0,
        skip_last_n: int = 0,
    ):
        self.model = model
        self.mode = str(mode).lower().strip()
        self.keep_ratio = float(keep_ratio)
        self.keep_tokens = int(keep_tokens)
        self.sink_tokens = int(sink_tokens)
        self.local_tokens = int(local_tokens)
        self.prefill_only = bool(prefill_only)
        self.layers = _find_decoder_layers(model)
        n_layers = len(self.layers)
        lo = max(0, int(start_layer), int(skip_first_n))
        hi = n_layers if int(end_layer) < 0 else min(n_layers, int(end_layer))
        hi = min(hi, max(0, n_layers - int(skip_last_n)))
        if hi < lo:
            hi = lo
        self.active_layer_indices = set(range(lo, hi))
        self.patched_modules = []

    def _sparsify(self, attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        return _topk_sparse_attention(
            attn_weights,
            value_states,
            mode=self.mode,
            keep_ratio=self.keep_ratio,
            keep_tokens=self.keep_tokens,
            sink_tokens=self.sink_tokens,
            local_tokens=self.local_tokens,
            prefill_only=self.prefill_only,
        )

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

            scaling = getattr(attn_self, "scaling", float(attn_self.head_dim) ** -0.5)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = self._sparsify(attn_weights, value_states)
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
            attn_weights = self._sparsify(attn_weights, value_states)
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
                    "Skipping sparse-attention patch for layer %s unsupported attention class %s",
                    layer_idx,
                    type(module).__name__,
                )

    def close(self) -> None:
        for module, orig_forward in reversed(self.patched_modules):
            module.forward = orig_forward
        self.patched_modules = []
