"""
Full definition of a GPT language model in a single file.

This implementation is adapted from standard public GPT-2 style references and
has been further modified for the experiments in this supplement.
"""

import math
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        head_dim_cfg = int(getattr(config, "n_head_dim", 0) or 0)
        if head_dim_cfg <= 0:
            if config.n_embd % config.n_head != 0:
                raise ValueError(
                    "n_embd must be divisible by n_head when n_head_dim is not set; "
                    f"got n_embd={config.n_embd}, n_head={config.n_head}"
                )
            head_dim_cfg = config.n_embd // config.n_head
        attn_dim = config.n_head * head_dim_cfg
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * attn_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(attn_dim, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = bool(getattr(config, "use_rope", False))
        self.rope_base = float(getattr(config, "rope_base", 10000.0))
        self.head_dim = head_dim_cfg
        self.attn_dim = attn_dim
        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}")
        if self.use_rope:
            inv_freq = 1.0 / (
                self.rope_base
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            pos = torch.arange(config.block_size, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            self.register_buffer("rope_cos_cached", emb.cos().view(1, 1, config.block_size, self.head_dim), persistent=False)
            self.register_buffer("rope_sin_cached", emb.sin().view(1, 1, config.block_size, self.head_dim), persistent=False)
        self.attn_diag_mode = str(getattr(config, "attn_diag_mode", "none")).lower().strip()
        self.attn_diag_bias = float(getattr(config, "attn_diag_bias", 0.0) or 0.0)
        self.attn_diag_keep_first = bool(getattr(config, "attn_diag_keep_first", True))
        if self.attn_diag_mode not in {"none", "hard", "bias"}:
            raise ValueError("attn_diag_mode must be one of: none, hard, bias")
        if self.attn_diag_mode == "bias" and self.attn_diag_bias <= 0.0:
            raise ValueError("attn_diag_bias must be > 0 when attn_diag_mode=bias")
        # In causal attention, token 0 has no previous key. Hard-masking (0,0) would create
        # an all -inf row and NaNs, so hard diagonal suppression must preserve it.
        if self.attn_diag_mode == "hard":
            self.attn_diag_keep_first = True
        self.use_diag_attention = self.attn_diag_mode != "none"
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        causal_keep = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("bias", causal_keep.view(1, 1, config.block_size, config.block_size))
        diag = torch.eye(config.block_size, dtype=torch.bool)
        diag_keep_first = diag.clone()
        diag_keep_first[0, 0] = False

        additive = torch.zeros(config.block_size, config.block_size, dtype=torch.float32)
        additive = additive.masked_fill(~causal_keep, float('-inf'))
        if self.use_diag_attention:
            diag_for_mask = diag_keep_first if self.attn_diag_keep_first else diag
            if self.attn_diag_mode == "hard":
                additive = additive.masked_fill(diag_for_mask, float('-inf'))
            elif self.attn_diag_mode == "bias":
                additive = additive - diag_for_mask.to(dtype=additive.dtype) * self.attn_diag_bias
        self.register_buffer("attn_additive_mask", additive.view(1, 1, config.block_size, config.block_size))
        self.track_preproj_self_value_metric = False
        self._xsa_preproj_self_value_proj_ratio = None
        self._xsa_preproj_self_value_proj_ratio_detach = None
        self._xsa_preproj_self_value_proj_sq = None
        self._xsa_preproj_self_value_y_sq = None
        self._xsa_preproj_self_value_x_sq = None
        self._last_attn_probs = None
        self._last_value = None
        self._last_y_preproj = None

    def _get_rope_cos_sin(self, T, device, dtype, position_ids=None):
        if position_ids is None:
            cos = self.rope_cos_cached[:, :, :T, :].to(device=device, dtype=dtype)
            sin = self.rope_sin_cached[:, :, :T, :].to(device=device, dtype=dtype)
            return cos, sin
        pos = position_ids.to(device=device, dtype=torch.long).clamp_min(0)
        flat = pos.reshape(-1)
        cos = self.rope_cos_cached[0, 0].index_select(0, flat).view(
            pos.size(0), pos.size(1), self.head_dim
        )
        sin = self.rope_sin_cached[0, 0].index_select(0, flat).view(
            pos.size(0), pos.size(1), self.head_dim
        )
        cos = cos.unsqueeze(1).to(dtype=dtype)
        sin = sin.unsqueeze(1).to(dtype=dtype)
        return cos, sin

    @staticmethod
    def _project_parallel_per_head(y, ref):
        dot = (y * ref).sum(dim=-1, keepdim=True)
        r_sq = ref.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (dot / r_sq) * ref

    @classmethod
    def _apply_parallel_op_per_head(cls, y, ref, op="remove_parallel", alpha=1.0, subtract_scale=1.0):
        op = str(op).lower().strip()
        alpha_t = torch.as_tensor(alpha, dtype=y.dtype, device=y.device)
        scale_t = torch.as_tensor(subtract_scale, dtype=y.dtype, device=y.device)
        dot = (y * ref).sum(dim=-1, keepdim=True)
        r_sq = ref.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coeff = dot / r_sq
        if op == "remove_parallel":
            return y - (scale_t * alpha_t * coeff) * ref
        if op == "keep_parallel":
            return (alpha_t * coeff) * ref
        if op == "add_parallel":
            return y + (alpha_t * coeff) * ref
        if op == "negate_parallel":
            return y - (scale_t * (1.0 + alpha_t) * coeff) * ref
        raise ValueError(f"Unsupported xsa_forward_op={op}")

    @staticmethod
    def _proj_stats_per_head(y, ref):
        y_f = y.float()
        r_f = ref.float()
        dot = (y_f * r_f).sum(dim=-1, keepdim=True)
        r_sq = (r_f * r_f).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = (dot / r_sq) * r_f
        proj_sq = (proj * proj).sum(dim=-1)
        y_sq = (y_f * y_f).sum(dim=-1).clamp_min(1e-6)
        ratio = (proj_sq / y_sq).mean()
        ratio_detach = (proj_sq / y_sq.detach()).mean()
        return ratio, ratio_detach, proj_sq.mean(), y_sq.mean(), r_sq.mean()

    def forward(
        self,
        x,
        attention_mask=None,
        position_ids=None,
        return_self_value=False,
        pre_proj_self_value_op=None,
        pre_proj_self_value_alpha=1.0,
        pre_proj_gamma=None,
        pre_proj_subtract_scale=1.0,
        output_attentions=False,
    ):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.attn_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        if self.use_rope:
            cos, sin = self._get_rope_cos_sin(T, q.device, q.dtype, position_ids=position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        self._last_attn_probs = None
        self._last_value = None
        self._last_y_preproj = None
        key_padding_additive_mask = None
        query_padding_mask = None
        if attention_mask is not None:
            attn_valid = attention_mask.to(device=x.device)
            if attn_valid.ndim != 2:
                raise ValueError(f"attention_mask must have shape [B, T], got {tuple(attn_valid.shape)}")
            query_padding_mask = attn_valid.unsqueeze(-1).to(dtype=x.dtype)
            key_padding_additive_mask = (1.0 - attn_valid[:, None, None, :].to(dtype=q.dtype)) * torch.finfo(q.dtype).min
        if self.flash and not output_attentions:
            # efficient attention using Flash Attention CUDA kernels when possible. For diagonal
            # suppression, use an additive logits mask instead of post-hoc projection.
            attn_mask = None
            if self.use_diag_attention:
                attn_mask = self.attn_additive_mask[:, :, :T, :T].to(dtype=q.dtype)
            if key_padding_additive_mask is not None:
                attn_mask = key_padding_additive_mask if attn_mask is None else attn_mask + key_padding_additive_mask
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=(attn_mask is None),
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + self.attn_additive_mask[:, :, :T, :T].to(dtype=att.dtype)
            if key_padding_additive_mask is not None:
                att = att + key_padding_additive_mask.to(dtype=att.dtype)
            att = F.softmax(att, dim=-1)
            if output_attentions:
                self._last_attn_probs = att.detach()
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        if self.track_preproj_self_value_metric:
            ratio, ratio_detach, proj_sq, y_sq, x_sq = self._proj_stats_per_head(y, v)
            self._xsa_preproj_self_value_proj_ratio = ratio
            self._xsa_preproj_self_value_proj_ratio_detach = ratio_detach
            self._xsa_preproj_self_value_proj_sq = proj_sq
            self._xsa_preproj_self_value_y_sq = y_sq
            self._xsa_preproj_self_value_x_sq = x_sq
        else:
            self._xsa_preproj_self_value_proj_ratio = None
            self._xsa_preproj_self_value_proj_ratio_detach = None
            self._xsa_preproj_self_value_proj_sq = None
            self._xsa_preproj_self_value_y_sq = None
            self._xsa_preproj_self_value_x_sq = None
        if pre_proj_self_value_op is not None:
            # Paper-aligned XSA family: manipulate the component parallel to self value v_i
            # before the output projection, independently for each attention head.
            y = self._apply_parallel_op_per_head(
                y,
                v,
                op=pre_proj_self_value_op,
                alpha=pre_proj_self_value_alpha,
                subtract_scale=pre_proj_subtract_scale,
            )
        if pre_proj_gamma is not None:
            g = torch.as_tensor(pre_proj_gamma, dtype=y.dtype, device=y.device)
            if g.ndim == 1:
                g = g.view(1, -1, 1, 1)
            y = y * g
        if query_padding_mask is not None:
            y = y * query_padding_mask[:, None, :, :]

        y = y.transpose(1, 2).contiguous().view(B, T, self.attn_dim) # re-assemble all head outputs side by side
        if output_attentions:
            self._last_value = v.transpose(1, 2).contiguous().view(B, T, self.attn_dim).detach()
            self._last_y_preproj = y.detach()
        self_v = None
        if return_self_value:
            # self value vector v_i = W_v x_i (concatenated across heads), used by strict XSA.
            self_v = v.transpose(1, 2).contiguous().view(B, T, self.attn_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if return_self_value:
            return y, self_v
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.track_xsa_self_metric = False
        self._xsa_attn_proj_ratio = None
        self._xsa_mlp_proj_ratio = None
        self._xsa_attn_proj_ratio_detach = None
        self._xsa_mlp_proj_ratio_detach = None
        self._xsa_attn_proj_sq = None
        self._xsa_mlp_proj_sq = None
        self._xsa_attn_y_sq = None
        self._xsa_mlp_y_sq = None
        self._xsa_attn_x_sq = None
        self._xsa_mlp_x_sq = None
        self._xsa_attn_state_cos = None
        self._xsa_mlp_state_cos = None
        self.xsa_forward_only = False
        self.xsa_forward_target = "attn" # attn | both | mlp
        self.xsa_forward_ref = "residual" # residual | self_value
        self.xsa_forward_space = "post_o_proj" # post_o_proj | pre_o_proj
        self.xsa_forward_op = "remove_parallel" # remove_parallel | keep_parallel | add_parallel | negate_parallel
        self.xsa_forward_alpha = 1.0
        self.xsa_forward_learnable_alpha = bool(getattr(config, "xsa_forward_learnable_alpha", False))
        self.xsa_forward_alpha_min = 0.0
        self.xsa_forward_alpha_max = 1.0
        self._xsa_forward_alpha_value = None
        self.xsa_forward_subtract_scale = 1.0
        self.xsa_forward_learnable_gate = bool(getattr(config, "xsa_forward_learnable_gate", False))
        self.xsa_forward_gate_mode = "static" # static | token
        self._xsa_forward_gate_value = None
        self.xsa_forward_learnable_gamma = bool(getattr(config, "xsa_forward_learnable_gamma", False))
        self.xsa_forward_gamma_per_head = False
        self._xsa_forward_gamma_value = None
        self.xsa_forward_enabled = True
        if self.xsa_forward_learnable_alpha:
            self.xsa_forward_alpha_raw = nn.Parameter(torch.tensor(0.0))
        if self.xsa_forward_learnable_gate:
            self.xsa_forward_gate_raw = nn.Parameter(torch.tensor(0.0))
            self.xsa_forward_gate_proj = nn.Linear(config.n_embd, 1, bias=True)
        if self.xsa_forward_learnable_gamma:
            self.xsa_forward_gamma_raw = nn.Parameter(torch.zeros(config.n_head))

    @staticmethod
    def _proj_stats(y, ref):
        y_f = y.float()
        r_f = ref.float()
        dot = (y_f * r_f).sum(dim=-1, keepdim=True)
        r_sq = (r_f * r_f).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = (dot / r_sq) * r_f
        proj_sq = (proj * proj).sum(dim=-1)
        y_sq = (y_f * y_f).sum(dim=-1).clamp_min(1e-6)
        ratio = (proj_sq / y_sq).mean()
        ratio_detach = (proj_sq / y_sq.detach()).mean()
        return ratio, ratio_detach, proj_sq.mean(), y_sq.mean(), r_sq.mean()

    @staticmethod
    def _state_cos(x, x_plus_f):
        x_f = x.float()
        y_f = x_plus_f.float()
        return F.cosine_similarity(x_f, y_f, dim=-1).mean()

    @staticmethod
    def _project_parallel(y, ref):
        dot = (y * ref).sum(dim=-1, keepdim=True)
        r_sq = ref.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (dot / r_sq) * ref

    @classmethod
    def _apply_parallel_op(cls, y, ref, op="remove_parallel", alpha=1.0, subtract_scale=1.0):
        op = str(op).lower().strip()
        alpha_t = torch.as_tensor(alpha, dtype=y.dtype, device=y.device)
        scale_t = torch.as_tensor(subtract_scale, dtype=y.dtype, device=y.device)
        dot = (y * ref).sum(dim=-1, keepdim=True)
        r_sq = ref.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coeff = dot / r_sq
        if op == "remove_parallel":
            return y - (scale_t * alpha_t * coeff) * ref
        if op == "keep_parallel":
            return (alpha_t * coeff) * ref
        if op == "add_parallel":
            return y + (alpha_t * coeff) * ref
        if op == "negate_parallel":
            return y - (scale_t * (1.0 + alpha_t) * coeff) * ref
        raise ValueError(f"Unsupported xsa_forward_op={op}")

    def _resolve_xsa_alpha(self, fallback_alpha):
        alpha_raw = getattr(self, "xsa_forward_alpha_raw", None)
        if bool(self.xsa_forward_learnable_alpha) and alpha_raw is not None:
            alpha01 = torch.sigmoid(alpha_raw)
            amin = float(self.xsa_forward_alpha_min)
            amax = float(self.xsa_forward_alpha_max)
            alpha = amin + (amax - amin) * alpha01
            self._xsa_forward_alpha_value = alpha.detach()
            return alpha
        self._xsa_forward_alpha_value = torch.tensor(float(fallback_alpha))
        return float(fallback_alpha)

    def _resolve_xsa_gamma(self):
        gamma_raw = getattr(self, "xsa_forward_gamma_raw", None)
        if bool(self.xsa_forward_learnable_gamma) and gamma_raw is not None:
            gamma_all = torch.exp(gamma_raw)
            gamma = gamma_all if bool(self.xsa_forward_gamma_per_head) else gamma_all.mean()
            self._xsa_forward_gamma_value = gamma.detach()
            return gamma
        self._xsa_forward_gamma_value = torch.tensor(1.0)
        return 1.0

    def _resolve_xsa_gate(self, x):
        mode = str(getattr(self, "xsa_forward_gate_mode", "static")).lower().strip()
        gate_proj = getattr(self, "xsa_forward_gate_proj", None)
        gate_raw = getattr(self, "xsa_forward_gate_raw", None)
        if bool(self.xsa_forward_learnable_gate) and mode == "token" and gate_proj is not None:
            gate = torch.sigmoid(gate_proj(x.float())).to(dtype=x.dtype)
            self._xsa_forward_gate_value = gate.detach().float().mean()
            return gate
        if bool(self.xsa_forward_learnable_gate) and gate_raw is not None:
            gate = torch.sigmoid(gate_raw)
            self._xsa_forward_gate_value = gate.detach()
            return gate
        self._xsa_forward_gate_value = torch.tensor(1.0)
        return 1.0

    def forward(self, x, attention_mask=None, position_ids=None, output_attentions=False):
        residual = x
        active_xsa_forward = bool(self.xsa_forward_only) and bool(self.xsa_forward_enabled)
        target = str(self.xsa_forward_target).lower().strip() if active_xsa_forward else ""
        ref_mode = str(self.xsa_forward_ref).lower().strip() if active_xsa_forward else ""
        xsa_space = str(self.xsa_forward_space).lower().strip() if active_xsa_forward else "post_o_proj"
        xsa_op = str(self.xsa_forward_op).lower().strip() if active_xsa_forward else "remove_parallel"
        xsa_alpha = self._resolve_xsa_alpha(self.xsa_forward_alpha) if active_xsa_forward else 1.0
        xsa_subtract_scale = float(self.xsa_forward_subtract_scale) if active_xsa_forward else 1.0
        xsa_gate = self._resolve_xsa_gate(x) if active_xsa_forward else 1.0
        xsa_effective_subtract_scale = xsa_subtract_scale * xsa_gate
        if ref_mode == "self_value" and target != "attn":
            target = "attn"
        self.attn.track_preproj_self_value_metric = bool(
            self.track_xsa_self_metric and ref_mode == "self_value" and xsa_space == "pre_o_proj"
        )
        xsa_gamma = self._resolve_xsa_gamma() if active_xsa_forward else 1.0
        if active_xsa_forward and target in {"attn", "both"}:
            if ref_mode == "self_value":
                if xsa_space == "pre_o_proj":
                    # Strict paper-aligned XSA family: manipulate y component parallel to v_i
                    # before c_proj/o_proj, in per-head value space.
                    # No need to return an extra self_v copy here: the pre-o_proj transform
                    # already happens inside attention and materializing self_v at block scope
                    # adds a large unused activation for long-context 2.7B runs.
                    attn_out = self.attn(
                        self.ln_1(x),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        pre_proj_self_value_op=xsa_op,
                        pre_proj_self_value_alpha=xsa_alpha,
                        pre_proj_gamma=xsa_gamma,
                        pre_proj_subtract_scale=(
                            xsa_effective_subtract_scale.unsqueeze(1)
                            if isinstance(xsa_effective_subtract_scale, torch.Tensor)
                            and xsa_effective_subtract_scale.ndim == 3
                            else xsa_effective_subtract_scale
                        ),
                        output_attentions=output_attentions,
                    )
                else:
                    attn_out, attn_self_v = self.attn(
                        self.ln_1(x),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_self_value=True,
                        output_attentions=output_attentions,
                    )
                    # Legacy self-value variant: post-projection attention output vs raw v_i.
                    attn_out = self._apply_parallel_op(
                        attn_out,
                        attn_self_v,
                        op=xsa_op,
                        alpha=xsa_alpha,
                        subtract_scale=xsa_effective_subtract_scale,
                    )
                    attn_out = attn_out * xsa_gamma
            else:
                # Original local variant: remove component parallel to residual state x.
                attn_out = self.attn(
                    self.ln_1(x),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
                attn_out = self._apply_parallel_op(
                    attn_out,
                    residual,
                    op=xsa_op,
                    alpha=xsa_alpha,
                    subtract_scale=xsa_effective_subtract_scale,
                )
                attn_out = attn_out * xsa_gamma
        else:
            attn_out = self.attn(
                self.ln_1(x),
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
        if self.track_xsa_self_metric:
            ratio, ratio_detach, proj_sq, y_sq, x_sq = self._proj_stats(attn_out, residual)
            self._xsa_attn_proj_ratio = ratio
            self._xsa_attn_proj_ratio_detach = ratio_detach
            self._xsa_attn_proj_sq = proj_sq
            self._xsa_attn_y_sq = y_sq
            self._xsa_attn_x_sq = x_sq
            self._xsa_attn_state_cos = self._state_cos(residual, residual + attn_out)
            self._xsa_attn_preproj_self_value_proj_ratio = getattr(self.attn, "_xsa_preproj_self_value_proj_ratio", None)
            self._xsa_attn_preproj_self_value_proj_ratio_detach = getattr(self.attn, "_xsa_preproj_self_value_proj_ratio_detach", None)
            self._xsa_attn_preproj_self_value_proj_sq = getattr(self.attn, "_xsa_preproj_self_value_proj_sq", None)
            self._xsa_attn_preproj_self_value_y_sq = getattr(self.attn, "_xsa_preproj_self_value_y_sq", None)
            self._xsa_attn_preproj_self_value_x_sq = getattr(self.attn, "_xsa_preproj_self_value_x_sq", None)
        else:
            self._xsa_attn_proj_ratio = None
            self._xsa_attn_proj_ratio_detach = None
            self._xsa_attn_proj_sq = None
            self._xsa_attn_y_sq = None
            self._xsa_attn_x_sq = None
            self._xsa_attn_state_cos = None
            self._xsa_attn_preproj_self_value_proj_ratio = None
            self._xsa_attn_preproj_self_value_proj_ratio_detach = None
            self._xsa_attn_preproj_self_value_proj_sq = None
            self._xsa_attn_preproj_self_value_y_sq = None
            self._xsa_attn_preproj_self_value_x_sq = None
        x = residual + attn_out

        residual = x
        mlp_out = self.mlp(self.ln_2(x))
        if active_xsa_forward:
            if target in {"both", "mlp"}:
                mlp_out = self._apply_parallel_op(
                    mlp_out,
                    residual,
                    op=xsa_op,
                    alpha=xsa_alpha,
                    subtract_scale=xsa_effective_subtract_scale,
                )
        if self.track_xsa_self_metric:
            ratio, ratio_detach, proj_sq, y_sq, x_sq = self._proj_stats(mlp_out, residual)
            self._xsa_mlp_proj_ratio = ratio
            self._xsa_mlp_proj_ratio_detach = ratio_detach
            self._xsa_mlp_proj_sq = proj_sq
            self._xsa_mlp_y_sq = y_sq
            self._xsa_mlp_x_sq = x_sq
            self._xsa_mlp_state_cos = self._state_cos(residual, residual + mlp_out)
        else:
            self._xsa_mlp_proj_ratio = None
            self._xsa_mlp_proj_ratio_detach = None
            self._xsa_mlp_proj_sq = None
            self._xsa_mlp_y_sq = None
            self._xsa_mlp_x_sq = None
            self._xsa_mlp_state_cos = None
        x = residual + mlp_out
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_head_dim: int = 0 # 0 => infer as n_embd // n_head (legacy behavior)
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    enable_attnres: bool = False
    attnres_eps: float = 1e-6
    attn_diag_mode: str = 'none' # none | hard | bias
    attn_diag_bias: float = 0.0
    attn_diag_keep_first: bool = True
    activation_checkpointing: bool = False
    xsa_forward_learnable_gate: bool = False
    xsa_forward_learnable_alpha: bool = False
    xsa_forward_learnable_gamma: bool = False
    use_rope: bool = True
    rope_base: float = 10000.0
    embedding_layernorm: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.use_rope = bool(getattr(config, "use_rope", False))
        self.embedding_layernorm = bool(getattr(config, "embedding_layernorm", False))
        self.activation_checkpointing = bool(getattr(config, "activation_checkpointing", False))

        transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        if not self.use_rope:
            transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        if self.embedding_layernorm:
            transformer_dict["ln_e"] = LayerNorm(config.n_embd, bias=config.bias)
        self.transformer = nn.ModuleDict(transformer_dict)
        for layer_idx, block in enumerate(self.transformer.h):
            block._xsa_layer_idx = layer_idx
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.enable_attnres = bool(getattr(config, "enable_attnres", False))
        self.attnres_eps = float(getattr(config, "attnres_eps", 1e-6))
        if self.enable_attnres:
            self.attnres_query = nn.Parameter(torch.empty(config.n_layer, config.n_embd))
            torch.nn.init.normal_(self.attnres_query, mean=0.0, std=0.02)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, "wpe"):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _attnres_rmsnorm(self, x):
        # x: [N, B, T, C], normalize over C with no affine params.
        denom = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.attnres_eps).rsqrt()
        return x * denom.to(dtype=x.dtype)

    def forward(self, idx, targets=None, attention_mask=None, position_ids=None, return_layer_outputs=False, return_xsa_metrics=False, output_attentions=False, use_cache=False, return_dict=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
            if attention_mask.shape != idx.shape:
                raise ValueError(
                    f"attention_mask shape must match idx shape, got {tuple(attention_mask.shape)} vs {tuple(idx.shape)}"
                )
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(dim=1) - 1
                position_ids = position_ids.clamp_min(0)
        if position_ids is not None:
            position_ids = position_ids.to(device=device, dtype=torch.long)
            if position_ids.shape != idx.shape:
                raise ValueError(
                    f"position_ids shape must match idx shape, got {tuple(position_ids.shape)} vs {tuple(idx.shape)}"
                )
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if hasattr(self.transformer, "wpe"):
            if position_ids is None:
                pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
                pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            else:
                pos_emb = self.transformer.wpe(position_ids)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        if hasattr(self.transformer, "ln_e"):
            x = self.transformer.ln_e(x)
        x = self.transformer.drop(x)
        layer_outputs = [] if return_layer_outputs else None
        xsa_layer_branch_proj_ratio = [] if return_xsa_metrics else None
        xsa_layer_branch_state_cos = [] if return_xsa_metrics else None
        all_attentions = [] if output_attentions else None
        # Attention Residuals (AttnRes) over depth: aggregate embedding + previous layer deltas.
        attnres_values = [x] if self.enable_attnres else None
        for lid, block in enumerate(self.transformer.h):
            if self.enable_attnres:
                # V: [Nsrc, B, T, C], K=RMSNorm(V), q_l in R^C, alpha softmax over depth axis.
                V = torch.stack(attnres_values, dim=0)
                K = self._attnres_rmsnorm(V)
                q = self.attnres_query[lid].to(dtype=K.dtype)
                logits_depth = torch.einsum('d,nbtd->nbt', q, K)
                alpha_depth = logits_depth.softmax(dim=0)
                x_in = torch.einsum('nbt,nbtd->btd', alpha_depth, V)
            else:
                x_in = x
            use_block_checkpoint = (
                bool(self.activation_checkpointing)
                and self.training
                and (not return_layer_outputs)
                and (not return_xsa_metrics)
                and (not output_attentions)
            )
            if use_block_checkpoint:
                def _block_forward(hidden_states, _block=block, _attention_mask=attention_mask, _position_ids=position_ids):
                    return _block(
                        hidden_states,
                        attention_mask=_attention_mask,
                        position_ids=_position_ids,
                        output_attentions=False,
                    )
                x = activation_checkpoint(_block_forward, x_in, use_reentrant=False)
            else:
                x = block(
                    x_in,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            if self.enable_attnres:
                attnres_values.append(x - x_in)
            if return_layer_outputs:
                layer_outputs.append(x)
            if output_attentions:
                all_attentions.append(block.attn._last_attn_probs)
            if return_xsa_metrics:
                xsa_layer_branch_proj_ratio.append(
                    {
                        "attn": getattr(block, "_xsa_attn_proj_ratio", None),
                        "mlp": getattr(block, "_xsa_mlp_proj_ratio", None),
                        "attn_ratio_detach": getattr(block, "_xsa_attn_proj_ratio_detach", None),
                        "mlp_ratio_detach": getattr(block, "_xsa_mlp_proj_ratio_detach", None),
                        "attn_proj_sq": getattr(block, "_xsa_attn_proj_sq", None),
                        "mlp_proj_sq": getattr(block, "_xsa_mlp_proj_sq", None),
                        "attn_y_sq": getattr(block, "_xsa_attn_y_sq", None),
                        "mlp_y_sq": getattr(block, "_xsa_mlp_y_sq", None),
                        "attn_x_sq": getattr(block, "_xsa_attn_x_sq", None),
                        "mlp_x_sq": getattr(block, "_xsa_mlp_x_sq", None),
                        "attn_preproj_self_value": getattr(block, "_xsa_attn_preproj_self_value_proj_ratio", None),
                        "attn_preproj_self_value_ratio_detach": getattr(block, "_xsa_attn_preproj_self_value_proj_ratio_detach", None),
                        "attn_preproj_self_value_proj_sq": getattr(block, "_xsa_attn_preproj_self_value_proj_sq", None),
                        "attn_preproj_self_value_y_sq": getattr(block, "_xsa_attn_preproj_self_value_y_sq", None),
                        "attn_preproj_self_value_x_sq": getattr(block, "_xsa_attn_preproj_self_value_x_sq", None),
                    }
                )
                xsa_layer_branch_state_cos.append(
                    {
                        "attn": getattr(block, "_xsa_attn_state_cos", None),
                        "mlp": getattr(block, "_xsa_mlp_state_cos", None),
                    }
                )
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_dict:
            payload = {"logits": logits, "loss": loss}
            if return_layer_outputs:
                payload["layer_outputs"] = layer_outputs
            if return_xsa_metrics:
                payload["xsa_layer_branch_proj_ratio"] = xsa_layer_branch_proj_ratio
                payload["xsa_layer_branch_state_cos"] = xsa_layer_branch_state_cos
            if output_attentions:
                payload["attentions"] = tuple(all_attentions)
            if use_cache:
                payload["past_key_values"] = None
            return SimpleNamespace(**payload)
        if return_layer_outputs or return_xsa_metrics:
            extras = {}
            if return_layer_outputs:
                extras["layer_outputs"] = layer_outputs
            if return_xsa_metrics:
                extras["xsa_layer_branch_proj_ratio"] = xsa_layer_branch_proj_ratio
                extras["xsa_layer_branch_state_cos"] = xsa_layer_branch_state_cos
            if output_attentions:
                extras["attentions"] = tuple(all_attentions)
            return logits, loss, extras
        if output_attentions:
            return logits, loss, {"attentions": tuple(all_attentions)}
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, "wpe"):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # keep pretrained GPT-2 path architecture-compatible with learned position embeddings.
        assert all(k in {'dropout', 'use_rope', 'embedding_layernorm'} for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        config_args['use_rope'] = False
        config_args['embedding_layernorm'] = False
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        if 'use_rope' in override_args:
            config_args['use_rope'] = bool(override_args['use_rope'])
        if 'embedding_layernorm' in override_args:
            config_args['embedding_layernorm'] = bool(override_args['embedding_layernorm'])
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, residual_lr_ratio=1.0):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        residual_lr_ratio = float(residual_lr_ratio)
        residual_names = {
            n for n in param_dict.keys()
            if ("residual_" in n) or ("attn_residual_" in n) or ("mlp_residual_" in n) or ("_residual_" in n)
        }

        base_decay_params = [p for n, p in param_dict.items() if (n not in residual_names) and (p.dim() >= 2)]
        base_nodecay_params = [p for n, p in param_dict.items() if (n not in residual_names) and (p.dim() < 2)]
        # Keep all residual parameters in no_decay; decay tends to collapse scales toward identity.
        residual_decay_params = []
        residual_nodecay_params = [p for n, p in param_dict.items() if n in residual_names]

        optim_groups = []
        if base_decay_params:
            optim_groups.append({"params": base_decay_params, "weight_decay": weight_decay, "lr_scale": 1.0})
        if base_nodecay_params:
            optim_groups.append({"params": base_nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0})

        has_residual = len(residual_decay_params) + len(residual_nodecay_params) > 0
        if has_residual:
            if residual_decay_params:
                optim_groups.append(
                    {
                        "params": residual_decay_params,
                        "weight_decay": weight_decay,
                        "lr_scale": residual_lr_ratio,
                    }
                )
            if residual_nodecay_params:
                optim_groups.append(
                    {
                        "params": residual_nodecay_params,
                        "weight_decay": 0.0,
                        "lr_scale": residual_lr_ratio,
                    }
                )

        num_base_decay = sum(p.numel() for p in base_decay_params)
        num_base_nodecay = sum(p.numel() for p in base_nodecay_params)
        num_residual_decay = sum(p.numel() for p in residual_decay_params)
        num_residual_nodecay = sum(p.numel() for p in residual_nodecay_params)
        print(f"num base decayed parameter tensors: {len(base_decay_params)}, with {num_base_decay:,} parameters")
        print(f"num base non-decayed parameter tensors: {len(base_nodecay_params)}, with {num_base_nodecay:,} parameters")
        print(f"num residual decayed parameter tensors: {len(residual_decay_params)}, with {num_residual_decay:,} parameters")
        print(f"num residual non-decayed parameter tensors: {len(residual_nodecay_params)}, with {num_residual_nodecay:,} parameters")
        print(f"residual_lr_ratio={residual_lr_ratio}")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        Q = int(getattr(cfg, "n_head_dim", 0) or 0)
        if Q <= 0:
            Q = cfg.n_embd // cfg.n_head
        L, H, T = cfg.n_layer, cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, attention_mask=None, position_ids=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) <= self.config.block_size else attention_mask[:, -self.config.block_size:]
            pos_cond = None
            if position_ids is not None:
                pos_cond = position_ids if position_ids.size(1) <= self.config.block_size else position_ids[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, attention_mask=mask_cond, position_ids=pos_cond)
            # pluck the logits at the final valid step and scale by desired temperature
            if mask_cond is not None:
                last_positions = mask_cond.long().sum(dim=1).sub(1).clamp_min(0)
                logits = logits[torch.arange(logits.size(0), device=logits.device), last_positions, :]
            else:
                logits = logits[:, -1, :]
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.size(0), 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
            if position_ids is not None:
                next_pos = position_ids[:, -1:].clone() + 1
                position_ids = torch.cat((position_ids, next_pos), dim=1)

        return idx
