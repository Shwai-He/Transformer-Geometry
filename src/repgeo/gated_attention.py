from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x, dim=-1).clamp_min(eps)


def _project_parallel(delta: torch.Tensor, base: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = (base * base).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = (delta * base).sum(dim=-1, keepdim=True) / denom
    return coeff * base


@dataclass
class UpdateGeometryStats:
    dz_para_mean: float
    dz_perp_mean: float
    log_para_perp_mean: float
    delta_norm_mean: float
    output_norm_mean: float


class VanillaAttentionBlock(nn.Module):
    """
    Minimal pre-norm residual block:
    x -> x + MHA(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + attn_out


class GatedAttentionBlock(nn.Module):
    """
    A simple gated variant inspired by the paper direction:
    x -> x + g(x) * MHA(LN(x))
    where g(x) in (0, 1) is token-wise, channel-wise gate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        g = self.gate(h)
        return x + g * attn_out


@torch.no_grad()
def compute_update_geometry(x_in: torch.Tensor, x_out: torch.Tensor) -> UpdateGeometryStats:
    """
    Compute parallel/orthogonal decomposition stats for token representations.
    x_in, x_out: [batch, seq, dim]
    """
    delta = x_out - x_in
    dz_para = _project_parallel(delta, x_in)
    dz_perp = delta - dz_para

    para_norm = _safe_norm(dz_para)
    perp_norm = _safe_norm(dz_perp)
    delta_norm = _safe_norm(delta)
    out_norm = _safe_norm(x_out)
    log_ratio = torch.log((para_norm + 1e-12) / (perp_norm + 1e-12))

    return UpdateGeometryStats(
        dz_para_mean=float(para_norm.mean().item()),
        dz_perp_mean=float(perp_norm.mean().item()),
        log_para_perp_mean=float(log_ratio.mean().item()),
        delta_norm_mean=float(delta_norm.mean().item()),
        output_norm_mean=float(out_norm.mean().item()),
    )


@torch.no_grad()
def compare_vanilla_vs_gated(
    d_model: int = 512,
    n_heads: int = 8,
    seq_len: int = 64,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Run a synthetic comparison to get first-order geometry signals.
    Useful for fast local iteration before plugging into large LMs.
    """
    torch.manual_seed(seed)

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    vanilla = VanillaAttentionBlock(d_model=d_model, n_heads=n_heads).to(device).eval()
    gated = GatedAttentionBlock(d_model=d_model, n_heads=n_heads).to(device).eval()

    y_vanilla = vanilla(x)
    y_gated = gated(x)

    s_vanilla = compute_update_geometry(x, y_vanilla)
    s_gated = compute_update_geometry(x, y_gated)

    return {
        "vanilla": s_vanilla.__dict__,
        "gated": s_gated.__dict__,
        "delta_gated_minus_vanilla": {
            k: s_gated.__dict__[k] - s_vanilla.__dict__[k]
            for k in s_vanilla.__dict__.keys()
        },
    }

