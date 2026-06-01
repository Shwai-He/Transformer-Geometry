from __future__ import annotations

import torch

from analysis.vlm_geometry.hooks import _expand_value_ref, _scale_update, _scale_update_multihead


class DummyAttention:
    def __init__(self, heads: int, dim_head: int, num_key_value_heads: int | None = None):
        self.heads = heads
        self.dim_head = dim_head
        self.num_key_value_heads = num_key_value_heads


class DummyPhiAttention(DummyAttention):
    class Config:
        model_type = "phi3"

    config = Config()


def test_value_scaling_is_per_head_for_diffusers_style_attention():
    attn = DummyAttention(heads=2, dim_head=2)
    y = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    ref = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])

    per_head, _ = _scale_update_multihead(
        y,
        ref,
        attn,
        para_scale=0.0,
        perp_scale=1.0,
        eps=1e-6,
    )
    flat, _ = _scale_update(y, ref, para_scale=0.0, perp_scale=1.0, eps=1e-6)

    assert torch.allclose(per_head, torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]))
    assert not torch.allclose(per_head, flat)


def test_head_aware_value_ref_expands_by_head_not_by_scalar():
    attn = DummyAttention(heads=4, dim_head=2, num_key_value_heads=2)
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    y_pre = torch.zeros(1, 1, 8)

    expanded = _expand_value_ref(value, y_pre, attn, expansion_mode="head_aware", attr_source="head_aware")

    assert expanded is not None
    assert torch.equal(expanded, torch.tensor([[[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]]]))


def test_model_type_uses_legacy_expansion_for_non_phi_modules():
    attn = DummyAttention(heads=4, dim_head=2, num_key_value_heads=2)
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    y_pre = torch.zeros(1, 1, 8)

    expanded = _expand_value_ref(value, y_pre, attn, expansion_mode="model_type", attr_source="model_type")

    assert expanded is not None
    assert torch.equal(expanded, torch.tensor([[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]]))


def test_model_type_uses_head_aware_expansion_for_phi_modules():
    attn = DummyPhiAttention(heads=4, dim_head=2, num_key_value_heads=2)
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    y_pre = torch.zeros(1, 1, 8)

    expanded = _expand_value_ref(value, y_pre, attn, expansion_mode="model_type", attr_source="model_type")

    assert expanded is not None
    assert torch.equal(expanded, torch.tensor([[[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]]]))


def test_legacy_value_ref_expansion_is_explicit_scalar_repeat():
    attn = DummyAttention(heads=4, dim_head=2, num_key_value_heads=2)
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    y_pre = torch.zeros(1, 1, 8)

    expanded = _expand_value_ref(value, y_pre, attn, expansion_mode="legacy", attr_source="legacy")

    assert expanded is not None
    assert torch.equal(expanded, torch.tensor([[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]]))
