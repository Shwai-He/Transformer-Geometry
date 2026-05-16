from __future__ import annotations

from residual_patch_core import (  # noqa: F401
    _find_decoder_layers,
    apply_qwen3_forward_patch,
    configure_residual_scaling,
    fix_scalar_residual_params,
)

__all__ = [
    "_find_decoder_layers",
    "apply_qwen3_forward_patch",
    "configure_residual_scaling",
    "fix_scalar_residual_params",
]
