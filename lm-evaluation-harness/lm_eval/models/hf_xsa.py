from __future__ import annotations

import logging
from typing import Any, Literal

from lm_eval.api.registry import register_model
from lm_eval.models.attn_diag_hooks import AttentionDiagonalHooks
from lm_eval.models.huggingface import HFLM
from lm_eval.models.xsa_hooks import QwenXSAForwardHooks


eval_logger = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in {"1", "true", "yes", "y", "on"}:
            return True
        if value_norm in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


@register_model("hf-xsa")
class HFXSALM(HFLM):
    """Hugging Face backend with optional XSA/residual-output forward hooks."""

    _XSA_KWARG_KEYS = {
        "xsa_enabled",
        "xsa_target",
        "xsa_intervention_site",
        "xsa_start_layer",
        "xsa_end_layer",
        "xsa_skip_first_n",
        "xsa_skip_last_n",
        "xsa_forward_op",
        "xsa_forward_alpha",
        "xsa_track_stats",
        "xsa_layerwise_stats",
        "attn_diag_enabled",
        "attn_diag_mode",
        "attn_diag_keep_first",
    }

    def __init__(
        self,
        *args,
        xsa_enabled: bool = True,
        xsa_target: Literal["none", "attn", "mlp", "both"] = "attn",
        xsa_intervention_site: Literal[
            "xsa_middle",
            "xsa_middle_multihead",
            "residual_output",
        ] = "xsa_middle_multihead",
        xsa_start_layer: int = 0,
        xsa_end_layer: int = -1,
        xsa_skip_first_n: int = 0,
        xsa_skip_last_n: int = 0,
        xsa_forward_op: Literal[
            "remove_parallel",
            "keep_parallel",
            "add_parallel",
            "negate_parallel",
        ] = "remove_parallel",
        xsa_forward_alpha: float = 1.0,
        xsa_track_stats: bool = False,
        xsa_layerwise_stats: bool = False,
        attn_diag_enabled: bool = False,
        attn_diag_mode: Literal["none", "zero_no_renorm", "zero_renorm"] = "none",
        attn_diag_keep_first: bool = True,
        **kwargs,
    ) -> None:
        # Be defensive about upstream arg plumbing: if any XSA-only knobs are
        # still present in kwargs, consume them here so they never leak into
        # HFLM -> _create_model -> AutoModel.from_pretrained(...).
        xsa_enabled = kwargs.pop("xsa_enabled", xsa_enabled)
        xsa_target = kwargs.pop("xsa_target", xsa_target)
        xsa_intervention_site = kwargs.pop(
            "xsa_intervention_site", xsa_intervention_site
        )
        xsa_start_layer = kwargs.pop("xsa_start_layer", xsa_start_layer)
        xsa_end_layer = kwargs.pop("xsa_end_layer", xsa_end_layer)
        xsa_skip_first_n = kwargs.pop("xsa_skip_first_n", xsa_skip_first_n)
        xsa_skip_last_n = kwargs.pop("xsa_skip_last_n", xsa_skip_last_n)
        xsa_forward_op = kwargs.pop("xsa_forward_op", xsa_forward_op)
        xsa_forward_alpha = kwargs.pop("xsa_forward_alpha", xsa_forward_alpha)
        xsa_track_stats = kwargs.pop("xsa_track_stats", xsa_track_stats)
        xsa_layerwise_stats = kwargs.pop(
            "xsa_layerwise_stats", xsa_layerwise_stats
        )
        attn_diag_enabled = kwargs.pop("attn_diag_enabled", attn_diag_enabled)
        attn_diag_mode = kwargs.pop("attn_diag_mode", attn_diag_mode)
        attn_diag_keep_first = kwargs.pop(
            "attn_diag_keep_first", attn_diag_keep_first
        )

        self.xsa_enabled = _as_bool(xsa_enabled)
        self.xsa_target = xsa_target
        self.xsa_intervention_site = xsa_intervention_site
        self.xsa_start_layer = int(xsa_start_layer)
        self.xsa_end_layer = int(xsa_end_layer)
        self.xsa_skip_first_n = int(xsa_skip_first_n)
        self.xsa_skip_last_n = int(xsa_skip_last_n)
        self.xsa_forward_op = str(xsa_forward_op).lower().strip()
        self.xsa_forward_alpha = float(xsa_forward_alpha)
        self.xsa_track_stats = _as_bool(xsa_track_stats)
        self.xsa_layerwise_stats = _as_bool(xsa_layerwise_stats)
        self.attn_diag_enabled = _as_bool(attn_diag_enabled)
        self.attn_diag_mode = str(attn_diag_mode).lower().strip()
        self.attn_diag_keep_first = _as_bool(attn_diag_keep_first)
        self.xsa_hooks = None
        self.attn_diag_hooks = None
        super().__init__(*args, **kwargs)
        self._attach_xsa_if_needed()
        self._attach_attn_diag_if_needed()

    def _create_model(self, *args, **kwargs) -> None:
        # Extra guardrail: regardless of how kwargs reach model creation, keep
        # XSA-only controls away from HF model constructors.
        for key in self._XSA_KWARG_KEYS:
            kwargs.pop(key, None)
        super()._create_model(*args, **kwargs)

    def _attach_xsa_if_needed(self) -> None:
        if not self.xsa_enabled or self.xsa_target == "none":
            eval_logger.info("XSA disabled for this model instance.")
            return
        self.xsa_hooks = QwenXSAForwardHooks(
            self.model,
            target=self.xsa_target,
            start_layer=self.xsa_start_layer,
            end_layer=self.xsa_end_layer,
            skip_first_n=self.xsa_skip_first_n,
            skip_last_n=self.xsa_skip_last_n,
            intervention_site=self.xsa_intervention_site,
            xsa_forward_op=self.xsa_forward_op,
            xsa_forward_alpha=self.xsa_forward_alpha,
            track_stats=self.xsa_track_stats,
            track_layerwise_stats=self.xsa_layerwise_stats,
        )
        self.xsa_hooks.attach()
        eval_logger.info(
            "Attached XSA hooks: target=%s site=%s op=%s alpha=%s layers=[%s,%s) skip_first=%s skip_last=%s track_stats=%s layerwise=%s",
            self.xsa_target,
            self.xsa_intervention_site,
            self.xsa_forward_op,
            self.xsa_forward_alpha,
            self.xsa_start_layer,
            self.xsa_end_layer,
            self.xsa_skip_first_n,
            self.xsa_skip_last_n,
            self.xsa_track_stats,
            self.xsa_layerwise_stats,
        )

    def _attach_attn_diag_if_needed(self) -> None:
        if not self.attn_diag_enabled or self.attn_diag_mode == "none":
            eval_logger.info("Attention-diagonal patch disabled for this model instance.")
            return
        self.attn_diag_hooks = AttentionDiagonalHooks(
            self.model,
            mode=self.attn_diag_mode,
            keep_first=self.attn_diag_keep_first,
            start_layer=self.xsa_start_layer,
            end_layer=self.xsa_end_layer,
            skip_first_n=self.xsa_skip_first_n,
            skip_last_n=self.xsa_skip_last_n,
        )
        self.attn_diag_hooks.attach()
        eval_logger.info(
            "Attached attention-diagonal patch: mode=%s keep_first=%s layers=[%s,%s) skip_first=%s skip_last=%s",
            self.attn_diag_mode,
            self.attn_diag_keep_first,
            self.xsa_start_layer,
            self.xsa_end_layer,
            self.xsa_skip_first_n,
            self.xsa_skip_last_n,
        )

    def cleanup(self) -> None:
        if self.xsa_hooks is not None:
            self.xsa_hooks.close()
            self.xsa_hooks = None
        if self.attn_diag_hooks is not None:
            self.attn_diag_hooks.close()
            self.attn_diag_hooks = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
