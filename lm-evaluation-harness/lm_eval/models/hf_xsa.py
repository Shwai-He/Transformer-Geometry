from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.attn_diag_hooks import AttentionDiagonalHooks
from lm_eval.models.huggingface import HFLM
from lm_eval.models.sparse_attention_hooks import SparseAttentionHooks
from lm_eval.models.token_compression import PrefillTokenCompressor, TokenCompressionConfig
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
        "xsa_layer_scale_mode",
        "xsa_layer_scale_seed",
        "xsa_layer_para_scale_min",
        "xsa_layer_para_scale_max",
        "xsa_track_stats",
        "xsa_layerwise_stats",
        "attn_diag_enabled",
        "attn_diag_mode",
        "attn_diag_keep_first",
        "token_compression_enabled",
        "token_compression_mode",
        "token_compression_keep_ratio",
        "token_compression_keep_tokens",
        "token_compression_layer",
        "token_compression_layers",
        "token_compression_ref",
        "token_compression_preserve_first",
        "token_compression_preserve_last",
        "token_compression_min_context",
        "token_compression_parallel_weight",
        "sparse_attention_enabled",
        "sparse_attention_mode",
        "sparse_attention_keep_ratio",
        "sparse_attention_keep_tokens",
        "sparse_attention_sink_tokens",
        "sparse_attention_local_tokens",
        "sparse_attention_prefill_only",
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
        xsa_layer_scale_mode: Literal[
            "none",
            "layer_para_uniform",
            "layer_para_choice",
            "sample_para_uniform",
            "sample_para_choice",
        ] = "none",
        xsa_layer_scale_seed: int = 0,
        xsa_layer_para_scale_min: float = -20.0,
        xsa_layer_para_scale_max: float = 20.0,
        xsa_track_stats: bool = False,
        xsa_layerwise_stats: bool = False,
        attn_diag_enabled: bool = False,
        attn_diag_mode: Literal["none", "zero_no_renorm", "zero_renorm"] = "none",
        attn_diag_keep_first: bool = True,
        token_compression_enabled: bool = False,
        token_compression_mode: str = "value_parallel_drop",
        token_compression_keep_ratio: float = 1.0,
        token_compression_keep_tokens: int = 0,
        token_compression_layer: int = 0,
        token_compression_layers: str = "",
        token_compression_ref: str = "last_value",
        token_compression_preserve_first: int = 32,
        token_compression_preserve_last: int = 256,
        token_compression_min_context: int = 512,
        token_compression_parallel_weight: float = 0.25,
        sparse_attention_enabled: bool = False,
        sparse_attention_mode: str = "contribution_perp",
        sparse_attention_keep_ratio: float = 1.0,
        sparse_attention_keep_tokens: int = 0,
        sparse_attention_sink_tokens: int = 32,
        sparse_attention_local_tokens: int = 128,
        sparse_attention_prefill_only: bool = True,
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
        xsa_layer_scale_mode = kwargs.pop("xsa_layer_scale_mode", xsa_layer_scale_mode)
        xsa_layer_scale_seed = kwargs.pop("xsa_layer_scale_seed", xsa_layer_scale_seed)
        xsa_layer_para_scale_min = kwargs.pop(
            "xsa_layer_para_scale_min", xsa_layer_para_scale_min
        )
        xsa_layer_para_scale_max = kwargs.pop(
            "xsa_layer_para_scale_max", xsa_layer_para_scale_max
        )
        xsa_track_stats = kwargs.pop("xsa_track_stats", xsa_track_stats)
        xsa_layerwise_stats = kwargs.pop(
            "xsa_layerwise_stats", xsa_layerwise_stats
        )
        attn_diag_enabled = kwargs.pop("attn_diag_enabled", attn_diag_enabled)
        attn_diag_mode = kwargs.pop("attn_diag_mode", attn_diag_mode)
        attn_diag_keep_first = kwargs.pop(
            "attn_diag_keep_first", attn_diag_keep_first
        )
        token_compression_enabled = kwargs.pop(
            "token_compression_enabled", token_compression_enabled
        )
        token_compression_mode = kwargs.pop(
            "token_compression_mode", token_compression_mode
        )
        token_compression_keep_ratio = kwargs.pop(
            "token_compression_keep_ratio", token_compression_keep_ratio
        )
        token_compression_keep_tokens = kwargs.pop(
            "token_compression_keep_tokens", token_compression_keep_tokens
        )
        token_compression_layer = kwargs.pop(
            "token_compression_layer", token_compression_layer
        )
        token_compression_layers = kwargs.pop(
            "token_compression_layers", token_compression_layers
        )
        token_compression_ref = kwargs.pop(
            "token_compression_ref", token_compression_ref
        )
        token_compression_preserve_first = kwargs.pop(
            "token_compression_preserve_first", token_compression_preserve_first
        )
        token_compression_preserve_last = kwargs.pop(
            "token_compression_preserve_last", token_compression_preserve_last
        )
        token_compression_min_context = kwargs.pop(
            "token_compression_min_context", token_compression_min_context
        )
        token_compression_parallel_weight = kwargs.pop(
            "token_compression_parallel_weight", token_compression_parallel_weight
        )
        sparse_attention_enabled = kwargs.pop(
            "sparse_attention_enabled", sparse_attention_enabled
        )
        sparse_attention_mode = kwargs.pop(
            "sparse_attention_mode", sparse_attention_mode
        )
        sparse_attention_keep_ratio = kwargs.pop(
            "sparse_attention_keep_ratio", sparse_attention_keep_ratio
        )
        sparse_attention_keep_tokens = kwargs.pop(
            "sparse_attention_keep_tokens", sparse_attention_keep_tokens
        )
        sparse_attention_sink_tokens = kwargs.pop(
            "sparse_attention_sink_tokens", sparse_attention_sink_tokens
        )
        sparse_attention_local_tokens = kwargs.pop(
            "sparse_attention_local_tokens", sparse_attention_local_tokens
        )
        sparse_attention_prefill_only = kwargs.pop(
            "sparse_attention_prefill_only", sparse_attention_prefill_only
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
        self.xsa_layer_scale_mode = str(xsa_layer_scale_mode).lower().strip()
        self.xsa_layer_scale_seed = int(xsa_layer_scale_seed)
        self.xsa_layer_para_scale_min = float(xsa_layer_para_scale_min)
        self.xsa_layer_para_scale_max = float(xsa_layer_para_scale_max)
        self.xsa_track_stats = _as_bool(xsa_track_stats)
        self.xsa_layerwise_stats = _as_bool(xsa_layerwise_stats)
        self.attn_diag_enabled = _as_bool(attn_diag_enabled)
        self.attn_diag_mode = str(attn_diag_mode).lower().strip()
        self.attn_diag_keep_first = _as_bool(attn_diag_keep_first)
        self.token_compression_config = TokenCompressionConfig(
            enabled=_as_bool(token_compression_enabled),
            mode=str(token_compression_mode).lower().strip(),
            keep_ratio=float(token_compression_keep_ratio),
            keep_tokens=int(token_compression_keep_tokens),
            layer=int(token_compression_layer),
            layers=str(token_compression_layers),
            ref=str(token_compression_ref).lower().strip(),
            preserve_first=int(token_compression_preserve_first),
            preserve_last=int(token_compression_preserve_last),
            min_context=int(token_compression_min_context),
            parallel_weight=float(token_compression_parallel_weight),
        )
        self.sparse_attention_enabled = _as_bool(sparse_attention_enabled)
        self.sparse_attention_mode = str(sparse_attention_mode).lower().strip()
        self.sparse_attention_keep_ratio = float(sparse_attention_keep_ratio)
        self.sparse_attention_keep_tokens = int(sparse_attention_keep_tokens)
        self.sparse_attention_sink_tokens = int(sparse_attention_sink_tokens)
        self.sparse_attention_local_tokens = int(sparse_attention_local_tokens)
        self.sparse_attention_prefill_only = _as_bool(sparse_attention_prefill_only)
        self.xsa_hooks = None
        self.attn_diag_hooks = None
        self.sparse_attention_hooks = None
        self.token_compressor = None
        super().__init__(*args, **kwargs)
        self._attach_xsa_if_needed()
        self._attach_attn_diag_if_needed()
        self._attach_sparse_attention_if_needed()
        self._attach_token_compressor_if_needed()

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
            xsa_layer_scale_mode=self.xsa_layer_scale_mode,
            xsa_layer_scale_seed=self.xsa_layer_scale_seed,
            xsa_layer_para_scale_min=self.xsa_layer_para_scale_min,
            xsa_layer_para_scale_max=self.xsa_layer_para_scale_max,
            track_stats=self.xsa_track_stats,
            track_layerwise_stats=self.xsa_layerwise_stats,
        )
        self.xsa_hooks.attach()
        eval_logger.info(
            "Attached XSA hooks: target=%s site=%s op=%s alpha=%s layer_scale=%s seed=%s range=[%s,%s] layers=[%s,%s) skip_first=%s skip_last=%s track_stats=%s layerwise=%s",
            self.xsa_target,
            self.xsa_intervention_site,
            self.xsa_forward_op,
            self.xsa_forward_alpha,
            self.xsa_layer_scale_mode,
            self.xsa_layer_scale_seed,
            self.xsa_layer_para_scale_min,
            self.xsa_layer_para_scale_max,
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

    def _attach_token_compressor_if_needed(self) -> None:
        cfg = self.token_compression_config
        if not cfg.enabled or cfg.keep_ratio >= 1.0 and cfg.keep_tokens <= 0:
            eval_logger.info("Prefill token compression disabled for this model instance.")
            return
        self.token_compressor = PrefillTokenCompressor(self.model, self.tokenizer, cfg)
        eval_logger.info(
            "Attached prefill token compressor: mode=%s keep_ratio=%s keep_tokens=%s layer=%s ref=%s preserve_first=%s preserve_last=%s min_context=%s",
            cfg.mode,
            cfg.keep_ratio,
            cfg.keep_tokens,
            cfg.layer,
            cfg.ref,
            cfg.preserve_first,
            cfg.preserve_last,
            cfg.min_context,
        )

    def _attach_sparse_attention_if_needed(self) -> None:
        if (
            not self.sparse_attention_enabled
            or self.sparse_attention_keep_ratio >= 1.0
            and self.sparse_attention_keep_tokens <= 0
        ):
            eval_logger.info("Sparse-attention patch disabled for this model instance.")
            return
        self.sparse_attention_hooks = SparseAttentionHooks(
            self.model,
            mode=self.sparse_attention_mode,
            keep_ratio=self.sparse_attention_keep_ratio,
            keep_tokens=self.sparse_attention_keep_tokens,
            sink_tokens=self.sparse_attention_sink_tokens,
            local_tokens=self.sparse_attention_local_tokens,
            prefill_only=self.sparse_attention_prefill_only,
            start_layer=self.xsa_start_layer,
            end_layer=self.xsa_end_layer,
            skip_first_n=self.xsa_skip_first_n,
            skip_last_n=self.xsa_skip_last_n,
        )
        self.sparse_attention_hooks.attach()
        eval_logger.info(
            "Attached sparse-attention patch: mode=%s keep_ratio=%s keep_tokens=%s sink=%s local=%s prefill_only=%s layers=[%s,%s) skip_first=%s skip_last=%s",
            self.sparse_attention_mode,
            self.sparse_attention_keep_ratio,
            self.sparse_attention_keep_tokens,
            self.sparse_attention_sink_tokens,
            self.sparse_attention_local_tokens,
            self.sparse_attention_prefill_only,
            self.xsa_start_layer,
            self.xsa_end_layer,
            self.xsa_skip_first_n,
            self.xsa_skip_last_n,
        )

    def _model_generate(self, context, max_length: int, stop: list[str], **generation_kwargs):
        if self.token_compressor is None:
            return super()._model_generate(context, max_length, stop, **generation_kwargs)
        original_context = context
        original_context_len = int(context.shape[1])
        max_new_tokens = max(0, int(max_length) - original_context_len)
        attention_mask = generation_kwargs.pop("attention_mask", None)
        compressed_context, compressed_mask, stats = self.token_compressor.compress(
            context,
            attention_mask=attention_mask,
        )
        compressed_max_length = int(compressed_context.shape[1]) + max_new_tokens
        eval_logger.info(
            "Prefill token compression: old_len=%.1f new_len=%.1f keep_ratio=%.3f max_new_tokens=%s",
            stats["old_len"],
            stats["new_len"],
            stats["keep_ratio"],
            max_new_tokens,
        )
        generated = super()._model_generate(
            compressed_context,
            compressed_max_length,
            stop,
            attention_mask=compressed_mask,
            **generation_kwargs,
        )
        continuation = generated[:, int(compressed_context.shape[1]) :]
        return torch.cat([original_context, continuation.to(original_context.device)], dim=1)

    def cleanup(self) -> None:
        if self.xsa_hooks is not None:
            self.xsa_hooks.close()
            self.xsa_hooks = None
        if self.attn_diag_hooks is not None:
            self.attn_diag_hooks.close()
            self.attn_diag_hooks = None
        if self.sparse_attention_hooks is not None:
            self.sparse_attention_hooks.close()
            self.sparse_attention_hooks = None
        self.token_compressor = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
