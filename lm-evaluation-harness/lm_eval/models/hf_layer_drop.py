from __future__ import annotations

import logging
from typing import Any

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.layer_drop_hooks import LayerDropHooks, load_selection_lists, parse_layer_list


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


@register_model("hf-layer-drop")
class HFLayerDropLM(HFLM):
    _LAYER_DROP_KWARG_KEYS = {
        "layer_drop_enabled",
        "layer_drop_attn",
        "layer_drop_mlp",
        "layer_drop_config",
        "layer_drop_component",
        "layer_drop_count",
    }

    def __init__(
        self,
        *args,
        layer_drop_enabled: bool = True,
        layer_drop_attn: str = "",
        layer_drop_mlp: str = "",
        layer_drop_config: str = "",
        layer_drop_component: str = "both",
        layer_drop_count: int = 0,
        **kwargs,
    ) -> None:
        layer_drop_enabled = kwargs.pop("layer_drop_enabled", layer_drop_enabled)
        layer_drop_attn = kwargs.pop("layer_drop_attn", layer_drop_attn)
        layer_drop_mlp = kwargs.pop("layer_drop_mlp", layer_drop_mlp)
        layer_drop_config = kwargs.pop("layer_drop_config", layer_drop_config)
        layer_drop_component = kwargs.pop("layer_drop_component", layer_drop_component)
        layer_drop_count = kwargs.pop("layer_drop_count", layer_drop_count)

        self.layer_drop_enabled = _as_bool(layer_drop_enabled)
        self.layer_drop_attn = str(layer_drop_attn)
        self.layer_drop_mlp = str(layer_drop_mlp)
        self.layer_drop_config = str(layer_drop_config)
        self.layer_drop_component = str(layer_drop_component).lower().strip()
        self.layer_drop_count = int(layer_drop_count)
        self.layer_drop_hooks = None
        super().__init__(*args, **kwargs)
        self._attach_layer_drop_if_needed()

    def _create_model(self, *args, **kwargs) -> None:
        for key in self._LAYER_DROP_KWARG_KEYS:
            kwargs.pop(key, None)
        super()._create_model(*args, **kwargs)

    def _attach_layer_drop_if_needed(self) -> None:
        if not self.layer_drop_enabled:
            eval_logger.info("Layer-drop hooks disabled for this model instance.")
            return
        drop_attn = parse_layer_list(self.layer_drop_attn)
        drop_mlp = parse_layer_list(self.layer_drop_mlp)
        if self.layer_drop_config and self.layer_drop_count > 0:
            cfg_attn, cfg_mlp = load_selection_lists(
                self.layer_drop_config,
                self.layer_drop_component,
                self.layer_drop_count,
            )
            drop_attn = drop_attn or cfg_attn
            drop_mlp = drop_mlp or cfg_mlp
        self.layer_drop_hooks = LayerDropHooks(self.model, drop_attn=drop_attn, drop_mlp=drop_mlp)
        self.layer_drop_hooks.attach()
        eval_logger.info("Attached layer-drop hooks: attn=%s mlp=%s", drop_attn, drop_mlp)

    def cleanup(self) -> None:
        if self.layer_drop_hooks is not None:
            self.layer_drop_hooks.close()
            self.layer_drop_hooks = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
