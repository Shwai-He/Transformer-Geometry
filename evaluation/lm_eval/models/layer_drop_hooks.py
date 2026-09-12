from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

eval_logger = logging.getLogger(__name__)


def parse_layer_list(text: str | None) -> list[int]:
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.replace("+", ",").replace(";", ",").split(",") if item.strip()]


def find_decoder_layers(model) -> list[torch.nn.Module]:
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise ValueError("Could not find decoder layers for layer-drop hooks.")


def load_selection_lists(config_path: str, component: str, drop_count: int) -> tuple[list[int], list[int]]:
    if not config_path:
        return [], []
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    recs = data.get("recommendations", {})
    key = f"drop_{int(drop_count)}"
    attn = recs.get("attn", {}).get(key, []) if component in {"attn", "both"} else []
    mlp = recs.get("mlp", {}).get(key, []) if component in {"mlp", "both"} else []
    return [int(x) for x in attn], [int(x) for x in mlp]


class LayerDropHooks:
    def __init__(self, model, *, drop_attn: list[int] | None = None, drop_mlp: list[int] | None = None):
        self.model = model
        self.layers = find_decoder_layers(model)
        self.drop_attn = set(int(x) for x in (drop_attn or []))
        self.drop_mlp = set(int(x) for x in (drop_mlp or []))
        self.handles = []

    @staticmethod
    def _zero_first_output(output):
        if isinstance(output, tuple):
            if not output:
                return output
            first = output[0]
            if torch.is_tensor(first):
                return (torch.zeros_like(first),) + output[1:]
            return output
        if torch.is_tensor(output):
            return torch.zeros_like(output)
        return output

    def attach(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.drop_attn:
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
                if attn is None:
                    eval_logger.warning("Requested attention drop for layer %s, but no attention module was found.", layer_idx)
                else:
                    self.handles.append(attn.register_forward_hook(lambda _m, _a, out: self._zero_first_output(out)))
            if layer_idx in self.drop_mlp:
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    eval_logger.warning("Requested MLP drop for layer %s, but no MLP module was found.", layer_idx)
                else:
                    self.handles.append(mlp.register_forward_hook(lambda _m, _a, out: self._zero_first_output(out)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
