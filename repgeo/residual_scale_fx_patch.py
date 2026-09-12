from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ResidualScaleFxConfig:
    mode: str = "sum"  # sum | param | wx
    param_init: float = 0.0  # alpha = 1 + p
    param_learnable: bool = True
    wx_init: float = 1.0  # used as base scale target for wx init
    wx_learnable: bool = True
    wx_use_bias: bool = False


def _new_branch_stats() -> Dict[str, Any]:
    return {
        "sum": 0.0,
        "count": 0,
        "layer_sum": {},
        "layer_count": {},
    }


def _unwrap_module(m: nn.Module) -> nn.Module:
    cur = m
    for _ in range(8):
        nxt = getattr(cur, "module", None)
        if nxt is None:
            break
        cur = nxt
    return cur


def _reset_scale_on_fx_runtime_stats(model: nn.Module) -> None:
    model = _unwrap_module(model)
    setattr(
        model,
        "_scale_fx_runtime_stats",
        {
            "attn": _new_branch_stats(),
            "mlp": _new_branch_stats(),
        },
    )


def _update_scale_on_fx_runtime_stats(
    model: nn.Module,
    branch: str,
    layer_idx: int,
    alpha: torch.Tensor,
) -> None:
    model = _unwrap_module(model)
    stats = getattr(model, "_scale_fx_runtime_stats", None)
    if not isinstance(stats, dict):
        _reset_scale_on_fx_runtime_stats(model)
        stats = getattr(model, "_scale_fx_runtime_stats")

    bstats = stats[branch]
    alpha_mean = float(alpha.detach().float().mean().item())
    bstats["sum"] += alpha_mean
    bstats["count"] += 1

    lsum = bstats["layer_sum"]
    lcount = bstats["layer_count"]
    lsum[layer_idx] = float(lsum.get(layer_idx, 0.0) + alpha_mean)
    lcount[layer_idx] = int(lcount.get(layer_idx, 0) + 1)


def _find_decoder_layers(model: nn.Module) -> Optional[List[nn.Module]]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "layers")
    ):
        return list(model.base_model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    return None


def _get_attn_mlp_modules(layer: nn.Module) -> tuple[Optional[nn.Module], Optional[nn.Module]]:
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
    else:
        attn = getattr(layer, "attn", None)
    mlp = getattr(layer, "mlp", None)
    return attn, mlp


def _extract_main_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_main_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_tensor, *output[1:])
    return new_tensor


def _ensure_scalar_param(layer: nn.Module, name: str, init_value: float, learnable: bool) -> None:
    old = getattr(layer, name, None)
    if isinstance(old, nn.Parameter):
        with torch.no_grad():
            old.fill_(float(init_value))
        old.requires_grad = learnable
        return
    if hasattr(layer, name):
        delattr(layer, name)
    setattr(
        layer,
        name,
        nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32), requires_grad=learnable),
    )


def _ensure_wx_linear(
    layer: nn.Module, name: str, hidden_size: int, base_scale: float, learnable: bool, use_bias: bool
) -> None:
    mod = getattr(layer, name, None)
    if not isinstance(mod, nn.Linear):
        if hasattr(layer, name):
            delattr(layer, name)
        mod = nn.Linear(hidden_size, 1, bias=use_bias)
        setattr(layer, name, mod)
    nn.init.zeros_(mod.weight)
    if mod.bias is not None:
        # near-linear inverse init: base_scale ~= 1 + silu(z), z ~= base_scale - 1
        nn.init.constant_(mod.bias, float(base_scale - 1.0))
    mod.weight.requires_grad = learnable
    if mod.bias is not None:
        mod.bias.requires_grad = learnable


def remove_scale_on_fx_patch(model: nn.Module) -> None:
    handles = getattr(model, "_scale_fx_hook_handles", None)
    if handles:
        for h in handles:
            h.remove()
    setattr(model, "_scale_fx_hook_handles", [])


def collect_scale_on_fx_stats(
    model: nn.Module,
    reset: bool = False,
    prefix: str = "residual",
) -> Dict[str, float]:
    model = _unwrap_module(model)
    stats = getattr(model, "_scale_fx_runtime_stats", None)
    if not isinstance(stats, dict):
        return {}

    out: Dict[str, float] = {}
    for branch in ("attn", "mlp"):
        bstats = stats.get(branch)
        if not isinstance(bstats, dict):
            continue
        total = int(bstats.get("count", 0))
        if total > 0:
            out[f"{prefix}/{branch}_alpha_mean"] = float(bstats["sum"] / total)

        layer_sum = bstats.get("layer_sum", {})
        layer_count = bstats.get("layer_count", {})
        if isinstance(layer_sum, dict) and isinstance(layer_count, dict):
            for lid in sorted(layer_sum.keys()):
                cnt = int(layer_count.get(lid, 0))
                if cnt > 0:
                    out[f"{prefix}/{branch}_alpha_layer_{lid}"] = float(layer_sum[lid] / cnt)

    if reset:
        _reset_scale_on_fx_runtime_stats(model)
    return out


def log_scale_on_fx_stats_to_wandb(
    model: nn.Module,
    step: Optional[int] = None,
    reset: bool = True,
    prefix: str = "residual",
    commit: bool = False,
) -> Dict[str, float]:
    payload = collect_scale_on_fx_stats(model, reset=reset, prefix=prefix)
    if not payload:
        return {}
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            if step is None:
                wandb.log(payload, commit=commit)
            else:
                wandb.log(payload, step=step, commit=commit)
    except Exception:
        pass
    return payload


def apply_scale_on_fx_patch(model: nn.Module, config: ResidualScaleFxConfig) -> bool:
    model = _unwrap_module(model)
    mode = str(config.mode).lower()
    if mode not in {"sum", "param", "wx"}:
        raise ValueError(f"Unsupported mode: {mode}. Choose from ['sum', 'param', 'wx'].")

    remove_scale_on_fx_patch(model)
    if mode == "sum":
        return True

    layers = _find_decoder_layers(model)
    if not layers:
        return False

    _reset_scale_on_fx_runtime_stats(model)

    handles = []
    patched = 0

    for layer_idx, layer in enumerate(layers):
        attn_mod, mlp_mod = _get_attn_mlp_modules(layer)
        if attn_mod is None or mlp_mod is None:
            continue

        if mode == "param":
            # alpha = 1 + p; default p=0 aligns with baseline.
            _ensure_scalar_param(layer, "residual_attn_param", float(config.param_init), bool(config.param_learnable))
            _ensure_scalar_param(layer, "residual_mlp_param", float(config.param_init), bool(config.param_learnable))
            for stale in ("residual_attn_wx", "residual_mlp_wx"):
                if hasattr(layer, stale):
                    delattr(layer, stale)
        else:
            hidden_size = int(getattr(layer, "hidden_size", 0) or getattr(getattr(model, "config", None), "hidden_size", 0))
            if hidden_size <= 0:
                continue
            _ensure_wx_linear(
                layer,
                "residual_attn_wx",
                hidden_size=hidden_size,
                base_scale=float(config.wx_init),
                learnable=bool(config.wx_learnable),
                use_bias=bool(config.wx_use_bias),
            )
            _ensure_wx_linear(
                layer,
                "residual_mlp_wx",
                hidden_size=hidden_size,
                base_scale=float(config.wx_init),
                learnable=bool(config.wx_learnable),
                use_bias=bool(config.wx_use_bias),
            )
            for stale in ("residual_attn_param", "residual_mlp_param"):
                if hasattr(layer, stale):
                    delattr(layer, stale)

        def _make_hook(branch: str, current_layer: nn.Module, current_layer_idx: int):
            def _hook(_module, module_inputs, output):
                main = _extract_main_tensor(output)
                if mode == "param":
                    p = getattr(current_layer, f"residual_{branch}_param")
                    alpha = 1.0 + p.to(device=main.device, dtype=main.dtype)
                else:
                    x = module_inputs[0] if module_inputs else main
                    wx = getattr(current_layer, f"residual_{branch}_wx")
                    alpha = 1.0 + F.silu(wx(x)).to(device=main.device, dtype=main.dtype)
                _update_scale_on_fx_runtime_stats(model, branch, current_layer_idx, alpha)
                scaled = main * alpha
                return _replace_main_tensor(output, scaled)

            return _hook

        handles.append(attn_mod.register_forward_hook(_make_hook("attn", layer, layer_idx)))
        handles.append(mlp_mod.register_forward_hook(_make_hook("mlp", layer, layer_idx)))
        patched += 1

    setattr(model, "_scale_fx_hook_handles", handles)
    return patched > 0
