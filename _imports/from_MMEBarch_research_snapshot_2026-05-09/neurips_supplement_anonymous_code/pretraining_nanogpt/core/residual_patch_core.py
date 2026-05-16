from __future__ import annotations

import inspect
import math
from types import MethodType
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

_RESIDUAL_ATTRS = (
    "attn_residual_scale",
    "mlp_residual_scale",
    "attn_residual_wx",
    "mlp_residual_wx",
    "residual_attn_scale",
    "residual_mlp_scale",
    "residual_attn_wx",
    "residual_mlp_wx",
    "residual_attn_branch_scale",
    "residual_mlp_branch_scale",
    "residual_attn_branch_wx",
    "residual_mlp_branch_wx",
)


def _is_compiling() -> bool:
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            return bool(torch.compiler.is_compiling())
    except Exception:
        pass
    try:
        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


def _print_residual_param_summary(model: nn.Module, mode: str) -> None:
    rows = []
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        if ("residual" not in name) and ("wx" not in name):
            continue
        t = p.detach().float()
        rows.append(
            (
                name,
                tuple(t.shape),
                bool(p.requires_grad),
                float(t.mean().item()),
                float(t.std().item()) if t.numel() > 1 else 0.0,
                float(t.min().item()),
                float(t.max().item()),
                float(t.reshape(-1)[0].item()),
                int(p.numel()),
            )
        )

    print("[INFO] residual_patch_summary:")
    print(f"  mode={mode}")
    print(f"  residual_param_count={len(rows)}")
    print(f"  trainable/total={trainable}/{max(total, 1)}")
    if not rows:
        print("  residual params: none")
        return
    for name, shape, req_grad, mean_v, std_v, min_v, max_v, first_v, n in rows:
        print(
            "  "
            f"{name} shape={shape} requires_grad={req_grad} numel={n} "
            f"mean={mean_v:.6f} std={std_v:.6f} min={min_v:.6f} max={max_v:.6f} first={first_v:.6f}"
        )


def _set_residual_penalty_lambda(model: nn.Module, value: float) -> None:
    setattr(model, "_residual_penalty_lambda", float(value))
    if hasattr(model, "config"):
        setattr(model.config, "residual_penalty_lambda", float(value))


def _reset_residual_penalty_state(model: nn.Module) -> None:
    setattr(model, "_residual_alpha_penalty_sum", None)
    setattr(model, "_residual_alpha_penalty_count", 0)
    setattr(model, "_residual_alpha_penalty", None)
    # Backward-compatible attr used in trainer code.
    setattr(model, "_residual_ratio_penalty", None)
    setattr(model, "_residual_alpha_stat", {"attn_sum": 0.0, "mlp_sum": 0.0, "attn_n": 0, "mlp_n": 0})
    setattr(model, "_residual_alpha_runtime_stats", None)
    setattr(model, "_residual_alpha_layer_stat", {"attn": {}, "mlp": {}})
    setattr(model, "_residual_alpha_layer_runtime_stats", None)


def _accumulate_alpha_penalty(model: nn.Module, alpha: torch.Tensor, branch: str, layer_idx: int) -> None:
    penalty = alpha.float().pow(2).mean()
    cur = getattr(model, "_residual_alpha_penalty_sum", None)
    if cur is None:
        setattr(model, "_residual_alpha_penalty_sum", penalty)
    else:
        setattr(model, "_residual_alpha_penalty_sum", cur + penalty)
    setattr(model, "_residual_alpha_penalty_count", int(getattr(model, "_residual_alpha_penalty_count", 0)) + 1)

    # Avoid Python-side stats updates during torch.compile graph capture.
    if _is_compiling():
        return

    stat = getattr(model, "_residual_alpha_stat", None)
    if not isinstance(stat, dict):
        stat = {"attn_sum": 0.0, "mlp_sum": 0.0, "attn_n": 0, "mlp_n": 0}
    a_mean = float(alpha.detach().float().mean().item())
    if branch == "attn":
        stat["attn_sum"] = float(stat.get("attn_sum", 0.0)) + a_mean
        stat["attn_n"] = int(stat.get("attn_n", 0)) + 1
    else:
        stat["mlp_sum"] = float(stat.get("mlp_sum", 0.0)) + a_mean
        stat["mlp_n"] = int(stat.get("mlp_n", 0)) + 1
    setattr(model, "_residual_alpha_stat", stat)

    layer_stat = getattr(model, "_residual_alpha_layer_stat", None)
    if not isinstance(layer_stat, dict):
        layer_stat = {"attn": {}, "mlp": {}}
    branch_dict = layer_stat.get(branch)
    if not isinstance(branch_dict, dict):
        branch_dict = {}
    cur = branch_dict.get(int(layer_idx), {"sum": 0.0, "n": 0})
    cur["sum"] = float(cur.get("sum", 0.0)) + a_mean
    cur["n"] = int(cur.get("n", 0)) + 1
    branch_dict[int(layer_idx)] = cur
    layer_stat[branch] = branch_dict
    setattr(model, "_residual_alpha_layer_stat", layer_stat)


def _proj_stats(y: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _remove_parallel_component(y: torch.Tensor, ref: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Remove the component of y parallel to ref on the last dimension."""
    if strength == 0.0:
        return y
    y_f = y.float()
    r_f = ref.float()
    dot = (y_f * r_f).sum(dim=-1, keepdim=True)
    r_sq = (r_f * r_f).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    proj = (dot / r_sq) * r_f
    return (y_f - float(strength) * proj).to(dtype=y.dtype)


def _xsa_forward_enabled(config: Any) -> bool:
    target = str(getattr(config, "xsa_forward_target", "none")).lower()
    strength = float(getattr(config, "xsa_forward_strength", 1.0) or 0.0)
    return target in {"attn", "mlp", "both"} and strength != 0.0


def _attn_diag_enabled(config: Any) -> bool:
    mode = str(getattr(config, "attn_diag_mode", "none")).lower()
    return mode == "hard"


def _apply_attn_diag_hard_mask(
    attention_mask: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    cache_position: Optional[torch.LongTensor] = None,
    keep_first: bool = True,
) -> torch.Tensor:
    """Block each query from attending to the key at the same absolute position."""
    bsz, q_len = hidden_states.shape[:2]
    device = hidden_states.device
    dtype = hidden_states.dtype

    if attention_mask is not None:
        k_len = int(attention_mask.shape[-1])
        base_mask = attention_mask
    else:
        k_len = int(q_len)
        base_mask = torch.zeros((bsz, 1, q_len, k_len), device=device, dtype=dtype)

    if cache_position is not None:
        q_pos = cache_position.to(device=device)
        if q_pos.numel() != q_len:
            q_pos = q_pos.reshape(-1)[-q_len:]
    else:
        q_pos = torch.arange(k_len - q_len, k_len, device=device)

    k_pos = torch.arange(k_len, device=device)
    diag = q_pos.reshape(q_len, 1).eq(k_pos.reshape(1, k_len))
    if keep_first:
        diag = diag & q_pos.reshape(q_len, 1).ne(0)

    neg = torch.finfo(base_mask.dtype if base_mask.is_floating_point() else dtype).min
    diag_mask = torch.zeros((1, 1, q_len, k_len), device=device, dtype=(base_mask.dtype if base_mask.is_floating_point() else dtype))
    diag_mask = diag_mask.masked_fill(diag.reshape(1, 1, q_len, k_len), neg)

    if base_mask.dtype == torch.bool:
        # Bool masks are keep/drop masks in some backends; convert to additive for compatibility.
        additive = torch.zeros(base_mask.shape, device=base_mask.device, dtype=dtype)
        additive = additive.masked_fill(~base_mask, torch.finfo(dtype).min)
        return additive + diag_mask.to(dtype=dtype)
    return base_mask + diag_mask.to(dtype=base_mask.dtype)


def _xsa_branch_enabled(layer: nn.Module, branch: str) -> bool:
    target = str(getattr(layer, "_xsa_forward_target", "none")).lower()
    return target == "both" or target == branch


def _state_cos(x: torch.Tensor, x_plus_f: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    y_f = x_plus_f.float()
    return F.cosine_similarity(x_f, y_f, dim=-1).mean()


def _finalize_residual_penalty_state(model: nn.Module, output: Any) -> None:
    total = getattr(model, "_residual_alpha_penalty_sum", None)
    count = int(getattr(model, "_residual_alpha_penalty_count", 0))
    if total is not None and count > 0:
        penalty = total / float(count)
        setattr(model, "_residual_alpha_penalty", penalty)
        setattr(model, "_residual_ratio_penalty", penalty)
        if _is_compiling():
            setattr(model, "_residual_alpha_runtime_stats", None)
            setattr(model, "_residual_alpha_layer_runtime_stats", None)
            return
        stat = getattr(model, "_residual_alpha_stat", None) or {}
        attn_n = int(stat.get("attn_n", 0))
        mlp_n = int(stat.get("mlp_n", 0))
        attn_mean = float(stat.get("attn_sum", 0.0)) / max(attn_n, 1)
        mlp_mean = float(stat.get("mlp_sum", 0.0)) / max(mlp_n, 1)
        setattr(
            model,
            "_residual_alpha_runtime_stats",
            {
                "attn_alpha_mean": attn_mean,
                "mlp_alpha_mean": mlp_mean,
                "alpha_mean": (attn_mean + mlp_mean) / 2.0,
            },
        )
        layer_stat = getattr(model, "_residual_alpha_layer_stat", None) or {}
        out_layer = {"attn": {}, "mlp": {}}
        for branch in ("attn", "mlp"):
            bd = layer_stat.get(branch, {})
            if isinstance(bd, dict):
                for lid, v in bd.items():
                    n = int(v.get("n", 0))
                    out_layer[branch][int(lid)] = float(v.get("sum", 0.0)) / max(n, 1)
        setattr(model, "_residual_alpha_layer_runtime_stats", out_layer)
        return

    dev = None
    if isinstance(output, tuple) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            dev = first.device
    elif isinstance(output, torch.Tensor):
        dev = output.device
    if dev is None:
        dev = torch.device("cpu")

    zero = torch.tensor(0.0, device=dev)
    setattr(model, "_residual_alpha_penalty", zero)
    setattr(model, "_residual_ratio_penalty", zero)
    if _is_compiling():
        setattr(model, "_residual_alpha_runtime_stats", None)
        setattr(model, "_residual_alpha_layer_runtime_stats", None)
        return
    setattr(
        model,
        "_residual_alpha_runtime_stats",
        {
            "attn_alpha_mean": 0.0,
            "mlp_alpha_mean": 0.0,
            "alpha_mean": 0.0,
        },
    )
    setattr(model, "_residual_alpha_layer_runtime_stats", {"attn": {}, "mlp": {}})


def _find_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return list(model.base_model.model.layers)
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "model")
        and hasattr(model.base_model.model.model, "layers")
    ):
        return list(model.base_model.model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    return None


def _extract_main_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _resolve_mode_from_config(config) -> str:
    raw_mode = getattr(config, "residual_mode", None)
    if raw_mode is None:
        raw_mode = getattr(config, "residual_scale_mode", "param")
    mode = str(raw_mode).lower()
    if mode == "parameter":
        mode = "param"
    if mode not in {"sum", "wx", "param"}:
        raise ValueError(f"Unsupported residual mode: {mode}. Choose from ['sum', 'wx', 'param', 'parameter'].")
    return mode


def _resolve_wx_activation(config) -> str:
    act = str(getattr(config, "residual_wx_activation", "silu")).lower()
    if act not in {"silu", "exp", "sigmoid2"}:
        raise ValueError(f"Unsupported residual_wx_activation: {act}. Choose from ['silu', 'exp', 'sigmoid2'].")
    return act


def _set_wx_linear(
    layer: nn.Module,
    attr_name: str,
    hidden_size: int,
    base_scale: float,
    learnable: bool,
    activation: str,
) -> None:
    mod = getattr(layer, attr_name, None)
    if not isinstance(mod, nn.Linear):
        if hasattr(layer, attr_name):
            delattr(layer, attr_name)
        # Save/restore RNG state: nn.Linear.__init__ calls reset_parameters() (kaiming_uniform_)
        # which advances the global RNG. Since we immediately override all values below, the
        # random init is discarded — but the RNG advance would silently misalign dropout masks
        # between patched and baseline runs.
        _rng_cpu = torch.get_rng_state()
        _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        mod = nn.Linear(hidden_size, 1, bias=True)
        torch.set_rng_state(_rng_cpu)
        if _rng_cuda:
            torch.cuda.set_rng_state_all(_rng_cuda)
        setattr(layer, attr_name, mod)
    # Initialize so x=0 gives alpha ~= base_scale.
    nn.init.zeros_(mod.weight)
    if activation == "exp":
        if base_scale <= 0:
            raise ValueError("residual_wx must be > 0 when residual_wx_activation='exp'.")
        nn.init.constant_(mod.bias, float(torch.log(torch.tensor(base_scale)).item()))
    elif activation == "sigmoid2":
        # alpha(x) = 2 * sigmoid(Wx+b), so for x=0 target alpha=base_scale:
        # b = logit(base_scale / 2)
        v = float(base_scale) / 2.0
        eps = 1e-6
        v = min(max(v, eps), 1.0 - eps)
        b = math.log(v / (1.0 - v))
        nn.init.constant_(mod.bias, b)
    else:
        nn.init.constant_(mod.bias, float(base_scale - 1.0))
    mod.weight.requires_grad = learnable
    mod.bias.requires_grad = learnable


def _set_scale_parameter_or_buffer(layer: nn.Module, attr_name: str, base_scale: float, learnable: bool) -> None:
    existing = getattr(layer, attr_name, None)

    if learnable:
        if isinstance(existing, nn.Parameter):
            with torch.no_grad():
                existing.data.fill_(base_scale)
        else:
            if hasattr(layer, attr_name):
                delattr(layer, attr_name)
            setattr(layer, attr_name, nn.Parameter(torch.tensor([base_scale], dtype=torch.float32)))
        return

    if isinstance(existing, nn.Parameter):
        delattr(layer, attr_name)

    if attr_name in layer._buffers:
        layer._buffers[attr_name] = torch.tensor([base_scale], dtype=torch.float32)
    else:
        layer.register_buffer(attr_name, torch.tensor([base_scale], dtype=torch.float32), persistent=False)


def _compute_alpha(layer: nn.Module, branch: str, x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "sum":
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    if mode == "wx":
        wx = layer.residual_attn_wx if branch == "attn" else layer.residual_mlp_wx
        act = str(getattr(layer, "_residual_wx_activation", "silu")).lower()
        z = wx(x)
        if act == "exp":
            return torch.exp(z)
        if act == "sigmoid2":
            return 2.0 * torch.sigmoid(z)
        return 1.0 + F.silu(z)

    scale = layer.residual_attn_scale if branch == "attn" else layer.residual_mlp_scale
    return scale.to(device=x.device, dtype=x.dtype)


def _compute_branch_gamma(layer: nn.Module, branch: str, x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "sum":
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    if not bool(getattr(layer, "_residual_branch_gate", False)):
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    if mode == "wx":
        wx = layer.residual_attn_branch_wx if branch == "attn" else layer.residual_mlp_branch_wx
        act = str(getattr(layer, "_residual_wx_activation", "silu")).lower()
        z = wx(x)
        if act == "exp":
            return torch.exp(z)
        if act == "sigmoid2":
            return 2.0 * torch.sigmoid(z)
        return 1.0 + F.silu(z)
    scale = layer.residual_attn_branch_scale if branch == "attn" else layer.residual_mlp_branch_scale
    return scale.to(device=x.device, dtype=x.dtype)


def _layer_returns_tensor(layer: nn.Module) -> bool:
    # Dropped-Llama custom layer returns `hidden_states` tensor directly.
    if hasattr(layer, "drop_attn") or hasattr(layer, "drop_mlp"):
        return True
    try:
        src = inspect.getsource(layer.forward)
        # nanoGPT/Qwen-style decoder blocks often return a single tensor under
        # a local name such as `x`, while HF-style layers build `outputs = (...)`.
        if (
            ("return hidden_states" in src or "return x" in src)
            and "outputs = (hidden_states" not in src
            and "outputs = (x" not in src
        ):
            return True
    except Exception:
        pass
    return False


def fix_scalar_residual_params(model: nn.Module) -> int:
    """Upgrade scalar residual parameters to shape [1] for FSDP compatibility."""
    fixed = 0
    for module in model.modules():
        for attr in _RESIDUAL_ATTRS:
            if not hasattr(module, attr):
                continue
            obj = getattr(module, attr)
            if isinstance(obj, nn.Parameter) and obj.ndim == 0:
                setattr(module, attr, nn.Parameter(obj.detach().reshape(1), requires_grad=obj.requires_grad))
                fixed += 1
    return fixed


def _make_llama_qwen_forward(layer: nn.Module, model: nn.Module, mode: str):
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        returns_tensor = bool(getattr(self, "_residual_returns_tensor", False))
        attn_mod = getattr(self, "self_attn", None)
        mlp_mod = getattr(self, "mlp", None)
        input_ln = getattr(self, "input_layernorm", None)
        post_ln = getattr(self, "post_attention_layernorm", None)

        attn_rest: Tuple[Any, ...] = tuple()
        router_logits = None

        if attn_mod is not None and input_ln is not None:
            residual = hidden_states
            attn_input = input_ln(hidden_states)
            attn_attention_mask = attention_mask
            if bool(getattr(self, "_attn_diag_hard", False)):
                attn_attention_mask = _apply_attn_diag_hard_mask(
                    attention_mask,
                    attn_input,
                    cache_position=cache_position,
                    keep_first=bool(getattr(self, "_attn_diag_keep_first", True)),
                )
            attn_kwargs = dict(kwargs)
            attn_kwargs.update(
                {
                    "hidden_states": attn_input,
                    "attention_mask": attn_attention_mask,
                    "position_ids": position_ids,
                    "use_cache": use_cache,
                    "cache_position": cache_position,
                    "position_embeddings": position_embeddings,
                }
            )
            # Support both HF naming variants.
            sig = inspect.signature(attn_mod.forward)
            if "past_key_values" in sig.parameters:
                attn_kwargs["past_key_values"] = past_key_value
            else:
                attn_kwargs["past_key_value"] = past_key_value
            if "output_attentions" in sig.parameters:
                attn_kwargs["output_attentions"] = output_attentions
            attn_out = attn_mod(**attn_kwargs)
            attn_hidden = _extract_main_tensor(attn_out)
            attn_rest = tuple(attn_out[1:]) if isinstance(attn_out, tuple) else tuple()
            if bool(getattr(self, "track_xsa_self_metric", False)):
                ratio, ratio_detach, proj_sq, y_sq, x_sq = _proj_stats(attn_hidden, residual)
                self._xsa_attn_proj_ratio = ratio
                self._xsa_attn_proj_ratio_detach = ratio_detach
                self._xsa_attn_proj_sq = proj_sq
                self._xsa_attn_y_sq = y_sq
                self._xsa_attn_x_sq = x_sq
                self._xsa_attn_state_cos = _state_cos(residual, residual + attn_hidden)
            else:
                self._xsa_attn_proj_ratio = None
                self._xsa_attn_proj_ratio_detach = None
                self._xsa_attn_proj_sq = None
                self._xsa_attn_y_sq = None
                self._xsa_attn_x_sq = None
                self._xsa_attn_state_cos = None
            if _xsa_branch_enabled(self, "attn"):
                attn_hidden = _remove_parallel_component(
                    attn_hidden,
                    residual,
                    float(getattr(self, "_xsa_forward_strength", 1.0)),
                )
            attn_alpha = _compute_alpha(self, "attn", residual, mode).to(device=residual.device, dtype=residual.dtype)
            attn_gamma = _compute_branch_gamma(self, "attn", residual, mode).to(device=residual.device, dtype=residual.dtype)
            hidden_states = residual * attn_alpha + attn_hidden * attn_gamma
            if not _is_compiling():
                self._residual_last_attn_alpha_mean = float(attn_alpha.detach().float().mean().item())
                self._residual_last_attn_branch_gamma_mean = float(attn_gamma.detach().float().mean().item())
            _accumulate_alpha_penalty(model, attn_alpha, "attn", int(getattr(self, "_residual_layer_idx", -1)))

        if mlp_mod is not None and post_ln is not None:
            residual = hidden_states
            mlp_input = post_ln(hidden_states)
            mlp_out = mlp_mod(mlp_input)
            if isinstance(mlp_out, tuple):
                mlp_hidden, router_logits = mlp_out
            else:
                mlp_hidden, router_logits = mlp_out, None

            if bool(getattr(self, "track_xsa_self_metric", False)):
                ratio, ratio_detach, proj_sq, y_sq, x_sq = _proj_stats(mlp_hidden, residual)
                self._xsa_mlp_proj_ratio = ratio
                self._xsa_mlp_proj_ratio_detach = ratio_detach
                self._xsa_mlp_proj_sq = proj_sq
                self._xsa_mlp_y_sq = y_sq
                self._xsa_mlp_x_sq = x_sq
                self._xsa_mlp_state_cos = _state_cos(residual, residual + mlp_hidden)
            else:
                self._xsa_mlp_proj_ratio = None
                self._xsa_mlp_proj_ratio_detach = None
                self._xsa_mlp_proj_sq = None
                self._xsa_mlp_y_sq = None
                self._xsa_mlp_x_sq = None
                self._xsa_mlp_state_cos = None
            if _xsa_branch_enabled(self, "mlp"):
                mlp_hidden = _remove_parallel_component(
                    mlp_hidden,
                    residual,
                    float(getattr(self, "_xsa_forward_strength", 1.0)),
                )
            mlp_alpha = _compute_alpha(self, "mlp", residual, mode).to(device=residual.device, dtype=residual.dtype)
            mlp_gamma = _compute_branch_gamma(self, "mlp", residual, mode).to(device=residual.device, dtype=residual.dtype)
            hidden_states = residual * mlp_alpha + mlp_hidden * mlp_gamma
            if not _is_compiling():
                self._residual_last_mlp_alpha_mean = float(mlp_alpha.detach().float().mean().item())
                self._residual_last_mlp_branch_gamma_mean = float(mlp_gamma.detach().float().mean().item())
            _accumulate_alpha_penalty(model, mlp_alpha, "mlp", int(getattr(self, "_residual_layer_idx", -1)))

        if returns_tensor:
            return hidden_states

        outputs = (hidden_states,)
        if output_attentions and len(attn_rest) > 0:
            outputs += (attn_rest[0],)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs

    return patched_forward


def _make_gpt2_forward(layer: nn.Module, model: nn.Module, mode: str):
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        returns_tensor = bool(getattr(self, "_residual_returns_tensor", False))
        if isinstance(hidden_states, tuple):
            hidden_states = _extract_main_tensor(hidden_states)
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_attention_mask = attention_mask
        if bool(getattr(self, "_attn_diag_hard", False)):
            attn_attention_mask = _apply_attn_diag_hard_mask(
                attention_mask,
                hidden_states,
                cache_position=None,
                keep_first=bool(getattr(self, "_attn_diag_keep_first", True)),
            )
        attn_sig = inspect.signature(self.attn.forward)
        attn_params = attn_sig.parameters
        if "layer_past" in attn_params or "attention_mask" in attn_params:
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attn_attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        else:
            # nanoGPT-style attention forward(x, output_attentions=...) -> tensor or tuple
            if "output_attentions" in attn_params:
                attn_outputs = self.attn(hidden_states, output_attentions=output_attentions)
            else:
                attn_outputs = self.attn(hidden_states)

        if isinstance(attn_outputs, tuple):
            attn_hidden = attn_outputs[0]
            outputs = attn_outputs[1:]
        else:
            attn_hidden = attn_outputs
            outputs = tuple()
        if bool(getattr(self, "track_xsa_self_metric", False)):
            ratio, ratio_detach, proj_sq, y_sq, x_sq = _proj_stats(attn_hidden, residual)
            self._xsa_attn_proj_ratio = ratio
            self._xsa_attn_proj_ratio_detach = ratio_detach
            self._xsa_attn_proj_sq = proj_sq
            self._xsa_attn_y_sq = y_sq
            self._xsa_attn_x_sq = x_sq
            self._xsa_attn_state_cos = _state_cos(residual, residual + attn_hidden)
        else:
            self._xsa_attn_proj_ratio = None
            self._xsa_attn_proj_ratio_detach = None
            self._xsa_attn_proj_sq = None
            self._xsa_attn_y_sq = None
            self._xsa_attn_x_sq = None
            self._xsa_attn_state_cos = None
        if _xsa_branch_enabled(self, "attn"):
            attn_hidden = _remove_parallel_component(
                attn_hidden,
                residual,
                float(getattr(self, "_xsa_forward_strength", 1.0)),
            )
        alpha = _compute_alpha(self, "attn", residual, mode).to(device=residual.device, dtype=residual.dtype)
        gamma = _compute_branch_gamma(self, "attn", residual, mode).to(device=residual.device, dtype=residual.dtype)
        hidden_states = residual * alpha + attn_hidden * gamma
        if not _is_compiling():
            self._residual_last_attn_alpha_mean = float(alpha.detach().float().mean().item())
            self._residual_last_attn_branch_gamma_mean = float(gamma.detach().float().mean().item())
        _accumulate_alpha_penalty(model, alpha, "attn", int(getattr(self, "_residual_layer_idx", -1)))

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = residual + cross_attn_outputs[0]
            outputs = outputs + cross_attn_outputs[2:]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_hidden = self.mlp(hidden_states)
        if bool(getattr(self, "track_xsa_self_metric", False)):
            ratio, ratio_detach, proj_sq, y_sq, x_sq = _proj_stats(mlp_hidden, residual)
            self._xsa_mlp_state_cos = _state_cos(residual, residual + mlp_hidden)
            self._xsa_mlp_proj_ratio = ratio
            self._xsa_mlp_proj_ratio_detach = ratio_detach
            self._xsa_mlp_proj_sq = proj_sq
            self._xsa_mlp_y_sq = y_sq
            self._xsa_mlp_x_sq = x_sq
        else:
            self._xsa_mlp_state_cos = None
            self._xsa_mlp_proj_ratio = None
            self._xsa_mlp_proj_ratio_detach = None
            self._xsa_mlp_proj_sq = None
            self._xsa_mlp_y_sq = None
            self._xsa_mlp_x_sq = None
        if _xsa_branch_enabled(self, "mlp"):
            mlp_hidden = _remove_parallel_component(
                mlp_hidden,
                residual,
                float(getattr(self, "_xsa_forward_strength", 1.0)),
            )
        alpha = _compute_alpha(self, "mlp", residual, mode).to(device=residual.device, dtype=residual.dtype)
        gamma = _compute_branch_gamma(self, "mlp", residual, mode).to(device=residual.device, dtype=residual.dtype)
        hidden_states = residual * alpha + mlp_hidden * gamma
        if not _is_compiling():
            self._residual_last_mlp_alpha_mean = float(alpha.detach().float().mean().item())
            self._residual_last_mlp_branch_gamma_mean = float(gamma.detach().float().mean().item())
        _accumulate_alpha_penalty(model, alpha, "mlp", int(getattr(self, "_residual_layer_idx", -1)))

        # Keep tensor-returning layers tensor-returning even when attention probs are requested.
        # nanoGPT callers read block.attn._last_attn_probs directly and expect block(x) -> Tensor.
        if returns_tensor:
            return hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            if len(outputs) > 0:
                outputs = (hidden_states,) + outputs[1:]
            else:
                outputs = (hidden_states,)
        return outputs

    return patched_forward


def _clear_residual_patch(model: nn.Module) -> None:
    # Remove handles
    for attr in ("_residual_patch_hook_handles", "_residual_penalty_state_hook_handles", "_residual_scaling_hook_handles"):
        handles = getattr(model, attr, None)
        if handles:
            for h in handles:
                h.remove()
        setattr(model, attr, [])

    # Restore original forward if previously patched
    layers = _find_decoder_layers(model) or []
    for layer in layers:
        orig = getattr(layer, "_residual_original_forward", None)
        if orig is not None:
            layer.forward = orig
            delattr(layer, "_residual_original_forward")

    _reset_residual_penalty_state(model)


def apply_qwen3_forward_patch(model, config) -> bool:
    """Patch to trunk-scaling form: y = alpha * x + f(x)."""
    print("[DEBUG][residual-patch] model before patch:")
    print(model)
    layers = _find_decoder_layers(model)
    if layers is None:
        return False

    _clear_residual_patch(model)

    mode = _resolve_mode_from_config(config)
    xsa_forward_target = str(getattr(config, "xsa_forward_target", "none")).lower()
    xsa_forward_strength = float(getattr(config, "xsa_forward_strength", 1.0) or 0.0)
    if xsa_forward_target not in {"none", "attn", "mlp", "both"}:
        raise ValueError("xsa_forward_target must be one of: none, attn, mlp, both.")
    xsa_enabled = _xsa_forward_enabled(config)
    attn_diag_mode = str(getattr(config, "attn_diag_mode", "none")).lower()
    if attn_diag_mode not in {"none", "hard"}:
        raise ValueError("attn_diag_mode must be one of: none, hard.")
    attn_diag_hard = _attn_diag_enabled(config)
    attn_diag_keep_first = bool(getattr(config, "attn_diag_keep_first", True))
    if mode == "sum" and not xsa_enabled and not attn_diag_hard:
        return False

    residual_penalty_lambda = float(getattr(config, "residual_penalty_lambda", 0.0) or 0.0)
    if residual_penalty_lambda < 0:
        raise ValueError("residual_penalty_lambda must be >= 0.")
    _set_residual_penalty_lambda(model, residual_penalty_lambda)

    param_learnable = bool(getattr(config, "residual_scale_learnable", False))
    wx_learnable = bool(getattr(config, "residual_wx_learnable", param_learnable))
    branch_gate = bool(getattr(config, "residual_branch_gate", False))
    branch_scale_learnable = bool(getattr(config, "residual_branch_scale_learnable", param_learnable))
    branch_wx_learnable = bool(getattr(config, "residual_branch_wx_learnable", wx_learnable))
    wx_activation = _resolve_wx_activation(config) if mode == "wx" else "silu"
    if mode == "wx" and not wx_learnable:
        wx_learnable = True

    num_layers = len(layers)
    denom = max(num_layers - 1, 1)
    post_norm_start_layer = int(getattr(config, "post_norm_start_layer", 10**9))
    use_post_norm = bool(getattr(config, "use_post_norm", False))

    patched_layers = 0

    for idx, layer in enumerate(layers):
        has_attn_branch = (getattr(layer, "self_attn", None) is not None) and hasattr(layer, "input_layernorm")
        has_mlp_branch = (getattr(layer, "mlp", None) is not None) and hasattr(layer, "post_attention_layernorm")
        has_qwen_llama = has_attn_branch or has_mlp_branch
        has_gpt2 = all(hasattr(layer, n) for n in ("attn", "mlp", "ln_1", "ln_2"))
        if not (has_qwen_llama or has_gpt2):
            continue

        if use_post_norm and idx < post_norm_start_layer:
            continue

        depth_ratio = idx / denom
        base_scale = float(getattr(config, "residual_scale_init", 1.0)) + float(
            getattr(config, "residual_scale_depth_slope", 0.0)
        ) * depth_ratio

        if mode == "sum":
            for attr_name in (
                "residual_attn_scale",
                "residual_mlp_scale",
                "residual_attn_wx",
                "residual_mlp_wx",
                "residual_attn_branch_scale",
                "residual_mlp_branch_scale",
                "residual_attn_branch_wx",
                "residual_mlp_branch_wx",
                "attn_residual_scale",
                "mlp_residual_scale",
                "attn_residual_wx",
                "mlp_residual_wx",
            ):
                if hasattr(layer, attr_name):
                    delattr(layer, attr_name)
        elif mode == "wx":
            hidden_size = int(getattr(layer, "hidden_size", 0) or getattr(getattr(model, "config", None), "hidden_size", 0))
            if hidden_size <= 0 and hasattr(layer, "ln_1"):
                ln1 = layer.ln_1
                if hasattr(ln1, "normalized_shape"):
                    hidden_size = int(ln1.normalized_shape[0])
                elif hasattr(ln1, "weight") and ln1.weight is not None:
                    hidden_size = int(ln1.weight.shape[0])
            if hidden_size <= 0:
                continue
            wx_init = float(getattr(config, "residual_wx", base_scale))
            branch_wx_init = float(getattr(config, "residual_branch_wx", 1.0))
            if getattr(layer, "self_attn", None) is not None or hasattr(layer, "attn"):
                _set_wx_linear(layer, "residual_attn_wx", hidden_size, wx_init, wx_learnable, wx_activation)
                if branch_gate:
                    _set_wx_linear(layer, "residual_attn_branch_wx", hidden_size, branch_wx_init, branch_wx_learnable, wx_activation)
            if getattr(layer, "mlp", None) is not None:
                _set_wx_linear(layer, "residual_mlp_wx", hidden_size, wx_init, wx_learnable, wx_activation)
                if branch_gate:
                    _set_wx_linear(layer, "residual_mlp_branch_wx", hidden_size, branch_wx_init, branch_wx_learnable, wx_activation)
            layer._residual_wx_activation = wx_activation
            for legacy_attr in (
                "attn_residual_wx", "mlp_residual_wx", "attn_residual_scale", "mlp_residual_scale",
                "residual_attn_scale", "residual_mlp_scale"
            ):
                if hasattr(layer, legacy_attr):
                    delattr(layer, legacy_attr)
            if not branch_gate:
                for a in ("residual_attn_branch_wx", "residual_mlp_branch_wx", "residual_attn_branch_scale", "residual_mlp_branch_scale"):
                    if hasattr(layer, a):
                        delattr(layer, a)
        else:
            attn_init = float(getattr(config, "residual_attn_param", base_scale))
            mlp_init = float(getattr(config, "residual_mlp_param", base_scale))
            branch_init = float(getattr(config, "residual_branch_scale_init", 1.0))
            if getattr(layer, "self_attn", None) is not None or hasattr(layer, "attn"):
                _set_scale_parameter_or_buffer(layer, "residual_attn_scale", attn_init, param_learnable)
                if branch_gate:
                    _set_scale_parameter_or_buffer(layer, "residual_attn_branch_scale", branch_init, branch_scale_learnable)
            if getattr(layer, "mlp", None) is not None:
                _set_scale_parameter_or_buffer(layer, "residual_mlp_scale", mlp_init, param_learnable)
                if branch_gate:
                    _set_scale_parameter_or_buffer(layer, "residual_mlp_branch_scale", branch_init, branch_scale_learnable)
            for legacy_attr in ("attn_residual_scale", "mlp_residual_scale", "attn_residual_wx", "mlp_residual_wx"):
                if hasattr(layer, legacy_attr):
                    delattr(layer, legacy_attr)
            if not branch_gate:
                for a in ("residual_attn_branch_wx", "residual_mlp_branch_wx", "residual_attn_branch_scale", "residual_mlp_branch_scale"):
                    if hasattr(layer, a):
                        delattr(layer, a)

        if not hasattr(layer, "_residual_original_forward"):
            layer._residual_original_forward = layer.forward
        layer._residual_layer_idx = int(idx)
        layer._residual_returns_tensor = _layer_returns_tensor(layer)
        layer._residual_branch_gate = bool(branch_gate)
        layer._xsa_forward_target = xsa_forward_target
        layer._xsa_forward_strength = xsa_forward_strength
        layer._attn_diag_hard = bool(attn_diag_hard)
        layer._attn_diag_keep_first = bool(attn_diag_keep_first)

        if has_qwen_llama:
            layer.forward = MethodType(_make_llama_qwen_forward(layer, model, mode), layer)
        else:
            layer.forward = MethodType(_make_gpt2_forward(layer, model, mode), layer)
        # Initialize runtime attrs so trainer can always discover patched layers.
        layer._residual_last_attn_alpha_mean = float("nan")
        layer._residual_last_mlp_alpha_mean = float("nan")
        layer._residual_last_attn_branch_gamma_mean = float("nan")
        layer._residual_last_mlp_branch_gamma_mean = float("nan")
        layer._xsa_attn_x_sq = None
        layer._xsa_mlp_x_sq = None

        patched_layers += 1

    penalty_state_handles = [
        model.register_forward_pre_hook(lambda _m, _in: _reset_residual_penalty_state(_m)),
        model.register_forward_hook(lambda _m, _in, out: _finalize_residual_penalty_state(_m, out)),
    ]

    setattr(model, "_residual_penalty_state_hook_handles", penalty_state_handles)
    setattr(model, "_residual_patch_hook_handles", [])

    patched = patched_layers > 0
    if patched:
        if mode != "sum":
            _print_residual_param_summary(model, mode)
        print(f"[INFO] residual_penalty_lambda={residual_penalty_lambda} (penalty = E[alpha^2])")
        if xsa_enabled:
            print(f"[INFO] xsa_forward_target={xsa_forward_target}, xsa_forward_strength={xsa_forward_strength}")
        if attn_diag_hard:
            print(f"[INFO] attn_diag_mode=hard, attn_diag_keep_first={attn_diag_keep_first}")
    return patched


def configure_residual_scaling(model, model_args, is_trainable: bool) -> bool:
    """Patch residual branch scaling via ModelArguments-style args."""
    mode = str(getattr(model_args, "residual_mode", "sum")).lower()
    if mode == "parameter":
        mode = "param"
    xsa_target = str(getattr(model_args, "xsa_forward_target", "none")).lower()
    xsa_strength = float(getattr(model_args, "xsa_forward_strength", 1.0) or 0.0)
    xsa_enabled = xsa_target in {"attn", "mlp", "both"} and xsa_strength != 0.0
    attn_diag_mode = str(getattr(model_args, "attn_diag_mode", "none")).lower()
    attn_diag_enabled = attn_diag_mode == "hard"

    if mode == "sum" and not xsa_enabled and not attn_diag_enabled:
        _clear_residual_patch(model)
        return False

    if mode not in {"sum", "wx", "param"}:
        raise ValueError(f"Unsupported residual_mode: {mode}. Choose from ['sum', 'wx', 'param'].")
    if xsa_target not in {"none", "attn", "mlp", "both"}:
        raise ValueError("xsa_forward_target must be one of: none, attn, mlp, both.")
    if attn_diag_mode not in {"none", "hard"}:
        raise ValueError("attn_diag_mode must be one of: none, hard.")

    cfg = type("ResidualPatchConfig", (), {})()
    cfg.residual_mode = mode
    cfg.residual_wx = float(getattr(model_args, "residual_wx", 1.0))
    cfg.residual_branch_gate = bool(getattr(model_args, "residual_branch_gate", False))
    cfg.residual_branch_wx = float(getattr(model_args, "residual_branch_wx", 1.0))
    cfg.residual_branch_scale_init = float(getattr(model_args, "residual_branch_scale_init", 1.0))
    cfg.residual_wx_activation = str(getattr(model_args, "residual_wx_activation", "silu"))
    cfg.residual_wx_learnable = bool(getattr(model_args, "residual_wx_learnable", False)) and bool(is_trainable)
    cfg.residual_branch_wx_learnable = bool(getattr(model_args, "residual_branch_wx_learnable", cfg.residual_wx_learnable)) and bool(is_trainable)
    if mode == "wx" and not cfg.residual_wx_learnable:
        cfg.residual_wx_learnable = True
    if mode == "wx" and cfg.residual_branch_gate and not cfg.residual_branch_wx_learnable:
        cfg.residual_branch_wx_learnable = True
    cfg.residual_attn_param = float(getattr(model_args, "residual_attn_param", 1.0))
    cfg.residual_mlp_param = float(getattr(model_args, "residual_mlp_param", 1.0))
    cfg.residual_scale_init = 1.0
    cfg.residual_scale_depth_slope = 0.0
    cfg.residual_scale_learnable = bool(is_trainable)
    cfg.residual_branch_scale_learnable = bool(getattr(model_args, "residual_branch_scale_learnable", is_trainable)) and bool(is_trainable)
    cfg.use_post_norm = bool(getattr(model_args, "use_post_norm", False))
    cfg.post_norm_start_layer = int(getattr(model_args, "post_norm_start_layer", 10**9))
    cfg.residual_penalty_lambda = float(getattr(model_args, "residual_penalty_lambda", 0.0) or 0.0)
    cfg.xsa_forward_target = xsa_target
    cfg.xsa_forward_strength = xsa_strength
    cfg.attn_diag_mode = attn_diag_mode
    cfg.attn_diag_keep_first = bool(getattr(model_args, "attn_diag_keep_first", True))

    patched = apply_qwen3_forward_patch(model, cfg)
    # Keep legacy handle attr name for generative-llm call sites.
    setattr(model, "_residual_scaling_hook_handles", getattr(model, "_residual_patch_hook_handles", []))
    return patched
