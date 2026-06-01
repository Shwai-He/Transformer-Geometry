from __future__ import annotations

import re
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from .presets import resolve_preset


@dataclass
class GeometryScaleConfig:
    """Configuration for para/perp scaling.

    The modified update is

        y' = para_scale * y_parallel + perp_scale * y_perp,

    where y is either a residual update, an attention/MLP branch output, or a
    value-projection output.  The reference x is the corresponding module input.
    """

    model_preset: str = "qwen-image"
    side: str = "gen"
    space: str = "residual"
    target: str = "block"
    para_scale: float = 1.0
    perp_scale: float = 1.0
    layer_paths: tuple[str, ...] = ()
    layer_indices: tuple[int, ...] | None = None
    skip_first_n: int = 0
    skip_last_n: int = 0
    attn_name_regex: str | None = None
    mlp_name_regex: str | None = None
    value_name_regex: str | None = None
    target_module_regex: str | None = None
    modify_tuple_index: int = 0
    eps: float = 1e-6
    track_stats: bool = True
    fail_on_missing_target: bool = True
    value_head_mode: str = "multihead"
    value_ref_expansion: str = "model_type"
    attn_attr_source: str = "model_type"
    scale_mode: str = "none"
    scale_seed: int = 0
    para_scale_min: float = -1.5
    para_scale_max: float = 1.5
    perp_scale_min: float = -1.5
    perp_scale_max: float = 1.5


def _get_by_path(root: Any, path: str) -> Any | None:
    cur = root
    for part in path.split("."):
        if not part:
            continue
        if isinstance(cur, (list, tuple, torch.nn.ModuleList)):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _as_module_list(obj: Any) -> list[torch.nn.Module]:
    if obj is None:
        return []
    if isinstance(obj, torch.nn.ModuleList):
        return list(obj)
    if isinstance(obj, (list, tuple)) and all(isinstance(x, torch.nn.Module) for x in obj):
        return list(obj)
    return []


def _first_tensor(obj: Any, tuple_index: int = 0) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)) and obj:
        idx = int(tuple_index)
        if 0 <= idx < len(obj) and isinstance(obj[idx], torch.Tensor):
            return obj[idx]
    return None


def _replace_first_tensor(obj: Any, value: torch.Tensor, tuple_index: int = 0) -> Any:
    if isinstance(obj, torch.Tensor):
        return value
    if isinstance(obj, tuple):
        idx = int(tuple_index)
        if 0 <= idx < len(obj) and isinstance(obj[idx], torch.Tensor):
            items = list(obj)
            items[idx] = value
            return tuple(items)
    if isinstance(obj, list):
        idx = int(tuple_index)
        if 0 <= idx < len(obj) and isinstance(obj[idx], torch.Tensor):
            items = list(obj)
            items[idx] = value
            return items
    return obj


def _find_tensor_index_by_shape(obj: Any, shape: torch.Size, tuple_index: int = 0) -> int | None:
    if isinstance(obj, torch.Tensor):
        return None if obj.shape == shape else -1
    if isinstance(obj, (list, tuple)) and obj:
        candidates = [idx for idx, value in enumerate(obj) if isinstance(value, torch.Tensor) and value.shape == shape]
        if not candidates:
            return -1
        if tuple_index in candidates:
            return tuple_index
        return candidates[0]
    return -1


def _tensor_at_index(obj: Any, tuple_index: int | None) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj if tuple_index is None else None
    if isinstance(obj, (list, tuple)) and tuple_index is not None and 0 <= tuple_index < len(obj):
        value = obj[tuple_index]
        return value if isinstance(value, torch.Tensor) else None
    return None


def _replace_tensor_at_index(obj: Any, value: torch.Tensor, tuple_index: int | None) -> Any:
    if isinstance(obj, torch.Tensor):
        return value if tuple_index is None else obj
    if isinstance(obj, tuple) and tuple_index is not None and 0 <= tuple_index < len(obj):
        items = list(obj)
        items[tuple_index] = value
        return tuple(items)
    if isinstance(obj, list) and tuple_index is not None and 0 <= tuple_index < len(obj):
        items = list(obj)
        items[tuple_index] = value
        return items
    return obj


def _replace_first_arg(args: tuple[Any, ...], kwargs: dict[str, Any], value: torch.Tensor) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if args:
        return (value,) + tuple(args[1:]), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["input"] = value
    return args, new_kwargs


def _module_input_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    for value in args:
        if isinstance(value, torch.Tensor):
            return value
    for key in ("hidden_states", "input", "x", "sample", "encoder_hidden_states"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            return value
    return None


def _is_phi_attention(attn_module: torch.nn.Module) -> bool:
    config = getattr(attn_module, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    module_name = attn_module.__class__.__name__.lower()
    return model_type.startswith("phi") or "phi" in module_name


def _get_attn_attr(attn_module: torch.nn.Module, *names: str, source: str = "model_type") -> Any | None:
    for name in names:
        value = getattr(attn_module, name, None)
        if value is not None:
            return value
    source = str(source or "model_type").lower().strip()
    if source in {"module_only", "module", "legacy", "legacy_repeat"}:
        return None
    if source in {"model_type", "auto_phi", "phi"} and not _is_phi_attention(attn_module):
        return None
    config = getattr(attn_module, "config", None)
    if config is not None:
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return value
    return None


def _as_positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _linear_out_features(module: Any) -> int | None:
    value = getattr(module, "out_features", None)
    if value is not None:
        return _as_positive_int(value)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
        return _as_positive_int(weight.shape[0])
    return None


def _infer_attn_head_shape(
    attn_module: torch.nn.Module,
    *,
    y_pre: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    source: str = "model_type",
) -> tuple[int | None, int | None, int | None]:
    source = str(source or "model_type").lower().strip()
    num_heads = _as_positive_int(
        _get_attn_attr(attn_module, "num_heads", "num_attention_heads", "n_heads", "heads", source=source)
    )
    head_dim = _as_positive_int(
        _get_attn_attr(attn_module, "head_dim", "attention_head_size", "dim_head", "attention_head_dim", source=source)
    )
    num_kv_heads = _as_positive_int(
        _get_attn_attr(
            attn_module,
            "num_key_value_heads",
            "num_kv_heads",
            "n_kv_heads",
            "kv_heads",
            source=source,
        )
    )
    allow_inference = source in {"head_aware", "auto", "config_fallback"} or (
        source in {"model_type", "auto_phi", "phi"} and _is_phi_attention(attn_module)
    )

    if allow_inference and num_heads is None and head_dim is not None and y_pre is not None and y_pre.size(-1) % head_dim == 0:
        num_heads = y_pre.size(-1) // head_dim
    if allow_inference and head_dim is None and num_heads is not None and y_pre is not None and y_pre.size(-1) % num_heads == 0:
        head_dim = y_pre.size(-1) // num_heads

    inner_dim = _as_positive_int(_get_attn_attr(attn_module, "inner_dim", source=source))
    if allow_inference and inner_dim is not None:
        if num_heads is None and head_dim is not None and inner_dim % head_dim == 0:
            num_heads = inner_dim // head_dim
        if head_dim is None and num_heads is not None and inner_dim % num_heads == 0:
            head_dim = inner_dim // num_heads

    if allow_inference and num_kv_heads is None and head_dim is not None and value is not None and value.size(-1) % head_dim == 0:
        num_kv_heads = value.size(-1) // head_dim
    if allow_inference and num_kv_heads is None:
        to_v = getattr(attn_module, "to_v", None) or getattr(attn_module, "v_proj", None) or getattr(attn_module, "v_proj_moe_gen", None)
        value_width = _linear_out_features(to_v)
        if head_dim is not None and value_width is not None and value_width % head_dim == 0:
            num_kv_heads = value_width // head_dim

    if allow_inference and num_kv_heads is None:
        num_kv_heads = num_heads
    return num_heads, head_dim, num_kv_heads


def _expand_value_ref(
    value: torch.Tensor,
    y_pre: torch.Tensor,
    attn_module: torch.nn.Module,
    *,
    expansion_mode: str = "model_type",
    attr_source: str = "model_type",
) -> torch.Tensor | None:
    if value.shape == y_pre.shape:
        return value
    if value.dim() not in {2, 3} or y_pre.dim() != value.dim():
        return None
    if value.shape[:-1] != y_pre.shape[:-1]:
        return None

    expansion_mode = str(expansion_mode or "model_type").lower().strip()
    use_head_aware = expansion_mode in {"head_aware", "auto", "config_fallback"} or (
        expansion_mode in {"model_type", "auto_phi", "phi"} and _is_phi_attention(attn_module)
    )
    if not use_head_aware:
        if y_pre.size(-1) % value.size(-1) != 0:
            return None
        return value.repeat_interleave(y_pre.size(-1) // value.size(-1), dim=-1)

    num_heads, head_dim, num_key_value_heads = _infer_attn_head_shape(
        attn_module,
        y_pre=y_pre,
        value=value,
        source=attr_source,
    )
    if head_dim is None or num_heads is None or num_key_value_heads is None:
        return None
    if value.size(-1) != num_key_value_heads * head_dim:
        return None
    if y_pre.size(-1) != num_heads * head_dim:
        return None
    if num_heads % num_key_value_heads != 0:
        return None

    ref = value.reshape(*value.shape[:-1], num_key_value_heads, head_dim)
    ref = ref.repeat_interleave(num_heads // num_key_value_heads, dim=-2)
    return ref.reshape_as(y_pre)


def _extract_value_projection(
    output: Any,
    attn_module: torch.nn.Module,
    tuple_index: int = 0,
    *,
    attr_source: str = "model_type",
) -> torch.Tensor | None:
    value = _first_tensor(output, tuple_index)
    if not isinstance(value, torch.Tensor):
        return None

    num_heads, head_dim, num_key_value_heads = _infer_attn_head_shape(
        attn_module,
        value=value,
        source=attr_source,
    )
    if head_dim is None or num_heads is None or num_key_value_heads is None:
        return value

    fused_qkv_dim = (num_heads + 2 * num_key_value_heads) * head_dim
    value_dim = num_key_value_heads * head_dim
    if value.size(-1) == fused_qkv_dim:
        return value[..., -value_dim:]
    return value


def _scale_update(
    y: torch.Tensor,
    ref: torch.Tensor,
    *,
    para_scale: float | torch.Tensor,
    perp_scale: float | torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y_f = y.float()
    ref_f = ref.float()
    dot = (y_f * ref_f).sum(dim=-1, keepdim=True)
    ref_sq = (ref_f * ref_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    para = dot / ref_sq * ref_f
    perp = y_f - para
    para_value = _scale_like_batch(para_scale, para)
    perp_value = _scale_like_batch(perp_scale, perp)
    out = para_value * para + perp_value * perp

    y_norm_sq = (y_f * y_f).sum(dim=-1).clamp_min(eps)
    ref_norm = torch.sqrt(ref_sq.squeeze(-1).clamp_min(eps))
    para_norm = torch.sqrt((para * para).sum(dim=-1).clamp_min(0.0))
    perp_norm = torch.sqrt((perp * perp).sum(dim=-1).clamp_min(0.0))
    stats = {
        "para_ratio": (para_norm.square() / y_norm_sq).detach(),
        "perp_ratio": (perp_norm.square() / y_norm_sq).detach(),
        "para_over_ref": (para_norm / ref_norm).detach(),
        "perp_over_ref": (perp_norm / ref_norm).detach(),
        "update_over_ref": (torch.sqrt(y_norm_sq) / ref_norm).detach(),
    }
    return out.to(dtype=y.dtype), stats


def _scale_like_batch(scale: float | torch.Tensor, target: torch.Tensor) -> float | torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        return float(scale)
    if scale.dim() == 0:
        return scale.to(device=target.device, dtype=target.dtype)
    if target.dim() > 0 and scale.shape[0] == target.shape[0]:
        return scale.reshape((int(scale.shape[0]),) + (1,) * (target.dim() - 1)).to(
            device=target.device,
            dtype=target.dtype,
        )
    return scale.to(device=target.device, dtype=target.dtype)


def _scale_update_multihead(
    y: torch.Tensor,
    ref: torch.Tensor,
    attn_module: torch.nn.Module,
    *,
    para_scale: float | torch.Tensor,
    perp_scale: float | torch.Tensor,
    eps: float,
    attr_source: str = "model_type",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_heads, head_dim, _ = _infer_attn_head_shape(attn_module, y_pre=y, value=ref, source=attr_source)
    if head_dim is None or num_heads is None:
        return _scale_update(y, ref, para_scale=para_scale, perp_scale=perp_scale, eps=eps)

    if y.dim() not in {2, 3} or ref.dim() != y.dim():
        return _scale_update(y, ref, para_scale=para_scale, perp_scale=perp_scale, eps=eps)
    if y.size(-1) != num_heads * head_dim or ref.size(-1) != num_heads * head_dim:
        return _scale_update(y, ref, para_scale=para_scale, perp_scale=perp_scale, eps=eps)

    y_heads = y.reshape(*y.shape[:-1], num_heads, head_dim)
    ref_heads = ref.reshape(*ref.shape[:-1], num_heads, head_dim)
    new_heads, stats = _scale_update(
        y_heads,
        ref_heads,
        para_scale=para_scale,
        perp_scale=perp_scale,
        eps=eps,
    )
    return new_heads.reshape_as(y), stats


class RunningStats:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.sums: dict[tuple[str, int, str], float] = defaultdict(float)
        self.counts: dict[tuple[str, int, str], int] = defaultdict(int)

    def update(self, target: str, layer_idx: int, stats: dict[str, torch.Tensor]) -> None:
        if not self.enabled:
            return
        for name, value in stats.items():
            v = value.detach().float()
            key = (target, int(layer_idx), name)
            self.sums[key] += float(v.sum().item())
            self.counts[key] += int(v.numel())

    def summary(self) -> dict[str, Any]:
        by_target: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        overall_sums: dict[tuple[str, str], float] = defaultdict(float)
        overall_counts: dict[tuple[str, str], int] = defaultdict(int)
        for (target, layer_idx, name), total in sorted(self.sums.items()):
            count = max(1, self.counts[(target, layer_idx, name)])
            by_target[target].setdefault(str(layer_idx), {})[name] = total / count
            overall_sums[(target, name)] += total
            overall_counts[(target, name)] += count
        overall: dict[str, dict[str, float]] = defaultdict(dict)
        for (target, name), total in sorted(overall_sums.items()):
            overall[target][name] = total / max(1, overall_counts[(target, name)])
        return {"overall": dict(overall), "layers": dict(by_target)}


class HookDiagnostics:
    def __init__(self):
        self.counters: dict[tuple[str, int, str], int] = defaultdict(int)

    def incr(self, target: str, layer_idx: int, name: str) -> None:
        self.counters[(target, int(layer_idx), name)] += 1

    def summary(self) -> dict[str, Any]:
        by_target: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        overall: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (target, layer_idx, name), value in sorted(self.counters.items()):
            by_target[target].setdefault(str(layer_idx), {})[name] = value
            overall[target][name] += value
        return {
            "overall": {target: dict(values) for target, values in overall.items()},
            "layers": dict(by_target),
        }


class VLMGeometryScaler:
    """Attach para/perp scaling hooks to VLM modules.

    This class intentionally avoids model-specific imports.  It discovers layer
    modules through presets or explicit paths, then modifies residual-space
    updates or value-projection outputs by comparing each module's output with
    its input.
    """

    def __init__(self, model: torch.nn.Module, config: GeometryScaleConfig):
        self.model = model
        self.config = config
        self.handles: list[Any] = []
        self.stats = RunningStats(config.track_stats)
        self.diagnostics = HookDiagnostics()
        self.resolved_layer_path = ""
        self.layer_groups = self._resolve_layer_groups()
        self.layers = [layer for _, layers in self.layer_groups for layer in layers]
        self.active = self._active_indices(len(self.layers))
        self.sample_rng = random.Random(int(config.scale_seed))
        self.current_sample_seed: int | None = None
        self.job_para_scale, self.job_perp_scale = self._sample_job_scales()

    def set_sample_seed(self, seed: int | None) -> None:
        self.current_sample_seed = None if seed is None else int(seed)

    def _sample_from_range(self, rng: random.Random, lo: float, hi: float, *, choice: bool) -> float:
        if choice:
            return float(rng.choice([float(lo), float(hi)]))
        return float(rng.uniform(float(lo), float(hi)))

    def _sample_job_scales(self) -> tuple[float, float]:
        mode = str(self.config.scale_mode).lower().strip()
        rng = random.Random(int(self.config.scale_seed))
        para = float(self.config.para_scale)
        perp = float(self.config.perp_scale)
        if mode in {"job_para_uniform", "job_both_uniform"}:
            para = self._sample_from_range(rng, self.config.para_scale_min, self.config.para_scale_max, choice=False)
        if mode in {"job_para_choice", "job_both_choice"}:
            para = self._sample_from_range(rng, self.config.para_scale_min, self.config.para_scale_max, choice=True)
        if mode in {"job_perp_uniform", "job_both_uniform"}:
            perp = self._sample_from_range(rng, self.config.perp_scale_min, self.config.perp_scale_max, choice=False)
        if mode in {"job_perp_choice", "job_both_choice"}:
            perp = self._sample_from_range(rng, self.config.perp_scale_min, self.config.perp_scale_max, choice=True)
        return para, perp

    def _scales_for_update(self, ref: torch.Tensor) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        mode = str(self.config.scale_mode).lower().strip()
        if mode in {"none", ""}:
            return float(self.config.para_scale), float(self.config.perp_scale)
        if mode.startswith("job_"):
            return self.job_para_scale, self.job_perp_scale
        if not mode.startswith("sample_"):
            raise ValueError(f"Unsupported scale_mode={self.config.scale_mode!r}")
        if ref.dim() == 0:
            return float(self.config.para_scale), float(self.config.perp_scale)
        seed = int(self.config.scale_seed) if self.current_sample_seed is None else int(self.current_sample_seed)
        rng = random.Random(seed)
        batch = int(ref.shape[0])
        para_values = None
        perp_values = None
        if mode in {"sample_para_uniform", "sample_both_uniform"}:
            para_values = [
                self._sample_from_range(rng, self.config.para_scale_min, self.config.para_scale_max, choice=False)
                for _ in range(batch)
            ]
        if mode in {"sample_para_choice", "sample_both_choice"}:
            para_values = [
                self._sample_from_range(rng, self.config.para_scale_min, self.config.para_scale_max, choice=True)
                for _ in range(batch)
            ]
        if mode in {"sample_perp_uniform", "sample_both_uniform"}:
            perp_values = [
                self._sample_from_range(rng, self.config.perp_scale_min, self.config.perp_scale_max, choice=False)
                for _ in range(batch)
            ]
        if mode in {"sample_perp_choice", "sample_both_choice"}:
            perp_values = [
                self._sample_from_range(rng, self.config.perp_scale_min, self.config.perp_scale_max, choice=True)
                for _ in range(batch)
            ]
        para: float | torch.Tensor = float(self.config.para_scale)
        perp: float | torch.Tensor = float(self.config.perp_scale)
        if para_values is not None:
            para = torch.tensor(para_values, device=ref.device, dtype=torch.float32)
        if perp_values is not None:
            perp = torch.tensor(perp_values, device=ref.device, dtype=torch.float32)
        return para, perp

    def _is_static_identity(self) -> bool:
        mode = str(self.config.scale_mode).lower().strip()
        return mode in {"none", ""} and float(self.config.para_scale) == 1.0 and float(self.config.perp_scale) == 1.0

    def _candidate_paths(self) -> tuple[str, ...]:
        if self.config.layer_paths:
            return self.config.layer_paths
        spec = resolve_preset(self.config.model_preset)
        if self.config.side == "und":
            return spec.und_layer_paths
        if self.config.side == "gen":
            return spec.gen_layer_paths
        if self.config.side == "both":
            seen = set()
            paths = []
            for path in (*spec.und_layer_paths, *spec.gen_layer_paths):
                if path in seen:
                    continue
                seen.add(path)
                paths.append(path)
            return tuple(paths)
        raise ValueError(f"side must be und, gen, or both, got {self.config.side!r}")

    def _resolve_layer_groups(self) -> list[tuple[str, list[torch.nn.Module]]]:
        tried = []
        groups: list[tuple[str, list[torch.nn.Module]]] = []
        for path in self._candidate_paths():
            tried.append(path)
            layers = _as_module_list(_get_by_path(self.model, path))
            if layers:
                groups.append((path, layers))
                if self.config.side != "both":
                    break
        if groups:
            self.resolved_layer_path = groups[0][0] if len(groups) == 1 else ";".join(path for path, _ in groups)
            return groups
        tried_msg = ", ".join(tried) or "<none>"
        raise ValueError(
            f"Could not locate {self.config.side} layers for preset={self.config.model_preset}. "
            f"Tried: {tried_msg}. Pass --layer-paths with the exact ModuleList path."
        )

    def _resolve_layers(self) -> list[torch.nn.Module]:
        return [layer for _, layers in self._resolve_layer_groups() for layer in layers]

    def _active_indices(self, n_layers: int) -> set[int]:
        if self.config.layer_indices is not None:
            return {idx for idx in self.config.layer_indices if 0 <= idx < n_layers}
        lo = max(0, int(self.config.skip_first_n))
        hi = max(lo, n_layers - max(0, int(self.config.skip_last_n)))
        return set(range(lo, hi))

    def _regex_for_target(self) -> str:
        if self.config.target_module_regex:
            return self.config.target_module_regex
        spec = resolve_preset(self.config.model_preset)
        if self.config.target == "attn":
            return self.config.attn_name_regex or spec.attn_name_regex
        if self.config.target == "mlp":
            return self.config.mlp_name_regex or spec.mlp_name_regex
        if self.config.target == "value":
            return self.config.value_name_regex or spec.value_name_regex
        raise ValueError(f"target regex requested for unsupported target={self.config.target!r}")

    def _find_child_target(self, layer: torch.nn.Module) -> torch.nn.Module | None:
        pattern = re.compile(self._regex_for_target(), re.IGNORECASE)
        matches: list[tuple[str, torch.nn.Module]] = []
        for name, module in layer.named_modules():
            if not name:
                continue
            if pattern.search(name):
                matches.append((name, module))
        # Prefer shallower modules; matching a linear projection such as to_q
        # is usually less useful than the containing attention module.
        matches.sort(key=lambda item: (item[0].count("."), len(item[0])))
        return matches[0][1] if matches else None

    def _find_value_pairs(
        self,
        layer: torch.nn.Module,
    ) -> list[tuple[str, torch.nn.Module, torch.nn.Module, torch.nn.Module]]:
        candidates: list[tuple[str, str, torch.nn.Module, torch.nn.Module, torch.nn.Module]] = []
        for name, module in layer.named_modules():
            value_module = None
            out_module = None
            to_out = getattr(module, "to_out", None)
            if out_module is None and isinstance(to_out, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)) and to_out:
                maybe_out = to_out[0]
                if isinstance(maybe_out, torch.nn.Module):
                    out_module = maybe_out

            pair_specs = [
                ("gen", getattr(module, "v_proj_moe_gen", None), getattr(module, "o_proj_moe_gen", None)),
                ("main", getattr(module, "v_proj", None), getattr(module, "o_proj", None)),
                ("main", getattr(module, "to_v", None), out_module),
                ("main", getattr(module, "qkv_proj", None), getattr(module, "dense", None)),
                ("main", getattr(module, "in_proj_qkv", None), getattr(module, "out_proj", None)),
                ("main", getattr(module, "query_key_value", None), getattr(module, "dense", None)),
            ]
            seen_pairs: set[tuple[int, int]] = set()
            for pair_label, value_module, out_module in pair_specs:
                if not isinstance(value_module, torch.nn.Module) or not isinstance(out_module, torch.nn.Module):
                    continue
                key = (id(value_module), id(out_module))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                candidates.append((name, pair_label, module, value_module, out_module))

        if not candidates:
            return []
        candidates.sort(key=lambda item: (item[0].count("."), len(item[0])))
        min_depth = candidates[0][0].count(".")
        min_len = len(candidates[0][0])
        best = [item for item in candidates if item[0].count(".") == min_depth and len(item[0]) == min_len]
        return [(pair_label, value_module, out_module, attn_module) for _, pair_label, attn_module, value_module, out_module in best]

    def attach(self) -> None:
        target = self.config.target
        space = self.config.space
        if space not in {"residual", "value"}:
            raise ValueError(f"space must be residual/value, got {space!r}")
        if target not in {"block", "attn", "mlp", "value"}:
            raise ValueError(f"target must be block/attn/mlp/value, got {target!r}")
        if space == "value" and target != "value":
            raise ValueError("space=value currently expects target=value.")
        if space == "residual" and target == "value":
            raise ValueError("target=value currently expects space=value.")
        missing = []
        global_idx = 0
        for group_path, layers in self.layer_groups:
            active = self._active_indices(len(layers))
            for local_idx, layer in enumerate(layers):
                idx = global_idx + local_idx
                if local_idx not in active:
                    continue
                target_label = target if len(self.layer_groups) == 1 else f"{target}:{group_path}"
                if target == "block":
                    self._attach_module(layer, idx, target_label, block_mode=True)
                    continue
                if target == "value":
                    pairs = self._find_value_pairs(layer)
                    if not pairs:
                        missing.append(f"{group_path}[{local_idx}]")
                        continue
                    for pair_label, value_module, out_module, attn_module in pairs:
                        pair_target_label = target_label if pair_label == "main" else f"{target_label}:{pair_label}"
                        self._attach_value_pair(
                            value_module,
                            out_module,
                            attn_module,
                            idx,
                            target_label=pair_target_label,
                        )
                    continue
                module = self._find_child_target(layer)
                if module is None:
                    missing.append(f"{group_path}[{local_idx}]")
                    continue
                self._attach_module(module, idx, target_label, block_mode=False)
            global_idx += len(layers)
        if missing and self.config.fail_on_missing_target:
            raise ValueError(
                f"Could not find target={target!r} modules in layers {missing[:12]}"
                f"{'...' if len(missing) > 12 else ''}. "
                "Use --target-module-regex or --fail-on-missing-target false."
            )

    def _attach_module(self, module: torch.nn.Module, layer_idx: int, target_name: str, *, block_mode: bool) -> None:
        cache: dict[str, torch.Tensor] = {}

        def pre_hook(_module, args, kwargs):
            self.diagnostics.incr(target_name, layer_idx, "pre_calls")
            ref = _module_input_tensor(args, kwargs)
            if isinstance(ref, torch.Tensor):
                cache["ref"] = ref.detach()
            else:
                self.diagnostics.incr(target_name, layer_idx, "missing_ref")

        def post_hook(_module, args, kwargs, output):
            self.diagnostics.incr(target_name, layer_idx, "post_calls")
            ref = cache.pop("ref", None)
            if not isinstance(ref, torch.Tensor):
                self.diagnostics.incr(target_name, layer_idx, "missing_cached_ref")
                return output

            if block_mode:
                output_index = _find_tensor_index_by_shape(output, ref.shape, self.config.modify_tuple_index)
                out_tensor = _tensor_at_index(output, output_index)
            else:
                out_tensor = _first_tensor(output, self.config.modify_tuple_index)
                output_index = None if isinstance(output, torch.Tensor) else self.config.modify_tuple_index

            if not isinstance(out_tensor, torch.Tensor):
                self.diagnostics.incr(target_name, layer_idx, "missing_output_tensor")
                return output
            if out_tensor.shape != ref.shape:
                self.diagnostics.incr(target_name, layer_idx, "shape_mismatch")
                return output
            y = out_tensor - ref if block_mode else out_tensor
            para_scale, perp_scale = self._scales_for_update(ref)
            y_new, stats = _scale_update(
                y,
                ref,
                para_scale=para_scale,
                perp_scale=perp_scale,
                eps=self.config.eps,
            )
            self.stats.update(target_name, layer_idx, stats)
            self.diagnostics.incr(target_name, layer_idx, "stat_updates")
            if self._is_static_identity():
                return output
            new_tensor = ref + y_new if block_mode else y_new
            if block_mode:
                return _replace_tensor_at_index(output, new_tensor, output_index)
            return _replace_first_tensor(output, new_tensor, self.config.modify_tuple_index)

        self.handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self.handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    def _attach_value_pair(
        self,
        value_module: torch.nn.Module,
        out_module: torch.nn.Module,
        attn_module: torch.nn.Module,
        layer_idx: int,
        *,
        target_label: str = "value",
    ) -> None:
        cache: dict[str, torch.Tensor] = {}
        target_name = target_label

        def value_hook(_module, _args, _kwargs, output):
            self.diagnostics.incr(target_name, layer_idx, "value_post_calls")
            value = _extract_value_projection(
                output,
                attn_module,
                self.config.modify_tuple_index,
                attr_source=self.config.attn_attr_source,
            )
            if isinstance(value, torch.Tensor):
                cache["value"] = value.detach()
            else:
                self.diagnostics.incr(target_name, layer_idx, "missing_value_tensor")
            return output

        def out_pre_hook(_module, args, kwargs):
            self.diagnostics.incr(target_name, layer_idx, "pre_calls")
            y_pre = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("input")
            value = cache.pop("value", None)
            if not isinstance(y_pre, torch.Tensor):
                self.diagnostics.incr(target_name, layer_idx, "missing_output_tensor")
                return None
            if not isinstance(value, torch.Tensor):
                self.diagnostics.incr(target_name, layer_idx, "missing_cached_value")
                return None
            ref = _expand_value_ref(
                value,
                y_pre,
                attn_module,
                expansion_mode=self.config.value_ref_expansion,
                attr_source=self.config.attn_attr_source,
            )
            if ref is None or ref.shape != y_pre.shape:
                self.diagnostics.incr(target_name, layer_idx, "shape_mismatch")
                return None
            para_scale, perp_scale = self._scales_for_update(ref)
            if self.config.value_head_mode == "merged":
                y_new, stats = _scale_update(
                    y_pre,
                    ref,
                    para_scale=para_scale,
                    perp_scale=perp_scale,
                    eps=self.config.eps,
                )
            else:
                y_new, stats = _scale_update_multihead(
                    y_pre,
                    ref,
                    attn_module,
                    para_scale=para_scale,
                    perp_scale=perp_scale,
                    eps=self.config.eps,
                    attr_source=self.config.attn_attr_source,
                )
            self.stats.update(target_name, layer_idx, stats)
            self.diagnostics.incr(target_name, layer_idx, "stat_updates")
            if self._is_static_identity():
                return None
            return _replace_first_arg(args, kwargs, y_new)

        self.handles.append(value_module.register_forward_hook(value_hook, with_kwargs=True))
        self.handles.append(out_module.register_forward_pre_hook(out_pre_hook, with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> "VLMGeometryScaler":
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def summary(self) -> dict[str, Any]:
        return {
            "config": {
                "model_preset": self.config.model_preset,
                "side": self.config.side,
                "space": self.config.space,
                "target": self.config.target,
                "para_scale": self.config.para_scale,
                "perp_scale": self.config.perp_scale,
                "n_layers": len(self.layers),
                "active_layers": sorted(self.active),
                "layer_paths": list(self._candidate_paths()),
                "resolved_layer_path": self.resolved_layer_path,
                "resolved_layer_groups": [
                    {"path": path, "n_layers": len(layers)}
                    for path, layers in self.layer_groups
                ],
                "value_head_mode": self.config.value_head_mode,
                "value_ref_expansion": self.config.value_ref_expansion,
                "attn_attr_source": self.config.attn_attr_source,
                "scale_mode": self.config.scale_mode,
                "scale_seed": self.config.scale_seed,
                "para_scale_min": self.config.para_scale_min,
                "para_scale_max": self.config.para_scale_max,
                "perp_scale_min": self.config.perp_scale_min,
                "perp_scale_max": self.config.perp_scale_max,
                "job_para_scale": self.job_para_scale,
                "job_perp_scale": self.job_perp_scale,
            },
            "stats": self.stats.summary(),
            "diagnostics": self.diagnostics.summary(),
        }


def parse_int_list(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_path_list(text: str | None) -> tuple[str, ...]:
    if text is None or not text.strip():
        return ()
    return tuple(x.strip() for x in text.split(",") if x.strip())
