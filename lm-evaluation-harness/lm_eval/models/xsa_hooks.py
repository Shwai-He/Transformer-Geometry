from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, List, Optional

import torch

eval_logger = logging.getLogger(__name__)


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return list(layers)
    raise ValueError(
        "Could not find decoder layers. Expected model.layers, transformer.h, or gpt_neox.layers."
    )


def _extract_hidden(args, kwargs):
    return kwargs.get("hidden_states", args[0] if args else None)


def _get_token_mixer_kind_and_module(layer):
    candidates = []
    for attr_name in (
        "self_attn",
        "linear_attn",
        "attn",
        "attention",
        "token_mixer",
        "mixer",
    ):
        module = getattr(layer, attr_name, None)
        if module is not None:
            candidates.append((attr_name, module))

    for _, module in candidates:
        if hasattr(module, "v_proj") and hasattr(module, "o_proj"):
            return "full_attention", module
    for _, module in candidates:
        if hasattr(module, "in_proj_qkv") and hasattr(module, "out_proj"):
            return "linear_attention", module
    for attr_name, module in candidates:
        class_name = type(module).__name__.lower()
        if "linear" in class_name and "att" in class_name:
            return "linear_attention", module
        if "att" in class_name or "mixer" in class_name:
            eval_logger.warning(
                "Treating layer module %s (%s) as full_attention by name fallback; "
                "expected explicit v_proj/o_proj or in_proj_qkv/out_proj.",
                attr_name,
                type(module).__name__,
            )
            return "full_attention", module
    return None, None


def _remove_parallel_and_stats(y: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6):
    y_f = y.float()
    r_f = ref.float()
    dot = (y_f * r_f).sum(dim=-1, keepdim=True)
    ref_sq = (r_f * r_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = dot / ref_sq
    proj = coeff * r_f
    perp = y_f - proj

    proj_sq = (proj * proj).sum(dim=-1)
    perp_sq = (perp * perp).sum(dim=-1)
    y_sq = (y_f * y_f).sum(dim=-1).clamp_min(eps)
    ref_norm = torch.sqrt(ref_sq.squeeze(-1))
    y_norm = torch.sqrt(y_sq)

    stats = {
        "para_ratio": (proj_sq / y_sq).detach(),
        "perp_ratio": (perp_sq / y_sq).detach(),
        "y_over_ref": (y_norm / ref_norm.clamp_min(eps)).detach(),
        "perp_over_ref": (
            torch.sqrt(perp_sq.clamp_min(0.0)) / ref_norm.clamp_min(eps)
        ).detach(),
        "para_over_ref": (
            torch.sqrt(proj_sq.clamp_min(0.0)) / ref_norm.clamp_min(eps)
        ).detach(),
    }
    return (y_f - proj).to(dtype=y.dtype), stats


def _project_parallel(y: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y_f = y.float()
    r_f = ref.float()
    dot = (y_f * r_f).sum(dim=-1, keepdim=True)
    ref_sq = (r_f * r_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    proj = (dot / ref_sq) * r_f
    return proj.to(dtype=y.dtype)


def _apply_parallel_op(
    y: torch.Tensor,
    ref: torch.Tensor,
    *,
    op: str = "remove_parallel",
    alpha: float = 1.0,
) -> torch.Tensor:
    proj = _project_parallel(y, ref)
    op_norm = str(op).lower().strip()
    alpha_value = float(alpha)
    if op_norm == "remove_parallel":
        return y - alpha_value * proj
    if op_norm == "keep_parallel":
        return alpha_value * proj
    if op_norm == "add_parallel":
        return y + alpha_value * proj
    if op_norm == "negate_parallel":
        return y - (1.0 + alpha_value) * proj
    raise ValueError(f"Unsupported xsa_forward_op={op}")


def _replace_first_arg(args, kwargs, new_value):
    if args:
        return (new_value,) + tuple(args[1:]), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["input"] = new_value
    return args, new_kwargs


def _expand_attn_value_ref(
    value: torch.Tensor, y_pre: torch.Tensor, attn_module
) -> Optional[torch.Tensor]:
    if value.shape == y_pre.shape:
        return value
    if value.dim() != 3 or y_pre.dim() != 3:
        return None
    if value.shape[:2] != y_pre.shape[:2]:
        return None

    head_dim = getattr(attn_module, "head_dim", None)
    num_heads = getattr(attn_module, "num_heads", None)
    num_key_value_heads = getattr(attn_module, "num_key_value_heads", None)
    if head_dim is None or num_heads is None or num_key_value_heads is None:
        if y_pre.size(-1) % value.size(-1) != 0:
            return None
        group = y_pre.size(-1) // value.size(-1)
        return value.repeat_interleave(group, dim=-1)

    if value.size(-1) != num_key_value_heads * head_dim:
        return None
    if y_pre.size(-1) != num_heads * head_dim:
        return None
    if num_heads % num_key_value_heads != 0:
        return None

    group = num_heads // num_key_value_heads
    ref = value.reshape(*value.shape[:-1], num_key_value_heads, head_dim)
    ref = ref.repeat_interleave(group, dim=-2)
    return ref.reshape_as(y_pre)


def _remove_parallel_attn_multihead(y_pre: torch.Tensor, ref: torch.Tensor, attn_module):
    head_dim = getattr(attn_module, "head_dim", None)
    num_heads = getattr(attn_module, "num_heads", None)
    return _apply_parallel_multihead(
        y_pre,
        ref,
        num_heads=num_heads,
        head_dim=head_dim,
        op="remove_parallel",
        alpha=1.0,
    )


def _remove_parallel_multihead(
    y_pre: torch.Tensor,
    ref: torch.Tensor,
    *,
    num_heads: Optional[int],
    head_dim: Optional[int],
):
    return _apply_parallel_multihead(
        y_pre,
        ref,
        num_heads=num_heads,
        head_dim=head_dim,
        op="remove_parallel",
        alpha=1.0,
    )


def _apply_parallel_multihead(
    y_pre: torch.Tensor,
    ref: torch.Tensor,
    *,
    num_heads: Optional[int],
    head_dim: Optional[int],
    op: str = "remove_parallel",
    alpha: float = 1.0,
):
    if head_dim is None or num_heads is None:
        return _apply_parallel_op(y_pre, ref, op=op, alpha=alpha), _remove_parallel_and_stats(y_pre, ref)[1]
    if y_pre.dim() != 3 or ref.dim() != 3:
        return _apply_parallel_op(y_pre, ref, op=op, alpha=alpha), _remove_parallel_and_stats(y_pre, ref)[1]
    if y_pre.size(-1) != num_heads * head_dim or ref.size(-1) != num_heads * head_dim:
        return _apply_parallel_op(y_pre, ref, op=op, alpha=alpha), _remove_parallel_and_stats(y_pre, ref)[1]

    y_heads = y_pre.reshape(*y_pre.shape[:-1], num_heads, head_dim)
    ref_heads = ref.reshape(*ref.shape[:-1], num_heads, head_dim)
    _, stats = _remove_parallel_and_stats(y_heads, ref_heads)
    new_heads = _apply_parallel_op(y_heads, ref_heads, op=op, alpha=alpha)
    return new_heads.reshape_as(y_pre), stats


def _extract_linear_value_ref(mixed_qkv: torch.Tensor, linear_module) -> Optional[torch.Tensor]:
    value_dim = getattr(linear_module, "value_dim", None)
    key_dim = getattr(linear_module, "key_dim", None)
    if not isinstance(mixed_qkv, torch.Tensor):
        return None
    if mixed_qkv.dim() != 3 or value_dim is None or key_dim is None:
        return None
    qkv_dim = key_dim * 2 + value_dim
    if mixed_qkv.size(-1) != qkv_dim:
        return None
    return mixed_qkv[..., 2 * key_dim :]


class RunningStats:
    def __init__(self, enabled: bool = False, track_layerwise: bool = False):
        self.enabled = bool(enabled)
        self.track_layerwise = bool(track_layerwise)
        self.global_sums = defaultdict(float)
        self.global_counts = defaultdict(int)
        self.global_mins = {}
        self.global_maxs = {}
        self.layer_sums = defaultdict(float)
        self.layer_counts = defaultdict(int)
        self.layer_mins = {}
        self.layer_maxs = {}

    def update_tensor(self, branch: str, layer_idx: int, name: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        v = value.detach().float()
        v_min = float(v.min().item())
        v_max = float(v.max().item())
        total = float(v.sum().item())
        count = int(v.numel())

        global_key = (branch, name)
        self.global_sums[global_key] += total
        self.global_counts[global_key] += count
        self.global_mins[global_key] = (
            v_min if global_key not in self.global_mins else min(self.global_mins[global_key], v_min)
        )
        self.global_maxs[global_key] = (
            v_max if global_key not in self.global_maxs else max(self.global_maxs[global_key], v_max)
        )

        if self.track_layerwise:
            layer_key = (branch, int(layer_idx), name)
            self.layer_sums[layer_key] += total
            self.layer_counts[layer_key] += count
            self.layer_mins[layer_key] = (
                v_min if layer_key not in self.layer_mins else min(self.layer_mins[layer_key], v_min)
            )
            self.layer_maxs[layer_key] = (
                v_max if layer_key not in self.layer_maxs else max(self.layer_maxs[layer_key], v_max)
            )

    def summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if not self.enabled:
            return {}
        layer_branches = {branch for (branch, _, _) in self.layer_sums.keys()}
        global_branches = {branch for (branch, _) in self.global_sums.keys()}
        all_branches = sorted(layer_branches | global_branches)

        layers: Dict[str, Dict[str, Dict[str, float]]] = {branch: {} for branch in all_branches}
        if self.track_layerwise:
            for (branch, layer_idx, name), total in sorted(self.layer_sums.items()):
                count = max(1, self.layer_counts[(branch, layer_idx, name)])
                layer_stats = layers.setdefault(branch, {}).setdefault(str(layer_idx), {})
                layer_stats[name] = total / count
                layer_stats[f"{name}_min"] = self.layer_mins[(branch, layer_idx, name)]
                layer_stats[f"{name}_max"] = self.layer_maxs[(branch, layer_idx, name)]

        overall: Dict[str, Dict[str, float]] = {branch: {} for branch in all_branches}
        for (branch, name), total in self.global_sums.items():
            count = max(1, self.global_counts[(branch, name)])
            branch_stats = overall.setdefault(branch, {})
            branch_stats[name] = total / count
            branch_stats[f"{name}_min"] = self.global_mins[(branch, name)]
            branch_stats[f"{name}_max"] = self.global_maxs[(branch, name)]
        return {"overall": overall, "layers": layers}


class QwenXSAForwardHooks:
    def __init__(
        self,
        model,
        target: str,
        start_layer: int = 0,
        end_layer: int = -1,
        skip_first_n: int = 0,
        skip_last_n: int = 0,
        intervention_site: str = "xsa_middle",
        xsa_forward_op: str = "remove_parallel",
        xsa_forward_alpha: float = 1.0,
        track_stats: bool = False,
        track_layerwise_stats: bool = False,
    ):
        if target not in {"attn", "mlp", "both"}:
            raise ValueError(f"target must be attn/mlp/both, got {target}")
        if intervention_site not in {"residual_output", "xsa_middle", "xsa_middle_multihead"}:
            raise ValueError(
                "intervention_site must be residual_output/xsa_middle/xsa_middle_multihead, "
                f"got {intervention_site}"
            )
        if intervention_site in {"xsa_middle", "xsa_middle_multihead"} and target != "attn":
            raise ValueError(
                f"{intervention_site} is attention-only and only supports target=attn, got target={target}"
            )
        self.model = model
        self.target = target
        self.intervention_site = intervention_site
        self.xsa_forward_op = str(xsa_forward_op).lower().strip()
        self.xsa_forward_alpha = float(xsa_forward_alpha)
        if self.xsa_forward_op not in {
            "remove_parallel",
            "keep_parallel",
            "add_parallel",
            "negate_parallel",
        }:
            raise ValueError(
                "xsa_forward_op must be one of remove_parallel/keep_parallel/add_parallel/negate_parallel, "
                f"got {xsa_forward_op}"
            )
        self.track_stats = bool(track_stats)
        self.track_layerwise_stats = bool(track_layerwise_stats)
        self.layers = _find_decoder_layers(model)
        n_layers = len(self.layers)
        lo = max(0, int(start_layer), int(skip_first_n))
        hi = n_layers if int(end_layer) < 0 else min(n_layers, int(end_layer))
        hi = min(hi, max(0, n_layers - int(skip_last_n)))
        if hi < lo:
            hi = lo
        self.active_layer_indices = set(range(lo, hi))
        self.layer_window = {
            "start_layer": lo,
            "end_layer_exclusive": hi,
            "skip_first_n": int(skip_first_n),
            "skip_last_n": int(skip_last_n),
            "n_layers": n_layers,
            "n_active_layers": len(self.active_layer_indices),
            "intervention_site": self.intervention_site,
            "track_stats": self.track_stats,
            "track_layerwise_stats": self.track_layerwise_stats,
            "xsa_forward_op": self.xsa_forward_op,
            "xsa_forward_alpha": self.xsa_forward_alpha,
            "intervention_pair": (
                "x_to_y_post"
                if self.intervention_site == "residual_output"
                else (
                    "token_mixer_value_to_out_proj_input_merged_heads"
                    if self.intervention_site == "xsa_middle"
                    else "token_mixer_value_to_out_proj_input_multihead"
                )
            ),
        }
        self.handles = []
        self.attn_residual = {}
        self.mlp_residual = {}
        self.attn_value = {}
        self.stats = RunningStats(
            enabled=self.track_stats,
            track_layerwise=self.track_layerwise_stats,
        )

    def _record_stats(self, branch: str, layer_idx: int, stats: Dict[str, torch.Tensor]) -> None:
        for name, value in stats.items():
            self.stats.update_tensor(branch, layer_idx, name, value)

    def attach(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in self.active_layer_indices:
                continue
            mixer_kind, mixer_module = _get_token_mixer_kind_and_module(layer)
            if self.target in {"attn", "both"} and mixer_module is None:
                eval_logger.warning(
                    "Skipping layer %s for XSA: missing a supported token mixer "
                    "(self_attn / linear_attn / attn / attention / token_mixer).",
                    layer_idx,
                )
                continue
            if self.target in {"mlp", "both"} and not hasattr(layer, "mlp"):
                raise ValueError(f"Layer {layer_idx} is missing mlp; not a supported decoder layer.")

            if self.intervention_site == "residual_output":
                def layer_pre_hook(_mod, args, kwargs, _idx=layer_idx):
                    hidden = _extract_hidden(args, kwargs)
                    if hidden is not None:
                        self.attn_residual[_idx] = hidden.detach()

                self.handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

            if self.target in {"attn", "both"}:
                if mixer_kind == "full_attention" and self.intervention_site == "residual_output":
                    v_proj = getattr(mixer_module, "v_proj", None)
                    o_proj = getattr(mixer_module, "o_proj", None)
                    if v_proj is not None and o_proj is not None:
                        def v_proj_stats_hook(_mod, _args, _kwargs, output, _idx=layer_idx):
                            if isinstance(output, torch.Tensor):
                                self.attn_value[_idx] = output.detach()
                            return output

                        def o_proj_stats_pre_hook(_mod, args, kwargs, _idx=layer_idx, _attn=mixer_module):
                            y_pre = args[0] if args else kwargs.get("input", None)
                            value = self.attn_value.get(_idx)
                            if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                                return None
                            ref = _expand_attn_value_ref(value, y_pre, _attn)
                            if ref is None or ref.shape != y_pre.shape:
                                return None
                            _, stats = _remove_parallel_and_stats(y_pre, ref)
                            self._record_stats("attn_pre_o_proj", _idx, stats)
                            self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                            return None

                        self.handles.append(v_proj.register_forward_hook(v_proj_stats_hook, with_kwargs=True))
                        self.handles.append(o_proj.register_forward_pre_hook(o_proj_stats_pre_hook, with_kwargs=True))

                    def attn_hook(_mod, args, kwargs, output, _idx=layer_idx):
                        residual = self.attn_residual.get(_idx)
                        if residual is None:
                            return output
                        attn_out = output[0] if isinstance(output, tuple) else output
                        if not isinstance(attn_out, torch.Tensor) or attn_out.shape != residual.shape:
                            return output
                        _, stats = _remove_parallel_and_stats(attn_out, residual)
                        new_attn = _apply_parallel_op(
                            attn_out,
                            residual,
                            op=self.xsa_forward_op,
                            alpha=self.xsa_forward_alpha,
                        )
                        self._record_stats("attn", _idx, stats)
                        self._record_stats("attn_post_o_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_post_out_proj", _idx, stats)
                        if isinstance(output, tuple):
                            return (new_attn,) + output[1:]
                        return new_attn

                    self.handles.append(mixer_module.register_forward_hook(attn_hook, with_kwargs=True))
                elif mixer_kind == "full_attention":
                    v_proj = getattr(mixer_module, "v_proj", None)
                    o_proj = getattr(mixer_module, "o_proj", None)
                    if v_proj is None or o_proj is None:
                        raise ValueError(
                            f"Layer {layer_idx} attention lacks v_proj/o_proj; cannot run {self.intervention_site}."
                        )

                    def v_proj_hook(_mod, _args, _kwargs, output, _idx=layer_idx):
                        if isinstance(output, torch.Tensor):
                            self.attn_value[_idx] = output.detach()
                        return output

                    def o_proj_pre_hook(_mod, args, kwargs, _idx=layer_idx, _attn=mixer_module):
                        y_pre = args[0] if args else kwargs.get("input", None)
                        value = self.attn_value.get(_idx)
                        if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                            return None
                        ref = _expand_attn_value_ref(value, y_pre, _attn)
                        if ref is None or ref.shape != y_pre.shape:
                            return None
                        if self.intervention_site == "xsa_middle":
                            _, stats = _remove_parallel_and_stats(y_pre, ref)
                            new_y_pre = _apply_parallel_op(
                                y_pre,
                                ref,
                                op=self.xsa_forward_op,
                                alpha=self.xsa_forward_alpha,
                            )
                        else:
                            new_y_pre, stats = _apply_parallel_multihead(
                                y_pre,
                                ref,
                                num_heads=getattr(_attn, "num_heads", None),
                                head_dim=getattr(_attn, "head_dim", None),
                                op=self.xsa_forward_op,
                                alpha=self.xsa_forward_alpha,
                            )
                        self._record_stats("attn", _idx, stats)
                        self._record_stats("attn_pre_o_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                        return _replace_first_arg(args, kwargs, new_y_pre)

                    self.handles.append(v_proj.register_forward_hook(v_proj_hook, with_kwargs=True))
                    self.handles.append(o_proj.register_forward_pre_hook(o_proj_pre_hook, with_kwargs=True))
                elif mixer_kind == "linear_attention" and self.intervention_site == "residual_output":
                    qkv_proj = getattr(mixer_module, "in_proj_qkv", None)
                    out_proj = getattr(mixer_module, "out_proj", None)
                    if qkv_proj is not None and out_proj is not None:
                        def qkv_stats_hook(_mod, _args, _kwargs, output, _idx=layer_idx, _mixer=mixer_module):
                            value = _extract_linear_value_ref(output, _mixer)
                            if isinstance(value, torch.Tensor):
                                self.attn_value[_idx] = value.detach()
                            return output

                        def linear_out_proj_stats_pre_hook(_mod, args, kwargs, _idx=layer_idx):
                            y_pre = args[0] if args else kwargs.get("input", None)
                            value = self.attn_value.get(_idx)
                            if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                                return None
                            if value.shape != y_pre.shape:
                                return None
                            _, stats = _remove_parallel_and_stats(y_pre, value)
                            self._record_stats("linear_pre_out_proj", _idx, stats)
                            self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                            return None

                        self.handles.append(qkv_proj.register_forward_hook(qkv_stats_hook, with_kwargs=True))
                        self.handles.append(out_proj.register_forward_pre_hook(linear_out_proj_stats_pre_hook, with_kwargs=True))

                    def linear_hook(_mod, args, kwargs, output, _idx=layer_idx):
                        residual = self.attn_residual.get(_idx)
                        if residual is None:
                            return output
                        linear_out = output[0] if isinstance(output, tuple) else output
                        if not isinstance(linear_out, torch.Tensor) or linear_out.shape != residual.shape:
                            return output
                        _, stats = _remove_parallel_and_stats(linear_out, residual)
                        new_linear = _apply_parallel_op(
                            linear_out,
                            residual,
                            op=self.xsa_forward_op,
                            alpha=self.xsa_forward_alpha,
                        )
                        self._record_stats("linear", _idx, stats)
                        self._record_stats("linear_post_out_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_post_out_proj", _idx, stats)
                        if isinstance(output, tuple):
                            return (new_linear,) + output[1:]
                        return new_linear

                    self.handles.append(mixer_module.register_forward_hook(linear_hook, with_kwargs=True))
                elif mixer_kind == "linear_attention":
                    qkv_proj = getattr(mixer_module, "in_proj_qkv", None)
                    out_proj = getattr(mixer_module, "out_proj", None)
                    if qkv_proj is None or out_proj is None:
                        raise ValueError(
                            f"Layer {layer_idx} linear attention lacks in_proj_qkv/out_proj; cannot run {self.intervention_site}."
                        )

                    def qkv_hook(_mod, _args, _kwargs, output, _idx=layer_idx, _mixer=mixer_module):
                        value = _extract_linear_value_ref(output, _mixer)
                        if isinstance(value, torch.Tensor):
                            self.attn_value[_idx] = value.detach()
                        return output

                    def linear_out_proj_pre_hook(_mod, args, kwargs, _idx=layer_idx, _mixer=mixer_module):
                        y_pre = args[0] if args else kwargs.get("input", None)
                        value = self.attn_value.get(_idx)
                        if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                            return None
                        if value.shape != y_pre.shape:
                            return None
                        if self.intervention_site == "xsa_middle":
                            _, stats = _remove_parallel_and_stats(y_pre, value)
                            new_y_pre = _apply_parallel_op(
                                y_pre,
                                value,
                                op=self.xsa_forward_op,
                                alpha=self.xsa_forward_alpha,
                            )
                        else:
                            new_y_pre, stats = _apply_parallel_multihead(
                                y_pre,
                                value,
                                num_heads=getattr(_mixer, "num_v_heads", None),
                                head_dim=getattr(_mixer, "head_v_dim", None),
                                op=self.xsa_forward_op,
                                alpha=self.xsa_forward_alpha,
                            )
                        self._record_stats("linear", _idx, stats)
                        self._record_stats("linear_pre_out_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                        return _replace_first_arg(args, kwargs, new_y_pre)

                    self.handles.append(qkv_proj.register_forward_hook(qkv_hook, with_kwargs=True))
                    self.handles.append(out_proj.register_forward_pre_hook(linear_out_proj_pre_hook, with_kwargs=True))

            if self.target in {"mlp", "both"}:
                norm = getattr(layer, "post_attention_layernorm", None)
                if norm is None:
                    raise ValueError(
                        f"Layer {layer_idx} has no post_attention_layernorm; cannot capture MLP residual."
                    )

                def mlp_pre_hook(_mod, args, kwargs, _idx=layer_idx):
                    hidden = args[0] if args else kwargs.get("hidden_states", None)
                    if hidden is not None:
                        self.mlp_residual[_idx] = hidden.detach()

                self.handles.append(norm.register_forward_pre_hook(mlp_pre_hook, with_kwargs=True))

                def mlp_hook(_mod, args, kwargs, output, _idx=layer_idx):
                    residual = self.mlp_residual.get(_idx)
                    if residual is None:
                        return output
                    mlp_out = output[0] if isinstance(output, tuple) else output
                    if not isinstance(mlp_out, torch.Tensor) or mlp_out.shape != residual.shape:
                        return output
                    _, stats = _remove_parallel_and_stats(mlp_out, residual)
                    new_mlp = _apply_parallel_op(
                        mlp_out,
                        residual,
                        op=self.xsa_forward_op,
                        alpha=self.xsa_forward_alpha,
                    )
                    self._record_stats("mlp", _idx, stats)
                    if isinstance(output, tuple):
                        return (new_mlp,) + output[1:]
                    return new_mlp

                self.handles.append(layer.mlp.register_forward_hook(mlp_hook, with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.attn_residual.clear()
        self.mlp_residual.clear()
        self.attn_value.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
