from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MIN_RECOMMENDED_PROMPT_TOKENS = 8
GEOMETRY_SPACE_CHOICES = ("hidden", "logits")


@dataclass
class LayerMetrics:
    layer: int
    z_post_norm: float
    z_mean: float
    z_median: float
    scale_gain_to_next: Optional[float]
    io_cos_sim: Optional[float]
    dz_norm: Optional[float]
    dz_mean: Optional[float]
    dz_median: Optional[float]
    dz_para_norm: Optional[float]
    dz_perp_norm: Optional[float]
    dz_perp_over_z_plus_dz_para: Optional[float]
    para_perp_ratio: Optional[float]


def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x, dim=-1).clamp_min(eps)


def _project_parallel(delta: torch.Tensor, base: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # projection of delta onto base
    denom = (base * base).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = (delta * base).sum(dim=-1, keepdim=True) / denom
    return coeff * base


def _topk_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1).indices


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def _delta_geometry(delta: torch.Tensor, base: torch.Tensor) -> Dict[str, float]:
    dz_para = _project_parallel(delta, base)
    dz_perp = delta - dz_para

    para_norm_t = _safe_norm(dz_para)
    perp_norm_t = _safe_norm(dz_perp)
    z_norm_t = _safe_norm(base)
    dz_norm_t = _safe_norm(delta)

    para_norm = float(para_norm_t.mean().item())
    perp_norm = float(perp_norm_t.mean().item())
    ratio = float(((para_norm_t + 1e-12) / (perp_norm_t + 1e-12)).mean().item())
    perp_over_z_plus_para = float((perp_norm_t / (z_norm_t + para_norm_t + 1e-12)).mean().item())
    return {
        "dz_norm": float(dz_norm_t.mean().item()),
        "dz_mean": float(delta.abs().mean().item()),
        "dz_median": float(delta.abs().median().item()),
        "dz_para_norm": para_norm,
        "dz_perp_norm": perp_norm,
        "dz_perp_over_z_plus_dz_para": perp_over_z_plus_para,
        "para_perp_ratio": ratio,
    }


def _dim_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.float()
    return {
        "z_post_dim_mean": float(x.mean().item()),
        "z_post_dim_min": float(x.min().item()),
        "z_post_dim_max": float(x.max().item()),
    }


def _repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep <= 1:
        return x
    bsz, n_kv_heads, seqlen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seqlen, head_dim)
    return x.reshape(bsz, n_kv_heads * n_rep, seqlen, head_dim)


def _split_hidden_to_heads(x: torch.Tensor, num_heads: int, head_dim: Optional[int] = None) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected hidden tensor with shape [B, T, D], got {tuple(x.shape)}")
    if head_dim is None:
        if x.shape[-1] % max(int(num_heads), 1) != 0:
            raise ValueError(f"Hidden size {x.shape[-1]} is not divisible by num_heads={num_heads}")
        head_dim = x.shape[-1] // int(num_heads)
    return x.view(x.shape[0], x.shape[1], int(num_heads), int(head_dim)).permute(0, 2, 1, 3).contiguous()


def _merge_heads_to_hidden(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected head tensor with shape [B, H, T, Dh], got {tuple(x.shape)}")
    return x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])


class GeometryAnalyzer:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        max_memory_per_gpu: Optional[str] = None,
        max_memory_gpu0: Optional[str] = None,
        offload_folder: Optional[str] = None,
        require_multi_gpu: bool = False,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        random_init: bool = False,
        random_seed: Optional[int] = None,
        attn_implementation: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.require_multi_gpu = require_multi_gpu
        self.random_init = random_init
        self.random_seed = random_seed
        self.attn_implementation = attn_implementation
        if self.random_init and self.device_map is not None:
            raise ValueError("random_init currently does not support --device_map. Please unset --device_map.")
        model_path = Path(model_name_or_path)
        local_files_only = model_path.exists()

        torch_dtype = None
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = None
        if attn_implementation:
            cfg = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            cfg.attn_implementation = attn_implementation
            cfg._attn_implementation = attn_implementation

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
            model_kwargs["config"] = cfg
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            if torch.cuda.is_available() and (max_memory_per_gpu is not None or max_memory_gpu0 is not None):
                max_memory: Dict[int, str] = {}
                for i in range(torch.cuda.device_count()):
                    if i == 0 and max_memory_gpu0 is not None:
                        max_memory[i] = max_memory_gpu0
                    elif max_memory_per_gpu is not None:
                        max_memory[i] = max_memory_per_gpu
                if max_memory:
                    model_kwargs["max_memory"] = max_memory
            if offload_folder is not None:
                Path(offload_folder).mkdir(parents=True, exist_ok=True)
                model_kwargs["offload_folder"] = offload_folder

        if self.random_init:
            if self.random_seed is not None:
                torch.manual_seed(int(self.random_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.random_seed))
            if cfg is None:
                cfg = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
            self.model = AutoModelForCausalLM.from_config(
                cfg,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.hf_device_map = getattr(self.model, "hf_device_map", None)
        self.loaded_attn_implementation = getattr(
            getattr(self.model, "config", None),
            "_attn_implementation",
            getattr(getattr(self.model, "config", None), "attn_implementation", None),
        )
        self.cuda_devices_used = self._extract_cuda_devices_used()
        if self.require_multi_gpu and self.device_map is not None and torch.cuda.device_count() > 1:
            if len(self.cuda_devices_used) < 2:
                raise RuntimeError(
                    "Model load did not shard across multiple GPUs. "
                    f"CUDA devices used: {self.cuda_devices_used or ['none']}. "
                    "Try --device_map balanced and set --max_memory_per_gpu (for example: 40GiB)."
                )
        self.input_device = self._resolve_input_device()
        self._warned_messages: set[str] = set()

        if not hasattr(self.model, "lm_head"):
            raise ValueError("Model has no lm_head; logits-space analysis requires a causal LM with lm_head.")

    def _lm_head_device(self) -> torch.device:
        try:
            return next(self.model.lm_head.parameters()).device
        except StopIteration:
            return self.input_device

    def _project_with_lm_head(self, h: torch.Tensor) -> torch.Tensor:
        # Hidden states may come from different shards/devices; always align both
        # device and dtype to output embedding weights before projection.
        lm_head = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
        if lm_head is None:
            lm_head = self.model.lm_head

        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor):
            target_device = weight.device
            target_dtype = weight.dtype
        else:
            param = next(lm_head.parameters(), None)
            if param is not None:
                target_device = param.device
                target_dtype = param.dtype
            else:
                target_device = self._lm_head_device()
                target_dtype = h.dtype

        h_proj = h.to(device=target_device, dtype=target_dtype)
        return lm_head(h_proj).float()

    def _resolve_input_device(self) -> torch.device:
        if self.device_map is None:
            return torch.device(self.device)

        hf_map = self.hf_device_map
        if not hf_map:
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                return torch.device(model_device)
            return torch.device(self.device)

        # Pick a real compute device (prefer CUDA) for input tensors.
        for value in hf_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value.startswith("cuda"):
                return torch.device(value)
        for value in hf_map.values():
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)
        return torch.device("cpu")

    def _extract_cuda_devices_used(self) -> List[int]:
        hf_map = self.hf_device_map
        if not hf_map:
            model_device = getattr(self.model, "device", None)
            if model_device is not None and str(model_device).startswith("cuda:"):
                return [int(str(model_device).split(":", 1)[1])]
            return []

        used = set()
        for value in hf_map.values():
            if isinstance(value, int):
                used.add(value)
            elif isinstance(value, str) and value.startswith("cuda:"):
                try:
                    used.add(int(value.split(":", 1)[1]))
                except ValueError:
                    continue
        return sorted(used)

    def _warn_once(self, msg: str) -> None:
        if msg in self._warned_messages:
            return
        self._warned_messages.add(msg)
        print(f"[WARN] {msg}")

    def _get_decoder_blocks(self) -> Optional[Any]:
        def _get_attr_path(root: Any, path: str) -> Any:
            cur = root
            for part in path.split("."):
                if cur is None or not hasattr(cur, part):
                    return None
                cur = getattr(cur, part)
            return cur

        candidate_paths = [
            "transformer.h",
            "layers",
            "model.layers",
            "model.model.layers",
            "base_model.layers",
            "base_model.model.layers",
            "language_model.layers",
            "language_model.model.layers",
        ]
        for p in candidate_paths:
            candidate = _get_attr_path(self.model, p)
            if candidate is not None and len(candidate) > 0:
                return candidate
        return None

    def _should_omit_attention_mask(self) -> bool:
        """
        Whitelist families that are known to hit strict/custom attention-mask
        shape checks in remote-code implementations during forward/generate.
        """
        name = str(self.model_name_or_path).lower()
        model_type = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        cls_name = type(self.model).__name__.lower()
        arch = " ".join(getattr(getattr(self.model, "config", None), "architectures", []) or []).lower()
        hay = " ".join([name, model_type, cls_name, arch])
        whitelist = (
            "moonlight",
            "deepseek",
        )
        return any(k in hay for k in whitelist)

    def _should_use_forward_decode(self) -> bool:
        """
        Some remote-code families have brittle generate()+cache+mask plumbing.
        For those, use explicit forward decoding.
        """
        name = str(self.model_name_or_path).lower()
        model_type = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        cls_name = type(self.model).__name__.lower()
        arch = " ".join(getattr(getattr(self.model, "config", None), "architectures", []) or []).lower()
        hay = " ".join([name, model_type, cls_name, arch])
        return "moonlight" in hay

    def _model_forward_with_mask_retry(self, keep_attention_mask: bool = False, **kwargs):
        # Only omit attention_mask for known-quirky remote-code paths when decoding with cache.
        has_kv_cache = kwargs.get("past_key_values") is not None
        if (
            "attention_mask" in kwargs
            and self._should_omit_attention_mask()
            and has_kv_cache
            and not keep_attention_mask
        ):
            kwargs = dict(kwargs)
            kwargs.pop("attention_mask", None)
        return self.model(**kwargs)

    def _model_generate_with_mask_retry(self, **kwargs):
        if "attention_mask" in kwargs and self._should_omit_attention_mask():
            kwargs = dict(kwargs)
            kwargs.pop("attention_mask", None)
        return self.model.generate(**kwargs)

    def _forward_with_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.Tensor] = None,
        keep_attention_mask: bool = False,
    ) -> Tuple[Any, Tuple[torch.Tensor, ...]]:
        """
        Returns (outputs, hidden_states_tuple).
        Fallback for models that ignore or do not return `output_hidden_states`.
        """
        outputs = self._model_forward_with_mask_retry(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            keep_attention_mask=keep_attention_mask,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return outputs, hidden_states

        blocks = self._get_decoder_blocks()
        if blocks is None:
            raise ValueError(
                "Model did not return hidden_states and decoder blocks were not found for hook-based fallback."
            )

        layer_inputs: List[torch.Tensor] = []
        hooks = []
        for block in blocks:
            def make_pre_hook():
                def _pre_hook(_module, inputs):
                    if inputs:
                        layer_inputs.append(inputs[0].detach())
                return _pre_hook

            hooks.append(block.register_forward_pre_hook(make_pre_hook()))

        try:
            outputs = self._model_forward_with_mask_retry(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=use_cache,
                past_key_values=past_key_values,
                cache_position=cache_position,
                keep_attention_mask=keep_attention_mask,
            )
        finally:
            for h in hooks:
                h.remove()

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Hook fallback failed: model outputs do not contain `last_hidden_state`.")
        if not layer_inputs:
            raise ValueError("Hook fallback failed: no decoder layer inputs were captured.")

        hidden_states = tuple(layer_inputs + [last_hidden.detach()])
        return outputs, hidden_states

    @torch.no_grad()
    def _decode_with_forward_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, ...]], List[torch.Tensor]]:
        """
        Manual autoregressive decoding with forward()+cache.
        Returns:
          full_ids, generated_token_ids, per_step_hidden_states, per_step_scores
        """
        full_ids = input_ids
        full_mask = attention_mask
        past_key_values = None

        per_step_hidden_states: List[Tuple[torch.Tensor, ...]] = []
        per_step_scores: List[torch.Tensor] = []
        generated_tokens: List[torch.Tensor] = []

        for step in range(max_new_tokens):
            if step == 0:
                cur_input_ids = full_ids
            else:
                cur_input_ids = generated_tokens[-1]

            outputs, step_hidden_states = self._forward_with_hidden_states(
                input_ids=cur_input_ids,
                attention_mask=full_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=None,
                keep_attention_mask=True,
            )

            score = outputs.logits[:, -1, :]
            per_step_scores.append(score.detach())
            per_step_hidden_states.append(tuple(h.detach() for h in step_hidden_states))

            if do_sample:
                t = max(float(temperature), 1e-6)
                probs = F.softmax(score / t, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(score, dim=-1, keepdim=True)

            generated_tokens.append(next_token)
            full_ids = torch.cat([full_ids, next_token], dim=1)
            one_mask = torch.ones(
                (full_mask.shape[0], 1),
                dtype=full_mask.dtype,
                device=full_mask.device,
            )
            full_mask = torch.cat([full_mask, one_mask], dim=1)
            past_key_values = outputs.past_key_values

            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and torch.all(next_token == eos_id):
                break

        if generated_tokens:
            generated_token_ids = torch.cat(generated_tokens, dim=1)
        else:
            generated_token_ids = full_ids.new_zeros((full_ids.shape[0], 0))

        return full_ids, generated_token_ids, per_step_hidden_states, per_step_scores

    def _get_attention_output_projection(self, attn_mod: Any) -> Any:
        for name in (
            "o_proj",
            "out_proj",
            "c_proj",
            "dense",
            "wo",
            "w_o",
            "out",
            "proj",
            "output_proj",
            "out_linear",
            "linear_out",
        ):
            if hasattr(attn_mod, name):
                return getattr(attn_mod, name)
        linear_children = [
            module for _, module in attn_mod.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if linear_children:
            return linear_children[-1]
        raise ValueError(f"Unsupported attention output projection for module {type(attn_mod).__name__}")

    def _attention_child_module_summary(self, attn_mod: Any, limit: int = 24) -> str:
        try:
            items = list(attn_mod.named_children())
        except Exception:  # noqa: BLE001
            return "<named_children unavailable>"
        if not items:
            return "<no direct child modules>"
        return ", ".join(f"{name}:{type(module).__name__}" for name, module in items[:limit])

    def _value_metric_capture_summary(
        self,
        attn_mods: Dict[int, Any],
        attn_inputs: Dict[int, torch.Tensor],
        attn_value_outputs: Dict[int, torch.Tensor],
        attn_pre_outputs: Dict[int, torch.Tensor],
        attn_weights: Dict[int, torch.Tensor],
    ) -> str:
        layer = min(attn_mods) if attn_mods else None
        if layer is None:
            module_summary = "<no attention modules captured>"
        else:
            attn_mod = attn_mods[layer]
            module_summary = (
                f"first_layer={layer}, module={type(attn_mod).__name__}, "
                f"children=[{self._attention_child_module_summary(attn_mod)}]"
            )
        return (
            f"attn_mod_layers={sorted(attn_mods)[:8]} "
            f"attn_input_layers={sorted(attn_inputs)[:8]} "
            f"attn_value_output_layers={sorted(attn_value_outputs)[:8]} "
            f"attn_pre_output_layers={sorted(attn_pre_outputs)[:8]} "
            f"attn_weight_layers={sorted(attn_weights)[:8]} "
            f"{module_summary}"
        )

    def _apply_output_projection(self, attn_mod: Any, x: torch.Tensor) -> torch.Tensor:
        return self._get_attention_output_projection(attn_mod)(x)

    def _attention_num_heads(self, attn_mod: Any) -> int:
        config = getattr(self.model, "config", None)
        for obj in (attn_mod, config):
            if obj is None:
                continue
            for name in ("num_heads", "n_head", "num_attention_heads", "n_heads"):
                value = getattr(obj, name, None)
                if value:
                    return int(value)
        return 0

    def _attention_num_kv_heads(self, attn_mod: Any, num_heads: int) -> int:
        config = getattr(self.model, "config", None)
        for obj in (attn_mod, config):
            if obj is None:
                continue
            for name in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
                value = getattr(obj, name, None)
                if value:
                    return int(value)
        return int(num_heads)

    def _attention_head_dim(self, attn_mod: Any) -> Optional[int]:
        config = getattr(self.model, "config", None)
        for obj in (attn_mod, config):
            if obj is None:
                continue
            for name in ("head_dim", "attention_head_size"):
                value = getattr(obj, name, None)
                if value:
                    return int(value)
        return None

    def _extract_value_heads(self, attn_mod: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(attn_mod, "v_proj"):
            value_proj = attn_mod.v_proj(hidden_states)
        elif hasattr(attn_mod, "c_attn"):
            qkv = attn_mod.c_attn(hidden_states)
            split_size = getattr(attn_mod, "split_size", hidden_states.shape[-1])
            if qkv.shape[-1] == split_size * 3:
                _, _, value_proj = qkv.split(split_size, dim=-1)
            else:
                chunks = torch.chunk(qkv, 3, dim=-1)
                if len(chunks) != 3:
                    raise ValueError(f"Could not split GPT-style qkv for module {type(attn_mod).__name__}")
                value_proj = chunks[2]
        else:
            raise ValueError(f"Unsupported attention value projection for module {type(attn_mod).__name__}")

        num_heads = self._attention_num_heads(attn_mod)
        if num_heads <= 0:
            raise ValueError(f"Attention module {type(attn_mod).__name__} is missing num_attention_heads")
        head_dim = self._attention_head_dim(attn_mod)
        num_kv_heads = self._attention_num_kv_heads(attn_mod, num_heads)
        value_heads = _split_hidden_to_heads(value_proj, num_kv_heads, head_dim=head_dim)
        if num_kv_heads != num_heads:
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads} "
                    f"for module {type(attn_mod).__name__}"
                )
            value_heads = _repeat_kv_heads(value_heads, num_heads // num_kv_heads)
        return value_heads

    def _value_projection_to_heads(
        self,
        attn_mod: Any,
        value_proj: torch.Tensor,
        query_num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        num_heads = query_num_heads or self._attention_num_heads(attn_mod)
        head_dim = self._attention_head_dim(attn_mod)
        if head_dim is None and num_heads > 0 and value_proj.shape[-1] % num_heads == 0:
            head_dim = value_proj.shape[-1] // num_heads
        if head_dim is None:
            hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
            cfg_heads = self._attention_num_heads(attn_mod)
            if hidden_size and cfg_heads:
                head_dim = int(hidden_size) // int(cfg_heads)
        if head_dim is None:
            raise ValueError(
                f"Attention module {type(attn_mod).__name__} is missing head_dim and it could not be inferred"
            )
        if num_heads <= 0:
            num_heads = query_num_heads or self._attention_num_heads(attn_mod)
        if num_heads <= 0:
            num_heads = max(int(getattr(getattr(self.model, "config", None), "num_attention_heads", 0) or 0), 0)
        if num_heads <= 0:
            num_heads = max(value_proj.shape[-1] // int(head_dim), 1)
        num_kv_heads = self._attention_num_kv_heads(attn_mod, num_heads)
        inferred_kv_heads = value_proj.shape[-1] // int(head_dim)
        if inferred_kv_heads > 0:
            num_kv_heads = inferred_kv_heads
        value_heads = _split_hidden_to_heads(value_proj, num_kv_heads, head_dim=head_dim)
        if num_kv_heads != num_heads:
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads} "
                    f"for module {type(attn_mod).__name__}"
                )
            value_heads = _repeat_kv_heads(value_heads, num_heads // num_kv_heads)
        return value_heads

    def _compute_attn_value_metrics(
        self,
        attn_mods: Dict[int, Any],
        attn_inputs: Dict[int, torch.Tensor],
        attn_value_outputs: Dict[int, torch.Tensor],
        attn_pre_outputs: Dict[int, torch.Tensor],
        attn_weights: Dict[int, torch.Tensor],
        hs: List[torch.Tensor],
    ) -> Dict[int, Dict[str, float]]:
        value_metrics: Dict[int, Dict[str, float]] = {}

        for layer_idx, attn_mod in attn_mods.items():
            if layer_idx >= len(hs) - 1:
                continue
            if layer_idx not in attn_inputs and layer_idx not in attn_value_outputs:
                continue

            try:
                if layer_idx in attn_value_outputs:
                    value_proj = attn_value_outputs[layer_idx].float()
                    if value_proj.ndim != 3:
                        continue
                    query_num_heads = None
                    if layer_idx in attn_weights and attn_weights[layer_idx].ndim == 4:
                        query_num_heads = int(attn_weights[layer_idx].shape[1])
                    value_heads = self._value_projection_to_heads(
                        attn_mod,
                        value_proj,
                        query_num_heads=query_num_heads,
                    )
                else:
                    hidden_states = attn_inputs[layer_idx].float()
                    if hidden_states.ndim != 3:
                        continue
                    value_heads = self._extract_value_heads(attn_mod, hidden_states)
                key_idx = value_heads.shape[-2] - 1
                self_value_heads = value_heads[:, :, key_idx:key_idx + 1, :]
                self_value = _merge_heads_to_hidden(self_value_heads)[:, 0, :].float()
                if layer_idx in attn_pre_outputs:
                    y_pre_tensor = attn_pre_outputs[layer_idx].float()
                    if y_pre_tensor.ndim == 3:
                        y_pre = y_pre_tensor[:, -1, :].float()
                    elif y_pre_tensor.ndim == 2:
                        y_pre = y_pre_tensor[-1:, :].float()
                    else:
                        raise ValueError(f"Unsupported o_proj input shape: {tuple(y_pre_tensor.shape)}")
                elif layer_idx in attn_weights:
                    weights = attn_weights[layer_idx].float()
                    if weights.ndim != 4 or weights.shape[-2] == 0 or weights.shape[-1] == 0:
                        continue
                    y_pre_heads = torch.matmul(
                        weights[:, :, -1:, :],
                        value_heads,
                    )
                    y_pre = _merge_heads_to_hidden(y_pre_heads)[:, 0, :].float()
                else:
                    if layer_idx == 0:
                        self._warn_once(
                            "Value-space attention pre-output was not captured. "
                            f"Layer-0 attention module: {type(attn_mod).__name__}; "
                            f"children: {self._attention_child_module_summary(attn_mod)}"
                        )
                    continue
            except Exception as e:  # noqa: BLE001
                self._warn_once(
                    f"Value-based sublayer metrics unavailable for layer {layer_idx} "
                    f"({type(attn_mod).__name__}): {e}"
                )
                continue

            z_in = hs[layer_idx].to(y_pre.device).float()
            value_pre_geo = _delta_geometry(y_pre, self_value)
            z_norm = float(_safe_norm(z_in).mean().item())
            y_pre_norm = float(_safe_norm(y_pre).mean().item())
            self_value_norm = float(_safe_norm(self_value).mean().item())
            value_pre_alignment = float(
                F.cosine_similarity(y_pre, self_value, dim=-1, eps=1e-8).mean().item()
            )

            value_metrics[layer_idx] = {
                "value_pre_norm": y_pre_norm,
                "value_pre_self_value_norm": self_value_norm,
                "value_pre_self_alignment": value_pre_alignment,
                "value_pre_para_norm": value_pre_geo["dz_para_norm"],
                "value_pre_perp_norm": value_pre_geo["dz_perp_norm"],
                "value_pre_para_perp_ratio": value_pre_geo["para_perp_ratio"],
                "value_pre_para_ratio": float(
                    value_pre_geo["dz_para_norm"] / max(y_pre_norm, 1e-12)
                ),
                "value_pre_perp_ratio": float(
                    value_pre_geo["dz_perp_norm"] / max(y_pre_norm, 1e-12)
                ),
            }

            if layer_idx in attn_weights:
                weights = attn_weights[layer_idx].float()
                if weights.ndim == 4 and weights.shape[-2] > 0 and weights.shape[-1] > 0:
                    diag = weights[:, :, -1, -1]
                    diag_mean = float(diag.mean().item())
                    diag_min = float(diag.min().item())
                    diag_max = float(diag.max().item())
                    self_context_heads = diag[:, :, None, None] * self_value_heads
                    self_context = _merge_heads_to_hidden(self_context_heads)
                    self_term = self._apply_output_projection(attn_mod, self_context)[:, 0, :].float()
                    z_for_self = z_in.to(self_term.device)
                    value_geo = _delta_geometry(self_term, z_for_self)
                    self_norm = float(_safe_norm(self_term).mean().item())
                    value_alignment = float(
                        F.cosine_similarity(self_term, z_for_self, dim=-1, eps=1e-8).mean().item()
                    )
                    value_metrics[layer_idx].update(
                        {
                            "diag_mean": diag_mean,
                            "diag_min": diag_min,
                            "diag_max": diag_max,
                            "attn_alpha_mean": diag_mean,
                            "value_self_norm": self_norm,
                            "value_self_over_z": self_norm / max(z_norm, 1e-12),
                            "value_self_alignment": value_alignment,
                            "diag_mean_times_value_self_alignment": diag_mean * value_alignment,
                            "value_self_term": diag_mean * value_alignment,
                            "value_self_para_norm": value_geo["dz_para_norm"],
                            "value_self_perp_norm": value_geo["dz_perp_norm"],
                            "value_self_para_over_z": value_geo["dz_para_norm"] / max(z_norm, 1e-12),
                            "value_self_perp_over_z": value_geo["dz_perp_norm"] / max(z_norm, 1e-12),
                            "value_self_para_perp_ratio": value_geo["para_perp_ratio"],
                        }
                    )

        return value_metrics

    @torch.no_grad()
    def _collect_gpt2_sublayer_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        collect_attn_weights: bool = False,
    ) -> Tuple[
        Any,
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, Any],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
    ]:
        # Best-effort support for common decoder stacks and wrappers:
        # - GPT2-style: model.transformer.h[i].attn / .mlp
        # - LLaMA/Qwen/Gemma-style: *.layers[i].self_attn / .mlp
        # The underlying model may be nested under wrapper attrs (base_model/language_model/model/...).
        blocks = self._get_decoder_blocks()
        if blocks is None:
            raise ValueError(
                "Sublayer analysis requires decoder blocks with attention+MLP (e.g. transformer.h or *.layers)."
            )

        attn_outs: Dict[int, torch.Tensor] = {}
        mlp_outs: Dict[int, torch.Tensor] = {}
        attn_mods: Dict[int, Any] = {}
        attn_inputs: Dict[int, torch.Tensor] = {}
        attn_value_outputs: Dict[int, torch.Tensor] = {}
        attn_pre_outputs: Dict[int, torch.Tensor] = {}
        attn_weights: Dict[int, torch.Tensor] = {}
        hooks = []

        for i, block in enumerate(blocks):
            attn_mod = None
            if hasattr(block, "attn"):
                attn_mod = block.attn
            elif hasattr(block, "self_attn"):
                attn_mod = block.self_attn
            elif hasattr(block, "linear_attn"):
                # Qwen3Next linear-attention layers
                attn_mod = block.linear_attn

            mlp_mod = None
            if hasattr(block, "mlp"):
                mlp_mod = block.mlp
            elif hasattr(block, "feed_forward"):
                mlp_mod = block.feed_forward
            elif hasattr(block, "ffn"):
                mlp_mod = block.ffn
            if attn_mod is None or mlp_mod is None:
                raise ValueError(
                    "Sublayer analysis requires each block to have token mixer "
                    "(.attn/.self_attn/.linear_attn) and MLP (.mlp/.feed_forward/.ffn)."
                )
            attn_mods[i] = attn_mod

            # Only treat post_*_layernorm as residual deltas for Gemma-style blocks.
            # For other families, keep hooks on raw attn/mlp modules.
            block_cls = type(block).__name__.lower()
            is_gemma_style = (
                "gemma" in block_cls
                or (
                    hasattr(block, "pre_feedforward_layernorm")
                    and hasattr(block, "post_feedforward_layernorm")
                    and hasattr(block, "post_attention_layernorm")
                )
            )
            if is_gemma_style:
                attn_delta_mod = (
                    block.post_attention_layernorm if hasattr(block, "post_attention_layernorm") else attn_mod
                )
                mlp_delta_mod = (
                    block.post_feedforward_layernorm if hasattr(block, "post_feedforward_layernorm") else mlp_mod
                )
            else:
                attn_delta_mod = attn_mod
                mlp_delta_mod = mlp_mod

            def make_attn_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    attn_outs[layer_idx] = out.detach()
                    if _inputs and layer_idx not in attn_inputs:
                        attn_inputs[layer_idx] = _inputs[0].detach()
                    if (
                        isinstance(output, tuple)
                        and len(output) > 1
                        and isinstance(output[1], torch.Tensor)
                        and layer_idx not in attn_weights
                    ):
                        attn_weights[layer_idx] = output[1].detach()
                return _hook

            def make_attn_value_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    if _inputs:
                        attn_inputs[layer_idx] = _inputs[0].detach()
                    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
                        attn_weights[layer_idx] = output[1].detach()
                return _hook

            def make_attn_pre_output_hook(layer_idx: int):
                def _pre_hook(_module, inputs):
                    if inputs:
                        attn_pre_outputs[layer_idx] = inputs[0].detach()
                return _pre_hook

            def make_value_output_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if isinstance(out, torch.Tensor):
                        attn_value_outputs[layer_idx] = out.detach()
                return _hook

            def make_mlp_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    mlp_outs[layer_idx] = out.detach()
                return _hook

            hooks.append(attn_delta_mod.register_forward_hook(make_attn_hook(i)))
            if attn_delta_mod is not attn_mod:
                hooks.append(attn_mod.register_forward_hook(make_attn_value_hook(i)))
            if hasattr(attn_mod, "v_proj"):
                hooks.append(attn_mod.v_proj.register_forward_hook(make_value_output_hook(i)))
            try:
                hooks.append(
                    self._get_attention_output_projection(attn_mod).register_forward_pre_hook(
                        make_attn_pre_output_hook(i)
                    )
                )
            except Exception as e:  # noqa: BLE001
                self._warn_once(
                    f"Could not hook attention pre-output for layer {i} "
                    f"({type(attn_mod).__name__}): {e}"
                )
            hooks.append(mlp_delta_mod.register_forward_hook(make_mlp_hook(i)))

        try:
            try:
                outputs = self._model_forward_with_mask_retry(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=collect_attn_weights,
                    return_dict=True,
                    use_cache=False,
                )
            except Exception:
                outputs = self._model_forward_with_mask_retry(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            for h in hooks:
                h.remove()

        model_attentions = getattr(outputs, "attentions", None)
        if model_attentions is not None:
            for i, attn in enumerate(model_attentions):
                if isinstance(attn, torch.Tensor) and i not in attn_weights:
                    attn_weights[i] = attn.detach()

        return outputs, attn_outs, mlp_outs, attn_mods, attn_inputs, attn_value_outputs, attn_pre_outputs, attn_weights

    def _build_sublayer_metrics(
        self,
        hs: List[torch.Tensor],
        attn_outs: Dict[int, torch.Tensor],
        mlp_outs: Dict[int, torch.Tensor],
        value_attn_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        sublayer_metrics: List[Dict[str, Any]] = []
        n_blocks = len(hs) - 1

        for l in range(n_blocks):
            if l not in attn_outs or l not in mlp_outs:
                continue

            z_in = hs[l].float()
            z_out = hs[l + 1].to(z_in.device).float()
            if z_in.ndim == 1:
                attn_delta = attn_outs[l][0, -1, :].to(z_in.device).float()
            else:
                attn_delta = attn_outs[l][0].to(z_in.device).float()

            z_after_attn = z_in + attn_delta
            if z_after_attn.ndim == 1:
                mlp_delta = mlp_outs[l][0, -1, :].to(z_after_attn.device).float()
            else:
                mlp_delta = mlp_outs[l][0].to(z_after_attn.device).float()
            attn_geo = _delta_geometry(attn_delta, z_in)
            mlp_geo = _delta_geometry(mlp_delta, z_after_attn)
            block_geo = _delta_geometry(z_out - z_in, z_in)
            block_geo["io_cos_sim"] = float(F.cosine_similarity(z_in, z_out, dim=-1, eps=1e-8).mean().item())

            attn_in_norm = float(_safe_norm(z_in).mean().item())
            attn_out_norm = float(_safe_norm(z_after_attn).mean().item())
            attn_geo["z_post_norm"] = attn_in_norm
            attn_geo.update(_dim_stats(z_in))
            attn_geo["scale_gain_to_next"] = attn_out_norm / max(attn_in_norm, 1e-12)
            attn_geo["io_cos_sim"] = float(F.cosine_similarity(z_in, z_after_attn, dim=-1, eps=1e-8).mean().item())
            attn_geo["dz_perp_over_z_plus_dz_para"] = float(
                attn_geo["dz_perp_norm"] / (attn_in_norm + attn_geo["dz_para_norm"] + 1e-12)
            )
            if value_attn_metrics and l in value_attn_metrics:
                attn_geo.update(value_attn_metrics[l])

            mlp_in_norm = float(_safe_norm(z_after_attn).mean().item())
            mlp_out_norm = float(_safe_norm(z_out).mean().item())
            mlp_geo["z_post_norm"] = mlp_in_norm
            mlp_geo.update(_dim_stats(z_after_attn))
            mlp_geo["scale_gain_to_next"] = mlp_out_norm / max(mlp_in_norm, 1e-12)
            mlp_geo["io_cos_sim"] = float(
                F.cosine_similarity(z_after_attn, z_out, dim=-1, eps=1e-8).mean().item()
            )
            mlp_geo["dz_perp_over_z_plus_dz_para"] = float(
                mlp_geo["dz_perp_norm"] / (mlp_in_norm + mlp_geo["dz_para_norm"] + 1e-12)
            )

            block_geo.update(_dim_stats(z_in))
            block_geo["dz_perp_over_z_plus_dz_para"] = float(
                block_geo["dz_perp_norm"] / (attn_in_norm + block_geo["dz_para_norm"] + 1e-12)
            )

            recomposed = z_after_attn + mlp_delta
            residual_error = float(_safe_norm(recomposed - z_out).mean().item())

            sublayer_metrics.append(
                {
                    "layer": l,
                    "attn_update": attn_geo,
                    "mlp_update": mlp_geo,
                    "block_update": block_geo,
                    "residual_recompose_error": residual_error,
                }
            )

        return sublayer_metrics

    @torch.no_grad()
    def _analyze_from_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 5,
        geometry_space: str = "hidden",
        include_sublayer_metrics: bool = False,
    ) -> Dict[str, Any]:
        attn_outs: Dict[int, torch.Tensor] = {}
        mlp_outs: Dict[int, torch.Tensor] = {}
        attn_mods: Dict[int, Any] = {}
        attn_inputs: Dict[int, torch.Tensor] = {}
        attn_value_outputs: Dict[int, torch.Tensor] = {}
        attn_pre_outputs: Dict[int, torch.Tensor] = {}
        attn_weights: Dict[int, torch.Tensor] = {}
        sublayer_warning: Optional[str] = None

        if include_sublayer_metrics:
            try:
                outputs, attn_outs, mlp_outs, attn_mods, attn_inputs, attn_value_outputs, attn_pre_outputs, attn_weights = self._collect_gpt2_sublayer_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    _, hidden_states = self._forward_with_hidden_states(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            except Exception as e:  # noqa: BLE001
                outputs, hidden_states = self._forward_with_hidden_states(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                sublayer_warning = f"Sublayer metrics unavailable: {e}"
                self._warn_once(sublayer_warning)
        else:
            outputs, hidden_states = self._forward_with_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        # Use current token (last position) representation for probe/generation consistency.
        hs = [h[:, -1, :].squeeze(0).float() for h in hidden_states]  # [(D,)]
        predicted_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        result = self._analyze_from_last_token_hs(
            hs=hs,
            top_k=top_k,
            geometry_space=geometry_space,
            predicted_token_id=predicted_token_id,
        )
        if include_sublayer_metrics:
            if attn_outs and mlp_outs:
                value_attn_metrics = self._compute_attn_value_metrics(
                    attn_mods=attn_mods,
                    attn_inputs=attn_inputs,
                    attn_value_outputs=attn_value_outputs,
                    attn_pre_outputs=attn_pre_outputs,
                    attn_weights=attn_weights,
                    hs=hs,
                )
                if not value_attn_metrics:
                    msg = (
                        "Value-space attention metrics unavailable: no attention pre-output was captured. "
                        "Capture summary: "
                        + self._value_metric_capture_summary(
                            attn_mods=attn_mods,
                            attn_inputs=attn_inputs,
                            attn_value_outputs=attn_value_outputs,
                            attn_pre_outputs=attn_pre_outputs,
                            attn_weights=attn_weights,
                        )
                    )
                    self._warn_once(msg)
                    sublayer_warning = msg if sublayer_warning is None else sublayer_warning
                result["sublayer_metrics"] = self._build_sublayer_metrics(
                    hs=hs,
                    attn_outs=attn_outs,
                    mlp_outs=mlp_outs,
                    value_attn_metrics=value_attn_metrics,
                )
            if sublayer_warning is not None:
                result["sublayer_warning"] = sublayer_warning
        return result

    def _analyze_from_last_token_hs(
        self,
        hs: List[torch.Tensor],
        top_k: int,
        geometry_space: str = "hidden",
        predicted_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        # hs: per-layer states across prompt tokens, len = n_layers + 1, each [T, D] (or [D]).
        hs = [h.float() for h in hs]

        if predicted_token_id is None:
            logits_last = self._project_with_lm_head(hs[-1])
            predicted_token_id = int(torch.argmax(logits_last, dim=-1).item())

        if geometry_space not in GEOMETRY_SPACE_CHOICES:
            raise ValueError(f"geometry_space must be one of {GEOMETRY_SPACE_CHOICES}, got: {geometry_space}")

        per_layer_logits: List[torch.Tensor] = [self._project_with_lm_head(h) for h in hs]
        geom_states = hs if geometry_space == "hidden" else per_layer_logits

        metrics: List[LayerMetrics] = []

        z_post_norms = [float(_safe_norm(v).mean().item()) for v in geom_states]

        dz_para_vals: List[float] = []
        dz_perp_vals: List[float] = []
        dz_perp_over_z_plus_dz_para_vals: List[float] = []
        dz_vals: List[float] = []
        ratio_vals: List[float] = []
        io_cos_vals: List[float] = []

        topk_changes: List[Dict[str, Any]] = []

        for l in range(len(geom_states) - 1):
            z_in = geom_states[l]
            z_out = geom_states[l + 1].to(z_in.device)
            delta = z_out - z_in

            dz_para = _project_parallel(delta, z_in)
            dz_perp = delta - dz_para

            para_norm_t = _safe_norm(dz_para)
            perp_norm_t = _safe_norm(dz_perp)
            z_norm_t = _safe_norm(z_in)
            dz_norm_t = _safe_norm(delta)
            para_norm = float(para_norm_t.mean().item())
            perp_norm = float(perp_norm_t.mean().item())
            dz_norm = float(dz_norm_t.mean().item())
            ratio = float(((para_norm_t + 1e-12) / (perp_norm_t + 1e-12)).mean().item())
            perp_over_z_plus_para = float((perp_norm_t / (z_norm_t + para_norm_t + 1e-12)).mean().item())
            io_cos = float(F.cosine_similarity(z_in, z_out, dim=-1, eps=1e-8).mean().item())

            dz_para_vals.append(para_norm)
            dz_perp_vals.append(perp_norm)
            dz_perp_over_z_plus_dz_para_vals.append(perp_over_z_plus_para)
            dz_vals.append(dz_norm)
            ratio_vals.append(ratio)
            io_cos_vals.append(io_cos)

            gain = z_post_norms[l + 1] / max(z_post_norms[l], 1e-12)

            metrics.append(
                LayerMetrics(
                    layer=l,
                    z_post_norm=z_post_norms[l],
                    z_mean=float(z_in.mean().item()),
                    z_median=float(z_in.median().item()),
                    scale_gain_to_next=gain,
                    io_cos_sim=io_cos,
                    dz_norm=dz_norm,
                    dz_mean=float(delta.abs().mean().item()),
                    dz_median=float(delta.abs().median().item()),
                    dz_para_norm=para_norm,
                    dz_perp_norm=perp_norm,
                    dz_perp_over_z_plus_dz_para=perp_over_z_plus_para,
                    para_perp_ratio=ratio,
                )
            )

            # logits/prob rank change for top-k
            logits_in = per_layer_logits[l]
            logits_out = per_layer_logits[l + 1]
            logits_in_last = logits_in[-1] if logits_in.ndim > 1 else logits_in
            logits_out_last = logits_out[-1] if logits_out.ndim > 1 else logits_out
            probs_in = F.softmax(logits_in_last, dim=-1)
            probs_out = F.softmax(logits_out_last, dim=-1)

            top_in = _topk_indices(logits_in_last, top_k)
            top_out = _topk_indices(logits_out_last, top_k)

            overlap = len(set(top_in.tolist()) & set(top_out.tolist()))

            topk_changes.append(
                {
                    "layer": l,
                    "topk_overlap": overlap,
                    "topk_in": top_in.tolist(),
                    "topk_out": top_out.tolist(),
                    "max_prob_in": float(probs_in.max().item()),
                    "max_prob_out": float(probs_out.max().item()),
                }
            )

        # add last layer post norm entry
        metrics.append(
            LayerMetrics(
                layer=len(hs) - 1,
                z_post_norm=z_post_norms[-1],
                z_mean=float(geom_states[-1].mean().item()),
                z_median=float(geom_states[-1].median().item()),
                scale_gain_to_next=None,
                io_cos_sim=None,
                dz_norm=None,
                dz_mean=None,
                dz_median=None,
                dz_para_norm=None,
                dz_perp_norm=None,
                dz_perp_over_z_plus_dz_para=None,
                para_perp_ratio=None,
            )
        )

        summary = {
            "z_mean": _mean_or_none(z_post_norms),
            "z_median": float(torch.tensor(z_post_norms).median().item()),
            "dz_mean": _mean_or_none(dz_vals),
            "dz_median": float(torch.tensor(dz_vals).median().item()) if dz_vals else None,
            "z_post_mean": _mean_or_none(z_post_norms),
            "dz_para_mean": _mean_or_none(dz_para_vals),
            "dz_perp_mean": _mean_or_none(dz_perp_vals),
            "dz_perp_over_z_plus_dz_para_mean": _mean_or_none(dz_perp_over_z_plus_dz_para_vals),
            "dz_para_dz_perp_ratio_mean": _mean_or_none(ratio_vals),
            "io_cos_sim_mean": _mean_or_none(io_cos_vals),
            "z_post_median": float(torch.tensor(z_post_norms).median().item()),
            "dz_para_median": float(torch.tensor(dz_para_vals).median().item()) if dz_para_vals else None,
            "dz_perp_median": float(torch.tensor(dz_perp_vals).median().item()) if dz_perp_vals else None,
            "dz_perp_over_z_plus_dz_para_median": (
                float(torch.tensor(dz_perp_over_z_plus_dz_para_vals).median().item())
                if dz_perp_over_z_plus_dz_para_vals
                else None
            ),
            "dz_para_dz_perp_ratio_median": float(torch.tensor(ratio_vals).median().item()) if ratio_vals else None,
            "n_layers_observed": len(hs) - 1,
        }

        decoded_top0 = self.tokenizer.decode([predicted_token_id])

        return {
            "summary": summary,
            "geometry_space": geometry_space,
            "predicted_next_token": decoded_top0,
            "layer_metrics": [m.__dict__ for m in metrics],
            "topk_changes": topk_changes,
        }

    def _print_layer_metrics(self, result: Dict[str, Any]) -> None:
        print("\nPer-layer metrics:")
        for m in result["layer_metrics"]:
            layer = m["layer"]
            z_post = m["z_post_norm"]
            z_mean = m["z_mean"]
            z_median = m["z_median"]
            gain = m["scale_gain_to_next"]
            dz_norm = m["dz_norm"]
            dz_mean = m["dz_mean"]
            dz_median = m["dz_median"]
            dz_para = m["dz_para_norm"]
            dz_perp = m["dz_perp_norm"]
            dz_perp_over_z_plus_dz_para = m["dz_perp_over_z_plus_dz_para"]
            para_perp_ratio = m["para_perp_ratio"]
            io_cos = m["io_cos_sim"]

            print(
                f"  layer={layer:>2} | "
                f"z_post_norm={z_post:.6f} | "
                f"z_mean={z_mean:.6f} | "
                f"z_median={z_median:.6f} | "
                f"scale_gain_to_next={gain if gain is not None else 'N/A'} | "
                f"io_cos_sim={io_cos if io_cos is not None else 'N/A'} | "
                f"dz_norm={dz_norm if dz_norm is not None else 'N/A'} | "
                f"dz_mean={dz_mean if dz_mean is not None else 'N/A'} | "
                f"dz_median={dz_median if dz_median is not None else 'N/A'} | "
                f"dz_para_norm={dz_para if dz_para is not None else 'N/A'} | "
                f"dz_perp_norm={dz_perp if dz_perp is not None else 'N/A'} | "
                f"dz_perp_over_z_plus_dz_para={dz_perp_over_z_plus_dz_para if dz_perp_over_z_plus_dz_para is not None else 'N/A'} | "
                f"para_perp_ratio={para_perp_ratio if para_perp_ratio is not None else 'N/A'}"
            )

    def _print_sublayer_metrics(self, result: Dict[str, Any]) -> None:
        sublayer_metrics = result.get("sublayer_metrics")
        if not sublayer_metrics:
            if result.get("sublayer_warning"):
                print(f"Sublayer warning: {result['sublayer_warning']}")
            return

        print("\nPer-layer sublayer metrics (attn/mlp, block-style):")
        for m in sublayer_metrics:
            l = m["layer"]
            a = m["attn_update"]
            p = m["mlp_update"]
            b = m["block_update"]
            print(
                f"  layer={l:>2} | part=attn | "
                f"z_post_norm={a['z_post_norm']:.6f} | "
                f"z_post_dim_mean={a['z_post_dim_mean']:.6f} | "
                f"z_post_dim_min={a['z_post_dim_min']:.6f} | "
                f"z_post_dim_max={a['z_post_dim_max']:.6f} | "
                f"scale_gain_to_next={a['scale_gain_to_next']:.6f} | "
                f"io_cos_sim={a['io_cos_sim']:.6f} | "
                f"dz_norm={a['dz_norm']:.6f} | "
                f"dz_mean={a['dz_mean']:.6f} | "
                f"dz_median={a['dz_median']:.6f} | "
                f"dz_para_norm={a['dz_para_norm']:.6f} | "
                f"dz_perp_norm={a['dz_perp_norm']:.6f} | "
                f"dz_perp_over_z_plus_dz_para={a['dz_perp_over_z_plus_dz_para']:.6f} | "
                f"para_perp_ratio={a['para_perp_ratio']:.6f}"
            )
            if "value_self_para_perp_ratio" in a:
                print(
                    f"  layer={l:>2} | value     | "
                    f"diag_mean={a['diag_mean']:.6f} | "
                    f"value_self_alignment={a['value_self_alignment']:.6f} | "
                    f"value_self_term={a['value_self_term']:.6f} | "
                    f"value_self_para_over_z={a['value_self_para_over_z']:.6f} | "
                    f"value_self_perp_over_z={a['value_self_perp_over_z']:.6f} | "
                    f"value_self_para_perp_ratio={a['value_self_para_perp_ratio']:.6f} | "
                    f"value_pre_para_perp_ratio={a.get('value_pre_para_perp_ratio', float('nan')):.6f} | "
                    f"value_pre_para_ratio={a.get('value_pre_para_ratio', float('nan')):.6f} | "
                    f"value_pre_perp_ratio={a.get('value_pre_perp_ratio', float('nan')):.6f}"
                )
            print(
                f"  layer={l:>2} | part=mlp  | "
                f"z_post_norm={p['z_post_norm']:.6f} | "
                f"z_post_dim_mean={p['z_post_dim_mean']:.6f} | "
                f"z_post_dim_min={p['z_post_dim_min']:.6f} | "
                f"z_post_dim_max={p['z_post_dim_max']:.6f} | "
                f"scale_gain_to_next={p['scale_gain_to_next']:.6f} | "
                f"io_cos_sim={p['io_cos_sim']:.6f} | "
                f"dz_norm={p['dz_norm']:.6f} | "
                f"dz_mean={p['dz_mean']:.6f} | "
                f"dz_median={p['dz_median']:.6f} | "
                f"dz_para_norm={p['dz_para_norm']:.6f} | "
                f"dz_perp_norm={p['dz_perp_norm']:.6f} | "
                f"dz_perp_over_z_plus_dz_para={p['dz_perp_over_z_plus_dz_para']:.6f} | "
                f"para_perp_ratio={p['para_perp_ratio']:.6f}"
            )
            print(
                f"  layer={l:>2} | part=block| "
                f"z_post_dim_mean={b['z_post_dim_mean']:.6f} | "
                f"z_post_dim_min={b['z_post_dim_min']:.6f} | "
                f"z_post_dim_max={b['z_post_dim_max']:.6f} | "
                f"io_cos_sim={b['io_cos_sim']:.6f} | "
                f"dz_norm={b['dz_norm']:.6f} | "
                f"dz_mean={b['dz_mean']:.6f} | "
                f"dz_median={b['dz_median']:.6f} | "
                f"dz_para_norm={b['dz_para_norm']:.6f} | "
                f"dz_perp_norm={b['dz_perp_norm']:.6f} | "
                f"dz_perp_over_z_plus_dz_para={b['dz_perp_over_z_plus_dz_para']:.6f} | "
                f"para_perp_ratio={b['para_perp_ratio']:.6f}"
            )

    def _build_short_prompt_warning(self, prompt_token_count: int) -> Optional[str]:
        if prompt_token_count < MIN_RECOMMENDED_PROMPT_TOKENS:
            return (
                f"Prompt is short ({prompt_token_count} tokens). "
                f"Per-layer geometry may be unstable; consider >= {MIN_RECOMMENDED_PROMPT_TOKENS} tokens."
            )
        return None

    @torch.no_grad()
    def analyze(
        self,
        prompt: str,
        top_k: int = 5,
        geometry_space: str = "hidden",
        print_per_layer: bool = True,
        include_sublayer_metrics: bool = False,
        prefill_token_stride: int = 1,
    ) -> Dict[str, Any]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.input_device)
        attention_mask = encoded.attention_mask.to(self.input_device)
        prompt_token_count = int(input_ids.shape[1])
        attn_outs: Dict[int, torch.Tensor] = {}
        mlp_outs: Dict[int, torch.Tensor] = {}
        attn_mods: Dict[int, Any] = {}
        attn_inputs: Dict[int, torch.Tensor] = {}
        attn_value_outputs: Dict[int, torch.Tensor] = {}
        attn_pre_outputs: Dict[int, torch.Tensor] = {}
        attn_weights: Dict[int, torch.Tensor] = {}
        sublayer_warning: Optional[str] = None

        if include_sublayer_metrics:
            try:
                outputs, attn_outs, mlp_outs, attn_mods, attn_inputs, attn_value_outputs, attn_pre_outputs, attn_weights = self._collect_gpt2_sublayer_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    _, hidden_states = self._forward_with_hidden_states(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            except Exception as e:  # noqa: BLE001
                outputs, hidden_states = self._forward_with_hidden_states(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                sublayer_warning = f"Sublayer metrics unavailable: {e}"
                self._warn_once(sublayer_warning)
        else:
            outputs, hidden_states = self._forward_with_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        hs_full = [h.squeeze(0).float() for h in hidden_states]  # [(T,D)]
        logits_full = outputs.logits.squeeze(0).float()  # (T,V)
        hs_last = [h[-1] if h.ndim == 2 else h for h in hs_full]  # [(D,)]
        predicted_last = int(torch.argmax(logits_full[-1], dim=-1).item())
        result = self._analyze_from_last_token_hs(
            hs=hs_last,
            top_k=top_k,
            geometry_space=geometry_space,
            predicted_token_id=predicted_last,
        )
        if include_sublayer_metrics:
            if attn_outs and mlp_outs:
                value_attn_metrics = self._compute_attn_value_metrics(
                    attn_mods=attn_mods,
                    attn_inputs=attn_inputs,
                    attn_value_outputs=attn_value_outputs,
                    attn_pre_outputs=attn_pre_outputs,
                    attn_weights=attn_weights,
                    hs=hs_last,
                )
                if not value_attn_metrics:
                    msg = (
                        "Value-space attention metrics unavailable: no attention pre-output was captured. "
                        "Capture summary: "
                        + self._value_metric_capture_summary(
                            attn_mods=attn_mods,
                            attn_inputs=attn_inputs,
                            attn_value_outputs=attn_value_outputs,
                            attn_pre_outputs=attn_pre_outputs,
                            attn_weights=attn_weights,
                        )
                    )
                    self._warn_once(msg)
                    sublayer_warning = msg if sublayer_warning is None else sublayer_warning
                result["sublayer_metrics"] = self._build_sublayer_metrics(
                    hs=hs_last,
                    attn_outs=attn_outs,
                    mlp_outs=mlp_outs,
                    value_attn_metrics=value_attn_metrics,
                )
            if sublayer_warning is not None:
                result["sublayer_warning"] = sublayer_warning
        result["prompt"] = prompt
        result["prompt_token_count"] = prompt_token_count
        short_prompt_warning = self._build_short_prompt_warning(prompt_token_count)
        if short_prompt_warning is not None:
            result["warning"] = short_prompt_warning
            print(f"Warning: {short_prompt_warning}")
        if print_per_layer:
            self._print_layer_metrics(result)
            if include_sublayer_metrics:
                self._print_sublayer_metrics(result)

        stride = max(int(prefill_token_stride), 1)
        if stride == 1:
            step_indices = list(range(prompt_token_count))
        else:
            step_indices = list(range(stride - 1, prompt_token_count, stride))
            if not step_indices or step_indices[-1] != prompt_token_count - 1:
                step_indices.append(prompt_token_count - 1)

        steps: List[Dict[str, Any]] = []
        for step in step_indices:
            hs_step = [h[step] if h.ndim == 2 else h for h in hs_full]
            predicted_step = int(torch.argmax(logits_full[step], dim=-1).item())
            probe = self._analyze_from_last_token_hs(
                hs=hs_step,
                top_k=top_k,
                geometry_space=geometry_space,
                predicted_token_id=predicted_step,
            )
            if include_sublayer_metrics:
                if attn_outs and mlp_outs:
                    attn_step = {k: v[:, step : step + 1, :] for k, v in attn_outs.items()}
                    mlp_step = {k: v[:, step : step + 1, :] for k, v in mlp_outs.items()}
                    attn_input_step = {k: v[:, : step + 1, :] for k, v in attn_inputs.items()}
                    attn_pre_output_step = {
                        k: v[:, step : step + 1, :] if v.ndim == 3 else v[step : step + 1, :]
                        for k, v in attn_pre_outputs.items()
                        if (v.ndim == 3 and v.shape[1] > step) or (v.ndim == 2 and v.shape[0] > step)
                    }
                    attn_value_output_step = {
                        k: v[:, : step + 1, :]
                        for k, v in attn_value_outputs.items()
                        if v.ndim == 3 and v.shape[1] > step
                    }
                    attn_weight_step = {
                        k: v[:, :, step : step + 1, : step + 1]
                        for k, v in attn_weights.items()
                        if v.ndim == 4 and v.shape[-2] > step and v.shape[-1] > step
                    }
                    value_attn_metrics = self._compute_attn_value_metrics(
                        attn_mods=attn_mods,
                        attn_inputs=attn_input_step,
                        attn_value_outputs=attn_value_output_step,
                        attn_pre_outputs=attn_pre_output_step,
                        attn_weights=attn_weight_step,
                        hs=hs_step,
                    )
                    probe["sublayer_metrics"] = self._build_sublayer_metrics(
                        hs=hs_step,
                        attn_outs=attn_step,
                        mlp_outs=mlp_step,
                        value_attn_metrics=value_attn_metrics,
                    )
                if sublayer_warning is not None:
                    probe["sublayer_warning"] = sublayer_warning

            token_id = int(input_ids[0, step].item())
            token_text = self.tokenizer.decode([token_id])
            step_entry: Dict[str, Any] = {
                "step": step,
                "token_id": token_id,
                "token_text": token_text,
                "summary": probe["summary"],
                "predicted_next_token": probe["predicted_next_token"],
            }
            if print_per_layer:
                step_entry["layer_metrics"] = probe["layer_metrics"]
            if include_sublayer_metrics:
                if "sublayer_metrics" in probe:
                    step_entry["sublayer_metrics"] = probe["sublayer_metrics"]
                if "sublayer_warning" in probe:
                    step_entry["sublayer_warning"] = probe["sublayer_warning"]
            steps.append(step_entry)
        result["steps"] = steps
        result["prefill_token_stride"] = stride

        return result

    @torch.no_grad()
    def analyze_prefill_batch(
        self,
        prompts: List[str],
        top_k: int = 5,
        geometry_space: str = "hidden",
        include_sublayer_metrics: bool = False,
        prefill_token_stride: int = 1,
    ) -> List[Dict[str, Any]]:
        if not prompts:
            return []

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = old_padding_side

        input_ids = encoded.input_ids.to(self.input_device)
        attention_mask = encoded.attention_mask.to(self.input_device)
        prompt_token_counts = attention_mask.sum(dim=1).to(torch.long).tolist()

        attn_outs: Dict[int, torch.Tensor] = {}
        mlp_outs: Dict[int, torch.Tensor] = {}
        attn_mods: Dict[int, Any] = {}
        attn_inputs: Dict[int, torch.Tensor] = {}
        attn_value_outputs: Dict[int, torch.Tensor] = {}
        attn_pre_outputs: Dict[int, torch.Tensor] = {}
        attn_weights: Dict[int, torch.Tensor] = {}
        sublayer_warning: Optional[str] = None

        if include_sublayer_metrics:
            try:
                outputs, attn_outs, mlp_outs, attn_mods, attn_inputs, attn_value_outputs, attn_pre_outputs, attn_weights = self._collect_gpt2_sublayer_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    _, hidden_states = self._forward_with_hidden_states(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            except Exception as e:  # noqa: BLE001
                outputs, hidden_states = self._forward_with_hidden_states(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                sublayer_warning = f"Sublayer metrics unavailable: {e}"
                self._warn_once(sublayer_warning)
        else:
            outputs, hidden_states = self._forward_with_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        hs_full = [h.float() for h in hidden_states]  # [(B,T,D)]
        logits_full = outputs.logits.float()  # (B,T,V)
        stride = max(int(prefill_token_stride), 1)

        def build_sublayer_for_step(batch_idx: int, step: int, hs_step: List[torch.Tensor]) -> Optional[List[Dict[str, Any]]]:
            if not (attn_outs and mlp_outs):
                return None
            attn_step = {k: v[batch_idx:batch_idx + 1, step:step + 1, :] for k, v in attn_outs.items()}
            mlp_step = {k: v[batch_idx:batch_idx + 1, step:step + 1, :] for k, v in mlp_outs.items()}
            attn_input_step = {
                k: v[batch_idx:batch_idx + 1, : step + 1, :]
                for k, v in attn_inputs.items()
                if v.ndim == 3 and v.shape[0] > batch_idx and v.shape[1] > step
            }
            attn_value_output_step = {
                k: v[batch_idx:batch_idx + 1, : step + 1, :]
                for k, v in attn_value_outputs.items()
                if v.ndim == 3 and v.shape[0] > batch_idx and v.shape[1] > step
            }
            attn_pre_output_step = {
                k: v[batch_idx:batch_idx + 1, step:step + 1, :]
                for k, v in attn_pre_outputs.items()
                if v.ndim == 3 and v.shape[0] > batch_idx and v.shape[1] > step
            }
            attn_weight_step = {
                k: v[batch_idx:batch_idx + 1, :, step:step + 1, : step + 1]
                for k, v in attn_weights.items()
                if v.ndim == 4 and v.shape[0] > batch_idx and v.shape[-2] > step and v.shape[-1] > step
            }
            value_attn_metrics = self._compute_attn_value_metrics(
                attn_mods=attn_mods,
                attn_inputs=attn_input_step,
                attn_value_outputs=attn_value_output_step,
                attn_pre_outputs=attn_pre_output_step,
                attn_weights=attn_weight_step,
                hs=hs_step,
            )
            return self._build_sublayer_metrics(
                hs=hs_step,
                attn_outs=attn_step,
                mlp_outs=mlp_step,
                value_attn_metrics=value_attn_metrics,
            )

        runs: List[Dict[str, Any]] = []
        for batch_idx, prompt in enumerate(prompts):
            prompt_token_count = int(prompt_token_counts[batch_idx])
            if prompt_token_count <= 0:
                continue
            last_idx = prompt_token_count - 1
            hs_last = [h[batch_idx, last_idx, :].float() for h in hs_full]
            predicted_last = int(torch.argmax(logits_full[batch_idx, last_idx, :], dim=-1).item())
            result = self._analyze_from_last_token_hs(
                hs=hs_last,
                top_k=top_k,
                geometry_space=geometry_space,
                predicted_token_id=predicted_last,
            )
            if include_sublayer_metrics:
                sub = build_sublayer_for_step(batch_idx, last_idx, hs_last)
                if sub is not None:
                    result["sublayer_metrics"] = sub
                if sublayer_warning is not None:
                    result["sublayer_warning"] = sublayer_warning
            result["prompt"] = prompt
            result["prompt_token_count"] = prompt_token_count
            result["prefill_token_stride"] = stride
            short_prompt_warning = self._build_short_prompt_warning(prompt_token_count)
            if short_prompt_warning is not None:
                result["warning"] = short_prompt_warning

            if stride == 1:
                step_indices = list(range(prompt_token_count))
            else:
                step_indices = list(range(stride - 1, prompt_token_count, stride))
                if not step_indices or step_indices[-1] != last_idx:
                    step_indices.append(last_idx)

            steps: List[Dict[str, Any]] = []
            for step in step_indices:
                hs_step = [h[batch_idx, step, :].float() for h in hs_full]
                predicted_step = int(torch.argmax(logits_full[batch_idx, step, :], dim=-1).item())
                probe = self._analyze_from_last_token_hs(
                    hs=hs_step,
                    top_k=top_k,
                    geometry_space=geometry_space,
                    predicted_token_id=predicted_step,
                )
                if include_sublayer_metrics:
                    sub = build_sublayer_for_step(batch_idx, step, hs_step)
                    if sub is not None:
                        probe["sublayer_metrics"] = sub
                    if sublayer_warning is not None:
                        probe["sublayer_warning"] = sublayer_warning

                token_id = int(input_ids[batch_idx, step].item())
                step_entry: Dict[str, Any] = {
                    "step": step,
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode([token_id]),
                    "summary": probe["summary"],
                    "predicted_next_token": probe["predicted_next_token"],
                }
                if include_sublayer_metrics:
                    if "sublayer_metrics" in probe:
                        step_entry["sublayer_metrics"] = probe["sublayer_metrics"]
                    if "sublayer_warning" in probe:
                        step_entry["sublayer_warning"] = probe["sublayer_warning"]
                steps.append(step_entry)
            result["steps"] = steps
            runs.append(result)

        return runs

    @torch.no_grad()
    def analyze_batch(self, prompts: List[str], top_k: int = 5) -> Dict[str, Any]:
        runs = [self.analyze(prompt=p, top_k=top_k, print_per_layer=False) for p in prompts]

        summary_keys = [
            "z_mean",
            "dz_mean",
            "z_post_mean",
            "dz_para_mean",
            "dz_perp_mean",
            "dz_perp_over_z_plus_dz_para_mean",
            "dz_para_dz_perp_ratio_mean",
            "io_cos_sim_mean",
            "z_median",
            "dz_median",
            "z_post_median",
            "dz_para_median",
            "dz_perp_median",
            "dz_perp_over_z_plus_dz_para_median",
            "dz_para_dz_perp_ratio_median",
        ]

        agg = {}
        for k in summary_keys:
            vals = [r["summary"][k] for r in runs if r["summary"].get(k) is not None]
            agg[f"mean_{k}"] = _mean_or_none(vals)

        return {
            "n_prompts": len(prompts),
            "aggregate": agg,
            "runs": runs,
        }

    @torch.no_grad()
    def analyze_generation(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        top_k: int = 5,
        geometry_space: str = "hidden",
        do_sample: bool = False,
        temperature: float = 1.0,
        print_per_layer: bool = False,
        include_sublayer_metrics: bool = False,
        prefill_token_stride: int = 1,
    ) -> Dict[str, Any]:
        # Use model.generate for both decoding and per-step hidden-state extraction.
        prefill_probe = self.analyze(
            prompt=prompt,
            top_k=top_k,
            geometry_space=geometry_space,
            print_per_layer=False,
            include_sublayer_metrics=include_sublayer_metrics,
            prefill_token_stride=prefill_token_stride,
        )
        prefill_steps = prefill_probe.get("steps", [])

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.input_device)
        attention_mask = encoded.attention_mask.to(self.input_device)
        prompt_token_count = int(input_ids.shape[1])

        decode_backend = "generate"
        if self._should_use_forward_decode():
            decode_backend = "forward_loop"
            full_ids, generated_token_ids, step_hidden_states_all, step_scores_all = self._decode_with_forward_loop(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            generated_token_count = int(generated_token_ids.shape[1])
        else:
            generate_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_hidden_states": True,
                "output_scores": True,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = max(temperature, 1e-6)

            generation = self._model_generate_with_mask_retry(**generate_kwargs)
            full_ids = generation.sequences
            generated_token_ids = full_ids[:, prompt_token_count:]
            generated_token_count = int(generated_token_ids.shape[1])
            step_hidden_states_all = list(generation.hidden_states)
            step_scores_all = list(generation.scores)

        steps: List[Dict[str, Any]] = []

        for step in range(generated_token_count):
            step_hidden_states = step_hidden_states_all[step]
            hs = [h[:, -1, :].squeeze(0).float() for h in step_hidden_states]

            score = step_scores_all[step]
            predicted_token_id = int(torch.argmax(score, dim=-1).item())
            probe = self._analyze_from_last_token_hs(
                hs=hs,
                top_k=top_k,
                geometry_space=geometry_space,
                predicted_token_id=predicted_token_id,
            )

            if include_sublayer_metrics:
                # Re-run step prefix once to collect attn/mlp internals (GPT2-like best effort).
                prefix_ids = full_ids[:, : prompt_token_count + step]
                prefix_mask = torch.ones_like(prefix_ids, device=prefix_ids.device)
                sub_probe = self._analyze_from_ids(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    top_k=top_k,
                    geometry_space=geometry_space,
                    include_sublayer_metrics=True,
                )
                if "sublayer_metrics" in sub_probe:
                    probe["sublayer_metrics"] = sub_probe["sublayer_metrics"]
                if "sublayer_warning" in sub_probe:
                    probe["sublayer_warning"] = sub_probe["sublayer_warning"]

            if print_per_layer:
                print(f"\n[Generation Step {step}]")
                self._print_layer_metrics(probe)
                if include_sublayer_metrics:
                    self._print_sublayer_metrics(probe)

            token_id = int(generated_token_ids[0, step].item())
            token_text = self.tokenizer.decode([token_id])

            step_entry = {
                "step": step,
                "token_id": token_id,
                "token_text": token_text,
                "summary": probe["summary"],
                "predicted_next_token": probe["predicted_next_token"],
            }
            if print_per_layer:
                step_entry["layer_metrics"] = probe["layer_metrics"]
            if include_sublayer_metrics:
                if "sublayer_metrics" in probe:
                    step_entry["sublayer_metrics"] = probe["sublayer_metrics"]
                if "sublayer_warning" in probe:
                    step_entry["sublayer_warning"] = probe["sublayer_warning"]
            steps.append(step_entry)

        generated = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        short_prompt_warning = self._build_short_prompt_warning(prompt_token_count)

        result = {
            "prompt": prompt,
            "geometry_space": geometry_space,
            "decode_backend": decode_backend,
            "prompt_token_count": prompt_token_count,
            "max_new_tokens": max_new_tokens,
            "generated_token_count": generated_token_count,
            "generated_text": generated,
            "steps": steps,
            "prefill_steps": prefill_steps,
            "timeline": [
                *[
                    {
                        "phase": "prefill",
                        "timestep": int(s["step"]),
                        **s,
                    }
                    for s in prefill_steps
                ],
                *[
                    {
                        "phase": "decode",
                        "timestep": int(s["step"]),
                        **s,
                    }
                    for s in steps
                ],
            ],
        }
        if short_prompt_warning is not None:
            result["warning"] = short_prompt_warning
        return result


def analyze_prompt(
    model_name_or_path: str,
    prompt: str,
    top_k: int = 5,
    device: Optional[str] = None,
    geometry_space: str = "hidden",
) -> Dict[str, Any]:
    analyzer = GeometryAnalyzer(model_name_or_path=model_name_or_path, device=device)
    return analyzer.analyze(prompt=prompt, top_k=top_k, geometry_space=geometry_space)
