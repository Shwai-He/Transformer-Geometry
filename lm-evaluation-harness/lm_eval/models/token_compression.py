from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from lm_eval.models.xsa_hooks import _find_decoder_layers, _get_token_mixer_kind_and_module


eval_logger = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


@dataclass
class TokenCompressionConfig:
    enabled: bool = False
    mode: str = "value_parallel_drop"
    keep_ratio: float = 1.0
    keep_tokens: int = 0
    layer: int = 0
    layers: str = ""
    ref: str = "last_value"
    preserve_first: int = 32
    preserve_last: int = 256
    min_context: int = 512
    parallel_weight: float = 0.25


class PrefillTokenCompressor:
    """Input-side token compaction for long-prefill generation.

    This is intentionally a lightweight first implementation. It compresses the
    prompt before `generate()` rather than modifying the model's dynamic KV cache.
    The geometry score uses first-layer value projections: tokens whose value
    vectors are most parallel to a reference are dropped first, while tokens with
    larger perpendicular value-space components are kept.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, config: TokenCompressionConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.layers = _find_decoder_layers(model)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0
        self.special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_input_embeddings"):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return emb(input_ids)
        for path in (("model", "embed_tokens"), ("transformer", "wte"), ("gpt_neox", "embed_in")):
            obj = self.model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj(input_ids)
        raise ValueError("Could not find input embeddings for token compression.")

    def _parse_layer_indices(self) -> list[int]:
        n_layers = len(self.layers)
        raw = str(self.config.layers or "").strip().lower()
        if raw in {"all", "*"} or int(self.config.layer) < 0:
            return list(range(n_layers))
        if raw:
            indices = []
            for item in raw.replace(";", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if item == "mid":
                    indices.append(n_layers // 2)
                elif item == "last":
                    indices.append(n_layers - 1)
                else:
                    indices.append(int(item))
            return sorted(set(max(0, min(n_layers - 1, idx)) for idx in indices))
        layer_idx = max(0, min(int(self.config.layer), n_layers - 1))
        return [layer_idx]

    def _attention_module(self, layer_idx: int) -> torch.nn.Module:
        layer = self.layers[layer_idx]
        _, attn = _get_token_mixer_kind_and_module(layer)
        if attn is None or not hasattr(attn, "v_proj"):
            raise ValueError(f"Layer {layer_idx} does not expose self-attention v_proj.")
        return attn

    def _capture_attention_inputs(
        self,
        input_ids: torch.Tensor,
        layer_indices: list[int],
        output_attentions: bool = False,
    ) -> tuple[dict[int, torch.Tensor], tuple[torch.Tensor, ...] | None]:
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, args, kwargs):
                hidden = kwargs.get("hidden_states") if kwargs else None
                if hidden is None and args:
                    hidden = args[0]
                if hidden is None:
                    raise ValueError(f"Could not capture hidden_states for layer {layer_idx}.")
                captured[layer_idx] = hidden.detach()

            return hook

        for layer_idx in layer_indices:
            handles.append(self._attention_module(layer_idx).register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_attentions=output_attentions,
                )
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(set(layer_indices) - set(captured))
        if missing:
            raise ValueError(f"Missing captured attention inputs for layers: {missing}")
        attentions = getattr(outputs, "attentions", None) if output_attentions else None
        return captured, attentions

    def _value_vectors_for_layer(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        attn = self._attention_module(layer_idx)
        v_proj = attn.v_proj
        param = next(v_proj.parameters(), None)
        if param is not None:
            hidden = hidden.to(device=param.device, dtype=param.dtype)
        return v_proj(hidden).float()

    def _head_counts(self, layer_idx: int) -> tuple[int, int]:
        attn = self._attention_module(layer_idx)
        n_heads = getattr(attn, "num_heads", None) or getattr(self.model.config, "num_attention_heads", None)
        n_kv_heads = (
            getattr(attn, "num_key_value_heads", None)
            or getattr(self.model.config, "num_key_value_heads", None)
            or n_heads
        )
        if n_heads is None or n_kv_heads is None:
            raise ValueError(f"Could not infer attention head counts for layer {layer_idx}.")
        return int(n_heads), int(n_kv_heads)

    def _reference(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.ref == "mean_value":
            denom = valid.float().sum().clamp_min(1.0)
            return (values * valid[:, None].float()).sum(dim=0) / denom
        if cfg.ref == "last_hidden":
            # For this first version, fall back to last value if the hidden
            # reference is not in the same space as the value projection.
            cfg_ref = "last_value"
        else:
            cfg_ref = cfg.ref
        if cfg_ref == "last_value":
            idx = torch.nonzero(valid, as_tuple=False).flatten()
            return values[idx[-1]] if idx.numel() else values[-1]
        raise ValueError(f"Unsupported token_compression_ref={cfg.ref}")

    def _score_tokens_from_values(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        ref = self._reference(values, valid)
        eps = 1e-8
        dot = (values * ref[None, :]).sum(dim=-1).abs()
        value_norm = values.norm(dim=-1).clamp_min(eps)
        ref_norm = ref.norm().clamp_min(eps)
        para_cos = (dot / (value_norm * ref_norm)).clamp(0.0, 1.0)
        perp_sin = (1.0 - para_cos.square()).clamp_min(0.0).sqrt()
        mode = self.config.mode
        if mode in {"value_parallel_drop", "drop_parallel"}:
            # Higher score is kept. Drop high-parallel tokens first.
            return perp_sin
        if mode in {"value_perp_keep", "keep_perp"}:
            return perp_sin
        if mode in {"value_norm_keep", "keep_norm"}:
            return value_norm
        if mode in {"value_perp_norm_keep", "drop_low_perp_low_norm", "low_perp_low_norm"}:
            # Keep tokens with large perpendicular value-space contribution,
            # while retaining a smaller weight on the parallel magnitude so
            # high-mass but mostly parallel tokens are not discarded too early.
            beta = float(self.config.parallel_weight)
            return value_norm * (perp_sin + beta * para_cos)
        if mode in {"value_perp_only_norm_keep", "drop_low_perp_norm"}:
            return value_norm * perp_sin
        if mode in {"value_parallel_keep", "keep_parallel"}:
            return para_cos
        if mode == "uniform":
            seq = torch.arange(values.shape[0], device=values.device, dtype=torch.float32)
            return -seq.remainder(997)
        raise ValueError(f"Unsupported token_compression_mode={mode}")

    def _score_tokens_from_contributions(self, contributions: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        ref = contributions.sum(dim=0)
        contrib_norm = contributions.norm(dim=-1).clamp_min(eps)
        ref_norm = ref.norm().clamp_min(eps)
        para = (contributions * ref[None, :]).sum(dim=-1).abs() / ref_norm
        para_ratio = (para / contrib_norm).clamp(0.0, 1.0)
        perp_ratio = (1.0 - para_ratio.square()).clamp_min(0.0).sqrt()
        mode = self.config.mode
        if mode in {"target_value_parallel_drop", "target_drop_parallel"}:
            return perp_ratio
        if mode in {"target_value_perp_keep", "target_keep_perp"}:
            return perp_ratio
        if mode in {"target_value_norm_keep", "target_keep_norm"}:
            return contrib_norm
        if mode in {"target_value_perp_norm_keep", "target_drop_low_perp_low_norm"}:
            beta = float(self.config.parallel_weight)
            return contrib_norm * (perp_ratio + beta * para_ratio)
        if mode in {"target_value_perp_only_norm_keep", "target_drop_low_perp_norm"}:
            return contrib_norm * perp_ratio
        raise ValueError(f"Unsupported target-query token_compression_mode={mode}")

    def _score_tokens_from_attention(
        self,
        attentions: tuple[torch.Tensor, ...],
        layer_indices: list[int],
        seq_len: int,
    ) -> torch.Tensor:
        mode = self.config.mode
        scores = []
        for layer_idx in layer_indices:
            attn = attentions[layer_idx][0].float()
            if mode in {"h2o", "h2o_keep", "heavy_hitter", "heavy_hitter_keep", "scissorhands"}:
                # H2O-style heavy hitter / Scissorhands-style persistence:
                # token importance is accumulated historical attention mass.
                score = attn.sum(dim=(0, 1))
            elif mode in {"snapkv", "snapkv_keep", "snapkv_observation", "observation_attention"}:
                # SnapKV-style observation window: estimate prompt-token
                # importance from the final prompt window before generation.
                obs = max(1, min(seq_len, int(self.config.preserve_last)))
                score = attn[:, -obs:, :].sum(dim=(0, 1))
            else:
                raise ValueError(f"Unsupported attention token_compression_mode={mode}")
            scores.append(score)
        return torch.stack(scores, dim=0).mean(dim=0)

    def _score_tokens_for_target_query(self, input_ids: torch.Tensor, layer_indices: list[int]) -> torch.Tensor:
        attention_inputs, attentions = self._capture_attention_inputs(input_ids, layer_indices, output_attentions=True)
        if attentions is None:
            raise ValueError("Model did not return attentions for target-query token compression.")
        seq_len = int(input_ids.shape[-1])
        scores = []
        for layer_idx in layer_indices:
            hidden = attention_inputs[layer_idx]
            values = self._value_vectors_for_layer(layer_idx, hidden).squeeze(0)
            n_heads, n_kv_heads = self._head_counts(layer_idx)
            head_dim = values.shape[-1] // n_kv_heads
            values = values.view(seq_len, n_kv_heads, head_dim)
            repeat = n_heads // n_kv_heads
            values = values.repeat_interleave(repeat, dim=1)
            attn_last = attentions[layer_idx][0, :, -1, :].float()
            contributions = (attn_last.transpose(0, 1).unsqueeze(-1) * values).reshape(seq_len, n_heads * head_dim)
            scores.append(self._score_tokens_from_contributions(contributions))
        return torch.stack(scores, dim=0).mean(dim=0)

    def _score_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        layer_indices = self._parse_layer_indices()
        mode = self.config.mode
        if mode == "uniform":
            seq = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.float32)
            return -seq.remainder(997)
        if mode in {"streamingllm", "streaming_llm", "sink_recent", "sink_recent_keep"}:
            # StreamingLLM-style prompt compaction: forced sink tokens are
            # handled in _keep_indices_for_row; this score fills the remaining
            # budget with the most recent tokens.
            return torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.float32)
        if mode in {
            "h2o",
            "h2o_keep",
            "heavy_hitter",
            "heavy_hitter_keep",
            "scissorhands",
            "snapkv",
            "snapkv_keep",
            "snapkv_observation",
            "observation_attention",
        }:
            _, attentions = self._capture_attention_inputs(input_ids, layer_indices, output_attentions=True)
            if attentions is None:
                raise ValueError("Model did not return attentions for attention-based token compression.")
            return self._score_tokens_from_attention(attentions, layer_indices, int(input_ids.shape[-1]))
        if mode.startswith("target_"):
            return self._score_tokens_for_target_query(input_ids, layer_indices)
        attention_inputs, _ = self._capture_attention_inputs(input_ids, layer_indices)
        valid = torch.ones(input_ids.shape[-1], device=input_ids.device, dtype=torch.bool)
        scores = []
        for layer_idx in layer_indices:
            values = self._value_vectors_for_layer(layer_idx, attention_inputs[layer_idx]).squeeze(0)
            scores.append(self._score_tokens_from_values(values, valid))
        stacked = torch.stack(scores, dim=0)
        return stacked.mean(dim=0)

    def _target_keep_count(self, valid_count: int) -> int:
        cfg = self.config
        if cfg.keep_tokens and cfg.keep_tokens > 0:
            keep = int(cfg.keep_tokens)
        else:
            keep = int(round(valid_count * float(cfg.keep_ratio)))
        keep = max(1, min(valid_count, keep))
        keep = max(keep, min(valid_count, int(cfg.preserve_first) + int(cfg.preserve_last)))
        return keep

    def _keep_indices_for_row(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.bool()
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid_count = int(valid_idx.numel())
        if valid_count <= 0:
            return valid_idx
        if valid_count < int(self.config.min_context):
            return valid_idx
        keep_count = self._target_keep_count(valid_count)
        if keep_count >= valid_count:
            return valid_idx

        score = self._score_tokens(input_ids[valid_idx].unsqueeze(0)).squeeze(0)

        forced = torch.zeros(valid_count, device=score.device, dtype=torch.bool)
        first = max(0, int(self.config.preserve_first))
        last = max(0, int(self.config.preserve_last))
        if first:
            forced[: min(first, valid_count)] = True
        if last:
            forced[max(0, valid_count - last) :] = True
        if self.special_token_ids:
            ids = input_ids[valid_idx].to(device=score.device)
            special = torch.zeros_like(forced)
            for tok in self.special_token_ids:
                special |= ids == int(tok)
            forced |= special

        forced_count = int(forced.sum().item())
        remaining = max(0, keep_count - forced_count)
        keep = forced.clone()
        if remaining > 0:
            candidate_score = score.masked_fill(forced, float("-inf"))
            top = torch.topk(candidate_score, k=min(remaining, valid_count - forced_count), largest=True).indices
            keep[top] = True
        kept_valid_idx = valid_idx[keep.detach().cpu().to(valid_idx.device)]
        return kept_valid_idx.sort().values

    def compress(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        rows = []
        device = input_ids.device
        for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
            keep_idx = self._keep_indices_for_row(row_ids, row_mask)
            rows.append(row_ids[keep_idx])
        max_len = max((int(r.numel()) for r in rows), default=0)
        if max_len <= 0:
            return input_ids, attention_mask, {"old_len": float(input_ids.shape[1]), "new_len": float(input_ids.shape[1]), "keep_ratio": 1.0}
        new_ids = input_ids.new_full((input_ids.shape[0], max_len), int(self.pad_token_id))
        new_mask = attention_mask.new_zeros((input_ids.shape[0], max_len))
        for i, row in enumerate(rows):
            n = int(row.numel())
            new_ids[i, -n:] = row.to(device=device)
            new_mask[i, -n:] = 1
        old_valid = attention_mask.long().sum(dim=1).float()
        new_valid = new_mask.long().sum(dim=1).float()
        stats = {
            "old_len": float(old_valid.mean().item()),
            "new_len": float(new_valid.mean().item()),
            "keep_ratio": float((new_valid / old_valid.clamp_min(1)).mean().item()),
        }
        return new_ids, new_mask, stats
