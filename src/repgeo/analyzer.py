from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LayerMetrics:
    layer: int
    z_post_norm: float
    scale_gain_to_next: Optional[float]
    dz_para_norm: Optional[float]
    dz_perp_norm: Optional[float]
    log_para_perp_ratio: Optional[float]


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


def _extract_hidden_from_module_output(output: Any) -> torch.Tensor:
    # HF blocks often return tuple where first item is hidden states.
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported module output type: {type(output)}")
    return output


class GeometryAnalyzer:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = None
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        if not hasattr(self.model, "lm_head"):
            raise ValueError("Model has no lm_head; logits-space analysis requires a causal LM with lm_head.")

    def _get_transformer_layers(self) -> Tuple[List[torch.nn.Module], str]:
        # LLaMA/Mistral style
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers), "llama_like"
        # GPT-2 style
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h), "gpt2_like"
        raise ValueError("Unsupported model architecture for sublayer hooks.")

    def _get_sublayer_modules(
        self, layer: torch.nn.Module, arch: str
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        if arch == "llama_like":
            return layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp
        if arch == "gpt2_like":
            return layer.ln_1, layer.attn, layer.ln_2, layer.mlp
        raise ValueError(f"Unsupported architecture tag: {arch}")

    def _compute_pair_metrics(
        self,
        z_in: torch.Tensor,
        z_out: torch.Tensor,
        layer_idx: int,
        stage: str,
        top_k: int,
    ) -> Dict[str, Any]:
        delta = z_out - z_in
        dz_para = _project_parallel(delta, z_in)
        dz_perp = delta - dz_para

        z_in_norm = float(_safe_norm(z_in).item())
        z_out_norm = float(_safe_norm(z_out).item())
        para_norm = float(_safe_norm(dz_para).item())
        perp_norm = float(_safe_norm(dz_perp).item())
        log_ratio = float(torch.log(torch.tensor((para_norm + 1e-12) / (perp_norm + 1e-12))).item())

        x_in = self.model.lm_head(z_in).float()
        x_out = self.model.lm_head(z_out).float()
        top_in = _topk_indices(x_in, top_k)
        top_out = _topk_indices(x_out, top_k)
        overlap = len(set(top_in.tolist()) & set(top_out.tolist()))

        return {
            "layer": layer_idx,
            "stage": stage,
            "z_in_norm": z_in_norm,
            "z_out_norm": z_out_norm,
            "scale_gain": z_out_norm / max(z_in_norm, 1e-12),
            "dz_para_norm": para_norm,
            "dz_perp_norm": perp_norm,
            "log_para_perp_ratio": log_ratio,
            "topk_overlap": overlap,
            "topk_in": top_in.tolist(),
            "topk_out": top_out.tolist(),
        }

    @torch.no_grad()
    def _analyze_sublayers_from_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, top_k: int = 5
    ) -> Dict[str, Any]:
        layers, arch = self._get_transformer_layers()
        n_layers = len(layers)

        cache: Dict[str, List[Optional[torch.Tensor]]] = {
            "attn_in": [None] * n_layers,
            "attn_out": [None] * n_layers,
            "mlp_in": [None] * n_layers,
            "mlp_out": [None] * n_layers,
        }
        handles: List[Any] = []

        for i, layer in enumerate(layers):
            pre_attn_norm, attn_mod, pre_mlp_norm, mlp_mod = self._get_sublayer_modules(layer, arch)

            def pre_attn_hook(_, __, output, idx=i):
                h = _extract_hidden_from_module_output(output)
                cache["attn_in"][idx] = h[:, -1, :].squeeze(0).detach().float()

            def attn_hook(_, __, output, idx=i):
                h = _extract_hidden_from_module_output(output)
                cache["attn_out"][idx] = h[:, -1, :].squeeze(0).detach().float()

            def pre_mlp_hook(_, __, output, idx=i):
                h = _extract_hidden_from_module_output(output)
                cache["mlp_in"][idx] = h[:, -1, :].squeeze(0).detach().float()

            def mlp_hook(_, __, output, idx=i):
                h = _extract_hidden_from_module_output(output)
                cache["mlp_out"][idx] = h[:, -1, :].squeeze(0).detach().float()

            handles.append(pre_attn_norm.register_forward_hook(pre_attn_hook))
            handles.append(attn_mod.register_forward_hook(attn_hook))
            handles.append(pre_mlp_norm.register_forward_hook(pre_mlp_hook))
            handles.append(mlp_mod.register_forward_hook(mlp_hook))

        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        finally:
            for h in handles:
                h.remove()

        sublayer_metrics: List[Dict[str, Any]] = []
        para_vals: List[float] = []
        perp_vals: List[float] = []
        log_ratio_vals: List[float] = []
        gain_vals: List[float] = []
        overlap_vals: List[float] = []

        for i in range(n_layers):
            attn_in = cache["attn_in"][i]
            attn_out = cache["attn_out"][i]
            mlp_in = cache["mlp_in"][i]
            mlp_out = cache["mlp_out"][i]

            if attn_in is not None and attn_out is not None:
                rec = self._compute_pair_metrics(attn_in, attn_out, i, "attn", top_k)
                sublayer_metrics.append(rec)
                para_vals.append(rec["dz_para_norm"])
                perp_vals.append(rec["dz_perp_norm"])
                log_ratio_vals.append(rec["log_para_perp_ratio"])
                gain_vals.append(rec["scale_gain"])
                overlap_vals.append(float(rec["topk_overlap"]))

            if mlp_in is not None and mlp_out is not None:
                rec = self._compute_pair_metrics(mlp_in, mlp_out, i, "mlp", top_k)
                sublayer_metrics.append(rec)
                para_vals.append(rec["dz_para_norm"])
                perp_vals.append(rec["dz_perp_norm"])
                log_ratio_vals.append(rec["log_para_perp_ratio"])
                gain_vals.append(rec["scale_gain"])
                overlap_vals.append(float(rec["topk_overlap"]))

        summary = {
            "n_layers_observed": n_layers,
            "n_sublayer_records": len(sublayer_metrics),
            "dz_para_median": float(torch.tensor(para_vals).median().item()) if para_vals else None,
            "dz_perp_median": float(torch.tensor(perp_vals).median().item()) if perp_vals else None,
            "dz_para_dz_perp_log_median": float(torch.tensor(log_ratio_vals).median().item()) if log_ratio_vals else None,
            "scale_gain_median": float(torch.tensor(gain_vals).median().item()) if gain_vals else None,
            "topk_overlap_mean": float(torch.tensor(overlap_vals).mean().item()) if overlap_vals else None,
        }
        return {"sublayer_summary": summary, "sublayer_metrics": sublayer_metrics}

    @torch.no_grad()
    def _analyze_from_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states  # tuple length = n_layers + 1
        last_token_states = [h[:, -1, :] for h in hidden_states]  # [(B=1, D)]
        hs = [x.squeeze(0).float() for x in last_token_states]

        metrics: List[LayerMetrics] = []

        z_post_norms = [_safe_norm(h).item() for h in hs]

        dz_para_vals: List[float] = []
        dz_perp_vals: List[float] = []
        log_ratio_vals: List[float] = []

        per_layer_logits: List[torch.Tensor] = []
        for h in hs:
            per_layer_logits.append(self.model.lm_head(h).float())

        topk_changes: List[Dict[str, Any]] = []

        for l in range(len(hs) - 1):
            z_in = hs[l]
            z_out = hs[l + 1]
            delta = z_out - z_in

            dz_para = _project_parallel(delta, z_in)
            dz_perp = delta - dz_para

            para_norm = _safe_norm(dz_para).item()
            perp_norm = _safe_norm(dz_perp).item()
            log_ratio = float(torch.log(torch.tensor((para_norm + 1e-12) / (perp_norm + 1e-12))))

            dz_para_vals.append(para_norm)
            dz_perp_vals.append(perp_norm)
            log_ratio_vals.append(log_ratio)

            gain = z_post_norms[l + 1] / max(z_post_norms[l], 1e-12)

            metrics.append(
                LayerMetrics(
                    layer=l,
                    z_post_norm=z_post_norms[l],
                    scale_gain_to_next=gain,
                    dz_para_norm=para_norm,
                    dz_perp_norm=perp_norm,
                    log_para_perp_ratio=log_ratio,
                )
            )

            # logits/prob rank change for top-k
            logits_in = per_layer_logits[l]
            logits_out = per_layer_logits[l + 1]
            probs_in = F.softmax(logits_in, dim=-1)
            probs_out = F.softmax(logits_out, dim=-1)

            top_in = _topk_indices(logits_in, top_k)
            top_out = _topk_indices(logits_out, top_k)

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
                scale_gain_to_next=None,
                dz_para_norm=None,
                dz_perp_norm=None,
                log_para_perp_ratio=None,
            )
        )

        summary = {
            "z_post_median": float(torch.tensor(z_post_norms).median().item()),
            "dz_para_median": float(torch.tensor(dz_para_vals).median().item()) if dz_para_vals else None,
            "dz_perp_median": float(torch.tensor(dz_perp_vals).median().item()) if dz_perp_vals else None,
            "dz_para_dz_perp_log_median": float(torch.tensor(log_ratio_vals).median().item()) if log_ratio_vals else None,
            "n_layers_observed": len(hs) - 1,
        }

        decoded_top0 = self.tokenizer.decode([int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())])

        return {
            "summary": summary,
            "predicted_next_token": decoded_top0,
            "layer_metrics": [m.__dict__ for m in metrics],
            "topk_changes": topk_changes,
        }

    @torch.no_grad()
    def analyze(self, prompt: str, top_k: int = 5, granularity: str = "block") -> Dict[str, Any]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        result = self._analyze_from_ids(input_ids=input_ids, attention_mask=attention_mask, top_k=top_k)
        if granularity in {"sublayer", "both"}:
            result.update(self._analyze_sublayers_from_ids(input_ids=input_ids, attention_mask=attention_mask, top_k=top_k))
        result["granularity"] = granularity
        result["prompt"] = prompt
        return result

    @torch.no_grad()
    def analyze_batch(self, prompts: List[str], top_k: int = 5) -> Dict[str, Any]:
        runs = [self.analyze(prompt=p, top_k=top_k) for p in prompts]

        summary_keys = [
            "z_post_median",
            "dz_para_median",
            "dz_perp_median",
            "dz_para_dz_perp_log_median",
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
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        # Step-wise decoding analysis: run full forward at each step and log geometry stats.
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)

        steps: List[Dict[str, Any]] = []

        for step in range(max_new_tokens):
            attention_mask = torch.ones_like(input_ids, device=self.device)
            probe = self._analyze_from_ids(input_ids=input_ids, attention_mask=attention_mask, top_k=top_k)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits[:, -1, :]
            if do_sample:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            token_text = self.tokenizer.decode([token_id])

            steps.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "token_text": token_text,
                    "summary": probe["summary"],
                    "predicted_next_token": probe["predicted_next_token"],
                }
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)

        generated = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "generated_text": generated,
            "steps": steps,
        }


def analyze_prompt(
    model_name_or_path: str,
    prompt: str,
    top_k: int = 5,
    device: Optional[str] = None,
    granularity: str = "block",
) -> Dict[str, Any]:
    analyzer = GeometryAnalyzer(model_name_or_path=model_name_or_path, device=device)
    return analyzer.analyze(prompt=prompt, top_k=top_k, granularity=granularity)
