from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

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

    @torch.no_grad()
    def analyze(self, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

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
            "prompt": prompt,
            "summary": summary,
            "predicted_next_token": decoded_top0,
            "layer_metrics": [m.__dict__ for m in metrics],
            "topk_changes": topk_changes,
        }


def analyze_prompt(model_name_or_path: str, prompt: str, top_k: int = 5, device: Optional[str] = None) -> Dict[str, Any]:
    analyzer = GeometryAnalyzer(model_name_or_path=model_name_or_path, device=device)
    return analyzer.analyze(prompt=prompt, top_k=top_k)
