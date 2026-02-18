from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x, dim=-1).clamp_min(eps)


def _project_parallel(delta: torch.Tensor, base: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = (base * base).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = (delta * base).sum(dim=-1, keepdim=True) / denom
    return coeff * base


@torch.no_grad()
def _mHC_decompose(z_in: torch.Tensor, z_out: torch.Tensor, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    """
    mHC-style decomposition:
      z_out ~= alpha * z_in + delta_perp
    where alpha is scalar fitted from projection and delta_perp is residual.
    """
    denom = (z_in * z_in).sum().clamp_min(eps)
    alpha = (z_out * z_in).sum() / denom
    delta_perp = z_out - alpha * z_in
    return {"alpha": alpha, "delta_perp": delta_perp}


class TechnicalReproducer:
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
            raise ValueError("Model has no lm_head; technical reproduction expects causal LM with lm_head.")

    @torch.no_grad()
    def reproduce(self, prompt: str, top_k: int = 5, translation_shift: float = 100.0) -> Dict[str, Any]:
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

        hs = [h[:, -1, :].squeeze(0).float() for h in outputs.hidden_states]
        n_layers = len(hs) - 1

        layer_records: List[Dict[str, Any]] = []

        for l in range(n_layers):
            z_in = hs[l]
            z_out = hs[l + 1]
            delta = z_out - z_in

            dz_para = _project_parallel(delta, z_in)
            dz_perp = delta - dz_para

            x_in = self.model.lm_head(z_in).float()
            x_out = self.model.lm_head(z_out).float()
            x_para = self.model.lm_head(dz_para).float()
            x_perp = self.model.lm_head(dz_perp).float()

            p_in = F.softmax(x_in, dim=-1)
            p_out = F.softmax(x_out, dim=-1)

            top_ids_in = torch.topk(x_in, k=min(top_k, x_in.shape[-1]), dim=-1).indices
            top_ids_out = torch.topk(x_out, k=min(top_k, x_out.shape[-1]), dim=-1).indices
            overlap = len(set(top_ids_in.tolist()) & set(top_ids_out.tolist()))

            # mHC-style alpha fitting and orthogonality check
            mhc = _mHC_decompose(z_in, z_out)
            alpha = float(mhc["alpha"].item())
            delta_perp_mhc = mhc["delta_perp"]
            ortho_cos = float(
                F.cosine_similarity(delta_perp_mhc.unsqueeze(0), z_in.unsqueeze(0), dim=-1).item()
            )

            selected = top_ids_in.tolist()
            layer_records.append(
                {
                    "layer": l,
                    "z_in_norm": float(_safe_norm(z_in).item()),
                    "z_out_norm": float(_safe_norm(z_out).item()),
                    "scale_gain": float((_safe_norm(z_out) / _safe_norm(z_in)).item()),
                    "dz_para_norm": float(_safe_norm(dz_para).item()),
                    "dz_perp_norm": float(_safe_norm(dz_perp).item()),
                    "log_para_perp_ratio": float(
                        torch.log((_safe_norm(dz_para) + 1e-12) / (_safe_norm(dz_perp) + 1e-12)).item()
                    ),
                    "topk_overlap": overlap,
                    "topk_ids_in": top_ids_in.tolist(),
                    "topk_ids_out": top_ids_out.tolist(),
                    "x_in_topk": [float(x_in[i].item()) for i in selected],
                    "x_out_topk": [float(x_out[i].item()) for i in selected],
                    "x_para_topk": [float(x_para[i].item()) for i in selected],
                    "x_perp_topk": [float(x_perp[i].item()) for i in selected],
                    "p_in_topk": [float(p_in[i].item()) for i in selected],
                    "p_out_topk": [float(p_out[i].item()) for i in selected],
                    "mhc_alpha": alpha,
                    "mhc_delta_perp_norm": float(_safe_norm(delta_perp_mhc).item()),
                    "mhc_orthogonality_cos": ortho_cos,
                }
            )

        # Softmax translation invariance check on final layer logits
        last_logits = self.model.lm_head(hs[-1]).float()
        p_ref = F.softmax(last_logits, dim=-1)
        p_shift = F.softmax(last_logits + translation_shift, dim=-1)
        translation_l1 = float(torch.abs(p_ref - p_shift).mean().item())
        translation_max_abs = float(torch.abs(p_ref - p_shift).max().item())

        return {
            "prompt": prompt,
            "n_layers_observed": n_layers,
            "translation_invariance": {
                "shift": translation_shift,
                "mean_abs_diff": translation_l1,
                "max_abs_diff": translation_max_abs,
            },
            "layer_records": layer_records,
        }

    @torch.no_grad()
    def reproduce_batch(
        self,
        prompts: List[str],
        top_k: int = 5,
        translation_shift: float = 100.0,
    ) -> Dict[str, Any]:
        runs = [
            self.reproduce(prompt=p, top_k=top_k, translation_shift=translation_shift)
            for p in prompts
        ]

        if not runs:
            return {"n_prompts": 0, "aggregate": {}, "runs": []}

        n_layers = runs[0]["n_layers_observed"]
        layerwise: List[Dict[str, Any]] = []

        keys = [
            "scale_gain",
            "dz_para_norm",
            "dz_perp_norm",
            "log_para_perp_ratio",
            "topk_overlap",
            "mhc_alpha",
            "mhc_orthogonality_cos",
        ]

        for layer_id in range(n_layers):
            recs = [r["layer_records"][layer_id] for r in runs]
            agg = {"layer": layer_id}
            for k in keys:
                vals = torch.tensor([float(rr[k]) for rr in recs], dtype=torch.float32)
                agg[f"{k}_mean"] = float(vals.mean().item())
                agg[f"{k}_std"] = float(vals.std(unbiased=False).item())
            layerwise.append(agg)

        trans_mean_abs = torch.tensor(
            [r["translation_invariance"]["mean_abs_diff"] for r in runs], dtype=torch.float32
        )
        trans_max_abs = torch.tensor(
            [r["translation_invariance"]["max_abs_diff"] for r in runs], dtype=torch.float32
        )

        aggregate = {
            "translation_invariance_mean_abs_diff_mean": float(trans_mean_abs.mean().item()),
            "translation_invariance_mean_abs_diff_std": float(trans_mean_abs.std(unbiased=False).item()),
            "translation_invariance_max_abs_diff_mean": float(trans_max_abs.mean().item()),
            "translation_invariance_max_abs_diff_std": float(trans_max_abs.std(unbiased=False).item()),
            "layerwise": layerwise,
        }

        return {
            "n_prompts": len(prompts),
            "top_k": top_k,
            "translation_shift": translation_shift,
            "aggregate": aggregate,
            "runs": runs,
        }
