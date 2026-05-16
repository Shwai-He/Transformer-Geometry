#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

# File-first config (edit here).
CKPT_PATH = "/mnt/hdfs/shwai.he/DepthBoost/nanoGPT/out/xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h1p0-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10"
CKPT_PICK = "latest"  # "latest" -> ckpt.pt, "best" -> ckpt_best.pt
SHOW_LAYERS = True
HEAD_PREVIEW = 8


def _resolve_ckpt(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        best = p / "ckpt_best.pt"
        latest = p / "ckpt.pt"
        pick = str(CKPT_PICK).lower().strip()
        if pick == "latest":
            if latest.is_file():
                return latest
            if best.is_file():
                return best
        elif pick == "best":
            if best.is_file():
                return best
            if latest.is_file():
                return latest
        else:
            raise ValueError(f"Unsupported CKPT_PICK={CKPT_PICK}; use latest or best")
        raise FileNotFoundError(f"No ckpt_best.pt or ckpt.pt found in directory: {p}")
    raise FileNotFoundError(f"Checkpoint path does not exist: {p}")


def _layer_id_from_key(key: str) -> int:
    # key example: transformer.h.10.xsa_forward_gamma_raw
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "h" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def _stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.float().view(-1)
    if x.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "std": float("nan")}
    return {
        "mean": float(x.mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def main() -> None:
    ckpt_path = _resolve_ckpt(CKPT_PATH)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unexpected checkpoint format: expected dict")

    config = checkpoint.get("config", {})
    state = checkpoint.get("model", {})
    if not isinstance(state, dict):
        raise RuntimeError("Unexpected checkpoint format: `model` state_dict missing")

    gamma_keys = sorted([k for k in state.keys() if k.endswith(".xsa_forward_gamma_raw")])
    alpha_keys = sorted([k for k in state.keys() if k.endswith(".xsa_forward_alpha_raw")])

    print(f"Checkpoint: {ckpt_path}")
    print(f"iter_num: {checkpoint.get('iter_num', 'NA')}")
    print(f"gamma_param_keys: {len(gamma_keys)}")
    print(f"alpha_param_keys: {len(alpha_keys)}")
    if isinstance(config, dict):
        print(f"config.xsa_forward_only={config.get('xsa_forward_only', None)}")
        print(f"config.xsa_forward_ref={config.get('xsa_forward_ref', None)}")
        print(f"config.xsa_forward_space={config.get('xsa_forward_space', None)}")
        print(f"config.xsa_forward_target={config.get('xsa_forward_target', None)}")
        print(f"config.xsa_forward_learnable_gamma={config.get('xsa_forward_learnable_gamma', None)}")
        print(f"config.xsa_forward_gamma_per_head={config.get('xsa_forward_gamma_per_head', None)}")
        print(f"config.xsa_forward_gamma_init={config.get('xsa_forward_gamma_init', None)}")

    if len(gamma_keys) == 0:
        print("No gamma params in this checkpoint.")
        return

    all_raw: List[torch.Tensor] = []
    all_gamma: List[torch.Tensor] = []
    per_layer: Dict[int, torch.Tensor] = {}
    for k in gamma_keys:
        raw = state[k].detach().float().view(-1)
        all_raw.append(raw)
        all_gamma.append(torch.exp(raw))
        lid = _layer_id_from_key(k)
        per_layer[lid] = torch.exp(raw)

    raw_cat = torch.cat(all_raw, dim=0)
    gamma_cat = torch.cat(all_gamma, dim=0)
    raw_s = _stats(raw_cat)
    g_s = _stats(gamma_cat)
    print(
        "raw_stats: "
        f"mean={raw_s['mean']:.6f} min={raw_s['min']:.6f} max={raw_s['max']:.6f} std={raw_s['std']:.6f}"
    )
    print(
        "gamma_stats(exp(raw)): "
        f"mean={g_s['mean']:.6f} min={g_s['min']:.6f} max={g_s['max']:.6f} std={g_s['std']:.6f}"
    )

    if SHOW_LAYERS:
        for lid in sorted(per_layer.keys()):
            g = per_layer[lid].view(-1)
            s = _stats(g)
            preview_n = max(1, int(HEAD_PREVIEW))
            preview_vals = ", ".join(f"{float(v):.6f}" for v in g[:preview_n].tolist())
            print(
                f"layer={lid:02d} gamma_mean={s['mean']:.6f} gamma_min={s['min']:.6f} "
                f"gamma_max={s['max']:.6f} gamma_std={s['std']:.6f} heads[:{preview_n}]=[{preview_vals}]"
            )


if __name__ == "__main__":
    main()
