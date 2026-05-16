#!/usr/bin/env python3
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM


def detect_awq(model_path: str) -> bool:
    cfg = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            j = json.load(f)
        qcfg = j.get("quantization_config", {})
        return isinstance(qcfg, dict) and str(qcfg.get("quant_method", "")).lower() == "awq"
    except Exception:
        return False


def load_model(model_name: str, dtype: str):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if detect_awq(model_name):
        print("[INFO] AWQ detected; loading via native AWQ path.")
        return AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="cpu", torch_dtype=dtype_map[dtype]
        ).eval()
    return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu").eval()


def get_layer_params(model, layer_idx):
    layer = model.model.layers[layer_idx]
    params = {}
    for name, p in layer.named_parameters():
        params[name] = p.detach().float().cpu()
    return params


def flatten_norm(x):
    return x.reshape(-1).norm().item()


def main():
    p = argparse.ArgumentParser("Compare dense vs quant parameters layer-wise.")
    p.add_argument("--dense_model", type=str, required=True)
    p.add_argument("--quant_model", type=str, required=True)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="dense_vs_quant_params")
    args = p.parse_args()

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Loading dense model on CPU...")
    dense = AutoModelForCausalLM.from_pretrained(args.dense_model, trust_remote_code=True, device_map="cpu").eval()
    print("[INFO] Loading quant model on CPU...")
    quant = load_model(args.quant_model, args.dtype)

    n_layers = min(len(dense.model.layers), len(quant.model.layers))
    rows = []

    for li in range(n_layers):
        pd = get_layer_params(dense, li)
        pq = get_layer_params(quant, li)
        common = sorted(set(pd.keys()) & set(pq.keys()))
        if not common:
            continue
        sum_delta_sq = 0.0
        sum_dense_sq = 0.0
        cos_weighted_num = 0.0
        cos_weighted_den = 0.0
        n_tensors = 0

        for k in common:
            xd = pd[k]
            xq = pq[k]
            if xd.shape != xq.shape:
                continue
            d = xq - xd
            dn = flatten_norm(d)
            xn = flatten_norm(xd)
            sum_delta_sq += dn * dn
            sum_dense_sq += xn * xn

            xd_f = xd.reshape(-1)
            xq_f = xq.reshape(-1)
            denom = (xd_f.norm() * xq_f.norm()).item()
            if denom > 0:
                cos = torch.dot(xd_f, xq_f).item() / denom
                w = float(xd.numel())
                cos_weighted_num += cos * w
                cos_weighted_den += w
            n_tensors += 1

        delta_norm = (sum_delta_sq ** 0.5) if sum_delta_sq > 0 else 0.0
        dense_norm = (sum_dense_sq ** 0.5) if sum_dense_sq > 0 else 0.0
        rel_delta = (delta_norm / dense_norm) if dense_norm > 0 else 0.0
        cos_mean = (cos_weighted_num / cos_weighted_den) if cos_weighted_den > 0 else 0.0

        rows.append(
            {
                "layer": li,
                "num_common_tensors": n_tensors,
                "delta_norm": delta_norm,
                "dense_norm": dense_norm,
                "rel_delta_norm": rel_delta,
                "cosine_mean": cos_mean,
            }
        )

    tsv_path = os.path.join(out_dir, "dense_vs_quant_params_layerwise.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "num_common_tensors",
                "delta_norm",
                "dense_norm",
                "rel_delta_norm",
                "cosine_mean",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    xs = [r["layer"] for r in rows]
    for metric, ylabel, fn in [
        ("delta_norm", "||W_q - W_d||", "fig_param_delta_norm.png"),
        ("rel_delta_norm", "||W_q-W_d|| / ||W_d||", "fig_param_rel_delta_norm.png"),
        ("cosine_mean", "mean cosine(W_d, W_q)", "fig_param_cosine_mean.png"),
    ]:
        ys = [r[metric] for r in rows]
        plt.figure(figsize=(10, 4.5))
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.3)
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fn), dpi=180)
        plt.close()

    print(f"[INFO] Done. Output dir: {os.path.abspath(out_dir)}")
    print(f"[INFO] TSV: {tsv_path}")


if __name__ == "__main__":
    main()
