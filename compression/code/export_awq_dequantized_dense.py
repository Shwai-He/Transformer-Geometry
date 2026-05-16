#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_awq(model_path: str) -> bool:
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        qcfg = data.get("quantization_config", {})
        return isinstance(qcfg, dict) and str(qcfg.get("quant_method", "")).lower() == "awq"
    except Exception:
        return False


def strip_quant_config(model_dir: Path):
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data.pop("quantization_config", None)
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser("Export AWQ model to dequantized dense model directory.")
    ap.add_argument("--awq_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save_safetensors", action="store_true")
    args = ap.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    target_dtype = dtype_map[args.dtype]

    if not detect_awq(args.awq_model):
        print("[WARN] quant_method!=awq (or missing). Continuing anyway.")

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.awq_model,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=target_dtype,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.awq_model, trust_remote_code=True)

    # Preferred path: official dequantize hook.
    if hasattr(model, "dequantize") and callable(getattr(model, "dequantize")):
        print("[INFO] Calling model.dequantize() ...")
        model = model.dequantize()
    else:
        # Fallback: fail loudly with useful introspection.
        methods = [x for x in dir(model) if "quant" in x.lower() or "dequant" in x.lower()]
        raise RuntimeError(
            "Model class does not expose dequantize(). "
            f"Available quant-related attrs: {methods}"
        )

    print(f"[INFO] Converting model to dtype={args.dtype} on CPU...")
    model = model.to(dtype=target_dtype, device=torch.device("cpu")).eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving dequantized model to: {out}")
    model.save_pretrained(out, safe_serialization=bool(args.save_safetensors))
    tokenizer.save_pretrained(out)
    strip_quant_config(out)

    print("[INFO] Done.")
    print(f"[INFO] Output: {out}")


if __name__ == "__main__":
    main()
