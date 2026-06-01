from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from analysis.vlm_geometry.hooks import GeometryScaleConfig, VLMGeometryScaler, parse_int_list, parse_path_list
from analysis.vlm_geometry.run_vlm_ppl import (
    _build_runner,
    _loss_to_record,
    _resolve_image_path,
)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_jsonl(path: Path, total: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if total is not None and len(rows) >= total:
                break
    return rows


def _candidate_answers(row: dict[str, Any], mode: str) -> list[str]:
    options = row.get("options") or row.get("option") or {}
    if isinstance(options, dict) and options:
        keys = [str(k) for k in options.keys()]
        if mode == "option_text":
            return [str(options[k]).strip() for k in keys]
        return keys
    if row.get("benchmark") == "mme" or row.get("answer_type") == "yes_no":
        return ["Yes", "No"]
    answer = row.get("answer") or row.get("gt_answer")
    return [str(answer).strip()] if answer is not None else []


def _canonical_answer(answer: str, row: dict[str, Any], mode: str) -> str:
    options = row.get("options") or row.get("option") or {}
    if isinstance(options, dict) and options and mode == "option_text":
        for key, value in options.items():
            if str(value).strip().lower() == str(answer).strip().lower():
                return str(key)
    return str(answer).strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forced-choice VLM understanding eval via conditional answer NLL.")
    p.add_argument("--model-name", required=True, choices=["qwenimage", "ming", "bagel"])
    p.add_argument("--model-path", required=True)
    p.add_argument("--processor-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-samples", type=int, default=32)
    p.add_argument("--candidate-mode", default="letter", choices=["letter", "option_text"])
    p.add_argument("--model-preset", default="")
    p.add_argument("--side", default="und", choices=["und", "gen"])
    p.add_argument("--space", default="residual", choices=["residual", "value"])
    p.add_argument("--target", default="block", choices=["block", "attn", "mlp", "value"])
    p.add_argument("--para-scale", type=float, default=1.0)
    p.add_argument("--perp-scale", type=float, default=1.0)
    p.add_argument("--layer-paths", default="")
    p.add_argument("--layer-indices", default="")
    p.add_argument("--skip-first-n", type=int, default=0)
    p.add_argument("--skip-last-n", type=int, default=0)
    p.add_argument("--target-module-regex", default="")
    p.add_argument("--attn-name-regex", default="")
    p.add_argument("--mlp-name-regex", default="")
    p.add_argument("--value-name-regex", default="")
    p.add_argument("--value-head-mode", default="multihead", choices=["multihead", "merged"])
    p.add_argument("--value-ref-expansion", default="model_type", choices=["model_type", "auto_phi", "phi", "legacy", "legacy_repeat", "module", "module_only", "head_aware", "auto", "config_fallback"])
    p.add_argument("--attn-attr-source", default="model_type", choices=["model_type", "auto_phi", "phi", "legacy", "legacy_repeat", "module", "module_only", "auto", "config_fallback", "head_aware"])
    p.add_argument("--disable-geometry", default="false", help="Run the model without installing geometry hooks.")
    p.add_argument(
        "--scale-mode",
        default="none",
        choices=[
            "none",
            "job_para_uniform",
            "job_perp_uniform",
            "job_both_uniform",
            "job_para_choice",
            "job_perp_choice",
            "job_both_choice",
            "sample_para_uniform",
            "sample_perp_uniform",
            "sample_both_uniform",
            "sample_para_choice",
            "sample_perp_choice",
            "sample_both_choice",
        ],
    )
    p.add_argument("--scale-seed", type=int, default=0)
    p.add_argument("--para-scale-min", type=float, default=-1.5)
    p.add_argument("--para-scale-max", type=float, default=1.5)
    p.add_argument("--perp-scale-min", type=float, default=-1.5)
    p.add_argument("--perp-scale-max", type=float, default=1.5)
    p.add_argument("--fail-on-missing-target", default="true")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparse-unified-root", default="/beacon-projects/traumallm/shwaihe/SparseUnifiedModel")
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    sparse_root = Path(args.sparse_unified_root).resolve()
    if str(sparse_root) not in sys.path:
        sys.path.insert(0, str(sparse_root))
    data_file = Path(args.data_file).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(data_file, args.total_samples)
    runner = _build_runner(args)
    preset = args.model_preset or ("qwen-image" if args.model_name == "qwenimage" else args.model_name)
    config = GeometryScaleConfig(
        model_preset=preset,
        side=args.side,
        space=args.space,
        target=args.target,
        para_scale=args.para_scale,
        perp_scale=args.perp_scale,
        layer_paths=parse_path_list(args.layer_paths),
        layer_indices=parse_int_list(args.layer_indices),
        skip_first_n=args.skip_first_n,
        skip_last_n=args.skip_last_n,
        attn_name_regex=args.attn_name_regex or None,
        mlp_name_regex=args.mlp_name_regex or None,
        value_name_regex=args.value_name_regex or None,
        target_module_regex=args.target_module_regex or None,
        fail_on_missing_target=(str(args.fail_on_missing_target).lower() == "true"),
        value_head_mode=args.value_head_mode,
        value_ref_expansion=args.value_ref_expansion,
        attn_attr_source=args.attn_attr_source,
        scale_mode=args.scale_mode,
        scale_seed=args.scale_seed,
        para_scale_min=args.para_scale_min,
        para_scale_max=args.para_scale_max,
        perp_scale_min=args.perp_scale_min,
        perp_scale_max=args.perp_scale_max,
    )

    result_path = out_dir / "results.jsonl"
    records = []
    scaler = None
    disable_geometry = _str_to_bool(args.disable_geometry)
    hook_context = (
        torch.no_grad()
        if disable_geometry
        else VLMGeometryScaler(runner.model, config)
    )
    with hook_context as maybe_scaler, result_path.open("w", encoding="utf-8") as f:
        scaler = None if disable_geometry else maybe_scaler
        for idx, row in enumerate(tqdm(rows, desc="choice-eval")):
            image_path_text = row.get("image") or row.get("image_path")
            if not image_path_text:
                continue
            image_path = _resolve_image_path(str(image_path_text), sparse_root=sparse_root, data_file=data_file)
            image = Image.open(image_path).convert("RGB")
            question = row.get("question") or row.get("prompt") or row.get("text") or "Describe the image."
            candidates = _candidate_answers(row, args.candidate_mode)
            if not candidates:
                continue
            scores = []
            for cand in candidates:
                if scaler is not None:
                    scaler.set_sample_seed(args.scale_seed + idx)
                loss, labels = runner.loss(image, str(question), str(cand))
                ce, n_tokens, nll = _loss_to_record(loss, labels)
                scores.append(
                    {
                        "answer": str(cand),
                        "canonical": _canonical_answer(str(cand), row, args.candidate_mode),
                        "ce": ce,
                        "target_tokens": n_tokens,
                        "nll": nll,
                    }
                )
            best = min(scores, key=lambda x: x["nll"])
            gt = str(row.get("answer", row.get("gt_answer"))).strip()
            record = {
                "idx": idx,
                "benchmark": row.get("benchmark"),
                "data_id": row.get("data_id"),
                "image": str(image_path),
                "question": question,
                "prediction": best["canonical"],
                "gt_answer": gt,
                "correct": best["canonical"].strip().lower() == gt.lower(),
                "scores": scores,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)

    correct = sum(1 for r in records if r["correct"])
    summary = {
        "model_name": args.model_name,
        "data_file": str(data_file),
        "n_examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "candidate_mode": args.candidate_mode,
        "geometry": {"disabled": True} if disable_geometry else scaler.summary(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
