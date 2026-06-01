from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
from pathlib import Path
from typing import Any

import torch

from analysis.vlm_geometry.hooks import (
    GeometryScaleConfig,
    VLMGeometryScaler,
    parse_int_list,
    parse_path_list,
)


def _resolve_dtype(name: str):
    name = (name or "auto").lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


def _env_flag(name: str, default: bool = False) -> bool:
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_model(args):
    if args.sparse_unified_root:
        root = Path(args.sparse_unified_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    if args.loader == "transformers":
        from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        dtype = _resolve_dtype(args.dtype)
        model_cls = AutoModelForCausalLM if args.causal_lm else AutoModel
        model = model_cls.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
        )
        processor = None
        tokenizer = None
        try:
            processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        except Exception:
            processor = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        except Exception:
            tokenizer = None
        return model, processor, tokenizer

    if args.loader == "diffusers":
        from diffusers import DiffusionPipeline

        dtype = _resolve_dtype(args.dtype)
        pipe = DiffusionPipeline.from_pretrained(
            args.model_name_or_path,
            torch_dtype=None if dtype == "auto" else dtype,
            trust_remote_code=True,
        )
        if args.device and args.device != "auto":
            pipe = pipe.to(args.device)
        model = pipe
        return model, pipe, None

    if args.loader == "sparse_qwenimage":
        from modeling.qwenimage.bridge import load_qwenimage_pipeline

        dtype = _resolve_dtype(args.dtype)
        pipe = load_qwenimage_pipeline(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if dtype == "auto" else dtype,
            device=None if args.device == "auto" else args.device,
            enable_model_cpu_offload=(
                _env_flag("QWENIMAGE_ENABLE_MODEL_CPU_OFFLOAD") or _env_flag("ENABLE_MODEL_CPU_OFFLOAD")
            ),
            enable_sequential_cpu_offload=(
                _env_flag("QWENIMAGE_ENABLE_SEQUENTIAL_CPU_OFFLOAD")
                or _env_flag("ENABLE_SEQUENTIAL_CPU_OFFLOAD")
            ),
        )
        return pipe, pipe, None

    if args.loader == "sparse_ming":
        from modeling.ming.bridge import load_ming_model, load_ming_processor

        dtype = _resolve_dtype(args.dtype)
        load_image_gen = args.ming_load_image_gen
        if load_image_gen == "auto":
            load_image_gen = "true" if args.mode == "ming_image_generate" or args.side in {"gen", "both"} else "false"
        model = load_ming_model(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if dtype == "auto" else dtype,
            device=None if args.device_map else (None if args.device == "auto" else args.device),
            device_map=args.device_map or None,
            load_image_gen=load_image_gen == "true",
        ).eval()
        processor = load_ming_processor()
        return model, processor, None

    if args.loader == "sparse_bagel":
        from modeling.adapters.bagel import BagelAdapter

        dtype = _resolve_dtype(args.dtype)
        bundle = BagelAdapter().load_bundle(
            args.model_name_or_path,
            dtype=torch.bfloat16 if dtype == "auto" else dtype,
            visual_gen=True,
            visual_und=True,
        )
        return bundle.model, bundle, None

    if args.loader == "sparse_adapter":
        from modeling import get_model_adapter, load_model_bundle

        dtype = _resolve_dtype(args.dtype)
        adapter = get_model_adapter(args.model_preset)
        bundle = load_model_bundle(
            args.model_preset,
            model_path=args.model_name_or_path,
            dtype=torch.bfloat16 if dtype == "auto" else dtype,
            device=None if args.device_map else (None if args.device == "auto" else args.device),
            device_map=args.device_map or None,
        )
        inferencer = bundle.inferencer_cls(**adapter.build_inferencer_kwargs(bundle))
        return bundle.model, inferencer, None

    raise ValueError(f"Unsupported loader={args.loader!r}")


def _run_text_forward(model, tokenizer, prompt: str, device: str | None) -> dict[str, Any]:
    if tokenizer is None:
        raise RuntimeError("Text smoke requires a tokenizer. Use --mode generate for pipeline smoke.")
    inputs = tokenizer(prompt, return_tensors="pt")
    if device and device != "auto":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = getattr(out, "logits", None)
    return {
        "mode": "text_forward",
        "logits_shape": list(logits.shape) if isinstance(logits, torch.Tensor) else None,
    }


def _save_images(images: Any, output_image_dir: str | None) -> list[str]:
    if not output_image_dir or images is None:
        return []
    image_dir = Path(output_image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(images, (list, tuple)):
        images = [images]
    saved = []
    for idx, image in enumerate(images):
        if isinstance(image, (list, tuple)):
            for sub_idx, sub_image in enumerate(image):
                if hasattr(sub_image, "save"):
                    path = image_dir / f"image_{idx:02d}_{sub_idx:02d}.png"
                    sub_image.save(path)
                    saved.append(str(path))
            continue
        if hasattr(image, "save"):
            path = image_dir / f"image_{idx:02d}.png"
            image.save(path)
            saved.append(str(path))
    return saved


def _save_sample_images(images: Any, sample_dir: Path, sample_offset: int = 0) -> list[str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        images = [images]
    saved = []
    sample_idx = sample_offset
    for image in images:
        if isinstance(image, (list, tuple)):
            for sub_image in image:
                if hasattr(sub_image, "save"):
                    path = sample_dir / f"{sample_idx:05d}.png"
                    sub_image.save(path)
                    saved.append(str(path))
                    sample_idx += 1
            continue
        if hasattr(image, "save"):
            path = sample_dir / f"{sample_idx:05d}.png"
            image.save(path)
            saved.append(str(path))
            sample_idx += 1
    return saved


def _run_pipeline_generate(pipe, prompt: str, args) -> dict[str, Any]:
    kwargs = {"prompt": prompt}
    if args.num_inference_steps is not None:
        kwargs["num_inference_steps"] = args.num_inference_steps
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width
    if args.seed is not None and torch.cuda.is_available():
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.no_grad():
        out = pipe(**kwargs)
    images = getattr(out, "images", None)
    return {
        "mode": "pipeline_generate",
        "n_images": len(images) if images is not None else None,
        "saved_images": _save_images(images, args.output_image_dir),
    }


def _run_generate_once(model, processor, prompt: str, args):
    if args.mode == "pipeline_generate":
        kwargs = {"prompt": prompt}
        if args.num_inference_steps is not None:
            kwargs["num_inference_steps"] = args.num_inference_steps
        if args.height is not None:
            kwargs["height"] = args.height
        if args.width is not None:
            kwargs["width"] = args.width
        if args.seed is not None and torch.cuda.is_available():
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.no_grad():
            out = processor(**kwargs)
        return getattr(out, "images", None)

    if args.mode == "ming_image_generate":
        if processor is None:
            raise RuntimeError("Ming image generation requires a processor.")
        messages = [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(_model_device(model))
        for key, value in list(inputs.items()):
            if key in {"pixel_values", "pixel_values_videos", "audio_feats"} and isinstance(value, torch.Tensor):
                inputs[key] = value.to(dtype=torch.bfloat16)
        with torch.no_grad():
            return model.generate(
                **inputs,
                image_gen=True,
                image_gen_cfg=args.cfg_scale,
                image_gen_steps=args.num_inference_steps,
                image_gen_width=args.width or 480,
                image_gen_height=args.height or 544,
                image_gen_seed=args.seed,
            )

    if args.mode == "bagel_interleave_generate":
        if processor is None:
            raise RuntimeError("BAGEL generation requires a LoadedModelBundle.")
        if hasattr(processor, "make_inferencer"):
            inferencer = processor.make_inferencer()
        elif getattr(processor, "inferencer_cls", None) is not None:
            inferencer_kwargs = {"model": model}
            for name in (
                "tokenizer",
                "vae_model",
                "vae_transform",
                "vit_transform",
                "new_token_ids",
            ):
                inferencer_kwargs[name] = processor.require_component(name)
            inferencer = processor.inferencer_cls(**inferencer_kwargs)
        else:
            raise RuntimeError("BAGEL bundle does not define an inferencer.")
        resolution = args.width or args.height or 512
        outputs = inferencer(
            text=prompt,
            image_shapes=(resolution, resolution),
            cfg_text_scale=args.cfg_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=(0.4, 1.0),
            timestep_shift=3.0,
            num_timesteps=args.num_inference_steps,
            think=False,
        )
        return [item for item in outputs if hasattr(item, "save")]

    if args.mode == "adapter_generate":
        if processor is None:
            raise RuntimeError("Adapter generation requires an inferencer.")
        outputs = processor(
            text=prompt,
            num_timesteps=args.num_inference_steps,
            cfg_text_scale=args.cfg_scale,
            cfg_img_scale=args.cfg_img_scale,
            seed=args.seed,
            num_images=max(1, args.num_images_per_prompt),
            img_size=args.width or args.height or 384,
        )
        if isinstance(outputs, dict):
            images = outputs.get("images")
            if images is None and outputs.get("image") is not None:
                images = [outputs["image"]]
            return images
        return outputs

    raise RuntimeError(f"Prompt-batch generation only supports image generation modes, got {args.mode!r}")


def _load_metadata_prompts(path: str, max_prompts: int, prompt_key: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get(prompt_key):
                continue
            records.append(rec)
            if max_prompts > 0 and len(records) >= max_prompts:
                break
    return records


def _run_metadata_generate(model, processor, args) -> dict[str, Any]:
    if not args.output_image_dir:
        raise RuntimeError("--output-image-dir is required when --metadata-file is used.")
    records = _load_metadata_prompts(args.metadata_file, args.max_prompts, args.prompt_key)
    image_root = Path(args.output_image_dir)
    metadata_out = image_root.parent / "metadata.jsonl"
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_summaries = []
    total_images = 0
    scalers = getattr(args, "_geometry_scalers", [])
    with metadata_out.open("w", encoding="utf-8") as meta_f:
        for prompt_idx, rec in enumerate(records):
            for scaler in scalers:
                scaler.set_sample_seed(int(args.seed) + prompt_idx)
            prompt = rec[args.prompt_key]
            prompt_dir = image_root / f"{prompt_idx:05d}"
            sample_dir = prompt_dir / "samples"
            saved = []
            for sample_idx in range(max(1, args.num_images_per_prompt)):
                images = _run_generate_once(model, processor, prompt, args)
                saved.extend(_save_sample_images(images, sample_dir, sample_offset=sample_idx))
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (prompt_dir / "metadata.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (prompt_dir / "metadata.jsonl").write_text(
                json.dumps(rec, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            prompt_summaries.append({"prompt_index": prompt_idx, "prompt": prompt, "saved_images": saved})
            total_images += len(saved)
    return {
        "mode": f"{args.mode}_metadata",
        "metadata_file": args.metadata_file,
        "n_prompts": len(records),
        "n_images": total_images,
        "metadata_out": str(metadata_out),
        "prompts": prompt_summaries,
    }


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_ming_image_generate(model, processor, prompt: str, args) -> dict[str, Any]:
    if processor is None:
        raise RuntimeError("Ming image generation requires a processor.")
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(_model_device(model))
    for key, value in list(inputs.items()):
        if key in {"pixel_values", "pixel_values_videos", "audio_feats"} and isinstance(value, torch.Tensor):
            inputs[key] = value.to(dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            image_gen=True,
            image_gen_cfg=args.cfg_scale,
            image_gen_steps=args.num_inference_steps,
            image_gen_width=args.width or 480,
            image_gen_height=args.height or 544,
            image_gen_seed=args.seed,
        )
    n_images = len(generated) if isinstance(generated, (list, tuple)) else 1
    return {
        "mode": "ming_image_generate",
        "n_images": n_images,
        "saved_images": _save_images(generated, args.output_image_dir),
    }


def _run_load_only(model) -> dict[str, Any]:
    return {
        "mode": "load_only",
        "model_class": type(model).__name__,
        "n_parameters": sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke-test VLM para/perp geometry scaling hooks.")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--model-preset", default="qwen-image", choices=["qwen-image", "bagel", "ming", "janus"])
    p.add_argument(
        "--loader",
        default="transformers",
        choices=["transformers", "diffusers", "sparse_qwenimage", "sparse_ming", "sparse_bagel", "sparse_adapter"],
    )
    p.add_argument(
        "--mode",
        default="text_forward",
        choices=[
            "text_forward",
            "pipeline_generate",
            "ming_image_generate",
            "bagel_interleave_generate",
            "adapter_generate",
            "load_only",
        ],
    )
    p.add_argument("--side", default="gen", choices=["und", "gen", "both"])
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
    p.add_argument("--fail-on-missing-target", default="true")
    p.add_argument("--prompt", default="A small red cube on a wooden table.")
    p.add_argument("--metadata-file", default="")
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--prompt-key", default="prompt")
    p.add_argument("--num-images-per-prompt", type=int, default=1)
    p.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--causal-lm", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--cfg-img-scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale-mode", default="none")
    p.add_argument("--scale-seed", type=int, default=0)
    p.add_argument("--para-scale-min", type=float, default=-1.5)
    p.add_argument("--para-scale-max", type=float, default=1.5)
    p.add_argument("--perp-scale-min", type=float, default=-1.5)
    p.add_argument("--perp-scale-max", type=float, default=1.5)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--sparse-unified-root", default="/beacon-projects/traumallm/shwaihe/SparseUnifiedModel")
    p.add_argument("--ming-load-image-gen", default="auto", choices=["auto", "true", "false"])
    p.add_argument("--output-json", default="")
    p.add_argument("--output-image-dir", default="")
    p.add_argument("--require-stats", default="true")
    p.add_argument("--disable-geometry", default="false")
    return p


def _make_geometry_configs(args) -> list[GeometryScaleConfig]:
    sides = (args.side,)
    return [
        GeometryScaleConfig(
            model_preset=args.model_preset,
            side=side,
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
            value_head_mode=args.value_head_mode,
            value_ref_expansion=args.value_ref_expansion,
            attn_attr_source=args.attn_attr_source,
            scale_mode=args.scale_mode,
            scale_seed=args.scale_seed,
            para_scale_min=args.para_scale_min,
            para_scale_max=args.para_scale_max,
            perp_scale_min=args.perp_scale_min,
            perp_scale_max=args.perp_scale_max,
            fail_on_missing_target=(str(args.fail_on_missing_target).lower() == "true"),
        )
        for side in sides
    ]


def _summaries_have_stats(summaries: list[dict[str, Any]]) -> bool:
    return all(summary.get("stats", {}).get("overall") for summary in summaries)


def _run_with_args(model, processor, tokenizer, args) -> dict[str, Any]:
    if args.mode == "text_forward":
        return _run_text_forward(model, tokenizer, args.prompt, args.device)
    if args.metadata_file:
        return _run_metadata_generate(model, processor, args)
    if args.mode == "pipeline_generate":
        return _run_pipeline_generate(processor, args.prompt, args)
    if args.mode == "ming_image_generate":
        return _run_ming_image_generate(model, processor, args.prompt, args)
    if args.mode == "adapter_generate":
        images = _run_generate_once(model, processor, args.prompt, args)
        return {
            "mode": "adapter_generate",
            "n_images": len(images) if isinstance(images, (list, tuple)) else (1 if images is not None else 0),
            "saved_images": _save_images(images, args.output_image_dir),
        }
    return _run_load_only(model)


def main() -> None:
    args = build_parser().parse_args()
    model, processor, tokenizer = _load_model(args)
    disable_geometry = str(args.disable_geometry).lower() == "true"
    summaries: list[dict[str, Any]] = []
    if disable_geometry:
        result = _run_with_args(model, processor, tokenizer, args)
        result["geometry"] = {"disabled": True}
    else:
        scalers = [VLMGeometryScaler(model, config) for config in _make_geometry_configs(args)]
        args._geometry_scalers = scalers
        with ExitStack() as stack:
            for scaler in scalers:
                stack.enter_context(scaler)
            result = _run_with_args(model, processor, tokenizer, args)
        summaries = [scaler.summary() for scaler in scalers]
        result["geometry"] = summaries[0] if len(summaries) == 1 else {"scalers": summaries}
    require_stats = str(args.require_stats).lower() == "true" and args.mode != "load_only"
    if require_stats and not disable_geometry and not _summaries_have_stats(summaries):
        diagnostics = [summary.get("diagnostics", {}).get("overall", {}) for summary in summaries]
        configs = [summary.get("config", {}) for summary in summaries]
        raise RuntimeError(
            "VLM geometry hooks produced no stats. "
            f"configs={configs!r}, diagnostics={diagnostics}"
        )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
