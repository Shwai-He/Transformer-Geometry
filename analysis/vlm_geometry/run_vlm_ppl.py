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


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_dtype(name: str) -> torch.dtype:
    name = (name or "bf16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


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


def _resolve_image_path(path_text: str, *, sparse_root: Path, data_file: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [
        sparse_root / path,
        data_file.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _answer_text(row: dict[str, Any], answer_mode: str) -> str:
    answer = row.get("answer")
    if answer is None:
        answer = row.get("label")
    if answer is None:
        return ""
    answer = str(answer).strip()
    if answer_mode == "option_text":
        options = row.get("options") or row.get("option") or {}
        if isinstance(options, dict) and answer in options:
            return str(options[answer]).strip()
    if answer_mode == "yes_no":
        return answer
    return answer


def _target_token_count(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


def _loss_to_record(loss: torch.Tensor, labels: torch.Tensor) -> tuple[float, int, float]:
    ce = float(loss.detach().float().item())
    n_tokens = _target_token_count(labels)
    nll = ce * n_tokens
    ppl = float(math.exp(min(ce, 50.0)))
    return ce, n_tokens, nll


def _filter_inputs(inputs: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key in allowed}


class QwenImagePPLRunner:
    def __init__(self, model_path: str, processor_path: str, dtype: torch.dtype, device_map: str):
        from modeling.qwenimage import load_qwenimage_text_encoder
        from transformers import AutoProcessor

        text_encoder_path = Path(model_path) / "text_encoder"
        if text_encoder_path.exists():
            model_path = str(text_encoder_path)
        self.model = load_qwenimage_text_encoder(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    def encode(self, image: Image.Image, question: str, answer: str):
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        full_inputs = full_inputs.to(self.model.device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        labels = full_inputs["input_ids"].clone()
        labels[:, : min(prompt_len, labels.shape[1])] = -100
        if "attention_mask" in full_inputs:
            labels = labels.masked_fill(full_inputs["attention_mask"] == 0, -100)
        return full_inputs, labels

    @torch.no_grad()
    def loss(self, image: Image.Image, question: str, answer: str):
        inputs, labels = self.encode(image, question, answer)
        allowed = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        }
        out = self.model(**_filter_inputs(inputs, allowed), labels=labels, use_cache=False)
        return out.loss, labels


class MingPPLRunner:
    def __init__(self, model_path: str, dtype: torch.dtype, device_map: str):
        from modeling.ming import load_ming_model, load_ming_processor

        self.model = load_ming_model(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            load_image_gen=False,
            low_cpu_mem_usage=True,
        ).eval()
        self.processor = load_ming_processor()

    def encode(self, image: Image.Image, question: str, answer: str):
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        prompt_messages = [{"role": "HUMAN", "content": user_content}]
        full_messages = [
            {"role": "HUMAN", "content": user_content},
            {"role": "ASSISTANT", "content": [{"type": "text", "text": answer}]},
        ]
        prompt_text = self.processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_messages, add_generation_prompt=False)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(full_messages)
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        labels = full_inputs["input_ids"].clone()
        labels[:, : min(prompt_len, labels.shape[1])] = -100
        if "attention_mask" in full_inputs:
            labels = labels.masked_fill(full_inputs["attention_mask"] == 0, -100)
        return full_inputs, labels

    @torch.no_grad()
    def loss(self, image: Image.Image, question: str, answer: str):
        inputs, labels = self.encode(image, question, answer)
        image_embeds = None
        video_embeds = None
        audio_embeds = None
        audio_embeds_lengths = None
        if inputs.get("pixel_values") is not None:
            image_embeds = self.model.extract_image_feature(
                inputs["pixel_values"],
                grid_thw=inputs.get("image_grid_thw"),
            )
        if inputs.get("pixel_values_videos") is not None:
            video_embeds = self.model.extract_image_feature(
                inputs["pixel_values_videos"],
                grid_thw=inputs.get("video_grid_thw"),
            )
        if inputs.get("audio_feats") is not None:
            audio_embeds, audio_embeds_lengths = self.model.extract_audio_feature(
                inputs["audio_feats"],
                inputs.get("audio_feats_lengths"),
            )

        clipped_ids = inputs["input_ids"].clip(
            0,
            self.model.model.get_input_embeddings().weight.shape[0] - 1,
        )
        if image_embeds is None and video_embeds is None and audio_embeds is None:
            words_embeddings = self.model.model.get_input_embeddings()(clipped_ids)
            image_mask = None
            audio_mask = None
        else:
            words_embeddings, image_mask, audio_mask = self.model.prompt_wrap_navit(
                clipped_ids,
                image_embeds,
                video_embeds,
                audio_embeds,
                audio_embeds_lengths,
                inputs.get("audio_placeholder_loc_lens"),
                None,
            )

        position_ids = inputs.get("position_ids")
        if (
            position_ids is None
            and self.model.config.llm_config.rope_scaling is not None
            and self.model.config.llm_config.rope_scaling.get("type") == "3D"
        ):
            position_ids, _ = self.model.get_rope_index(
                inputs["input_ids"],
                image_token_id=self.model.config.llm_config.image_patch_token,
                video_token_id=self.model.config.llm_config.image_patch_token,
                image_start_token_id=self.model.config.llm_config.image_start_token,
                video_start_token_id=self.model.config.llm_config.video_start_token,
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
            )

        out = self.model.model(
            input_ids=None,
            attention_mask=inputs.get("attention_mask"),
            position_ids=position_ids,
            inputs_embeds=words_embeddings,
            labels=labels,
            use_cache=False,
            return_dict=True,
            image_mask=image_mask,
            audio_mask=audio_mask,
        )
        return out.loss, labels


class BagelPPLRunner:
    def __init__(self, model_path: str, dtype: torch.dtype):
        from data.data_utils import prepare_attention_mask_per_sample
        from data.data_utils import pil_img2rgb
        from modeling import get_model_adapter, load_model_bundle

        adapter = get_model_adapter("bagel")
        bundle = load_model_bundle(
            "bagel",
            model_path=model_path,
            dtype=dtype,
            visual_gen=False,
            visual_und=True,
            max_mem_per_gpu="80GiB",
        )
        self.model = bundle.model.eval()
        self.tokenizer = bundle.require_component("tokenizer")
        self.vit_transform = bundle.require_component("vit_transform")
        self.new_token_ids = bundle.require_component("new_token_ids")
        self.pil_img2rgb = pil_img2rgb
        self.prepare_attention_mask_per_sample = prepare_attention_mask_per_sample
        self.device = self._infer_device()

    def _infer_device(self) -> torch.device:
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, list):
            return [self._to_device(x) for x in obj]
        return obj

    def encode(self, image: Image.Image, question: str, answer: str) -> tuple[dict[str, Any], torch.Tensor]:
        image = self.vit_transform.resize_transform(self.pil_img2rgb(image))
        img_inputs, _, _ = self.model.prepare_vit_images(
            curr_kvlens=[0],
            curr_rope=[0],
            images=[image],
            transforms=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )
        prompt_ids = self._encode_text(str(question))
        answer_ids = self._encode_text(str(answer))
        full_text_ids = prompt_ids + answer_ids
        shifted_text_ids = [self.new_token_ids["bos_token_id"]] + full_text_ids

        image_len = int(img_inputs["packed_seqlens"][0].item())
        text_start = image_len
        text_len = len(shifted_text_ids)

        packed_text_ids = torch.cat(
            [
                img_inputs["packed_text_ids"].long(),
                torch.tensor(shifted_text_ids, dtype=torch.long),
            ],
            dim=0,
        )
        packed_text_indexes = torch.cat(
            [
                img_inputs["packed_text_indexes"].long(),
                torch.arange(text_start, text_start + text_len, dtype=torch.long),
            ],
            dim=0,
        )
        packed_position_ids = torch.cat(
            [
                img_inputs["packed_position_ids"].long(),
                torch.arange(1, 1 + text_len, dtype=torch.long),
            ],
            dim=0,
        )
        answer_start = text_start + len(prompt_ids)
        ce_loss_indexes = torch.zeros(image_len + text_len, dtype=torch.bool)
        ce_loss_indexes[answer_start : answer_start + len(answer_ids)] = True
        packed_label_ids = torch.tensor(answer_ids, dtype=torch.long)

        model_inputs = {
            "sequence_length": image_len + text_len,
            "packed_text_ids": packed_text_ids,
            "packed_text_indexes": packed_text_indexes,
            "sample_lens": [image_len + text_len],
            "packed_position_ids": packed_position_ids,
            "nested_attention_masks": [
                self.prepare_attention_mask_per_sample(
                    [image_len, text_len],
                    ["full", "causal"],
                    device=self.device,
                )
            ],
            "ce_loss_indexes": ce_loss_indexes,
            "packed_label_ids": packed_label_ids,
            "packed_vit_tokens": img_inputs["packed_vit_tokens"],
            "packed_vit_token_indexes": img_inputs["packed_vit_token_indexes"].long(),
            "packed_vit_position_ids": img_inputs["packed_vit_position_ids"].long(),
            "vit_token_seqlens": img_inputs["vit_token_seqlens"].int(),
        }
        return {key: self._to_device(value) for key, value in model_inputs.items()}, packed_label_ids

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer.encode(text))

    @torch.no_grad()
    def loss(self, image: Image.Image, question: str, answer: str):
        inputs, labels = self.encode(image, question, answer)
        was_training = bool(self.model.language_model.training)
        original_forward = self.model.language_model.forward

        def hidden_only_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if isinstance(output, tuple):
                return output[0]
            return output

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            # BAGEL's language_model dispatches packed training inputs only when
            # its training flag is true.  We keep no-grad and restore the flag
            # immediately after this CE-only forward.
            self.model.language_model.train(True)
            self.model.language_model.forward = hidden_only_forward
            try:
                out = self.model(**inputs)
            finally:
                self.model.language_model.forward = original_forward
                self.model.language_model.train(was_training)
        ce = out["ce"]
        if ce is None or ce.numel() == 0:
            raise RuntimeError("BAGEL forward returned no CE values.")
        return ce.float().mean(), labels


def _build_runner(args):
    dtype = _resolve_dtype(args.dtype)
    if args.sparse_unified_root:
        root = Path(args.sparse_unified_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    if args.model_name == "qwenimage":
        return QwenImagePPLRunner(args.model_path, args.processor_path, dtype, args.device_map)
    if args.model_name == "ming":
        return MingPPLRunner(args.model_path, dtype, args.device_map)
    if args.model_name == "bagel":
        return BagelPPLRunner(args.model_path, dtype)
    raise ValueError(f"Unsupported model_name={args.model_name!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute image-text conditional answer PPL under VLM geometry scaling.")
    p.add_argument("--model-name", required=True, choices=["qwenimage", "ming", "bagel"])
    p.add_argument("--model-path", required=True)
    p.add_argument("--processor-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-samples", type=int, default=32)
    p.add_argument("--answer-mode", default="letter", choices=["letter", "option_text", "yes_no"])
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
    p.add_argument("--disable-geometry", default="false", help="Run without installing geometry hooks.")
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
    p.add_argument("--sparse-unified-root", default=os.environ.get("SPARSE_UNIFIED_ROOT", "SparseUnifiedModel"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    sparse_root = Path(args.sparse_unified_root).resolve()
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

    records = []
    total_nll = 0.0
    total_tokens = 0
    result_path = out_dir / "results.jsonl"
    disable_geometry = _str_to_bool(args.disable_geometry)
    hook_context = torch.no_grad() if disable_geometry else VLMGeometryScaler(runner.model, config)
    with hook_context as maybe_scaler, result_path.open("w", encoding="utf-8") as f:
        scaler = None if disable_geometry else maybe_scaler
        for idx, row in enumerate(tqdm(rows, desc="vlm-ppl")):
            image_path_text = row.get("image") or row.get("image_path")
            if not image_path_text:
                continue
            image_path = _resolve_image_path(str(image_path_text), sparse_root=sparse_root, data_file=data_file)
            question = row.get("question") or row.get("prompt") or row.get("text") or "Describe the image."
            answer = _answer_text(row, args.answer_mode)
            if not answer:
                continue
            image = Image.open(image_path).convert("RGB")
            if scaler is not None:
                scaler.set_sample_seed(args.scale_seed + idx)
            loss, labels = runner.loss(image, str(question), answer)
            ce, n_tokens, nll = _loss_to_record(loss, labels)
            record = {
                "index": idx,
                "benchmark": row.get("benchmark"),
                "data_id": row.get("data_id"),
                "image": str(image_path),
                "answer": answer,
                "ce": ce,
                "ppl": float(math.exp(min(ce, 50.0))),
                "target_tokens": n_tokens,
                "nll": nll,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            total_nll += nll
            total_tokens += n_tokens

    mean_ce = total_nll / max(1, total_tokens)
    summary = {
        "model_name": args.model_name,
        "data_file": str(data_file),
        "n_examples": len(records),
        "target_tokens": total_tokens,
        "mean_ce": mean_ce,
        "ppl": float(math.exp(min(mean_ce, 50.0))),
        "mean_example_ce": sum(r["ce"] for r in records) / max(1, len(records)),
        "geometry": {"disabled": True} if disable_geometry else scaler.summary(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
