#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repgeo import GeometryAnalyzer

DEFAULT_PROMPT_FILE = "results/prompts.txt"
DEFAULT_PROMPT_INDEX = 8


def normalize_drop_lists_in_model_config(model_name_or_path: str) -> None:
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return

    config_path = model_path / "config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    changed = False
    for key in ("drop_attn_list", "drop_mlp_list"):
        if config.get(key) != []:
            config[key] = []
            changed = True

    if changed:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {config_path}: set drop_attn_list/drop_mlp_list to []")


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in visible.split(",") if x.strip()]
    if ids:
        return len(ids)
    try:
        import torch  # Local import to keep startup light when CUDA is absent.

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-wise generation geometry probe")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt text. If omitted, load from --prompt_file.")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        help=f"Text file with one prompt per line (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help="Zero-based line index from --prompt_file (backward-compatible single index).",
    )
    parser.add_argument(
        "--prompt_indices",
        type=int,
        nargs="+",
        default=None,
        help="One or more zero-based line indices from --prompt_file, e.g. --prompt_indices 0 1 2",
    )
    parser.add_argument(
        "--all_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all prompts from --prompt_file (default: True).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--measure_mode",
        type=str,
        default="both",
        choices=["prefill", "decode", "both"],
        help="Measurement mode: prefill-only, decode-only, or both (timeline).",
    )
    parser.add_argument(
        "--geometry_space",
        type=str,
        default="hidden",
        choices=["hidden", "logits"],
        help="Space for para/perp decomposition.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="Optional HF device_map for multi-GPU sharding.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help='Optional per-GPU max memory when using --device_map (example: "80GiB").',
    )
    parser.add_argument(
        "--max_memory_gpu0",
        type=str,
        default=None,
        help='Optional max memory cap only for cuda:0 (example: "10GiB").',
    )
    parser.add_argument(
        "--offload_folder",
        type=str,
        default=None,
        help='Optional offload folder when using --device_map (example: "/tmp/offload").',
    )
    parser.add_argument(
        "--require_multi_gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail if --device_map is set but model still ends up on a single CUDA device. "
        "Default: auto-enable when multiple GPUs are visible.",
    )
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Optional HF attention implementation. Use eager when value_pre metrics need attention weights.",
    )
    parser.add_argument(
        "--random_init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize model from config with random weights instead of loading pretrained checkpoint.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed used when --random_init is enabled.",
    )
    parser.add_argument("--print_per_layer", action="store_true", help="Print per-layer metrics for every generation step.")
    parser.add_argument(
        "--prefill_token_stride",
        type=int,
        default=1,
        help="For prefill steps, keep every Nth token plus the last token. Default: 1 keeps all tokens.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Prefill-only prompt batch size. Default: 1.",
    )
    parser.add_argument(
        "--include_sublayer_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-step attn/mlp sublayer geometry (GPT2-like models). Default: True.",
    )
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Default: results/generation_<geometry_space>/<model_name>.json",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        model_name = args.model_name_or_path.split("/")[-1]
        if not args.random_init:
            output_path = (
                f"results/generation_{args.geometry_space}/"
                f"{model_name}-mode{args.measure_mode}-tokens{args.max_new_tokens}.json"
            )
        else:
            output_path = (
                f"results/generation_{args.geometry_space}_random_init/"
                f"{model_name}-mode{args.measure_mode}-tokens{args.max_new_tokens}.json"
            )
    print(f"[INFO] Output path: {output_path}")

    if args.prompt is not None:
        selected_prompts = [args.prompt]
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompts = [ln.strip() for ln in prompt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in: {prompt_path}")
        if args.all_prompts:
            selected_prompts = prompts
        elif args.prompt_indices:
            bad = [idx for idx in args.prompt_indices if idx < 0 or idx >= len(prompts)]
            if bad:
                raise IndexError(f"prompt_indices out of range: {bad}. Valid range: 0..{len(prompts)-1}")
            selected_prompts = [prompts[idx] for idx in args.prompt_indices]
        else:
            idx = args.prompt_index if args.prompt_index is not None else DEFAULT_PROMPT_INDEX
            if not (0 <= idx < len(prompts)):
                raise IndexError(f"prompt_index out of range: {idx}. Valid range: 0..{len(prompts)-1}")
            selected_prompts = [prompts[idx]]
    normalize_drop_lists_in_model_config(args.model_name_or_path)

    visible_gpu_count = _visible_gpu_count()
    if args.device_map is None and visible_gpu_count >= 2:
        args.device_map = "balanced"
        print(f"[INFO] Auto-set device_map=balanced (visible GPUs: {visible_gpu_count}).")
    require_multi_gpu = args.require_multi_gpu
    if require_multi_gpu is None:
        require_multi_gpu = args.device_map is not None and visible_gpu_count >= 2

    requested_kwargs = {
        "model_name_or_path": args.model_name_or_path,
        "device": args.device,
        "device_map": args.device_map,
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "max_memory_gpu0": args.max_memory_gpu0,
        "offload_folder": args.offload_folder,
        "require_multi_gpu": bool(require_multi_gpu),
        "dtype": args.dtype,
        "random_init": args.random_init,
        "random_seed": args.random_seed,
        "attn_implementation": args.attn_implementation,
    }
    accepted = set(inspect.signature(GeometryAnalyzer.__init__).parameters.keys())
    analyzer_kwargs = {k: v for k, v in requested_kwargs.items() if k in accepted}
    dropped = [k for k in requested_kwargs if k not in analyzer_kwargs]
    if dropped:
        print(f"[WARN] GeometryAnalyzer does not support args: {dropped}.")
    analyzer = GeometryAnalyzer(**analyzer_kwargs)
    cuda_devices_used = getattr(analyzer, "cuda_devices_used", None)
    hf_device_map = getattr(analyzer, "hf_device_map", None)
    loaded_attn_implementation = getattr(analyzer, "loaded_attn_implementation", None)
    print("CUDA devices used:", cuda_devices_used)
    print("Loaded attention implementation:", loaded_attn_implementation)
    if hf_device_map is not None:
        print("hf_device_map entries:", len(hf_device_map))
    def run_one_prompt(prompt: str) -> dict:
        if args.measure_mode == "prefill":
            probe = analyzer.analyze(
                prompt=prompt,
                top_k=args.top_k,
                geometry_space=args.geometry_space,
                print_per_layer=args.print_per_layer,
                include_sublayer_metrics=args.include_sublayer_metrics,
                prefill_token_stride=args.prefill_token_stride,
            )
            prefill_steps = probe.get("steps", [])
            timeline = [{"phase": "prefill", "timestep": int(s["step"]), **s} for s in prefill_steps]
            run = {
                "model_name_or_path": args.model_name_or_path,
                "prompt": prompt,
                "geometry_space": args.geometry_space,
                "prompt_token_count": probe.get("prompt_token_count", len(prefill_steps)),
                "max_new_tokens": 0,
                "generated_token_count": 0,
                "generated_text": "",
                "steps": [],
                "prefill_steps": prefill_steps,
                "timeline": timeline,
                "summary": probe.get("summary"),
                "layer_metrics": probe.get("layer_metrics"),
                "predicted_next_token": probe.get("predicted_next_token"),
            }
            if "warning" in probe:
                run["warning"] = probe["warning"]
            if "sublayer_metrics" in probe:
                run["sublayer_metrics"] = probe["sublayer_metrics"]
            if "sublayer_warning" in probe:
                run["sublayer_warning"] = probe["sublayer_warning"]
            return run

        run = analyzer.analyze_generation(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            geometry_space=args.geometry_space,
            do_sample=args.do_sample,
            temperature=args.temperature,
            print_per_layer=args.print_per_layer,
            include_sublayer_metrics=args.include_sublayer_metrics,
            prefill_token_stride=args.prefill_token_stride,
        )
        run["model_name_or_path"] = args.model_name_or_path
        if args.measure_mode == "decode":
            run["prefill_steps"] = []
            run["timeline"] = [s for s in (run.get("timeline") or []) if s.get("phase") == "decode"]
        return run

    def run_prefill_batch(batch_prompts: list[str], start_idx: int) -> list[dict]:
        probes = analyzer.analyze_prefill_batch(
            prompts=batch_prompts,
            top_k=args.top_k,
            geometry_space=args.geometry_space,
            include_sublayer_metrics=args.include_sublayer_metrics,
            prefill_token_stride=args.prefill_token_stride,
        )
        runs = []
        for offset, probe in enumerate(probes):
            prompt = batch_prompts[offset]
            prefill_steps = probe.get("steps", [])
            timeline = [{"phase": "prefill", "timestep": int(s["step"]), **s} for s in prefill_steps]
            run = {
                "model_name_or_path": args.model_name_or_path,
                "prompt": prompt,
                "geometry_space": args.geometry_space,
                "prompt_token_count": probe.get("prompt_token_count", len(prefill_steps)),
                "max_new_tokens": 0,
                "generated_token_count": 0,
                "generated_text": "",
                "steps": [],
                "prefill_steps": prefill_steps,
                "timeline": timeline,
                "summary": probe.get("summary"),
                "layer_metrics": probe.get("layer_metrics"),
                "predicted_next_token": probe.get("predicted_next_token"),
                "batch_prompt_index": start_idx + offset,
            }
            if "warning" in probe:
                run["warning"] = probe["warning"]
            if "sublayer_metrics" in probe:
                run["sublayer_metrics"] = probe["sublayer_metrics"]
            if "sublayer_warning" in probe:
                run["sublayer_warning"] = probe["sublayer_warning"]
            runs.append(run)
        return runs

    if args.measure_mode == "prefill" and len(selected_prompts) > 1 and args.batch_size > 1:
        runs = []
        batch_size = max(int(args.batch_size), 1)
        for start in range(0, len(selected_prompts), batch_size):
            batch_prompts = selected_prompts[start:start + batch_size]
            print(f"\n=== Prompt batch {start}..{start + len(batch_prompts) - 1} ===")
            runs.extend(run_prefill_batch(batch_prompts, start_idx=start))
        result = {"n_prompts": len(runs), "runs": runs}
    elif len(selected_prompts) == 1:
        result = run_one_prompt(selected_prompts[0])
    else:
        runs = []
        for i, prompt in enumerate(selected_prompts):
            print(f"\n=== Prompt {i} ===")
            runs.append(run_one_prompt(prompt))
        result = {"n_prompts": len(runs), "runs": runs}

    result_with_meta = {
        "run_meta": {
            "model_name_or_path": args.model_name_or_path,
            "device": analyzer.device,
            "input_device": str(analyzer.input_device),
            "device_map": analyzer.device_map,
            "require_multi_gpu": bool(require_multi_gpu),
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "max_memory_gpu0": args.max_memory_gpu0,
            "offload_folder": args.offload_folder,
            "cuda_devices_used": cuda_devices_used,
            "hf_device_map": hf_device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "loaded_attn_implementation": loaded_attn_implementation,
            "prefill_token_stride": args.prefill_token_stride,
            "batch_size": args.batch_size,
            "random_init": bool(args.random_init),
            "random_seed": args.random_seed,
            "measure_mode": args.measure_mode,
        },
        **result,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result_with_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("Device:", result_with_meta["run_meta"]["input_device"])
    if "runs" not in result:
        print("Prompt token count:", result["prompt_token_count"])
        if "warning" in result:
            print("Warning:", result["warning"])
        if "sublayer_warning" in result:
            print("Sublayer warning:", result["sublayer_warning"])
        print("Generated token count:", result["generated_token_count"])
        print("Measure mode:", args.measure_mode)
        if result["steps"]:
            first_step = result["steps"][0]
            if "sublayer_warning" in first_step:
                print("Sublayer warning:", first_step["sublayer_warning"])
            elif "sublayer_metrics" in first_step:
                print("Sublayer metrics: enabled")
        print("Generated text:")
        print(result["generated_text"])
    else:
        print("n_prompts:", result["n_prompts"])
        for i, run in enumerate(result["runs"]):
            print(f"[prompt {i}] Prompt token count: {run['prompt_token_count']}")
            if "warning" in run:
                print(f"[prompt {i}] Warning: {run['warning']}")
            if "sublayer_warning" in run:
                print(f"[prompt {i}] Sublayer warning: {run['sublayer_warning']}")
            print(f"[prompt {i}] Generated token count: {run['generated_token_count']}")
            if run["steps"]:
                first_step = run["steps"][0]
                if "sublayer_warning" in first_step:
                    print(f"[prompt {i}] Sublayer warning: {first_step['sublayer_warning']}")
                elif "sublayer_metrics" in first_step:
                    print(f"[prompt {i}] Sublayer metrics: enabled")


if __name__ == "__main__":
    main()
