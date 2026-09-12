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


DEFAULT_MODEL_NAME_OR_PATH = "Qwen/Qwen3-0.6B-Base"
DEFAULT_PROMPT_FILE = "results/prompts.txt"
DEFAULT_PROMPT_INDEX = 0


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
        if key in config and config.get(key) != []:
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
    parser = argparse.ArgumentParser(description="Representation geometry probe for Transformer LMs")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help=f"HuggingFace model id or local path (default: {DEFAULT_MODEL_NAME_OR_PATH})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Direct prompt text. If omitted, load from --prompt_file.",
    )
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
        action="store_true",
        help="Use all prompts from --prompt_file.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--geometry_space",
        type=str,
        default="hidden",
        choices=["hidden", "logits"],
        help="Space for para/perp decomposition.",
    )
    parser.add_argument(
        "--include_sublayer_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-layer attn/mlp sublayer geometry (GPT2-like models). Default: True.",
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Default: results/probe_<model_name>.json",
    )
    args = parser.parse_args()

    if args.prompt is not None:
        selected_prompts = [args.prompt]
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in file: {prompt_path}")
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
    }
    accepted = set(inspect.signature(GeometryAnalyzer.__init__).parameters.keys())
    analyzer_kwargs = {k: v for k, v in requested_kwargs.items() if k in accepted}
    dropped = [k for k in requested_kwargs if k not in analyzer_kwargs]
    if dropped:
        print(f"[WARN] GeometryAnalyzer does not support args: {dropped}.")
    analyzer = GeometryAnalyzer(**analyzer_kwargs)
    cuda_devices_used = getattr(analyzer, "cuda_devices_used", None)
    hf_device_map = getattr(analyzer, "hf_device_map", None)
    print("CUDA devices used:", cuda_devices_used)
    if hf_device_map is not None:
        print("hf_device_map entries:", len(hf_device_map))
    if len(selected_prompts) == 1:
        result = analyzer.analyze(
            prompt=selected_prompts[0],
            top_k=args.top_k,
            geometry_space=args.geometry_space,
            print_per_layer=True,
            include_sublayer_metrics=args.include_sublayer_metrics,
        )
    else:
        runs = []
        for i, prompt in enumerate(selected_prompts):
            print(f"\n=== Prompt {i} ===")
            run = analyzer.analyze(
                prompt=prompt,
                top_k=args.top_k,
                geometry_space=args.geometry_space,
                print_per_layer=True,
                include_sublayer_metrics=args.include_sublayer_metrics,
            )
            runs.append(run)
        result = {
            "n_prompts": len(runs),
            "runs": runs,
        }

    result_with_meta = {
        "run_meta": {
            "device": analyzer.device,
            "input_device": str(analyzer.input_device),
            "device_map": analyzer.device_map,
            "require_multi_gpu": bool(require_multi_gpu),
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "max_memory_gpu0": args.max_memory_gpu0,
            "offload_folder": args.offload_folder,
            "cuda_devices_used": analyzer.cuda_devices_used,
            "hf_device_map": analyzer.hf_device_map,
            "dtype": args.dtype,
            "random_init": bool(args.random_init),
            "random_seed": args.random_seed,
        },
        **result,
    }

    output_path = args.output
    if output_path is None:
        model_name = args.model_name_or_path.split("/")[-1]
        output_path = f"results/probe_{model_name}.json"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result_with_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("Device:", result_with_meta["run_meta"]["input_device"])
    if "summary" in result:
        print("Prompt token count:", result["prompt_token_count"])
        if "warning" in result:
            print("Warning:", result["warning"])
        if "sublayer_warning" in result:
            print("Sublayer warning:", result["sublayer_warning"])
        elif "sublayer_metrics" in result:
            print("Sublayer metrics: enabled")
        print("Summary:")
        for k, v in result["summary"].items():
            print(f"  {k}: {v}")
        print("Predicted next token:", repr(result["predicted_next_token"]))
    else:
        print("n_prompts:", result["n_prompts"])
        for i, run in enumerate(result["runs"]):
            print(f"[prompt {i}] Prompt token count: {run['prompt_token_count']}")
            if "warning" in run:
                print(f"[prompt {i}] Warning: {run['warning']}")
            if "sublayer_warning" in run:
                print(f"[prompt {i}] Sublayer warning: {run['sublayer_warning']}")
            elif "sublayer_metrics" in run:
                print(f"[prompt {i}] Sublayer metrics: enabled")
            print(f"[prompt {i}] Predicted next token: {repr(run['predicted_next_token'])}")


if __name__ == "__main__":
    main()
