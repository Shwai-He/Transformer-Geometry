from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import time
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

import torch
import tiktoken


def _resolve_dtype(dtype: str):
    key = str(dtype).lower()
    if key == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype={dtype}")
    return mapping[key]


class NanoGPTTokenizer:
    def __init__(self, dataset_meta_path: Optional[Path] = None):
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.chat_template = None
        self._encode = None
        self._decode = None
        self._token_to_text = None

        if dataset_meta_path is not None and dataset_meta_path.is_file():
            with dataset_meta_path.open("rb") as f:
                meta = pickle.load(f)
            stoi = meta.get("stoi")
            itos = meta.get("itos")
            if isinstance(stoi, dict) and isinstance(itos, dict):
                self._encode = lambda s: [stoi[c] for c in s]
                self._decode = lambda ids: "".join([itos[int(i)] for i in ids])
                self._token_to_text = lambda idx: str(itos.get(int(idx), f"<{int(idx)}>"))

        if self._encode is None:
            enc = tiktoken.get_encoding("gpt2")
            self._encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self._decode = lambda ids: enc.decode(list(map(int, ids)))

            def _tok(idx: int) -> str:
                try:
                    return enc.decode_single_token_bytes(int(idx)).decode("utf-8", errors="replace")
                except Exception:
                    return enc.decode([int(idx)])

            self._token_to_text = _tok

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = False, max_length: Optional[int] = None):
        ids = self._encode(text)
        if truncation and max_length is not None:
            ids = ids[: max_length]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        if return_tensors != "pt":
            raise ValueError("NanoGPTTokenizer only supports return_tensors='pt'")
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_ids_to_tokens(self, ids: Iterable[int]):
        return [self._token_to_text(int(idx)) for idx in ids]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if tokenize:
            raise ValueError("NanoGPTTokenizer.apply_chat_template only supports tokenize=False")
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


def load_nanogpt_checkpoint(ckpt_path: str, device: str, dtype: str, nanogpt_repo_root: Optional[str] = None):
    load_start = time.time()
    ckpt_file = Path(ckpt_path).expanduser().resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"nanoGPT checkpoint not found: {ckpt_file}")
    print(f"[STAGE] load_nanogpt_checkpoint_start ckpt={ckpt_file}", flush=True)

    def _stable_local_ckpt_path(src: Path) -> Path:
        cache_root = Path(os.getenv("NANOGPT_LOCAL_CKPT_CACHE_DIR", "/tmp/nanogpt_ckpt_cache")).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        stat = src.stat()
        key = f"{src}:{stat.st_size}:{int(stat.st_mtime)}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return cache_root / f"{src.stem}-{digest}{src.suffix}"

    def _maybe_stage_ckpt_to_local(src: Path) -> Path:
        # Default: stage to local tmp first to avoid flaky remote endpoint reads.
        use_local_stage = os.getenv("NANOGPT_STAGE_CKPT_LOCAL", "true").strip().lower() not in {"0", "false", "no", "off"}
        if not use_local_stage:
            return src
        dst = _stable_local_ckpt_path(src)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            print(f"[INFO] Staged checkpoint to local path: {dst}", flush=True)
        return dst

    def _torch_load_with_retry(path: Path, map_location: str):
        max_retries = int(os.getenv("NANOGPT_TORCH_LOAD_RETRIES", "4"))
        base_sleep = float(os.getenv("NANOGPT_TORCH_LOAD_RETRY_SLEEP", "1.5"))
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return torch.load(str(path), map_location=map_location, weights_only=False)
            except OSError as exc:
                last_error = exc
                msg = str(exc).lower()
                if "transport endpoint is not connected" not in msg:
                    raise
                if attempt == max_retries:
                    break
                sleep_s = base_sleep * attempt
                print(
                    f"[WARN] torch.load failed on attempt {attempt}/{max_retries} "
                    f"for {path}: {exc}. Retrying in {sleep_s:.1f}s...",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise last_error  # type: ignore[misc]

    repo_root = Path(nanogpt_repo_root).expanduser().resolve() if nanogpt_repo_root else ckpt_file.parents[1]
    if not (repo_root / "model.py").is_file():
        raise FileNotFoundError(f"Could not find nanoGPT repo root from {repo_root}")
    print(f"[STAGE] nanogpt_repo_root_ready repo_root={repo_root}", flush=True)

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    stage_start = time.time()
    load_path = _maybe_stage_ckpt_to_local(ckpt_file)
    print(
        f"[STAGE] local_ckpt_stage_done path={load_path} elapsed_sec={time.time() - stage_start:.1f}",
        flush=True,
    )
    print(f"[STAGE] torch_load_start path={load_path}", flush=True)
    torch_load_start = time.time()
    checkpoint = _torch_load_with_retry(load_path, map_location=device)
    print(
        f"[STAGE] torch_load_done path={load_path} elapsed_sec={time.time() - torch_load_start:.1f}",
        flush=True,
    )

    from model import GPT, GPTConfig  # pylint: disable=import-error

    # Infer use_rope / embedding_layernorm from checkpoint keys when model_args doesn't
    # include them (old checkpoints predate these config fields; GPTConfig defaults would
    # otherwise create the wrong architecture).
    ckpt_keys = set(checkpoint["model"].keys())
    model_args = dict(checkpoint["model_args"])
    if "use_rope" not in model_args:
        model_args["use_rope"] = not any("transformer.wpe" in k for k in ckpt_keys)
    if "embedding_layernorm" not in model_args:
        model_args["embedding_layernorm"] = any("transformer.ln_e" in k for k in ckpt_keys)
    if "enable_attnres" not in model_args:
        model_args["enable_attnres"] = any("attnres_query" in k for k in ckpt_keys)

    gptconf = GPTConfig(**model_args)
    print("[STAGE] nanogpt_model_build_start", flush=True)
    model_build_start = time.time()
    model = GPT(gptconf)
    print(
        f"[STAGE] nanogpt_model_build_done elapsed_sec={time.time() - model_build_start:.1f}",
        flush=True,
    )
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Detect residual-patch variant and apply patch before loading state dict so that
    # the wx / scale Linear modules exist and can receive their trained weights.
    _has_residual_wx = any("residual_attn_wx" in k or "residual_mlp_wx" in k for k in ckpt_keys)
    _has_residual_scale = any("residual_attn_scale" in k or "residual_mlp_scale" in k for k in ckpt_keys)
    _has_branch_wx = any("residual_attn_branch_wx" in k or "residual_mlp_branch_wx" in k for k in ckpt_keys)
    _has_branch_scale = any("residual_attn_branch_scale" in k or "residual_mlp_branch_scale" in k for k in ckpt_keys)
    if _has_residual_wx or _has_residual_scale:
        try:
            from residual_patch_core import apply_qwen3_forward_patch  # pylint: disable=import-error
            _patch_cfg = type("_PatchCfg", (), {})()
            _patch_cfg.residual_mode = "wx" if _has_residual_wx else "param"
            _patch_cfg.residual_branch_gate = bool(_has_branch_wx or _has_branch_scale)
            _patch_cfg.residual_wx = 1.0
            _patch_cfg.residual_branch_wx = 1.0
            _patch_cfg.residual_branch_scale_init = 1.0
            _ckpt_cfg = checkpoint.get("config", {})
            _patch_cfg.residual_wx_activation = (
                _ckpt_cfg.get("residual_wx_activation", "silu") if isinstance(_ckpt_cfg, dict) else "silu"
            )
            _patch_cfg.residual_wx_learnable = True
            _patch_cfg.residual_branch_wx_learnable = bool(_has_branch_wx or _has_branch_scale)
            _patch_cfg.residual_attn_param = 1.0
            _patch_cfg.residual_mlp_param = 1.0
            _patch_cfg.residual_scale_init = 1.0
            _patch_cfg.residual_scale_depth_slope = 0.0
            _patch_cfg.residual_scale_learnable = _has_residual_scale
            _patch_cfg.residual_branch_scale_learnable = _has_branch_scale
            _patch_cfg.use_post_norm = False
            _patch_cfg.post_norm_start_layer = 10 ** 9
            _patch_cfg.residual_penalty_lambda = 0.0
            _patch_cfg.xsa_forward_target = "none"
            _patch_cfg.xsa_forward_strength = 0.0
            _patch_cfg.attn_diag_mode = "none"
            _patch_cfg.attn_diag_keep_first = True
            apply_qwen3_forward_patch(model, _patch_cfg)
            print(f"[INFO] Applied residual patch (mode={_patch_cfg.residual_mode}, branch_gate={_patch_cfg.residual_branch_gate})")
        except ImportError:
            print("[WARN] residual_patch_core not found; residual/gate weights will not be loaded.")

    allowed_missing_suffixes = (
        ".attn.bias",
        ".attn.attn_additive_mask",
        ".xsa_forward_alpha_raw",
        ".xsa_forward_gamma_raw",
        ".xsa_forward_gate_raw",
        ".xsa_forward_gate_proj.weight",
        ".xsa_forward_gate_proj.bias",
    )
    model_state_dict = model.state_dict()
    dropped_optional_shape_mismatch = []
    for key in list(state_dict.keys()):
        if key not in model_state_dict:
            continue
        src_shape = tuple(state_dict[key].shape)
        dst_shape = tuple(model_state_dict[key].shape)
        if src_shape == dst_shape:
            continue
        if key.endswith(allowed_missing_suffixes):
            dropped_optional_shape_mismatch.append((key, src_shape, dst_shape))
            state_dict.pop(key)
    if dropped_optional_shape_mismatch:
        preview = dropped_optional_shape_mismatch[:8]
        print(
            "[INFO] Dropping optional XSA/residual keys with shape mismatch before load_state_dict: "
            f"count={len(dropped_optional_shape_mismatch)} preview={preview}",
            flush=True,
        )

    print("[STAGE] load_state_dict_start", flush=True)
    load_state_start = time.time()
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        f"[STAGE] load_state_dict_done elapsed_sec={time.time() - load_state_start:.1f}",
        flush=True,
    )
    # Keys that are legitimately absent in some architectures.
    _benign_missing_keys = {"transformer.ln_e.weight"}
    # Keys from other architectural variants that don't affect the base forward pass.
    _arch_unexpected_patterns = (
        "transformer.wpe",          # baseline: learned positional embedding (use_rope=False, old ckpt)
        # Legacy residual-patch key names (superseded by residual_patch_core):
        "attn_residual_scale",
        "attn_residual_wx",
        "mlp_residual_scale",
        "mlp_residual_wx",
    )
    benign_unexpected = [
        k for k in incompatible.unexpected_keys
        if any(pat in k for pat in _arch_unexpected_patterns)
    ]
    hard_unexpected = [k for k in incompatible.unexpected_keys if k not in benign_unexpected]
    disallowed_missing = [
        key for key in incompatible.missing_keys
        if key not in _benign_missing_keys and not key.endswith(allowed_missing_suffixes)
    ]
    if hard_unexpected or disallowed_missing:
        raise RuntimeError(
            "Failed to load nanoGPT checkpoint cleanly: "
            f"missing={disallowed_missing} unexpected={hard_unexpected}"
        )
    if benign_unexpected or incompatible.missing_keys:
        unexpected_preview = benign_unexpected[:5]
        missing_preview = list(incompatible.missing_keys[:8])
        print(
            "[INFO] Load complete with known arch differences — "
            f"ignored_unexpected_count={len(benign_unexpected)} "
            f"ignored_unexpected_preview={unexpected_preview}; "
            f"randomly_init_missing_count={len(incompatible.missing_keys)} "
            f"randomly_init_missing_preview={missing_preview}",
            flush=True,
        )
    print(f"[STAGE] model_to_device_start device={device} dtype={dtype}", flush=True)
    model_to_device_start = time.time()
    model.to(device=device, dtype=_resolve_dtype(dtype))
    print(
        f"[STAGE] model_to_device_done device={device} dtype={dtype} "
        f"elapsed_sec={time.time() - model_to_device_start:.1f}",
        flush=True,
    )
    model.eval()
    print("[STAGE] model_eval_ready", flush=True)

    dataset_name = None
    dataset_meta_path = None
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = list(model.transformer.h)
        xsa_only = bool(config.get("xsa_forward_only", False))
        xsa_ref = str(config.get("xsa_forward_ref", "self_value"))
        xsa_space = str(config.get("xsa_forward_space", "pre_o_proj"))
        xsa_target = str(config.get("xsa_forward_target", "attn"))
        xsa_op = str(config.get("xsa_forward_op", "remove_parallel"))
        xsa_alpha = float(config.get("xsa_forward_alpha", 1.0))
        xsa_lgamma = bool(config.get("xsa_forward_learnable_gamma", False))
        xsa_gph = bool(config.get("xsa_forward_gamma_per_head", False))
        xsa_ginit = float(config.get("xsa_forward_gamma_init", 1.0))
        xsa_lalpha = bool(config.get("xsa_forward_learnable_alpha", False))
        xsa_amin = float(config.get("xsa_forward_alpha_min", 0.0))
        xsa_amax = float(config.get("xsa_forward_alpha_max", 1.0))

        for blk in blocks:
            blk.xsa_forward_only = xsa_only
            blk.xsa_forward_ref = xsa_ref
            blk.xsa_forward_space = xsa_space
            blk.xsa_forward_target = xsa_target
            blk.xsa_forward_op = xsa_op
            blk.xsa_forward_alpha = xsa_alpha
            blk.xsa_forward_learnable_alpha = xsa_lalpha
            blk.xsa_forward_alpha_min = xsa_amin
            blk.xsa_forward_alpha_max = xsa_amax
            blk.xsa_forward_learnable_gamma = xsa_lgamma
            blk.xsa_forward_gamma_per_head = xsa_gph
            if hasattr(blk, "xsa_forward_gamma_raw"):
                blk.xsa_forward_gamma_raw.requires_grad_(False)
            if hasattr(blk, "xsa_forward_alpha_raw"):
                blk.xsa_forward_alpha_raw.requires_grad_(False)
            if hasattr(blk, "xsa_forward_gate_raw"):
                blk.xsa_forward_gate_raw.requires_grad_(False)
        print(
            "[INFO] Applied ckpt XSA flags to model: "
            f"forward_only={xsa_only} ref={xsa_ref} space={xsa_space} target={xsa_target} "
            f"op={xsa_op} alpha={xsa_alpha} learnable_gamma={xsa_lgamma} gamma_per_head={xsa_gph} gamma_init={xsa_ginit}",
            flush=True,
        )
        if xsa_lgamma:
            gamma_layer_means = []
            for lid, blk in enumerate(blocks):
                if hasattr(blk, "xsa_forward_gamma_raw"):
                    g = torch.exp(blk.xsa_forward_gamma_raw.detach().float().view(-1))
                    g_mean = float(g.mean().item())
                    gamma_layer_means.append(g_mean)
                    head_preview = ",".join(f"{float(v):.4f}" for v in g[: min(4, g.numel())].tolist())
                    print(
                        f"[INFO] loaded_gamma layer={lid} mean={g_mean:.6f} heads[:4]=[{head_preview}]",
                        flush=True,
                    )
            if gamma_layer_means:
                gmin = min(gamma_layer_means)
                gmax = max(gamma_layer_means)
                gavg = sum(gamma_layer_means) / len(gamma_layer_means)
                print(
                    f"[INFO] loaded_gamma_summary layers={len(gamma_layer_means)} mean={gavg:.6f} min={gmin:.6f} max={gmax:.6f}",
                    flush=True,
                )

    if isinstance(config, dict):
        dataset_name = config.get("dataset")
    if dataset_name:
        candidate = repo_root / "data" / str(dataset_name) / "meta.pkl"
        if candidate.is_file():
            dataset_meta_path = candidate
    tokenizer = NanoGPTTokenizer(dataset_meta_path=dataset_meta_path)
    print(
        f"[STAGE] load_nanogpt_checkpoint_done elapsed_sec={time.time() - load_start:.1f}",
        flush=True,
    )
    return model, tokenizer, checkpoint, repo_root


def map_intervention_site(intervention_site: str):
    site = str(intervention_site).lower().strip()
    if site == "xsa_middle_multihead":
        return {"ref": "self_value", "space": "pre_o_proj", "target": "attn"}
    if site == "xsa_middle":
        return {"ref": "self_value", "space": "post_o_proj", "target": "attn"}
    if site == "residual_output":
        return {"ref": "residual", "space": "post_o_proj", "target": "attn"}
    raise ValueError(f"Unsupported intervention_site for nanoGPT: {intervention_site}")


@contextmanager
def nanogpt_xsa_context(
    model,
    enable: bool,
    ref: str,
    space: str,
    target: str,
    start_layer: int,
    end_layer: int,
    single_layer: Optional[int] = None,
):
    blocks = list(model.transformer.h)
    n_layers = len(blocks)
    end = n_layers - 1 if int(end_layer) < 0 else min(int(end_layer), n_layers - 1)
    start = max(0, int(start_layer))
    enabled_layers = {int(single_layer)} if single_layer is not None else set(range(start, end + 1))

    saved = []
    for lid, block in enumerate(blocks):
        saved.append(
            (
                getattr(block, "xsa_forward_only", False),
                getattr(block, "xsa_forward_ref", "residual"),
                getattr(block, "xsa_forward_space", "post_o_proj"),
                getattr(block, "xsa_forward_op", "remove_parallel"),
                getattr(block, "xsa_forward_alpha", 1.0),
                getattr(block, "xsa_forward_learnable_alpha", False),
                getattr(block, "xsa_forward_alpha_min", 0.0),
                getattr(block, "xsa_forward_alpha_max", 1.0),
                getattr(block, "xsa_forward_learnable_gamma", False),
                getattr(block, "xsa_forward_gamma_per_head", False),
                getattr(block, "xsa_forward_target", "attn"),
                getattr(block, "xsa_forward_enabled", True),
            )
        )
        block.xsa_forward_only = bool(enable)
        block.xsa_forward_ref = ref
        block.xsa_forward_space = space
        block.xsa_forward_op = "remove_parallel"
        block.xsa_forward_alpha = 1.0
        block.xsa_forward_target = target
        block.xsa_forward_enabled = bool(enable) and (lid in enabled_layers)
    try:
        yield
    finally:
        for block, state in zip(blocks, saved):
            (
                block.xsa_forward_only,
                block.xsa_forward_ref,
                block.xsa_forward_space,
                block.xsa_forward_op,
                block.xsa_forward_alpha,
                block.xsa_forward_learnable_alpha,
                block.xsa_forward_alpha_min,
                block.xsa_forward_alpha_max,
                block.xsa_forward_learnable_gamma,
                block.xsa_forward_gamma_per_head,
                block.xsa_forward_target,
                block.xsa_forward_enabled,
            ) = state
