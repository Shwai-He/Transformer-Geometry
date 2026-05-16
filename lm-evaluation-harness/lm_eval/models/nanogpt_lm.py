from __future__ import annotations

import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import find_executable_batch_size
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, handle_stop_sequences
from lm_eval.models.utils_hf import clear_torch_cache, pad_and_concat

eval_logger = logging.getLogger(__name__)


def _emit_progress(message: str) -> None:
    print(message, flush=True)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        x = v.strip().lower()
        if x in {"1", "true", "yes", "y", "on"}:
            return True
        if x in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(v)


def _tqdm_disable(disable_tqdm: bool) -> bool:
    force_tqdm = os.getenv("FORCE_TQDM", "false")
    if _as_bool(force_tqdm):
        return False
    return disable_tqdm


def _resolve_ckpt(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        best = p / "ckpt_best.pt"
        latest = p / "ckpt.pt"
        if best.is_file():
            return best
        if latest.is_file():
            return latest
        raise FileNotFoundError(f"No ckpt_best.pt or ckpt.pt found in: {p}")
    raise FileNotFoundError(f"Checkpoint path does not exist: {p}")


@register_model("nanogpt")
class NanoGPTLM(TemplateLM):
    """lm-eval wrapper for local nanoGPT checkpoints (ckpt.pt / ckpt_best.pt)."""

    def __init__(
        self,
        ckpt_path: str,
        nanogpt_repo_root: str | None = None,
        device: str = "cuda",
        dtype: str = "bf16",
        batch_size: int | str = 1,
        max_batch_size: int | None = 64,
        max_length: int | None = None,
        xsa_forward_only: bool | str | None = None,
        xsa_forward_target: str | None = None,
        xsa_forward_ref: str | None = None,
        xsa_forward_space: str | None = None,
        xsa_forward_op: str | None = None,
        xsa_forward_alpha: float | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        init_start = time.time()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size) if str(batch_size).isdigit() else str(batch_size)
        self.max_batch_size = int(max_batch_size) if max_batch_size is not None else 64
        self._max_gen_toks = 256
        self.batch_sizes = {}
        self.batch_schedule = 1.0
        if isinstance(self.batch_size, str) and self.batch_size.startswith("auto"):
            parts = self.batch_size.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                self.batch_schedule = max(1.0, float(parts[1]))
            self.batch_size = "auto"

        from lm_eval.models.nanogpt_attention_utils import (  # pylint: disable=import-error
            load_nanogpt_checkpoint,
        )

        _emit_progress(f"[STAGE] nanogpt_init_start ckpt_path={ckpt_path}")
        ckpt_file = _resolve_ckpt(ckpt_path)
        _emit_progress(f"[STAGE] nanogpt_ckpt_resolved ckpt={ckpt_file}")
        _emit_progress("[STAGE] nanogpt_load_ckpt_start")
        self.model, self.tokenizer, self.checkpoint, _ = load_nanogpt_checkpoint(
            ckpt_path=str(ckpt_file),
            device=str(self._device),
            dtype=dtype,
            nanogpt_repo_root=nanogpt_repo_root,
        )
        _emit_progress(
            f"[STAGE] nanogpt_load_ckpt_done elapsed_sec={time.time() - init_start:.1f}"
        )
        self._supports_attention_mask = (
            "attention_mask" in inspect.signature(self.model.forward).parameters
        )

        self.backend = "causal"
        self.max_length = int(max_length) if max_length is not None else int(self.model.config.block_size)
        self._progress_log_every = max(
            1, int(os.getenv("NANOGPT_LM_PROGRESS_EVERY", "100"))
        )
        self._batch_log_every = max(
            1, int(os.getenv("NANOGPT_LM_BATCH_LOG_EVERY", "1"))
        )

        # Optional runtime XSA overrides (without touching ckpt file).
        blocks = list(self.model.transformer.h) if hasattr(self.model, "transformer") else []
        if blocks:
            if xsa_forward_only is not None:
                v = _as_bool(xsa_forward_only)
                for blk in blocks:
                    blk.xsa_forward_only = v
            if xsa_forward_target is not None:
                for blk in blocks:
                    blk.xsa_forward_target = str(xsa_forward_target)
            if xsa_forward_ref is not None:
                for blk in blocks:
                    blk.xsa_forward_ref = str(xsa_forward_ref)
            if xsa_forward_space is not None:
                for blk in blocks:
                    blk.xsa_forward_space = str(xsa_forward_space)
            if xsa_forward_op is not None:
                for blk in blocks:
                    blk.xsa_forward_op = str(xsa_forward_op)
            if xsa_forward_alpha is not None:
                for blk in blocks:
                    blk.xsa_forward_alpha = float(xsa_forward_alpha)

            enabled_layers = [
                idx for idx, blk in enumerate(blocks)
                if bool(getattr(blk, "xsa_forward_only", False))
            ]
            if enabled_layers:
                ref = str(getattr(blocks[0], "xsa_forward_ref", ""))
                space = str(getattr(blocks[0], "xsa_forward_space", ""))
                target = str(getattr(blocks[0], "xsa_forward_target", ""))
                op = str(getattr(blocks[0], "xsa_forward_op", ""))
                alpha = float(getattr(blocks[0], "xsa_forward_alpha", 1.0))
                _emit_progress(
                    "[INFO] Effective XSA forward active in lm-eval: "
                    f"layers={len(enabled_layers)}/{len(blocks)} first={enabled_layers[0]} last={enabled_layers[-1]} "
                    f"ref={ref} space={space} target={target} op={op} alpha={alpha}"
                )
            else:
                _emit_progress(
                    "[WARN] Effective XSA forward is disabled in lm-eval after checkpoint load/overrides"
                )

        # GPT-2 eot token id
        try:
            import tiktoken

            self._eot_token_id = int(tiktoken.get_encoding("gpt2").eot_token)
        except Exception:
            self._eot_token_id = 50256

        _emit_progress(
            f"[MODEL] loaded ckpt={ckpt_file} device={self._device} max_length={self.max_length} batch_size={self.batch_size}"
        )
        _emit_progress(
            f"[STAGE] nanogpt_model_ready elapsed_sec={time.time() - init_start:.1f}"
        )
        if not self._supports_attention_mask:
            _emit_progress(
                "[WARN] nanoGPT model.forward does not accept attention_mask; "
                "falling back to same-length batching compatibility mode"
            )

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def tokenizer_name(self) -> str:
        return "nanogpt-gpt2-tokenizer"

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        del add_special_tokens, kwargs
        out = self.tokenizer(string, return_tensors="pt")
        return out["input_ids"][0].tolist()

    def tok_decode(self, tokens: list[int]) -> str:
        if hasattr(self.tokenizer, "_decode"):
            return self.tokenizer._decode(tokens)  # pylint: disable=protected-access
        return ""

    @torch.no_grad()
    def _model_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Use targets=input_ids to force full-sequence logits in nanoGPT forward.
        targets = input_ids.clone()
        if attention_mask is not None and self._supports_attention_mask:
            targets = targets.masked_fill(attention_mask == 0, -1)
        if self._supports_attention_mask:
            logits, _ = self.model(
                input_ids, targets=targets, attention_mask=attention_mask
            )
        else:
            logits, _ = self.model(input_ids, targets=targets)
        return logits

    def _iter_same_length_batches(self, items: list[tuple[Any, ...]], lengths: list[int]):
        if not items:
            return
        i = 0
        max_batch = max(1, int(self.batch_size))
        n_items = len(items)
        while i < n_items:
            current_len = lengths[i]
            j = i + 1
            while j < n_items and lengths[j] == current_len and (j - i) < max_batch:
                j += 1
            yield items[i:j], current_len
            i = j

    def _batch_scheduler(self, pos: int, n_reordered_requests) -> int:
        total = max(1, len(n_reordered_requests))
        sched = pos // max(1, int(total / self.batch_schedule))
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        _emit_progress(
            f"[STAGE] auto_batch_detect_start schedule={sched} pos={pos} max_batch_size={self.max_batch_size}"
        )
        detected = self._detect_batch_size(n_reordered_requests, pos)
        self.batch_sizes[sched] = detected
        _emit_progress(
            f"[STAGE] auto_batch_detect_done schedule={sched} pos={pos} batch_size={detected}"
        )
        return detected

    def _detect_batch_size(self, requests: list | None = None, pos: int = 0) -> int:
        if requests:
            req = requests[pos]
            if len(req) == 3:
                _, context_enc, continuation_enc = req
                max_length = min(
                    self.max_length,
                    max(1, len((context_enc + continuation_enc[:-1])[-self.max_length :])),
                )
            elif len(req) == 2:
                context = req[0]
                max_length = min(self.max_length, max(1, len(self.tok_encode(context))))
            else:
                max_length = self.max_length
        else:
            max_length = self.max_length

        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size: int):
            test_batch = torch.ones(
                (batch_size, max_length), device=self._device, dtype=torch.long
            )
            attention_mask = (
                torch.ones_like(test_batch, dtype=torch.long)
                if self._supports_attention_mask
                else None
            )
            for _ in range(3):
                F.log_softmax(
                    self._model_logits(test_batch, attention_mask=attention_mask), dim=-1
                )
            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as exc:
            if "No executable batch size found" in str(exc):
                batch_size = 1
            else:
                raise
        clear_torch_cache()
        return batch_size

    def _maybe_log_progress(self, tag: str, done: int, total: int) -> None:
        if done == total or done % self._progress_log_every == 0:
            _emit_progress(f"[PROGRESS] {tag} {done}/{total}")

    @torch.no_grad()
    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], list[int], list[int]]], **kwargs
    ) -> list[tuple[float, bool]]:
        disable_tqdm = kwargs.pop("disable_tqdm", False)
        del kwargs
        res: list[tuple[float, bool]] = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        n_reordered_requests = len(re_ord)
        adaptive_batch_size = None
        if self.batch_size == "auto":
            adaptive_batch_size = self._detect_batch_size(requests, 0)
            _emit_progress(
                f"[STAGE] auto_batch_detect_done mode=loglikelihood batch_size={adaptive_batch_size}"
            )
        batch_size = (
            max(1, int(adaptive_batch_size))
            if adaptive_batch_size is not None
            else max(1, int(self.batch_size))
        )
        _emit_progress(
            f"[STAGE] loglikelihood_start requests={len(requests)} batch_size={batch_size}"
        )

        pbar = tqdm(
            total=len(requests),
            disable=_tqdm_disable(disable_tqdm),
            desc="Running loglikelihood requests",
        )
        processed = 0
        total_batches = (n_reordered_requests + batch_size - 1) // batch_size if batch_size > 0 else 0
        for batch_idx, chunk in enumerate(re_ord.get_batched(n=batch_size), start=1):
            batch_start = time.time()
            live_chunk = []
            inps = []
            cont_toks_list = []
            inplens = []
            padding_len_inp = None
            for req in chunk:
                _, context_enc, continuation_enc = req
                if len(continuation_enc) == 0:
                    res.append((0.0, True))
                    processed += 1
                    pbar.update(1)
                    self._maybe_log_progress("loglikelihood", processed, len(requests))
                    continue
                inp = context_enc + continuation_enc[:-1]
                if len(inp) > self.max_length:
                    inp = inp[-self.max_length :]
                inp_tensor = torch.tensor(inp, dtype=torch.long, device=self._device)
                live_chunk.append(req)
                inps.append(inp_tensor)
                cont_toks_list.append(continuation_enc)
                inplens.append(inp_tensor.shape[0])
                padding_len_inp = (
                    max(padding_len_inp, inp_tensor.shape[0])
                    if padding_len_inp is not None
                    else inp_tensor.shape[0]
                )

            if not live_chunk:
                continue

            if batch_idx == 1 or batch_idx % self._batch_log_every == 0:
                max_inp_len = max(inplens) if inplens else 0
                avg_inp_len = (sum(inplens) / len(inplens)) if inplens else 0.0
                _emit_progress(
                    f"[STAGE] loglikelihood_batch_start batch={batch_idx}/{total_batches} "
                    f"batch_items={len(live_chunk)} max_inp_len={max_inp_len} avg_inp_len={avg_inp_len:.1f}"
                )

            if self._supports_attention_mask:
                assert padding_len_inp is not None
                batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
                attention_mask = pad_and_concat(
                    padding_len_inp,
                    [torch.ones_like(inp, dtype=torch.long) for inp in inps],
                    padding_side="right",
                )
                forward_start = time.time()
                if batch_idx == 1 or batch_idx % self._batch_log_every == 0:
                    _emit_progress(
                        f"[STAGE] loglikelihood_forward_start batch={batch_idx}/{total_batches} "
                        f"shape={tuple(batched_inps.shape)}"
                    )
                log_probs = F.log_softmax(
                    self._model_logits(batched_inps, attention_mask=attention_mask), dim=-1
                )
                if batch_idx == 1 or batch_idx % self._batch_log_every == 0:
                    _emit_progress(
                        f"[STAGE] loglikelihood_forward_done batch={batch_idx}/{total_batches} "
                        f"elapsed_sec={time.time() - forward_start:.1f}"
                    )

                result_iter = zip(
                    live_chunk, log_probs, inplens, cont_toks_list, strict=True
                )
                for (_, _, _continuation_enc), lp, inp_len, cont_toks in result_iter:
                    cont_len = len(cont_toks)
                    start = max(0, inp_len - cont_len)
                    cont_lp = lp[start : start + cont_len, :]
                    tgt = torch.tensor(
                        cont_toks[-cont_lp.size(0) :],
                        device=cont_lp.device,
                        dtype=torch.long,
                    )
                    token_lp = cont_lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    greedy = (cont_lp.argmax(dim=-1) == tgt).all().item()
                    res.append((float(token_lp.sum().item()), bool(greedy)))
                    processed += 1
                    pbar.update(1)
                    self._maybe_log_progress("loglikelihood", processed, len(requests))
            else:
                batch_items = list(zip(live_chunk, inps, cont_toks_list, inplens, strict=True))
                same_lengths = [inp.shape[0] for inp in inps]
                for same_len_batch, _ in self._iter_same_length_batches(
                    batch_items, same_lengths
                ):
                    batch_inps = [inp for _, inp, _, _ in same_len_batch]
                    stacked = torch.stack(batch_inps, dim=0)
                    forward_start = time.time()
                    if batch_idx == 1 or batch_idx % self._batch_log_every == 0:
                        _emit_progress(
                            f"[STAGE] loglikelihood_forward_start batch={batch_idx}/{total_batches} "
                            f"shape={tuple(stacked.shape)}"
                        )
                    log_probs = F.log_softmax(self._model_logits(stacked), dim=-1)
                    if batch_idx == 1 or batch_idx % self._batch_log_every == 0:
                        _emit_progress(
                            f"[STAGE] loglikelihood_forward_done batch={batch_idx}/{total_batches} "
                            f"elapsed_sec={time.time() - forward_start:.1f}"
                        )
                    for ((_, _, _continuation_enc), _, cont_toks, inp_len), lp in zip(
                        same_len_batch, log_probs, strict=True
                    ):
                        cont_len = len(cont_toks)
                        start = max(0, inp_len - cont_len)
                        cont_lp = lp[start : start + cont_len, :]
                        tgt = torch.tensor(
                            cont_toks[-cont_lp.size(0) :],
                            device=cont_lp.device,
                            dtype=torch.long,
                        )
                        token_lp = cont_lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                        greedy = (cont_lp.argmax(dim=-1) == tgt).all().item()
                        res.append((float(token_lp.sum().item()), bool(greedy)))
                        processed += 1
                        pbar.update(1)
                        self._maybe_log_progress("loglikelihood", processed, len(requests))

            if batch_idx == 1 or batch_idx % self._batch_log_every == 0:
                _emit_progress(
                    f"[STAGE] loglikelihood_batch_done batch={batch_idx}/{total_batches} "
                    f"processed={processed}/{len(requests)} elapsed_sec={time.time() - batch_start:.1f}"
                )

        pbar.close()
        _emit_progress(
            f"[STAGE] loglikelihood_done requests={len(requests)} batch_size={batch_size}"
        )
        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        loglikelihoods = []
        _emit_progress(
            f"[STAGE] loglikelihood_rolling_start requests={len(requests)}"
        )
        for (string,) in tqdm(
            [req.args for req in requests],
            disable=_tqdm_disable(disable_tqdm),
            desc="loglikelihood_rolling",
            dynamic_ncols=False,
            mininterval=0.5,
        ):
            token_list = self.tok_encode(string)
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            rolling_requests = [
                (None, context_enc, continuation_enc)
                for context_enc, continuation_enc in rolling_token_windows
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_requests,
                disable_tqdm=True,
            )
            loglikelihoods.append(sum(nll for nll, _ in string_nll))
        _emit_progress(
            f"[STAGE] loglikelihood_rolling_done requests={len(requests)}"
        )
        return loglikelihoods

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        res = []

        def _collate(req):
            toks = self.tok_encode(req[0])
            return (-len(toks), req[0])

        re_ord = Collator([req.args for req in requests], sort_fn=_collate, group_by="gen_kwargs")

        total = len(re_ord)
        processed = 0
        _emit_progress(
            f"[STAGE] generate_until_start requests={total} batch_size={self.batch_size}"
        )
        pbar = tqdm(
            total=total,
            disable=_tqdm_disable(disable_tqdm),
            desc="Running generate_until requests",
        )

        adaptive_batch_size = None
        if self.batch_size == "auto":
            adaptive_batch_size = self._detect_batch_size([req.args for req in requests], 0)
            _emit_progress(
                f"[STAGE] auto_batch_detect_done mode=generate batch_size={adaptive_batch_size}"
            )
        batch_size = (
            max(1, int(adaptive_batch_size))
            if adaptive_batch_size is not None
            else max(1, int(self.batch_size))
        )
        for chunk in re_ord.get_batched(n=batch_size):
            contexts, all_gen_kwargs = zip(*chunk, strict=True)
            gen_kwargs = dict(all_gen_kwargs[0])
            until = handle_stop_sequences(gen_kwargs.pop("until", None), eos=self.tok_decode([self.eot_token_id]))
            max_gen_toks = int(gen_kwargs.pop("max_gen_toks", self._max_gen_toks))
            temperature = float(gen_kwargs.pop("temperature", 0.0))
            top_k = gen_kwargs.pop("top_k", None)
            top_k = None if top_k is None else int(top_k)

            encoded_contexts = []
            context_lengths = []
            for context in contexts:
                ctx = self.tok_encode(context)
                if len(ctx) == 0:
                    ctx = [self.prefix_token_id]
                if len(ctx) > self.max_length:
                    ctx = ctx[-self.max_length :]
                encoded_contexts.append(ctx)
                context_lengths.append(len(ctx))

            batch_items = list(zip(contexts, encoded_contexts, context_lengths, strict=True))
            if self._supports_attention_mask:
                max_ctx_len = max(context_lengths)
                inps = []
                masks = []
                pos_ids = []
                for ctx in encoded_contexts:
                    pad_len = max_ctx_len - len(ctx)
                    inps.append(
                        torch.tensor([0] * pad_len + ctx, dtype=torch.long, device=self._device)
                    )
                    masks.append(
                        torch.tensor([0] * pad_len + [1] * len(ctx), dtype=torch.long, device=self._device)
                    )
                    pos_ids.append(
                        torch.tensor([0] * pad_len + list(range(len(ctx))), dtype=torch.long, device=self._device)
                    )
                x = torch.stack(inps, dim=0)
                attention_mask = torch.stack(masks, dim=0)
                position_ids = torch.stack(pos_ids, dim=0)
                y = self.model.generate(
                    x,
                    max_new_tokens=max_gen_toks,
                    temperature=max(temperature, 1e-6) if temperature > 0 else 1.0,
                    top_k=top_k,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                for ctx_len, row in zip(context_lengths, y, strict=True):
                    gen = self.tok_decode(row[ctx_len:].tolist())
                    for term in until:
                        if term and term in gen:
                            gen = gen.split(term)[0]
                    res.append(gen)
                    processed += 1
                    pbar.update(1)
                    self._maybe_log_progress("generate_until", processed, total)
            else:
                for same_len_batch, _ in self._iter_same_length_batches(batch_items, context_lengths):
                    batch_contexts = [ctx for _, ctx, _ in same_len_batch]
                    x = torch.tensor(batch_contexts, dtype=torch.long, device=self._device)
                    y = self.model.generate(
                        x,
                        max_new_tokens=max_gen_toks,
                        temperature=max(temperature, 1e-6) if temperature > 0 else 1.0,
                        top_k=top_k,
                    )
                    for (_, _, ctx_len), row in zip(same_len_batch, y, strict=True):
                        gen = self.tok_decode(row[ctx_len:].tolist())
                        for term in until:
                            if term and term in gen:
                                gen = gen.split(term)[0]
                        res.append(gen)
                        processed += 1
                        pbar.update(1)
                        self._maybe_log_progress("generate_until", processed, total)

        pbar.close()
        _emit_progress(
            f"[STAGE] generate_until_done requests={total} batch_size={self.batch_size}"
        )
        return re_ord.get_original(res)
