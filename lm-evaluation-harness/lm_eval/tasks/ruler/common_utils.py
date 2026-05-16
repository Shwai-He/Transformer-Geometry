import hashlib
import json
import logging
import os
import re
import time
from functools import cache
from typing import TYPE_CHECKING, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]
RULER_CACHE_VERSION = 1


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def get_ruler_cache_dir() -> str:
    cache_dir = os.environ.get("RULER_CACHE_DIR")
    if cache_dir:
        return os.path.abspath(os.path.expanduser(cache_dir))
    return "/mnt/hdfs/shwai.he/DepthBoost/cache/ruler"


def _stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@cache
def get_tokenizer_fingerprint(
    tokenizer_name=None, pretrained=None, **kwargs
) -> str:
    tokenizer = get_tokenizer(tokenizer=tokenizer_name, pretrained=pretrained, **kwargs)
    digest = hashlib.sha256()
    digest.update(
        f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}\n".encode(
            "utf-8"
        )
    )

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend_tokenizer, "to_str", None)
    if callable(backend_to_str):
        digest.update(backend_to_str().encode("utf-8"))
    else:
        payload = {
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
            "all_special_tokens": list(getattr(tokenizer, "all_special_tokens", [])),
            "all_special_ids": list(getattr(tokenizer, "all_special_ids", [])),
            "added_vocab": dict(sorted(tokenizer.get_added_vocab().items())),
            "vocab": dict(sorted(tokenizer.get_vocab().items())),
        }
        digest.update(_stable_json(payload).encode("utf-8"))
    return digest.hexdigest()[:16]


def _get_cache_path(task_name: str, tokenizer_fingerprint: str, seq_length: int) -> str:
    safe_task_name = str(task_name).replace("/", "_")
    return os.path.join(
        get_ruler_cache_dir(),
        f"v{RULER_CACHE_VERSION}",
        safe_task_name,
        f"tok_{tokenizer_fingerprint}",
        f"seq_{int(seq_length)}.jsonl",
    )


def _read_cached_records(cache_path: str) -> list[dict]:
    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_cached_records(cache_path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(_stable_json(row))
            f.write("\n")
    os.replace(tmp_path, cache_path)


def _acquire_cache_lock(lock_dir: str, timeout_seconds: float = 600.0) -> bool:
    start = time.time()
    while True:
        try:
            os.makedirs(lock_dir)
            return True
        except FileExistsError:
            if time.time() - start >= timeout_seconds:
                return False
            time.sleep(0.2)


def _release_cache_lock(lock_dir: str) -> None:
    try:
        os.rmdir(lock_dir)
    except FileNotFoundError:
        pass


def load_or_generate_cached_samples(
    *,
    task_name: str,
    seq_length: int,
    tokenizer_name=None,
    pretrained=None,
    generator_fn,
    **kwargs,
) -> list[dict]:
    tokenizer_fingerprint = get_tokenizer_fingerprint(
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        **kwargs,
    )
    cache_path = _get_cache_path(task_name, tokenizer_fingerprint, seq_length)
    if os.path.isfile(cache_path):
        eval_logger.info(
            "RULER cache hit: task=%s seq=%s fingerprint=%s path=%s",
            task_name,
            seq_length,
            tokenizer_fingerprint,
            cache_path,
        )
        return _read_cached_records(cache_path)

    lock_dir = f"{cache_path}.lock"
    if not _acquire_cache_lock(lock_dir):
        raise TimeoutError(f"Timed out waiting for RULER cache lock: {lock_dir}")
    try:
        if os.path.isfile(cache_path):
            eval_logger.info(
                "RULER cache filled by peer: task=%s seq=%s fingerprint=%s path=%s",
                task_name,
                seq_length,
                tokenizer_fingerprint,
                cache_path,
            )
            return _read_cached_records(cache_path)

        eval_logger.info(
            "RULER cache miss: task=%s seq=%s fingerprint=%s; generating.",
            task_name,
            seq_length,
            tokenizer_fingerprint,
        )
        records = generator_fn()
        _write_cached_records(cache_path, records)
        return records
    finally:
        _release_cache_lock(lock_dir)


def build_cached_dataset(
    *,
    task_name: str,
    seq_lengths: list[int],
    tokenizer_name=None,
    pretrained=None,
    generator_fn,
    **kwargs,
) -> dict[str, datasets.Dataset]:
    rows = []
    for seq in seq_lengths:
        rows.extend(
            load_or_generate_cached_samples(
                task_name=task_name,
                seq_length=seq,
                tokenizer_name=tokenizer_name,
                pretrained=pretrained,
                generator_fn=lambda seq=seq: generator_fn(seq),
                **kwargs,
            )
        )
    return {"test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
