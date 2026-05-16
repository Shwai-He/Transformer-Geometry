#!/usr/bin/env python3
"""Collect nanoGPT lm-eval results into tidy CSV summaries.

Supports two common nanoGPT result layouts:
1. Per-task JSON files under each `lm_eval_results/` directory
2. A single `*-all_tasks.json` file containing multiple tasks

The collector recursively scans SEARCH_ROOTS for `lm_eval_results/*.json`.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


TASK_METRICS = {
    "arc_easy": "acc,none",
    "arc_challenge": "acc_norm,none",
    "boolq": "acc,none",
    "hellaswag": "acc_norm,none",
    "lambada_openai": "perplexity,none",
    "openbookqa": "acc_norm,none",
    "piqa": "acc_norm,none",
    "rte": "acc,none",
    "winogrande": "acc,none",
    "mmlu": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "humaneval": "pass@1,create_test",
    "mbpp": "pass@1,sanitized",
    "drop": "f1,none",
    "nq_open": "exact_match,remove_whitespace",
    "bbh_cot_zeroshot": "acc_norm,none",
}

HIGHER_IS_BETTER_DEFAULT = {
    "perplexity,none": False,
}

_CKPT_PROGRESS_CACHE: dict[str, tuple[int | None, float | None]] = {}
_CKPT_METADATA_CACHE: dict[str, dict[str, Any] | None] = {}


def default_search_roots() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    harness_dir = script_dir.parent
    repo_root = harness_dir.parent
    return [
        repo_root / "nanoGPT" / "out",
        harness_dir / "nanoGPT" / "out",
        harness_dir / "outputs",
    ]


def parse_search_roots() -> list[Path]:
    raw = os.environ.get("SEARCH_ROOTS", "").strip()
    if raw:
        return [Path(p.strip()).expanduser() for p in raw.split(":") if p.strip()]
    return default_search_roots()


def parse_result_dirs() -> list[Path]:
    raw = os.environ.get("RESULT_DIRS", "").strip()
    if raw:
        return [Path(p.strip()).expanduser() for p in raw.split(":") if p.strip()]
    return []


def prefer_ckpt_best() -> bool:
    raw = os.environ.get("PREFER_CKPT_BEST", "false").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def fallback_to_checkpoint_payload() -> bool:
    raw = os.environ.get("FALLBACK_TO_CKPT_PAYLOAD", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] invalid integer for {name}: {raw}")
        return None
    return value if value >= 0 else None


def iter_result_dirs(search_root: Path):
    """Yield lm_eval_results directories under search_root robustly.

    `Path.rglob()` is convenient but can be painfully slow and brittle on large
    remote-mounted directory trees. Here we use `os.walk` with error-tolerant
    traversal and only yield directories actually named `lm_eval_results`.
    """

    def _onerror(exc: OSError) -> None:
        print(f"[WARN] walk error under {search_root}: {exc}")

    max_depth = _parse_optional_int_env("SEARCH_MAX_DEPTH")
    root_depth = len(search_root.parts)

    try:
        for dirpath, dirnames, _ in os.walk(
            search_root, topdown=True, onerror=_onerror, followlinks=False
        ):
            current_depth = len(Path(dirpath).parts) - root_depth
            if max_depth is not None and current_depth >= max_depth:
                dirnames[:] = []

            if "lm_eval_results" in dirnames:
                result_dir = Path(dirpath) / "lm_eval_results"
                if result_dir.is_dir():
                    yield result_dir
                # No need to descend into the result directory during the main walk.
                dirnames.remove("lm_eval_results")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[WARN] failed to traverse {search_root}: {exc}")


def discover_result_dirs(search_roots: list[Path], direct_result_dirs: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    max_result_dirs = _parse_optional_int_env("MAX_RESULT_DIRS")
    discovery_timeout_sec = _parse_optional_int_env("DISCOVERY_TIMEOUT_SEC")
    deadline = (
        time.monotonic() + discovery_timeout_sec
        if discovery_timeout_sec is not None and discovery_timeout_sec > 0
        else None
    )

    def _should_stop() -> bool:
        if max_result_dirs is not None and len(discovered) >= max_result_dirs:
            print(f"[INFO] reached MAX_RESULT_DIRS={max_result_dirs}, stop discovery early")
            return True
        if deadline is not None and time.monotonic() >= deadline:
            print(f"[INFO] reached DISCOVERY_TIMEOUT_SEC={discovery_timeout_sec}, stop discovery early")
            return True
        return False

    if direct_result_dirs:
        print(f"[INFO] scanning direct result dirs: {len(direct_result_dirs)}")
        for result_dir in direct_result_dirs:
            print(f"[SCAN] result_dir={result_dir}")
            discovered.append(result_dir)
            if _should_stop():
                break

    for search_root in search_roots:
        if _should_stop():
            break
        print(f"[INFO] scanning search root: {search_root}")
        count_before = len(discovered)
        for result_dir in iter_result_dirs(search_root):
            print(f"[SCAN] found_result_dir={result_dir}")
            discovered.append(result_dir)
            if _should_stop():
                break
        print(
            f"[INFO] search root done: {search_root} newly_found={len(discovered) - count_before}"
        )
        if _should_stop():
            break

    unique: list[Path] = []
    seen: set[str] = set()
    for path in discovered:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def extract_run_tag(file_path: Path) -> str:
    stem = file_path.stem
    if stem.endswith("-all_tasks"):
        return stem[: -len("-all_tasks")]
    return stem.rsplit("-", 1)[0] if "-" in stem else stem


def _file_suffix_key(file_path: Path) -> str:
    stem = file_path.stem
    if stem.endswith("-all_tasks"):
        return "all_tasks"
    return stem.split("-", 1)[1] if "-" in stem else stem


def filter_preferred_json_paths(json_paths: list[Path]) -> list[Path]:
    if not prefer_ckpt_best():
        return json_paths

    suffix_to_best: dict[str, Path] = {}
    suffix_to_regular: dict[str, Path] = {}
    passthrough: list[Path] = []

    for path in json_paths:
        stem = path.stem
        suffix_key = _file_suffix_key(path)
        if stem.startswith("ckpt_best.pt-"):
            suffix_to_best[suffix_key] = path
        elif stem.startswith("ckpt.pt-"):
            suffix_to_regular[suffix_key] = path
        else:
            passthrough.append(path)

    selected: list[Path] = []
    chosen = set()

    for suffix_key, path in suffix_to_best.items():
        selected.append(path)
        chosen.add(path)
        if suffix_key in suffix_to_regular:
            print(
                f"[INFO] prefer ckpt_best over ckpt.pt for suffix={suffix_key}: "
                f"keep={path.name} skip={suffix_to_regular[suffix_key].name}"
            )

    for suffix_key, path in suffix_to_regular.items():
        if suffix_key in suffix_to_best:
            continue
        selected.append(path)
        chosen.add(path)

    for path in passthrough:
        if path not in chosen:
            selected.append(path)

    return sorted(selected)


def _load_checkpoint_metadata(ckpt_path: str | None) -> dict[str, Any] | None:
    if not isinstance(ckpt_path, str) or not ckpt_path:
        return None

    cache_key = str(Path(ckpt_path).expanduser())
    if cache_key in _CKPT_METADATA_CACHE:
        return _CKPT_METADATA_CACHE[cache_key]

    sidecar_path = Path(cache_key + ".json")
    if sidecar_path.exists():
        try:
            payload = json.loads(sidecar_path.read_text())
            if isinstance(payload, dict):
                _CKPT_METADATA_CACHE[cache_key] = payload
                return payload
        except Exception as exc:
            print(f"[WARN] Failed to read checkpoint sidecar {sidecar_path}: {exc}")

    if not fallback_to_checkpoint_payload():
        _CKPT_METADATA_CACHE[cache_key] = None
        return None

    ckpt_file = Path(cache_key)
    if not ckpt_file.exists():
        _CKPT_METADATA_CACHE[cache_key] = None
        return None

    try:
        import torch

        payload = torch.load(ckpt_file, map_location="cpu")
        if isinstance(payload, dict):
            _CKPT_METADATA_CACHE[cache_key] = payload
            return payload
    except Exception as exc:
        print(f"[WARN] Failed to read checkpoint payload {ckpt_file}: {exc}")

    _CKPT_METADATA_CACHE[cache_key] = None
    return None


def extract_ckpt_label(result_dir: Path, data: dict[str, Any], run_tag: str) -> str:
    ckpt_path = resolve_ckpt_path(result_dir, data, run_tag)
    metadata = _load_checkpoint_metadata(ckpt_path)
    if isinstance(metadata, dict):
        run_name = metadata.get("run_name")
        if isinstance(run_name, str) and run_name.strip():
            return run_name.strip()
        out_dir = metadata.get("out_dir")
        if isinstance(out_dir, str) and out_dir.strip():
            return Path(out_dir).name
        nested_cfg = metadata.get("config")
        if isinstance(nested_cfg, dict):
            nested_out_dir = nested_cfg.get("out_dir")
            if isinstance(nested_out_dir, str) and nested_out_dir.strip():
                return Path(nested_out_dir).name
    return result_dir.parent.name or run_tag


def extract_model_size_label(name: str) -> str:
    match = re.search(r"(\d+p\d+)(?:b)?(?:[^0-9]|$)", name)
    if match:
        return match.group(1)
    return name


def extract_model_name(result_dir: Path, data: dict[str, Any], run_tag: str) -> str:
    ckpt_path = resolve_ckpt_path(result_dir, data, run_tag)
    if isinstance(ckpt_path, str) and ckpt_path:
        ckpt_parent = Path(ckpt_path).expanduser().parent
        if ckpt_parent.name:
            return extract_model_size_label(ckpt_parent.name)
    return extract_model_size_label(result_dir.parent.name or run_tag)


def extract_checkpoint_name(result_dir: Path, data: dict[str, Any], run_tag: str) -> str:
    ckpt_path = resolve_ckpt_path(result_dir, data, run_tag)
    metadata = _load_checkpoint_metadata(ckpt_path)
    if isinstance(metadata, dict):
        checkpoint_name = metadata.get("checkpoint_name")
        if isinstance(checkpoint_name, str) and checkpoint_name.strip():
            return checkpoint_name.strip()
    if isinstance(ckpt_path, str) and ckpt_path:
        return Path(ckpt_path).name
    return run_tag


def resolve_ckpt_path(result_dir: Path, data: dict[str, Any], run_tag: str) -> str | None:
    cfg = data.get("config", {}) if isinstance(data, dict) else {}
    ckpt_path = cfg.get("ckpt_path") if isinstance(cfg, dict) else None
    if isinstance(ckpt_path, str) and ckpt_path:
        return ckpt_path

    inferred = result_dir.parent / run_tag
    if inferred.is_file():
        return str(inferred)
    if run_tag in {"ckpt.pt", "ckpt_best.pt"}:
        fallback = result_dir.parent / run_tag
        if fallback.is_file():
            return str(fallback)
    return None


def extract_checkpoint_progress(
    result_dir: Path, data: dict[str, Any], run_tag: str
) -> tuple[int | None, float | None]:
    ckpt_path = resolve_ckpt_path(result_dir, data, run_tag)
    if not isinstance(ckpt_path, str) or not ckpt_path:
        return None, None

    cache_key = str(Path(ckpt_path).expanduser())
    if cache_key in _CKPT_PROGRESS_CACHE:
        return _CKPT_PROGRESS_CACHE[cache_key]

    payload = _load_checkpoint_metadata(ckpt_path)
    if not isinstance(payload, dict):
        _CKPT_PROGRESS_CACHE[cache_key] = (None, None)
        return None, None

    iter_num = payload.get("iter_num")
    tokens_seen = payload.get("tokens_seen")
    save_step = int(iter_num) if isinstance(iter_num, (int, float)) else None
    tokens_seen_b = (
        round(float(tokens_seen) / 1e9, 3)
        if isinstance(tokens_seen, (int, float))
        else None
    )
    _CKPT_PROGRESS_CACHE[cache_key] = (save_step, tokens_seen_b)
    return save_step, tokens_seen_b


def metric_value(task: str, metrics: dict[str, Any], hib: dict[str, Any]) -> tuple[float | None, str | None]:
    preferred = TASK_METRICS.get(task)
    if preferred and preferred in metrics and isinstance(metrics[preferred], (int, float)):
        return float(metrics[preferred]), preferred

    for key, val in metrics.items():
        if "_stderr" in key or not isinstance(val, (int, float)):
            continue
        hib_flag = hib.get(key)
        if hib_flag is None:
            hib_flag = HIGHER_IS_BETTER_DEFAULT.get(key, True)
        return float(val), key
    return None, None


def normalize_metric(task: str, metric_key: str | None, value: float, hib: dict[str, Any]) -> float:
    hib_flag = None if metric_key is None else hib.get(metric_key)
    if hib_flag is None and metric_key is not None:
        hib_flag = HIGHER_IS_BETTER_DEFAULT.get(metric_key, True)
    if hib_flag is False:
        return round(value, 4)
    return round(value * 100.0, 2)


def collect_rows(search_roots: list[Path]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    direct_result_dirs = parse_result_dirs()

    if direct_result_dirs:
        existing_direct_dirs = [p for p in direct_result_dirs if p.exists()]
        for p in direct_result_dirs:
            if not p.exists():
                print(f"[INFO] Result dir missing, skip: {p}")
        result_dirs_to_scan = existing_direct_dirs
    else:
        result_dirs_to_scan = []

    existing_roots = [root for root in search_roots if root.exists()]
    for root in search_roots:
        if not root.exists():
            print(f"[INFO] Search root missing, skip: {root}")
    if not existing_roots and not result_dirs_to_scan:
        raise SystemExit(
            "No search roots or direct result dirs exist. Checked roots: "
            + ", ".join(str(root) for root in search_roots)
        )

    iterable_result_dirs = discover_result_dirs(existing_roots, result_dirs_to_scan)
    if not iterable_result_dirs:
        print("[WARN] No lm_eval_results directories found after scanning.")
        return []

    print(f"[INFO] total result dirs to read: {len(iterable_result_dirs)}")
    for result_dir in tqdm(iterable_result_dirs, desc="Reading result dirs", unit="dir"):
        try:
            json_paths = sorted(result_dir.glob("*.json"))
        except Exception as exc:
            print(f"[WARN] Failed to list json files in {result_dir}: {exc}")
            continue

        if not json_paths:
            print(f"[WARN] empty result dir: {result_dir}")
            continue

        json_paths = filter_preferred_json_paths(json_paths)

        print(f"[INFO] reading result dir: {result_dir} json_files={len(json_paths)}")
        for jpath in tqdm(
            json_paths,
            desc=f"JSONs in {result_dir.name}",
            unit="json",
            leave=False,
        ):
            try:
                data = json.loads(jpath.read_text())
            except Exception as exc:
                print(f"[ERROR] Failed to read {jpath}: {exc}")
                continue

            results = data.get("results", {}) if isinstance(data, dict) else {}
            higher_is_better = data.get("higher_is_better", {}) if isinstance(data, dict) else {}
            if not isinstance(results, dict) or not results:
                print(f"[WARN] Empty or invalid results in {jpath}")
                continue

            run_tag = extract_run_tag(jpath)
            group_key = (str(result_dir.resolve()), run_tag)
            row = rows_by_key.setdefault(
                group_key,
                {
                    "model": extract_model_name(result_dir, data, run_tag),
                    "run_tag": run_tag,
                    "ckpt_label": extract_ckpt_label(result_dir, data, run_tag),
                    "checkpoint_name": extract_checkpoint_name(result_dir, data, run_tag),
                },
            )
            save_step, tokens_seen_b = extract_checkpoint_progress(result_dir, data, run_tag)
            if save_step is not None:
                row["save_step"] = save_step
            if tokens_seen_b is not None:
                row["tokens_seen_b"] = tokens_seen_b

            for task_name, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                hib_for_task = higher_is_better.get(task_name, {})
                if not isinstance(hib_for_task, dict):
                    hib_for_task = {}
                value, metric_key = metric_value(task_name, metrics, hib_for_task)
                if value is None:
                    continue
                row[task_name] = normalize_metric(task_name, metric_key, value, hib_for_task)
                row[f"{task_name}__metric"] = metric_key

    rows: list[dict[str, Any]] = []
    for row in rows_by_key.values():
        rows.append(row)
    return rows


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    harness_dir = script_dir.parent
    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            harness_dir / "outputs" / "nanogpt_lm_eval_summary",
        )
    ).expanduser()

    rows = collect_rows(parse_search_roots())
    if not rows:
        raise SystemExit("No nanoGPT lm-eval results found.")

    df = pd.DataFrame(rows)
    task_cols = [task for task in TASK_METRICS if task in df.columns]
    discovered_task_cols = sorted(
        col for col in df.columns
        if col not in {
            "model", "run_tag", "ckpt_label", "checkpoint_name", "save_step", "tokens_seen_b"
        }
        and not col.endswith("__metric")
        and col not in task_cols
    )
    task_cols = task_cols + discovered_task_cols

    score_cols = []
    for col in task_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        metric_col = f"{col}__metric"
        if metric_col in df.columns:
            metric_names = {
                name for name in df[metric_col].dropna().astype(str).tolist() if name
            }
            if metric_names and all(
                HIGHER_IS_BETTER_DEFAULT.get(name, True) is False for name in metric_names
            ):
                continue
        score_cols.append(col)
    if score_cols:
        df["avg"] = df[score_cols].mean(axis=1).round(2)
    else:
        df["avg"] = None

    def _is_early_or_incomplete(row: pd.Series) -> int:
        step = row.get("save_step")
        tokens = row.get("tokens_seen_b")
        try:
            step_val = float(step) if pd.notna(step) else None
        except Exception:
            step_val = None
        try:
            tok_val = float(tokens) if pd.notna(tokens) else None
        except Exception:
            tok_val = None

        # Push obviously early / incomplete checkpoints to the bottom:
        # - missing both progress indicators
        # - step <= 0
        # - tokens <= 0
        if step_val is None and tok_val is None:
            return 1
        if step_val is not None and step_val <= 0:
            return 1
        if tok_val is not None and tok_val <= 0:
            return 1
        return 0

    df["_early_or_incomplete"] = df.apply(_is_early_or_incomplete, axis=1)

    front_cols = ["model", "ckpt_label"]
    if "checkpoint_name" in df.columns:
        front_cols.append("checkpoint_name")
    if "save_step" in df.columns:
        front_cols.append("save_step")
    if "tokens_seen_b" in df.columns:
        front_cols.append("tokens_seen_b")
    include_run_tag = "run_tag" in df.columns and df["run_tag"].nunique() > df["model"].nunique()
    if include_run_tag:
        front_cols.append("run_tag")
    ordered_cols = front_cols + task_cols + ["avg"]
    sort_cols = ["model", "_early_or_incomplete"]
    ascending = [True, True]
    if "tokens_seen_b" in df.columns:
        sort_cols.append("tokens_seen_b")
        ascending.append(False)
    if "save_step" in df.columns:
        sort_cols.append("save_step")
        ascending.append(False)
    sort_cols.extend(["ckpt_label"])
    ascending.extend([True])
    if "checkpoint_name" in df.columns:
        sort_cols.append("checkpoint_name")
        ascending.append(True)
    if include_run_tag:
        sort_cols.append("run_tag")
        ascending.append(True)

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    df = df[ordered_cols]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nanogpt_lm_eval_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"[OK] {len(df)} rows saved -> {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
