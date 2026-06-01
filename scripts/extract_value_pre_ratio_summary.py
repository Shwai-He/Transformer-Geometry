#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple


DEFAULT_METRICS = (
    "value_pre_para_perp_ratio",
    "value_pre_para_ratio",
    "value_pre_perp_ratio",
    "value_pre_self_alignment",
    "value_pre_norm",
    "value_pre_self_value_norm",
    "value_pre_projected_para_norm",
    "value_pre_projected_para_over_z",
    "value_pre_projected_para_alignment",
    "value_pre_scale2_projected_delta_over_z",
    "value_pre_scale2_post_rms_delta_over_normed_z",
    "value_pre_scale2_rms_damping_ratio",
    "value_pre_scale2_cos_z_edit",
    "value_pre_scale2_cos_normed_z_edit",
    "value_pre_scale5_projected_delta_over_z",
    "value_pre_scale5_post_rms_delta_over_normed_z",
    "value_pre_scale5_rms_damping_ratio",
    "value_pre_scale5_cos_z_edit",
    "value_pre_scale5_cos_normed_z_edit",
    "value_pre_scale10_projected_delta_over_z",
    "value_pre_scale10_post_rms_delta_over_normed_z",
    "value_pre_scale10_rms_damping_ratio",
    "value_pre_scale10_cos_z_edit",
    "value_pre_scale10_cos_normed_z_edit",
)


def _iter_json_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                parts = set(candidate.parts)
                if "summary" in parts or "extracted_value_pre" in parts:
                    continue
                if candidate.name == "manifest.json":
                    continue
                files.append(candidate)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input not found: {path}")
    return files


def _runs_from_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return data.get("runs") or [data]


def _steps_for_run(run: Dict[str, Any], phase: str) -> List[Dict[str, Any]]:
    timeline = run.get("timeline") or []
    if timeline:
        if phase == "all":
            return timeline
        return [step for step in timeline if step.get("phase") == phase]

    if run.get("sublayer_metrics") and phase in {"all", "prefill"}:
        return [{"phase": "prefill", "step": -1, "sublayer_metrics": run.get("sublayer_metrics")}]

    if phase == "prefill":
        return run.get("prefill_steps") or []
    if phase == "decode":
        return run.get("steps") or []
    return (run.get("prefill_steps") or []) + (run.get("steps") or [])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _stats(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def _write_stats_csv(path: Path, rows: List[Dict[str, Any]], group_fields: List[str]) -> None:
    stat_fields = ["n", "mean", "median", "p10", "p90", "min", "max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields + stat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_file(
    path: Path,
    metrics: Tuple[str, ...],
    phase: str,
) -> Tuple[
    Dict[Tuple[str, str], List[float]],
    Dict[Tuple[str, int, str, str], List[float]],
    Dict[Tuple[int, int, str, str], List[float]],
    Dict[str, Any],
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = _runs_from_data(data)

    overall: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    by_layer: DefaultDict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    by_prompt: DefaultDict[Tuple[int, int, str, str], List[float]] = defaultdict(list)
    attn_keys = set()
    debug: Dict[str, Any] = {
        "runs": len(runs),
        "steps": 0,
        "sublayer_items": 0,
        "matched_values": 0,
        "attn_update_keys_sample": [],
    }

    for prompt_idx, run in enumerate(runs):
        prompt_token_count = int(run.get("prompt_token_count") or 0)
        for step in _steps_for_run(run, phase):
            debug["steps"] += 1
            step_phase = str(step.get("phase") or phase)
            for item in step.get("sublayer_metrics") or []:
                debug["sublayer_items"] += 1
                layer = item.get("layer")
                if layer is None:
                    continue
                attn_update = item.get("attn_update") or {}
                attn_keys.update(str(k) for k in attn_update.keys())
                for metric in metrics:
                    value = _safe_float(attn_update.get(metric))
                    if value is None:
                        value = _safe_float(item.get(metric))
                    if value is None:
                        continue
                    debug["matched_values"] += 1
                    overall[(step_phase, metric)].append(value)
                    by_layer[(step_phase, int(layer), "attn_update", metric)].append(value)
                    by_prompt[(prompt_idx, prompt_token_count, step_phase, metric)].append(value)

    debug["attn_update_keys_sample"] = sorted(attn_keys)[:80]
    return dict(overall), dict(by_layer), dict(by_prompt), debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract compact CSV summaries from run_generation_probe value_pre JSON outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON files or directories. Directories are searched recursively for *.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <first-input-parent>/extracted_value_pre",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "prefill", "decode"],
        default="all",
        help="Which phase to summarize. Default: all.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="attn_update metric keys to extract.",
    )
    args = parser.parse_args()

    input_files = _iter_json_files(args.inputs)
    if not input_files:
        raise RuntimeError("No JSON inputs found.")

    output_dir = Path(args.output_dir) if args.output_dir else input_files[0].parent / "extracted_value_pre"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []
    debug_by_source: Dict[str, Any] = {}

    metrics = tuple(args.metrics)
    for input_file in input_files:
        print(f"[READ] {input_file}")
        overall, by_layer, by_prompt, debug = extract_file(input_file, metrics=metrics, phase=args.phase)
        source = input_file.stem
        debug_by_source[source] = debug
        print(
            "[INFO] "
            f"runs={debug['runs']} steps={debug['steps']} "
            f"sublayer_items={debug['sublayer_items']} matched_values={debug['matched_values']}"
        )
        if debug["matched_values"] == 0:
            print(
                "[WARN] No requested value_pre metrics found. "
                "This usually means the JSON was produced without the updated analyzer.py. "
                f"Sample attn_update keys: {debug['attn_update_keys_sample']}"
            )

        for (step_phase, metric), values in sorted(overall.items()):
            overall_rows.append(
                {
                    "source": source,
                    "phase": step_phase,
                    "metric": metric,
                    **_stats(values),
                }
            )

        for (step_phase, layer, part, metric), values in sorted(by_layer.items()):
            layer_rows.append(
                {
                    "source": source,
                    "phase": step_phase,
                    "layer": layer,
                    "part": part,
                    "metric": metric,
                    **_stats(values),
                }
            )

        for (prompt_idx, prompt_token_count, step_phase, metric), values in sorted(by_prompt.items()):
            prompt_rows.append(
                {
                    "source": source,
                    "prompt_idx": prompt_idx,
                    "prompt_token_count": prompt_token_count,
                    "phase": step_phase,
                    "metric": metric,
                    **_stats(values),
                }
            )

    overall_path = output_dir / "value_pre_overall_summary.csv"
    layer_path = output_dir / "value_pre_layer_summary.csv"
    prompt_path = output_dir / "value_pre_prompt_summary.csv"

    _write_stats_csv(overall_path, overall_rows, ["source", "phase", "metric"])
    _write_stats_csv(layer_path, layer_rows, ["source", "phase", "layer", "part", "metric"])
    _write_stats_csv(prompt_path, prompt_rows, ["source", "prompt_idx", "prompt_token_count", "phase", "metric"])

    manifest = {
        "inputs": [str(path) for path in input_files],
        "phase": args.phase,
        "metrics": list(metrics),
        "debug": debug_by_source,
        "outputs": {
            "overall": str(overall_path),
            "layer": str(layer_path),
            "prompt": str(prompt_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[SAVE] {overall_path}")
    print(f"[SAVE] {layer_path}")
    print(f"[SAVE] {prompt_path}")
    print(f"[SAVE] {manifest_path}")


if __name__ == "__main__":
    main()
