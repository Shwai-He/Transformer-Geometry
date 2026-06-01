from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# ==============================
# Edit these values directly.
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODE = "summary_tsv"  # "summary_tsv" or "sweep_dir"
SUMMARY_TSV = PROJECT_ROOT / "drawing" / "para_ablation" / "data" / "para_ppl_summary.tsv"
INPUT_DIR = Path("representation-analysis/outputs/para_ablation")
OUTPUT_DIR = None  # Set to a Path(...) to override the default output directory.
MAX_EXAMPLE_SETTINGS = 5
MAX_EXAMPLE_PROMPTS = 4


COMPARE_FILE_RE = re.compile(
    r"compare_para_scale_(?P<para>[-0-9p]+?)(?:_perp_scale_(?P<perp>[-0-9p]+))?\.jsonl$"
)
GENERATE_FILE_RE = re.compile(
    r"generate_para_scale_(?P<para>[-0-9.]+?)(?:_perp_scale_(?P<perp>[-0-9.]+))?\.jsonl$"
)


def decode_scale_token(token: Optional[str]) -> Optional[float]:
    if token is None:
        return None
    return float(token.replace("p", "."))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_compare_files(input_dir: Path) -> List[Path]:
    compare_dir = input_dir / "compare"
    if compare_dir.is_dir():
        return sorted(compare_dir.glob("*.jsonl"))
    return sorted(input_dir.glob("compare*.jsonl"))


def discover_generation_files(input_dir: Path) -> List[Path]:
    generate_dir = input_dir / "generate"
    candidates = list(generate_dir.glob("*.jsonl")) if generate_dir.is_dir() else list(input_dir.glob("generate*.jsonl"))
    return sorted(
        path
        for path in candidates
        if not path.name.endswith(".pretty.jsonl")
        and not path.name.endswith(".stats.json")
        and not path.name.endswith("generation_paths.jsonl")
    )


def parse_setting_from_compare_path(path: Path) -> Dict[str, Optional[float]]:
    match = COMPARE_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unrecognized compare filename format: {path}")
    return {
        "para_scale": decode_scale_token(match.group("para")),
        "perp_scale": decode_scale_token(match.group("perp")),
    }


def parse_setting_from_generation_path(path: Path) -> Dict[str, Optional[float]]:
    match = GENERATE_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unrecognized generation filename format: {path}")
    return {
        "para_scale": decode_scale_token(match.group("para")),
        "perp_scale": decode_scale_token(match.group("perp")),
    }


def choose_sweep_axis(rows: List[Dict]) -> str:
    para_values = {row.get("para_scale") for row in rows if row.get("para_scale") is not None}
    perp_values = {row.get("perp_scale") for row in rows if row.get("perp_scale") is not None}
    if len(para_values) > 1 and len(perp_values) <= 1:
        return "para_scale"
    if len(perp_values) > 1 and len(para_values) <= 1:
        return "perp_scale"
    if len(perp_values) > len(para_values):
        return "perp_scale"
    return "para_scale"


def load_compare_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_compare_file(path: Path) -> Dict:
    setting = parse_setting_from_compare_path(path)
    rows = load_compare_rows(path)
    if not rows:
        raise ValueError(f"Empty compare file: {path}")
    summary = dict(setting)
    summary["path"] = str(path)
    summary["n_prompts"] = len(rows)
    for key in [
        "baseline_ppl",
        "xsa_ppl",
        "delta_ppl",
        "residual_attn_ppl",
        "delta_ppl_residual_attn",
        "residual_mlp_ppl",
        "delta_ppl_residual_mlp",
        "residual_both_ppl",
        "delta_ppl_residual_both",
        "baseline_loss",
        "xsa_loss",
        "delta_loss",
        "residual_attn_loss",
        "delta_loss_residual_attn",
        "residual_mlp_loss",
        "delta_loss_residual_mlp",
        "residual_both_loss",
        "delta_loss_residual_both",
    ]:
        summary[f"avg_{key}"] = mean(float(row[key]) for row in rows)
    return summary


def load_generation_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _first_float(row: Dict[str, str], *keys: str) -> float:
    for key in keys:
        if key in row and row[key] not in {"", None}:
            return float(row[key])
    raise KeyError(keys[0])


def summarize_generation_file(path: Path) -> List[Dict]:
    setting = parse_setting_from_generation_path(path)
    rows = load_generation_rows(path)
    if not rows:
        return []

    by_target_chars: Dict[str, List[int]] = defaultdict(list)
    by_target_words: Dict[str, List[int]] = defaultdict(list)
    change_flags: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        generations = row.get("generations", {})
        baseline = generations.get("none", "")
        for target, text in generations.items():
            by_target_chars[target].append(len(text))
            by_target_words[target].append(len(text.split()))
            if target != "none":
                change_flags[target].append(0 if text == baseline else 1)

    summaries = []
    for target in sorted(by_target_chars):
        rec = dict(setting)
        rec["path"] = str(path)
        rec["target"] = target
        rec["n_prompts"] = len(by_target_chars[target])
        rec["avg_chars"] = mean(by_target_chars[target])
        rec["avg_words"] = mean(by_target_words[target])
        rec["change_rate_vs_none"] = mean(change_flags[target]) if target != "none" and change_flags[target] else 0.0
        summaries.append(rec)
    return summaries


def load_summary_tsv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="	"))
    parsed = []
    for row in rows:
        rec = {
            "para_scale": _first_float(row, "para_scale", "setting_value"),
            "n_prompts": int(float(row.get("n_prompts", row.get("n_rows", 0)))),
            "avg_baseline_ppl": _first_float(row, "avg_baseline_ppl", "mean_per_text_baseline_ppl", "weighted_baseline_ppl"),
            "avg_xsa_ppl": _first_float(row, "avg_xsa_ppl", "mean_per_text_xsa_ppl", "weighted_xsa_ppl"),
            "avg_delta_ppl": _first_float(row, "avg_delta_ppl_xsa", "mean_per_text_delta_ppl_xsa", "weighted_delta_loss_xsa"),
            "avg_residual_attn_ppl": _first_float(row, "avg_residual_attn_ppl", "mean_per_text_residual_attn_ppl", "weighted_residual_attn_ppl"),
            "avg_delta_ppl_residual_attn": _first_float(row, "avg_delta_ppl_residual_attn", "mean_per_text_delta_ppl_residual_attn", "weighted_delta_loss_residual_attn"),
            "avg_residual_mlp_ppl": _first_float(row, "avg_residual_mlp_ppl", "mean_per_text_residual_mlp_ppl", "weighted_residual_mlp_ppl"),
            "avg_delta_ppl_residual_mlp": _first_float(row, "avg_delta_ppl_residual_mlp", "mean_per_text_delta_ppl_residual_mlp", "weighted_delta_loss_residual_mlp"),
            "avg_residual_both_ppl": _first_float(row, "avg_residual_both_ppl", "mean_per_text_residual_both_ppl", "weighted_residual_both_ppl"),
            "avg_delta_ppl_residual_both": _first_float(row, "avg_delta_ppl_residual_both", "mean_per_text_delta_ppl_residual_both", "weighted_delta_loss_residual_both"),
            "path": str(path),
        }
        parsed.append(rec)
    return parsed


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plot_compare_absolute(compare_rows: List[Dict], axis_key: str, output_path: Path) -> None:
    if not compare_rows:
        return
    rows = sorted(compare_rows, key=lambda row: row[axis_key])
    xs = [row[axis_key] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    series = [
        ("avg_baseline_ppl", "Baseline / none", "#4C566A"),
        ("avg_xsa_ppl", "XSA middle / attn", "#D08770"),
        ("avg_residual_attn_ppl", "Residual / attn", "#5E81AC"),
        ("avg_residual_mlp_ppl", "Residual / mlp", "#A3BE8C"),
        ("avg_residual_both_ppl", "Residual / both", "#BF616A"),
    ]
    for key, label, color in series:
        ax.plot(xs, [row[key] for row in rows], marker="o", linewidth=2, label=label, color=color)
    ax.set_xlabel(axis_key)
    ax.set_ylabel("Average PPL")
    ax.set_title("Average PPL Across Sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_compare_delta(compare_rows: List[Dict], axis_key: str, output_path: Path) -> None:
    if not compare_rows:
        return
    rows = sorted(compare_rows, key=lambda row: row[axis_key])
    xs = [row[axis_key] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    series = [
        ("avg_delta_ppl", "XSA middle / attn", "#D08770"),
        ("avg_delta_ppl_residual_attn", "Residual / attn", "#5E81AC"),
        ("avg_delta_ppl_residual_mlp", "Residual / mlp", "#A3BE8C"),
        ("avg_delta_ppl_residual_both", "Residual / both", "#BF616A"),
    ]
    for key, label, color in series:
        ax.plot(xs, [row[key] for row in rows], marker="o", linewidth=2, label=label, color=color)
    ax.axhline(0.0, color="#2E3440", linewidth=1, alpha=0.6)
    ax.set_xlabel(axis_key)
    ax.set_ylabel("Average ΔPPL vs baseline")
    ax.set_title("PPL Change Relative to Baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_compare_relative(compare_rows: List[Dict], axis_key: str, output_path: Path) -> None:
    if not compare_rows:
        return
    rows = sorted(compare_rows, key=lambda row: row[axis_key])
    xs = [row[axis_key] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    series = [
        ("avg_xsa_ppl", "XSA middle / attn", "#D08770"),
        ("avg_residual_attn_ppl", "Residual / attn", "#5E81AC"),
        ("avg_residual_mlp_ppl", "Residual / mlp", "#A3BE8C"),
        ("avg_residual_both_ppl", "Residual / both", "#BF616A"),
    ]
    for key, label, color in series:
        rel_pct = [100.0 * (row[key] / row["avg_baseline_ppl"] - 1.0) for row in rows]
        ax.plot(xs, rel_pct, marker="o", linewidth=2, label=label, color=color)
    ax.axhline(0.0, color="#2E3440", linewidth=1, alpha=0.6)
    ax.set_xlabel(axis_key)
    ax.set_ylabel("Relative PPL vs baseline (%)")
    ax.set_title("PPL Relative To Baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_generation_length(generation_rows: List[Dict], axis_key: str, output_path: Path) -> None:
    if not generation_rows:
        return
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in generation_rows:
        grouped[row["target"]].append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = {
        "none": "#4C566A",
        "attn": "#D08770",
        "mlp": "#A3BE8C",
        "both": "#BF616A",
    }
    for target, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row[axis_key])
        ax.plot(
            [row[axis_key] for row in rows],
            [row["avg_chars"] for row in rows],
            marker="o",
            linewidth=2,
            label=target,
            color=palette.get(target),
        )
    ax.set_xlabel(axis_key)
    ax.set_ylabel("Average generated chars")
    ax.set_title("Generation Length Across Sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title="target")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_generation_change_rate(generation_rows: List[Dict], axis_key: str, output_path: Path) -> None:
    filtered = [row for row in generation_rows if row["target"] != "none"]
    if not filtered:
        return
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in filtered:
        grouped[row["target"]].append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = {
        "attn": "#D08770",
        "mlp": "#A3BE8C",
        "both": "#BF616A",
    }
    for target, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row[axis_key])
        ax.plot(
            [row[axis_key] for row in rows],
            [row["change_rate_vs_none"] for row in rows],
            marker="o",
            linewidth=2,
            label=target,
            color=palette.get(target),
        )
    ax.set_xlabel(axis_key)
    ax.set_ylabel("Fraction changed vs none")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Generation Change Rate Relative to Baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title="target")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_generation_examples(
    generation_files: List[Path],
    axis_key: str,
    output_path: Path,
    max_settings: int,
    max_prompts: int,
) -> None:
    if not generation_files:
        return
    with_settings = []
    for path in generation_files:
        setting = parse_setting_from_generation_path(path)
        axis_value = setting[axis_key]
        if axis_value is None:
            continue
        with_settings.append((axis_value, path))
    with_settings.sort(key=lambda item: item[0])
    if not with_settings:
        return

    picks = []
    if len(with_settings) <= max_settings:
        picks = with_settings
    else:
        indices = sorted(
            {
                0,
                len(with_settings) - 1,
                *[
                    round(i * (len(with_settings) - 1) / max(1, max_settings - 1))
                    for i in range(max_settings)
                ],
            }
        )
        picks = [with_settings[i] for i in indices[:max_settings]]

    lines = ["# XSA Generation Examples", ""]
    for axis_value, path in picks:
        rows = load_generation_rows(path)[:max_prompts]
        lines.append(f"## {axis_key} = {axis_value:g}")
        lines.append("")
        for row in rows:
            lines.append(f"### prompt_idx = {row['prompt_idx']}")
            lines.append("")
            lines.append("**Prompt**")
            lines.append("")
            lines.append(row["prompt"])
            lines.append("")
            for target, text in row.get("generations", {}).items():
                lines.append(f"**{target}**")
                lines.append("")
                lines.append(text.strip() or "<empty>")
                lines.append("")
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_overview(compare_rows: List[Dict], generation_rows: List[Dict], axis_key: str) -> Dict:
    axis_values = sorted({row[axis_key] for row in compare_rows if row.get(axis_key) is not None})
    overview = {
        "sweep_axis": axis_key,
        "num_compare_settings": len(compare_rows),
        "num_generation_target_rows": len(generation_rows),
        "axis_values": axis_values,
    }
    if compare_rows:
        best_xsa = min(compare_rows, key=lambda row: row["avg_xsa_ppl"])
        best_delta = min(compare_rows, key=lambda row: row["avg_delta_ppl"])
        overview["best_xsa_ppl_setting"] = {
            axis_key: best_xsa[axis_key],
            "avg_xsa_ppl": best_xsa["avg_xsa_ppl"],
            "avg_delta_ppl": best_xsa["avg_delta_ppl"],
        }
        overview["best_delta_ppl_setting"] = {
            axis_key: best_delta[axis_key],
            "avg_xsa_ppl": best_delta["avg_xsa_ppl"],
            "avg_delta_ppl": best_delta["avg_delta_ppl"],
        }
    return overview


def run_summary_tsv_mode() -> None:
    summary_tsv = Path(SUMMARY_TSV)
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else summary_tsv.parent / "viz_para"
    ensure_dir(output_dir)

    compare_rows = load_summary_tsv(summary_tsv)
    axis_key = "para_scale"

    save_csv(output_dir / "compare_aggregate.csv", compare_rows)
    plot_compare_absolute(compare_rows, axis_key, output_dir / "ppl_absolute.png")
    plot_compare_delta(compare_rows, axis_key, output_dir / "ppl_delta.png")
    plot_compare_relative(compare_rows, axis_key, output_dir / "ppl_relative_pct.png")
    overview = build_overview(compare_rows, [], axis_key)
    save_json(output_dir / "overview.json", overview)

    print(f"[INFO] MODE={MODE}")
    print(f"[INFO] SUMMARY_TSV={summary_tsv}")
    print(f"[INFO] OUTPUT_DIR={output_dir}")
    print(f"[INFO] Sweep axis: {axis_key}")
    print(f"[INFO] Compare settings: {len(compare_rows)}")
    print(f"[INFO] Wrote visualization bundle to {output_dir}")


def run_sweep_dir_mode() -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else input_dir / "viz"
    ensure_dir(output_dir)

    compare_files = discover_compare_files(input_dir)
    generation_files = discover_generation_files(input_dir)
    if not compare_files and not generation_files:
        raise SystemExit(f"No compare/generate jsonl files found under {input_dir}")

    compare_rows = [summarize_compare_file(path) for path in compare_files]
    generation_rows = [row for path in generation_files for row in summarize_generation_file(path)]

    axis_source = compare_rows if compare_rows else generation_rows
    axis_key = choose_sweep_axis(axis_source)

    save_csv(output_dir / "compare_aggregate.csv", compare_rows)
    save_csv(output_dir / "generation_aggregate.csv", generation_rows)

    plot_compare_absolute(compare_rows, axis_key, output_dir / "ppl_absolute.png")
    plot_compare_delta(compare_rows, axis_key, output_dir / "ppl_delta.png")
    plot_compare_relative(compare_rows, axis_key, output_dir / "ppl_relative_pct.png")
    plot_generation_length(generation_rows, axis_key, output_dir / "generation_avg_chars.png")
    plot_generation_change_rate(generation_rows, axis_key, output_dir / "generation_change_rate.png")
    write_generation_examples(
        generation_files,
        axis_key,
        output_dir / "generation_examples.md",
        max_settings=MAX_EXAMPLE_SETTINGS,
        max_prompts=MAX_EXAMPLE_PROMPTS,
    )

    overview = build_overview(compare_rows, generation_rows, axis_key)
    save_json(output_dir / "overview.json", overview)

    print(f"[INFO] MODE={MODE}")
    print(f"[INFO] INPUT_DIR={input_dir}")
    print(f"[INFO] OUTPUT_DIR={output_dir}")
    print(f"[INFO] MAX_EXAMPLE_SETTINGS={MAX_EXAMPLE_SETTINGS}")
    print(f"[INFO] MAX_EXAMPLE_PROMPTS={MAX_EXAMPLE_PROMPTS}")
    print(f"[INFO] Sweep axis: {axis_key}")
    print(f"[INFO] Compare settings: {len(compare_rows)}")
    print(f"[INFO] Generation files: {len(generation_files)}")
    print(f"[INFO] Wrote visualization bundle to {output_dir}")


def main() -> None:
    if MODE == "summary_tsv":
        run_summary_tsv_mode()
        return
    if MODE == "sweep_dir":
        run_sweep_dir_mode()
        return
    raise SystemExit(f"Unknown MODE={MODE!r}; expected 'summary_tsv' or 'sweep_dir'.")


if __name__ == "__main__":
    main()
