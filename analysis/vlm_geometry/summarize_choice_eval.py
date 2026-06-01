from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _parse_setting(path: Path) -> dict[str, str]:
    name = path.parent.name
    fields = {"setting": name}
    for key in ("space", "target"):
        match = re.search(rf"{key}_([^_]+)", name)
        if match:
            fields[key] = match.group(1)
    para = re.search(r"para_([^_]+)__perp_([^_]+)", name)
    if para:
        fields["para_scale"] = para.group(1).replace("m", "-").replace("p", ".")
        fields["perp_scale"] = para.group(2).replace("m", "-").replace("p", ".")
    head = re.search(r"vhead_([^_]+)", name)
    if head:
        fields["value_head_mode"] = head.group(1)
    ref = re.search(r"vref_(.*?)__attr_(.*?)(?:__mode_|__n)", name)
    if ref:
        fields["value_ref_expansion"] = ref.group(1)
        fields["attn_attr_source"] = ref.group(2)
    mode = re.search(r"mode_(.*?)__seed_([^_]+)", name)
    if mode:
        fields["scale_mode"] = mode.group(1)
        fields["scale_seed"] = mode.group(2)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VLM forced-choice geometry evaluation results.")
    parser.add_argument("roots", nargs="+", help="Result roots containing by_model/*/*/*/summary.json")
    parser.add_argument("--output", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for root_text in args.roots:
        root = Path(root_text)
        for summary_path in sorted(root.glob("by_model/*/*/*/summary.json")):
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            rel = summary_path.relative_to(root)
            model = rel.parts[1]
            task = rel.parts[2]
            row = {
                "root": str(root),
                "model": model,
                "task": task,
                "n_examples": data.get("n_examples"),
                "correct": data.get("correct"),
                "accuracy": data.get("accuracy"),
                "candidate_mode": data.get("candidate_mode"),
            }
            row.update(_parse_setting(summary_path))
            config = data.get("geometry", {}).get("config", {})
            row.setdefault("space", config.get("space"))
            row.setdefault("target", config.get("target"))
            row.setdefault("para_scale", config.get("para_scale"))
            row.setdefault("perp_scale", config.get("perp_scale"))
            row.setdefault("value_head_mode", config.get("value_head_mode", ""))
            row.setdefault("value_ref_expansion", config.get("value_ref_expansion", ""))
            row.setdefault("attn_attr_source", config.get("attn_attr_source", ""))
            row.setdefault("scale_mode", config.get("scale_mode", ""))
            row.setdefault("scale_seed", config.get("scale_seed", ""))
            rows.append(row)

    fieldnames = [
        "root",
        "model",
        "task",
        "setting",
        "space",
        "target",
        "para_scale",
        "perp_scale",
        "value_head_mode",
        "value_ref_expansion",
        "attn_attr_source",
        "scale_mode",
        "scale_seed",
        "n_examples",
        "correct",
        "accuracy",
        "candidate_mode",
    ]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
