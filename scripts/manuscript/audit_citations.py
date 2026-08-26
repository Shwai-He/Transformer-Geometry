#!/usr/bin/env python3
"""Audit citation coverage and bibliography hygiene for the camera-ready paper."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


CITE_PATTERN = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}", re.DOTALL)
ENTRY_START_PATTERN = re.compile(r"(?m)^@(\w+)\{\s*([^,]+),")


def normalize_title(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def field(block, name):
    match = re.search(
        r"(?im)^\s*{}\s*=\s*[{{\"](.*?)(?:[}}\"]\s*,?\s*$)".format(
            re.escape(name)
        ),
        block,
        re.DOTALL,
    )
    return match.group(1).strip() if match else None


def parse_entries(text):
    starts = list(ENTRY_START_PATTERN.finditer(text))
    entries = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        entries.append(
            {
                "type": match.group(1),
                "key": match.group(2).strip(),
                "block": text[match.start() : end],
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tex_paths = sorted(args.paper_root.glob("*.tex")) + sorted(
        (args.paper_root / "sections").glob("*.tex")
    )
    tex = "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)
    cited = set()
    for match in CITE_PATTERN.finditer(tex):
        cited.update(key.strip() for key in match.group(1).split(",") if key.strip())

    bib_path = args.paper_root / "references.bib"
    entries = parse_entries(bib_path.read_text(encoding="utf-8"))
    by_key = defaultdict(list)
    by_title = defaultdict(list)
    for entry in entries:
        by_key[entry["key"]].append(entry)
        title = field(entry["block"], "title")
        if title:
            by_title[normalize_title(title)].append(entry["key"])

    errors = []
    warnings = []
    missing = sorted(cited - set(by_key))
    if missing:
        errors.append("missing bibliography keys: {}".format(", ".join(missing)))
    duplicate_keys = sorted(key for key, rows in by_key.items() if len(rows) > 1)
    if duplicate_keys:
        errors.append("duplicate bibliography keys: {}".format(", ".join(duplicate_keys)))
    duplicate_titles = sorted(
        sorted(keys) for keys in by_title.values() if len(set(keys)) > 1
    )
    if duplicate_titles:
        errors.append("duplicate normalized titles: {!r}".format(duplicate_titles))

    cited_with_truncated_authors = []
    cited_without_locator = []
    for key in sorted(cited & set(by_key)):
        block = by_key[key][0]["block"]
        author = field(block, "author") or ""
        if re.search(r"\band\s+others\b", author, re.IGNORECASE):
            cited_with_truncated_authors.append(key)
        if not any(field(block, name) for name in ("url", "doi", "eprint")):
            cited_without_locator.append(key)
    if cited_with_truncated_authors:
        errors.append(
            "cited entries use 'and others': {}".format(
                ", ".join(cited_with_truncated_authors)
            )
        )
    if cited_without_locator:
        warnings.append(
            "cited entries lack URL/DOI/eprint: {}".format(
                ", ".join(cited_without_locator)
            )
        )

    output = {
        "paper_root": str(args.paper_root.resolve()),
        "tex_files": [str(path) for path in tex_paths],
        "bibliography": str(bib_path),
        "cited_keys": len(cited),
        "bibliography_entries": len(entries),
        "missing_keys": missing,
        "duplicate_keys": duplicate_keys,
        "duplicate_titles": duplicate_titles,
        "cited_with_truncated_authors": cited_with_truncated_authors,
        "cited_without_locator": cited_without_locator,
        "errors": errors,
        "warnings": warnings,
        "pass": not errors,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
