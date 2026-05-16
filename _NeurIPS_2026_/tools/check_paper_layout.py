#!/usr/bin/env python3
"""Lightweight paper layout checks for the NeurIPS draft.

Examples:
  python tools/check_paper_layout.py --pdf ~/Downloads/paper.pdf
  python tools/check_paper_layout.py --pdf ~/Downloads/paper.pdf --check-tex

The PDF short-line check uses `pdftotext -layout`, so install poppler first if
`pdftotext` is not available.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX_FILES = [*ROOT.glob("*.tex"), *ROOT.glob("sections/*.tex")]


def run_pdftotext(pdf: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf), str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return out_path.read_text(errors="ignore")
    except FileNotFoundError:
        raise SystemExit("pdftotext not found. Install poppler first.")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr.strip() or "pdftotext failed.")
    finally:
        out_path.unlink(missing_ok=True)


def is_probably_bad_short_line(line: str, *, max_words: int, max_chars: int) -> bool:
    text = line.strip()
    if not text or text.isdigit():
        return False
    if any(ch.isdigit() for ch in text):
        return False
    if text.startswith(("Figure", "Table", "References", "Appendix")):
        return False
    words = text.split()
    return 1 <= len(words) <= max_words and len(text) <= max_chars


def check_short_lines(pdf: Path, *, max_words: int, max_chars: int) -> int:
    text = run_pdftotext(pdf)
    pages = text.split("\f")
    hits = 0
    for page_no, page in enumerate(pages, 1):
        page_hits: list[tuple[int, str]] = []
        for line_no, line in enumerate(page.splitlines(), 1):
            if is_probably_bad_short_line(line, max_words=max_words, max_chars=max_chars):
                page_hits.append((line_no, line.strip()))
        if page_hits:
            hits += len(page_hits)
            print(f"\n[short-lines] page {page_no}")
            for line_no, text_line in page_hits:
                print(f"  {line_no:>3}: {text_line}")
    return hits


def check_includegraphics() -> int:
    missing = 0
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    for tex in TEX_FILES:
        for match in pattern.finditer(tex.read_text()):
            path = ROOT / match.group(1)
            if not path.exists():
                missing += 1
                print(f"[missing-graphic] {tex.relative_to(ROOT)}: {match.group(1)}")
    return missing


def check_labels() -> int:
    labels: dict[str, list[str]] = {}
    refs: list[tuple[str, str]] = []
    label_re = re.compile(r"\\label\{([^}]+)\}")
    ref_re = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
    for tex in TEX_FILES:
        rel = str(tex.relative_to(ROOT))
        text = tex.read_text()
        for match in label_re.finditer(text):
            labels.setdefault(match.group(1), []).append(rel)
        for match in ref_re.finditer(text):
            refs.append((rel, match.group(1)))

    problems = 0
    for label, files in sorted(labels.items()):
        if len(files) > 1:
            problems += 1
            print(f"[duplicate-label] {label}: {', '.join(files)}")
    for file_name, ref in refs:
        if ref not in labels:
            problems += 1
            print(f"[missing-ref] {file_name}: {ref}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="Compiled PDF to scan for short lines.")
    parser.add_argument("--check-tex", action="store_true", help="Check graphics and refs.")
    parser.add_argument("--max-words", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=28)
    args = parser.parse_args()

    total = 0
    if args.pdf:
        total += check_short_lines(args.pdf.expanduser(), max_words=args.max_words, max_chars=args.max_chars)
    if args.check_tex:
        total += check_includegraphics()
        total += check_labels()
    if not args.pdf and not args.check_tex:
        parser.error("provide --pdf and/or --check-tex")
    if total == 0:
        print("No issues found by the selected checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
