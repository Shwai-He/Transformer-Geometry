#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER_DIR="${PAPER_DIR:-$REPO_ROOT/_overleaf_/ARR_May_Revision_Overleaf}"
RENDER_DIR="$(mktemp -d /tmp/emnlp_2026_render.XXXXXX)"
OUTPUT_PDF="$REPO_ROOT/archive/local_exports/emnlp_2026_camera_ready.pdf"

mkdir -p "$(dirname "$OUTPUT_PDF")"

# Compile directly because the local latexmk wrapper currently depends on a
# missing Perl module. The explicit sequence matches the standard BibTeX build.
(
  cd "$PAPER_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error emnlp_2026.tex
  bibtex emnlp_2026
  pdflatex -interaction=nonstopmode -halt-on-error emnlp_2026.tex
  pdflatex -interaction=nonstopmode -halt-on-error emnlp_2026.tex
)

cp "$PAPER_DIR/emnlp_2026.pdf" "$OUTPUT_PDF"

python - "$OUTPUT_PDF" "$RENDER_DIR" <<'PY'
import pathlib
import sys

import fitz

pdf_path = pathlib.Path(sys.argv[1])
render_dir = pathlib.Path(sys.argv[2])
document = fitz.open(pdf_path)
for page_number, page in enumerate(document, start=1):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pixmap.save(render_dir / f"page-{page_number:02d}.png")
PY

echo "Wrote $OUTPUT_PDF"
echo "Rendered pages in $RENDER_DIR"
