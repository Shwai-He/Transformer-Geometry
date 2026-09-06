#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER_DIR="${PAPER_DIR:-$REPO_ROOT/_overleaf_/ARR_May_Revision_Overleaf}"
OUTPUT_PDF="$REPO_ROOT/archive/local_exports/emnlp_2026_camera_ready.pdf"

# Stabilize PDF creation dates and document IDs. Callers may override the epoch
# for an official release; otherwise use the checked-out camera-ready baseline.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$REPO_ROOT" log -1 --format=%ct)}"
export FORCE_SOURCE_DATE=1
export TZ=UTC

for command_name in pdflatex bibtex kpsewhich; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing manuscript build command: $command_name" >&2
    exit 1
  fi
done

for latex_package in multirow.sty placeins.sty; do
  if ! kpsewhich "$latex_package" >/dev/null 2>&1; then
    echo "Missing LaTeX package: $latex_package (see scripts/manuscript/README.md)" >&2
    exit 1
  fi
done

RENDER_DIR="$(mktemp -d /tmp/emnlp_2026_render.XXXXXX)"
TEX_CACHE_DIR="$(mktemp -d /tmp/emnlp_2026_texmf.XXXXXX)"
trap 'rm -rf -- "$TEX_CACHE_DIR"' EXIT

mkdir -p "$(dirname "$OUTPUT_PDF")"

# Compile directly because the local latexmk wrapper currently depends on a
# missing Perl module. The explicit sequence matches the standard BibTeX build.
(
  cd "$PAPER_DIR"
  export TEXMFVAR="$TEX_CACHE_DIR/var"
  export TEXMFCONFIG="$TEX_CACHE_DIR/config"
  export TEXMFHOME="$TEX_CACHE_DIR/home"
  pdflatex -interaction=nonstopmode -halt-on-error emnlp_2026.tex
  bibtex emnlp_2026
  pdflatex -interaction=nonstopmode -halt-on-error emnlp_2026.tex
  pdflatex -interaction=nonstopmode -halt-on-error emnlp_2026.tex
)

cp "$PAPER_DIR/emnlp_2026.pdf" "$OUTPUT_PDF"
cp "$PAPER_DIR/emnlp_2026.pdf" "$REPO_ROOT/emnlp_2026_camera_ready.pdf"

if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -png -r 144 "$OUTPUT_PDF" "$RENDER_DIR/page"
elif command -v gs >/dev/null 2>&1; then
  gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r144 \
    -sOutputFile="$RENDER_DIR/page-%02d.png" "$OUTPUT_PDF"
else
  echo "Missing PDF renderer: install pdftoppm or Ghostscript." >&2
  exit 1
fi

echo "Wrote $OUTPUT_PDF"
echo "Rendered pages in $RENDER_DIR"

# Keep the source tree free of transient TeX state by default. Set
# KEEP_BUILD_AUX=1 when the full final-pass log is needed for diagnosis.
if [[ "${KEEP_BUILD_AUX:-0}" != "1" ]]; then
  rm -f -- \
    "$PAPER_DIR/emnlp_2026.aux" \
    "$PAPER_DIR/emnlp_2026.blg" \
    "$PAPER_DIR/emnlp_2026.log" \
    "$PAPER_DIR/emnlp_2026.out"
fi
