#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$ROOT_DIR/_EMNLP_2026_"
BUILD_DIR="/tmp/emnlp_2026_build"
RENDER_DIR="/tmp/emnlp_render"
LOCAL_TEX_DIR="$ROOT_DIR/_EMNLP_2026_legacy/local-tex/lineno-v5.5"

rm -rf "$BUILD_DIR" "$RENDER_DIR"
mkdir -p "$BUILD_DIR" "$RENDER_DIR"

if [[ ! -f "$LOCAL_TEX_DIR/lineno.sty" ]]; then
  echo "Missing local lineno override: $LOCAL_TEX_DIR/lineno.sty" >&2
  exit 1
fi

echo "Using local lineno override: $LOCAL_TEX_DIR/lineno.sty"

# Homebrew TeX Live 2026 currently ships lineno v5.7, which can place ACL
# review line numbers in the two-column gutter. Use a build-only TeX input path
# so the Overleaf upload directory keeps the official ACL template untouched.
(
  cd "$PAPER_DIR"
  TEXINPUTS="$LOCAL_TEX_DIR//:" \
    latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
    -outdir="$BUILD_DIR" emnlp_2026.tex
)

pdftoppm -png -r 144 "$BUILD_DIR/emnlp_2026.pdf" "$RENDER_DIR/page"
cp "$BUILD_DIR/emnlp_2026.pdf" "$ROOT_DIR/emnlp_2026.pdf"

echo "Wrote $ROOT_DIR/emnlp_2026.pdf"
echo "Rendered pages in $RENDER_DIR"
