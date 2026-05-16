#!/usr/bin/env bash
set -euo pipefail

# backup_ppt.sh
# Usage: ./backup_ppt.sh [source_dir] [backup_dir]
# Default source_dir: current directory
# Default backup_dir: ./ppt_backups

SOURCE_DIR="${1:-.}"
BACKUP_DIR="${2:-./ppt_backups}"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
mkdir -p "$BACKUP_DIR"

find "$SOURCE_DIR" -maxdepth 3 \( -iname '*.ppt' -o -iname '*.pptx' \) | while IFS= read -r file; do
  rel=$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$file" "$SOURCE_DIR")
  dest="$BACKUP_DIR/${TIMESTAMP}_$rel"
  mkdir -p "$(dirname "$dest")"
  cp -p "$file" "$dest"
  echo "Backed up: $file -> $dest"
done

echo "PPT backup completed to $BACKUP_DIR"
