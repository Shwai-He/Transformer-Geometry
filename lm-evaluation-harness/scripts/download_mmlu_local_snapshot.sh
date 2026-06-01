#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Download a local MMLU snapshot for lm-eval.
#
# This avoids depending on Hugging Face dataset API calls at evaluation time. It
# uses git for the small dataset repo metadata and resolves Git-LFS parquet
# pointer files through /resolve/main URLs, which works on nodes without git-lfs.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"

TARGET_DIR="${TARGET_DIR:-$HARNESS_DIR/outputs/task_cache/cais_mmlu_git_tmp}"
REPO_URL="${REPO_URL:-https://huggingface.co/datasets/cais/mmlu}"
RESOLVE_BASE="${RESOLVE_BASE:-https://huggingface.co/datasets/cais/mmlu/resolve/main}"

mkdir -p "$(dirname "$TARGET_DIR")"
if [[ ! -d "$TARGET_DIR/.git" ]]; then
  rm -rf "$TARGET_DIR"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
else
  git -C "$TARGET_DIR" pull --ff-only
fi

pointer_list="$(mktemp)"
python - "$TARGET_DIR" "$pointer_list" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for path in sorted(root.rglob("*.parquet")):
    if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
        rows.append(str(path.relative_to(root)))
out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
print(f"[INFO] pointer_files={len(rows)}")
PY

count="$(wc -l < "$pointer_list" | tr -d ' ')"
i=0
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  i=$((i + 1))
  out="$TARGET_DIR/$rel"
  tmp="$out.tmp"
  echo "[$i/$count] $rel"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 --max-time 120 \
    -o "$tmp" "$RESOLVE_BASE/$rel"
  head4="$(head -c 4 "$tmp" | od -An -t x1 | tr -d ' \n')"
  if [[ "$head4" != "50415231" ]]; then
    echo "[ERROR] downloaded file is not parquet: $rel head=$head4" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$out"
done < "$pointer_list"
rm -f "$pointer_list"

HF_HOME="${HF_HOME:-/beacon-projects/traumallm/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "${PYTHON_BIN:-/beacon-projects/traumallm/shwaihe/envs/sparse-ug-sys/bin/python}" - <<PY
from datasets import load_dataset

path = "$TARGET_DIR"
for name in ("abstract_algebra", "high_school_geography", "professional_law"):
    ds = load_dataset(path, name, split="test", trust_remote_code=True)
    print(f"[CHECK] {name}: rows={len(ds)} columns={ds.column_names}")
PY

echo "[DONE] local MMLU snapshot: $TARGET_DIR"
