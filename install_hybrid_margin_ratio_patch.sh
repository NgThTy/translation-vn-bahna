#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/translation-vn-bahna" >&2
  exit 2
fi

ROOT="$(cd "$1" && pwd)"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$ROOT/backups/hybrid_margin_ratio_$TIMESTAMP"

mkdir -p "$ROOT/src" "$BACKUP_DIR/src"

for name in retrieval_scoring.py hybrid_lexical_neural_rerank.py; do
  if [[ -f "$ROOT/src/$name" ]]; then
    cp -a "$ROOT/src/$name" "$BACKUP_DIR/src/$name"
  fi
  install -m 755 "$PATCH_DIR/src/$name" "$ROOT/src/$name"
done

python -m py_compile \
  "$ROOT/src/retrieval_scoring.py" \
  "$ROOT/src/hybrid_lexical_neural_rerank.py"

echo "Installed hybrid margin-ratio patch into: $ROOT"
echo "Backup directory: $BACKUP_DIR"
