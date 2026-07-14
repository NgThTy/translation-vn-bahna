#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/translation-vn-bahna" >&2
    exit 2
fi

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$1" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$TARGET_ROOT/.margin_scoring_backup_$TIMESTAMP"

FILES=(
    src/retrieval_scoring.py
    src/fasttext_procrustes_baseline.py
    src/multilingual_encoder_baseline.py
    src/previous_pipeline_baseline.py
    src/full_encoder_contrastive_finetune_baseline.py
)

mkdir -p "$TARGET_ROOT/src" "$BACKUP_DIR/src"

for relative_path in "${FILES[@]}"; do
    source_path="$PATCH_DIR/$relative_path"
    target_path="$TARGET_ROOT/$relative_path"

    if [[ ! -f "$source_path" ]]; then
        echo "Missing patch file: $source_path" >&2
        exit 1
    fi

    if [[ -f "$target_path" ]]; then
        cp -a "$target_path" "$BACKUP_DIR/$relative_path"
    fi

    install -m 644 "$source_path" "$target_path"
done

python -m py_compile \
    "$TARGET_ROOT/src/retrieval_scoring.py" \
    "$TARGET_ROOT/src/fasttext_procrustes_baseline.py" \
    "$TARGET_ROOT/src/multilingual_encoder_baseline.py" \
    "$TARGET_ROOT/src/previous_pipeline_baseline.py" \
    "$TARGET_ROOT/src/full_encoder_contrastive_finetune_baseline.py"

echo "Installed margin-scoring files into: $TARGET_ROOT"
echo "Backup of replaced files: $BACKUP_DIR"
echo "Remember to update the active shell runner grids to include margin_ratio."
