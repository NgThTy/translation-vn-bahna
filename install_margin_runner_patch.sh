#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/translation-vn-bahna" >&2
  exit 2
fi

ROOT="$(cd "$1" && pwd)"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$ROOT/backups/margin_runner_patch_$TIMESTAMP"

FILES=(
  run_fasttext_procrustes_baselines.sh
  run_off_the_shelf_encoder_baselines.sh
  run_previous_pipeline_baselines.sh
  run_full_encoder_contrastive_baselines.sh
  run_previous_pipeline_additional_experiments
  run_previous_pipeline_clarification_experiments
  run_previous_pipeline_token_level_kabsch_experiments.sh
)

mkdir -p "$BACKUP_DIR"
for file in "${FILES[@]}"; do
  if [[ -e "$ROOT/$file" ]]; then
    cp -a "$ROOT/$file" "$BACKUP_DIR/$file"
  fi
  install -m 755 "$PATCH_DIR/$file" "$ROOT/$file"
  bash -n "$ROOT/$file"
done

cp -a "$PATCH_DIR/README_MARGIN_RUNNER_PATCH.md" "$ROOT/README_MARGIN_RUNNER_PATCH.md"

echo "Installed margin-ratio runner patch into: $ROOT"
echo "Backups saved under: $BACKUP_DIR"
echo "Shell syntax validation completed successfully."
