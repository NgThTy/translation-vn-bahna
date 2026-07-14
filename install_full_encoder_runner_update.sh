#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/translation-vn-bahna" >&2
  exit 2
fi

ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/run_full_encoder_contrastive_baselines.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$ROOT" ]]; then
  echo "Repository directory not found: $ROOT" >&2
  exit 1
fi

if [[ -f "$TARGET" ]]; then
  cp "$TARGET" "$TARGET.backup.$STAMP"
  echo "Backup created: $TARGET.backup.$STAMP"
fi

cp "$SCRIPT_DIR/run_full_encoder_contrastive_baselines.sh" "$TARGET"
chmod +x "$TARGET"
bash -n "$TARGET"

echo "Installed and validated: $TARGET"
