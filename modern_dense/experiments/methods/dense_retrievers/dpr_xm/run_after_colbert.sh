#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
METHOD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
X_SUMMARY="$ROOT/results/dense/dev_protocol/checkpoint_sweeps/colbert_x/dev_selection_summary.json"
XM_SUMMARY="$ROOT/results/dense/dev_protocol/checkpoint_sweeps/colbert_xm/dev_selection_summary.json"

while [[ ! -f "$X_SUMMARY" || ! -f "$XM_SUMMARY" ]]; do
    sleep 30
done

exec "$METHOD_ROOT/orchestration/wait_for_gpu.sh" 2 env DPR_GPU_ID=2 "$SCRIPT_DIR/run_dev_pool.sh"
