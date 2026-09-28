#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACT="${AACL2026_ROOT:?set AACL2026_ROOT to the downloaded dense-baseline bundle}"
PYTHON="${PYTHON_BIN:-python}"
DATA="$ROOT/artifacts/prepared_data/dense_retrievers/colbert_dev_eval_data"
EVALUATOR="$ARTIFACT/colbertx/evaluate_dev_protocol.py"
RUNTIME="$ROOT/artifacts/prepared_data/dense_retrievers/colbert_runtime"
CHECKPOINTS="$ROOT/checkpoints/dense_retrievers/colbert_x/seed_042/colbertx/none/seed42/checkpoints"
OUTPUT="$ROOT/results/dense/dev_protocol/checkpoint_sweeps/colbert_x"
GPU_ID="${COLBERT_X_GPU:-0}"

mkdir -p "$OUTPUT"
for step in $(seq 5000 5000 50000); do
    eval_dir="$OUTPUT/step_$(printf '%05d' "$step")"
    if [[ -f "$eval_dir/metrics.json" ]]; then
        continue
    fi
    if [[ -d "$eval_dir" && ! -f "$eval_dir/indexes/bahna_dev/metadata.json" ]]; then
        mv "$eval_dir" "${eval_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
    env CUDA_VISIBLE_DEVICES="$GPU_ID" PATH="$(dirname "$PYTHON"):$PATH" PYTHONPATH="$RUNTIME" \
        "$PYTHON" -u "$EVALUATOR" \
        --data_dir "$DATA" \
        --output_dir "$eval_dir" \
        --checkpoint "$CHECKPOINTS/colbert-$step"
done

"$PYTHON" "$SCRIPT_DIR/select_dev_checkpoints.py" --eval-dir "$OUTPUT"
