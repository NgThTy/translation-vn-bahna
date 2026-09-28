#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACT="${AACL2026_ROOT:?set AACL2026_ROOT to the downloaded dense-baseline bundle}"
PYTHON="${PYTHON_BIN:-python}"
DATA="$ROOT/artifacts/prepared_data/dense_retrievers/colbert_dev_eval_data"
EVALUATOR="$ARTIFACT/colbertx/evaluate_dev_protocol.py"
RUNTIME="$ROOT/artifacts/prepared_data/dense_retrievers/colbert_runtime"
XM_SOURCE="$ARTIFACT/xm-retrievers"
XM_TRAIN="$ROOT/checkpoints/dense_retrievers/colbert_xm/seed_042/bahna/modular-retrievers/none/2026-07-13_23.09-facebook-xmod-base-bahna-bana/checkpoints"
X_TRAIN="$ROOT/checkpoints/dense_retrievers/colbert_x/seed_042/colbertx/none/seed42/checkpoints"
EVAL_ROOT="$ROOT/results/dense/dev_protocol/checkpoint_sweeps"

mkdir -p "$EVAL_ROOT/colbert_xm" "$EVAL_ROOT/colbert_x"

for step in $(seq 5000 5000 60000); do
    output="$EVAL_ROOT/colbert_xm/step_$(printf '%05d' "$step")"
    if [[ -f "$output/metrics.json" ]]; then
        continue
    fi
    if [[ -d "$output" && ! -f "$output/indexes/bahna_dev/metadata.json" ]]; then
        mv "$output" "${output}.incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
    env CUDA_VISIBLE_DEVICES=2 PATH="$(dirname "$PYTHON"):$PATH" PYTHONPATH="$RUNTIME:$XM_SOURCE" \
        "$PYTHON" -u "$EVALUATOR" \
        --data_dir "$DATA" \
        --output_dir "$output" \
        --checkpoint "$XM_TRAIN/colbert-$step" \
        --xmod
done

"$PYTHON" "$SCRIPT_DIR/select_dev_checkpoints.py" \
    --eval-dir "$EVAL_ROOT/colbert_xm"

# ColBERT-X is still training when this sweep is launched. Its parent writes
# best_checkpoint.txt only after the final 50k checkpoint is safely closed.
while [[ ! -f "$ROOT/checkpoints/dense_retrievers/colbert_x/seed_042/best_checkpoint.txt" ]]; do
    sleep 30
done

for step in $(seq 5000 5000 50000); do
    output="$EVAL_ROOT/colbert_x/step_$(printf '%05d' "$step")"
    if [[ -f "$output/metrics.json" ]]; then
        continue
    fi
    if [[ -d "$output" && ! -f "$output/indexes/bahna_dev/metadata.json" ]]; then
        mv "$output" "${output}.incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
    env CUDA_VISIBLE_DEVICES=2 PATH="$(dirname "$PYTHON"):$PATH" PYTHONPATH="$RUNTIME" \
        "$PYTHON" -u "$EVALUATOR" \
        --data_dir "$DATA" \
        --output_dir "$output" \
        --checkpoint "$X_TRAIN/colbert-$step"
done

"$PYTHON" "$SCRIPT_DIR/select_dev_checkpoints.py" \
    --eval-dir "$EVAL_ROOT/colbert_x"
