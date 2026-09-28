#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
XM="${XM_RETRIEVERS_ROOT:?set XM_RETRIEVERS_ROOT to the xm-retrievers checkout}"
PYTHON="${PYTHON_BIN:-python}"
OUTPUT="$ROOT/checkpoints/dense_retrievers/dpr_xm/seed_042"
GPU_ID="${DPR_GPU_ID:-3}"

env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    AACL_ROOT="$ROOT/artifacts/prepared_data/dense_retrievers/dpr_dev_protocol" \
    PYTHONPATH="$XM" \
    "$PYTHON" -u "$XM/src/retrievers/dpr_xm_bahna.py" \
    --language bana \
    --model_name facebook/xmod-base \
    --max_seq_length 128 \
    --pooling mean \
    --sim cos_sim \
    --do_train \
    --do_test \
    --epochs 20 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --lr 2e-5 \
    --wd 0.01 \
    --scheduler warmuplinear \
    --warmup_ratio 0.1 \
    --seed 42 \
    --log_steps 50 \
    --output_dir "$OUTPUT"
