#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
AACL="${AACL2026_ROOT:?set AACL2026_ROOT to the downloaded dense-baseline bundle}"
PYTHON="${PYTHON_BIN:-python}"
DATA="$WORKSPACE/artifacts/prepared_data/dense_retrievers/colbert_20k_data"
RUNS="$WORKSPACE/checkpoints/dense_retrievers"

mkdir -p "$RUNS/colbert_x/seed_042_20k" "$RUNS/colbert_xm/seed_042_20k"

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$AACL/upstream" \
  nohup "$PYTHON" "$AACL/colbertx/run_dev_protocol.py" \
    --data_dir "$DATA" \
    --output_dir "$RUNS/colbert_x/seed_042_20k" \
    --model_name xlm-roberta-base \
    --seed 42 \
    --maxsteps 20000 \
    --bsize 16 \
    --checkpoint_steps 5000 10000 15000 20000 \
    >"$RUNS/colbert_x/seed_042_20k/run.log" 2>&1 &
COLBERT_X_PID=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$AACL/xm-retrievers:$AACL/upstream" \
  nohup "$PYTHON" "$AACL/xm-retrievers/src/retrievers/multi_vector_biencoder.py" \
    --dataset bahna \
    --language bana \
    --data_dir "$DATA" \
    --output_dir "$RUNS/colbert_xm/seed_042_20k" \
    --do_train \
    --model_name facebook/xmod-base \
    --dim 128 \
    --doc_maxlen 128 \
    --mask_punctuation \
    --query_maxlen 64 \
    --similarity cosine \
    --bsize 8 \
    --accumsteps 1 \
    --lr 3e-6 \
    --maxsteps 20000 \
    --warmup 2000 \
    --nway 8 \
    --use_ib_negatives \
    --ignore_scores \
    --nbits 2 \
    --kmeans_niters 4 \
    --checkpoint_steps 5000 10000 15000 20000 \
    >"$RUNS/colbert_xm/seed_042_20k/run.log" 2>&1 &
COLBERT_XM_PID=$!

printf 'ColBERT-X PID: %s\nColBERT-XM PID: %s\n' "$COLBERT_X_PID" "$COLBERT_XM_PID"
