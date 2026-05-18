#!/usr/bin/env bash
set -e

# ============================================================
# Reviewer response:
# Compact FLORES generalization experiment
#
# Usage:
#   ./run_flores_generalization_reviewer_response.sh khmkhmr_vielatn
#   ./run_flores_generalization_reviewer_response.sh laolaoo_vielatn
#
# Experiments:
#   1. Frozen XLM-R retrieval + cosine
#   2. Frozen XLM-R retrieval + CSLS
#   3. Baseline-6-style LoRA + projection + no Kabsch
#      with cosine and CSLS evaluation
#
# Important:
#   This script runs ONE language pair per SLURM job.
# ============================================================

SRC_MODEL="xlm-roberta-base"
TGT_MODEL="xlm-roberta-base"

PAIR="${1:-}"

if [ -z "$PAIR" ]; then
  echo "Usage: $0 <pair_slug>"
  echo "Available pair slugs:"
  echo "  khmkhmr_vielatn"
  echo "  laolaoo_vielatn"
  exit 1
fi

if [ "$PAIR" != "khmkhmr_vielatn" ] && [ "$PAIR" != "laolaoo_vielatn" ]; then
  echo "Unknown pair slug: $PAIR"
  echo "Expected one of:"
  echo "  khmkhmr_vielatn"
  echo "  laolaoo_vielatn"
  exit 1
fi

echo "Preparing FLORES data..."
python src/prepare_flores_generalization_pairs.py

TRAIN_CSV="data/flores_generalization/${PAIR}/train.csv"
TEST_CSV="data/flores_generalization/${PAIR}/test.csv"

test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

echo "============================================================"
echo "Pair: $PAIR"
echo "Train CSV: $TRAIN_CSV"
echo "Test CSV: $TEST_CSV"
echo "============================================================"

# ------------------------------------------------------------
# Experiment 1A: Frozen XLM-R + cosine
# ------------------------------------------------------------
python src/frozen_xlmr_flores_retrieval.py \
  --test_csv "$TEST_CSV" \
  --output_dir "results/baselines/flores_${PAIR}_frozen_xlmr_cosine" \
  --model_name "$SRC_MODEL" \
  --max_len 256 \
  --batch_size 16 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ------------------------------------------------------------
# Experiment 1B: Frozen XLM-R + CSLS
# ------------------------------------------------------------
python src/frozen_xlmr_flores_retrieval.py \
  --test_csv "$TEST_CSV" \
  --output_dir "results/baselines/flores_${PAIR}_frozen_xlmr_csls" \
  --model_name "$SRC_MODEL" \
  --max_len 256 \
  --batch_size 16 \
  --use_csls \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ------------------------------------------------------------
# Experiment 2: Baseline-6-style LoRA + projection
# no Kabsch + cosine/CSLS retrieval
#
# Reduced from 50 epochs to 10 epochs to fit the 1-day job limit.
# ------------------------------------------------------------
python src/lora_projection_generalization_train_eval.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_mean \
  --projection_dim 256 \
  --proj_dropout 0.1 \
  --epochs 10 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --eval_batch_size 16 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --temperature 0.07 \
  --warmup_ratio 0.06 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --eval_retrievals both \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --num_workers 0 \
  --output_dir "results/baselines/flores_${PAIR}_b6style_lora_projection_10ep_token_mean_both"

echo
echo "Finished FLORES reviewer-response experiment for pair: $PAIR"
