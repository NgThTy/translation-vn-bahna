#!/usr/bin/env bash
set -e

# ============================================================
# Baseline 8: Full encoder contrastive fine-tuning
# ============================================================
#
# This baseline fine-tunes the full XLM-R source and target encoders.
#
# Run this on a GPU compute node, not directly on the head node.
#
# Main comparison:
# - Baseline 7: LoRA contrastive fine-tuning
# - Baseline 8: Full encoder contrastive fine-tuning
#
# Recommended first setting:
# - XLM-R
# - symmetric InfoNCE
# - sentence_mean pooling
# - 3 epochs
# - small per-device batch size
# - gradient accumulation
# ============================================================

echo "Checking required data files..."

test -f data/train.csv || { echo "Missing data/train.csv"; exit 1; }
test -f data/test.csv || { echo "Missing data/test.csv"; exit 1; }

echo "All required data files found."
echo

echo "GPU status:"
nvidia-smi || true
echo

# -------------------------
# 1. Main run:
#    XLM-R full fine-tuning + cosine retrieval
# -------------------------
python src/full_encoder_contrastive_finetune_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --pooling sentence_mean \
  --output_dir results/baselines/full_encoder_xlmr_infonce_3ep_cosine \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --temperature 0.07 \
  --warmup_ratio 0.06 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --eval_batch_size 16 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --fp16

# -------------------------
# 2. Same trained setting, separate run with CSLS retrieval.
#    This retrains from scratch, so use it only if you want a direct
#    independent CSLS run.
# -------------------------
python src/full_encoder_contrastive_finetune_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --pooling sentence_mean \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/full_encoder_xlmr_infonce_3ep_csls \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --temperature 0.07 \
  --warmup_ratio 0.06 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --eval_batch_size 16 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --fp16

echo
echo "Finished Baseline 8: Full encoder contrastive fine-tuning."
