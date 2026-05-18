#!/usr/bin/env bash
set -e

# ============================================================
# Method 8: Hybrid IBM1 + XLM-R LoRA reranking
#
# IMPORTANT:
# Set NEURAL_TEST_PRED to the sentence_predictions.csv produced by
# Baseline 6:
# XLM-R 50ep + LoRA + no Kabsch + token_mean + CSLS
#
# The file must contain:
#   Bahnaric
#   Gold_VN or Vietnamese
#   TopK_Preds
#
# Optional:
#   TopK_Scores
# ============================================================

NEURAL_TEST_PRED="results/baselines/baseline6_xlmr_lora_no_kabsch_token_mean_csls/sentence_predictions.csv"

python src/hybrid_lexical_neural_rerank.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --test_neural_predictions "$NEURAL_TEST_PRED" \
  --output_dir results/baselines/hybrid_ibm1_xlmr_lora_rerank_test_fixed \
  --strip_accents \
  --remove_punct \
  --length_penalty 0.1 \
  --ibm_iters 5 \
  --alpha 0.5 \
  --beta 0.5 \
  --gamma 0.0
