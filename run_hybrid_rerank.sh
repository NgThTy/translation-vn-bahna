#!/usr/bin/env bash
set -e

# Method 8: Hybrid IBM1 + XLM-R LoRA reranking
# Neural candidate generator:
# XLM-R 50ep + LoRA + no Kabsch + CSLS

NEURAL_TEST_PRED="results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_csls_lora/sentence_predictions.csv"

python src/hybrid_lexical_neural_rerank.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --test_neural_predictions "$NEURAL_TEST_PRED" \
  --output_dir results/baselines/hybrid_ibm1_xlmr_lora_no_kabsch_csls_rerank_test_fixed \
  --strip_accents \
  --remove_punct \
  --length_penalty 0.1 \
  --ibm_iters 5 \
  --alpha 0.5 \
  --beta 0.5 \
  --gamma 0.0
