#!/usr/bin/env bash
set -e

# ============================================================
# Method 8 for FLORES generalization:
# Hybrid IBM1 + Levenshtein + XLM-R LoRA reranking
#
# Stage 1:
#   Existing LoRA/projection + no Kabsch + CSLS candidate generation.
#
# Stage 2:
#   IBM1 / Levenshtein / neural-score hybrid reranking.
#
# This matches the Bahnaric-Vietnamese Method 8 design.
# ============================================================

PAIRS=(
  "khmkhmr_vielatn"
  "laolaoo_vielatn"
)

for PAIR in "${PAIRS[@]}"; do
  echo "============================================================"
  echo "Method 8 hybrid reranking for FLORES pair: $PAIR"
  echo "============================================================"

  TRAIN_CSV="data/flores_generalization/${PAIR}/train.csv"
  TEST_CSV="data/flores_generalization/${PAIR}/test.csv"

  NEURAL_TEST_PRED="results/baselines/flores_${PAIR}_b6style_lora_projection_10ep_token_mean_both/test_csls/sentence_predictions.csv"

  HYBRID_INPUT="results/baselines/flores_${PAIR}_b6style_lora_projection_10ep_token_mean_both/test_csls/sentence_predictions_for_hybrid.csv"

  test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
  test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }
  test -f "$NEURAL_TEST_PRED" || { echo "Missing $NEURAL_TEST_PRED"; exit 1; }
  test -f "src/hybrid_lexical_neural_rerank.py" || { echo "Missing src/hybrid_lexical_neural_rerank.py"; exit 1; }

  echo "Converting TopK_indices to TopK_Preds..."
  python src/convert_flores_topk_indices_to_preds.py \
    --test_csv "$TEST_CSV" \
    --neural_predictions_csv "$NEURAL_TEST_PRED" \
    --output_csv "$HYBRID_INPUT"

  echo "Running Method 8 hybrid reranking..."
  python src/hybrid_lexical_neural_rerank.py \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --test_neural_predictions "$HYBRID_INPUT" \
    --output_dir "results/baselines/flores_${PAIR}_hybrid_ibm1_xlmr_lora_no_kabsch_csls_rerank" \
    --strip_accents \
    --remove_punct \
    --length_penalty 0.1 \
    --ibm_iters 5 \
    --alpha 0.5 \
    --beta 0.5 \
    --gamma 0.0

done

echo
echo "Finished FLORES Method 8 hybrid IBM1/neural reranking."
