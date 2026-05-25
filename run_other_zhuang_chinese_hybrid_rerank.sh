#!/usr/bin/env bash
set -e

# ============================================================
# Zhuang-Chinese Method 8:
# Hybrid IBM1 + neural LoRA/projection reranking
# ============================================================

TRAIN_CSV="data/other_low_resource_generalization/zhuang_chinese/train.csv"
TEST_CSV="data/other_low_resource_generalization/zhuang_chinese/test.csv"

NEURAL_TEST_PRED="results/baselines/other_zhuang_chinese_b6style_lora_projection_10ep_token_mean_both/test_csls/sentence_predictions.csv"
HYBRID_INPUT="results/baselines/other_zhuang_chinese_b6style_lora_projection_10ep_token_mean_both/test_csls/sentence_predictions_for_hybrid.csv"

OUTPUT_DIR="results/baselines/other_zhuang_chinese_hybrid_ibm1_xlmr_lora_no_kabsch_csls_rerank"

echo "Checking input files..."
test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }
test -f "$NEURAL_TEST_PRED" || { echo "Missing $NEURAL_TEST_PRED"; exit 1; }
test -f "src/convert_flores_topk_indices_to_preds.py" || { echo "Missing src/convert_flores_topk_indices_to_preds.py"; exit 1; }
test -f "src/hybrid_lexical_neural_rerank.py" || { echo "Missing src/hybrid_lexical_neural_rerank.py"; exit 1; }

mkdir -p "$OUTPUT_DIR"

echo "Train CSV: $TRAIN_CSV"
echo "Test CSV: $TEST_CSV"
echo "Neural prediction file: $NEURAL_TEST_PRED"
echo "Hybrid input file: $HYBRID_INPUT"
echo "Output dir: $OUTPUT_DIR"
echo

echo "Step 1: Convert TopK_indices to TopK_Preds..."
python src/convert_flores_topk_indices_to_preds.py \
  --test_csv "$TEST_CSV" \
  --neural_predictions_csv "$NEURAL_TEST_PRED" \
  --output_csv "$HYBRID_INPUT"

echo
echo "Step 2: Run hybrid IBM1 + neural reranking..."
python src/hybrid_lexical_neural_rerank.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --test_neural_predictions "$HYBRID_INPUT" \
  --output_dir "$OUTPUT_DIR" \
  --strip_accents \
  --remove_punct \
  --length_penalty 0.1 \
  --ibm_iters 5 \
  --alpha 0.5 \
  --beta 0.5 \
  --gamma 0.0

echo
echo "Finished Zhuang-Chinese hybrid IBM1/neural reranking."
