#!/usr/bin/env bash
set -e

# ============================================================
# Baseline 6 additional experiments
# ============================================================
#
# Purpose:
# Add the missing no-Kabsch pooling ablations:
#
#   1. no Kabsch + token_idf + cosine
#   2. no Kabsch + token_idf + CSLS
#   3. no Kabsch + sentence_mean + cosine
#   4. no Kabsch + sentence_mean + CSLS
#
# IMPORTANT:
# The XLM-R B2 checkpoint was trained with LoRA + projection heads.
# Therefore every run must include --use_lora.
# ============================================================

PROJ_DIR="results/models/b2_xlmr_50ep"
SRC_MODEL="xlm-roberta-base"
TGT_MODEL="xlm-roberta-base"
TEST_CSV="data/test.csv"

echo "Checking required files..."

test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

test -f "$PROJ_DIR/src_proj.pt" || { echo "Missing $PROJ_DIR/src_proj.pt"; exit 1; }
test -f "$PROJ_DIR/tgt_proj.pt" || { echo "Missing $PROJ_DIR/tgt_proj.pt"; exit 1; }

test -d "$PROJ_DIR/src_adapters" || { echo "Missing $PROJ_DIR/src_adapters"; exit 1; }
test -d "$PROJ_DIR/tgt_adapters" || { echo "Missing $PROJ_DIR/tgt_adapters"; exit 1; }

test -f "$PROJ_DIR/src_adapters/adapter_config.json" || { echo "Missing $PROJ_DIR/src_adapters/adapter_config.json"; exit 1; }
test -f "$PROJ_DIR/tgt_adapters/adapter_config.json" || { echo "Missing $PROJ_DIR/tgt_adapters/adapter_config.json"; exit 1; }

echo "All required files found."
echo


# ============================================================
# 1. token_idf + no Kabsch + cosine
# encoder -> projection per token -> IDF-weighted pool -> cosine
# ============================================================
python src/previous_pipeline_baseline.py \
  --test_csv "$TEST_CSV" \
  --proj_dir "$PROJ_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_idf \
  --use_lora \
  --no_kabsch \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_token_idf_cosine_lora \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# ============================================================
# 2. token_idf + no Kabsch + CSLS
# encoder -> projection per token -> IDF-weighted pool -> CSLS
# ============================================================
python src/previous_pipeline_baseline.py \
  --test_csv "$TEST_CSV" \
  --proj_dir "$PROJ_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_idf \
  --use_lora \
  --no_kabsch \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_token_idf_csls_lora \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# ============================================================
# 3. sentence_mean + no Kabsch + cosine
# encoder -> mean pool -> projection -> cosine
# ============================================================
python src/previous_pipeline_baseline.py \
  --test_csv "$TEST_CSV" \
  --proj_dir "$PROJ_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling sentence_mean \
  --use_lora \
  --no_kabsch \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_sentence_mean_cosine_lora \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# ============================================================
# 4. sentence_mean + no Kabsch + CSLS
# encoder -> mean pool -> projection -> CSLS
# ============================================================
python src/previous_pipeline_baseline.py \
  --test_csv "$TEST_CSV" \
  --proj_dir "$PROJ_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling sentence_mean \
  --use_lora \
  --no_kabsch \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_sentence_mean_csls_lora \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10

echo
echo "Finished Baseline 6 additional no-Kabsch pooling experiments."
