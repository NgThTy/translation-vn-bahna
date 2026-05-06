#!/usr/bin/env bash
set -e

# ============================================================
# Baseline 6: Previous pipeline / Projection + Procrustes
# ============================================================
#
# Recommended main setting:
# - Backbone: XLM-R
# - Epochs: 50
# - Kabsch alignment: 10K lexicon
#
# Main checkpoint:
#   results/models/b2_xlmr_50ep
#
# Main alignment:
#   results/alignment/alignment_B2_xlmr_10K_50ep
#
# This evaluates the old/current system as a baseline:
#
# encoder -> pooling -> projection head -> optional Kabsch -> retrieval
#
# Required existing files:
# - results/models/b2_xlmr_50ep/src_proj.pt
# - results/models/b2_xlmr_50ep/tgt_proj.pt
# - results/alignment/alignment_B2_xlmr_10K_50ep/R.npy
# - results/alignment/alignment_B2_xlmr_10K_50ep/t.npy
#
# Do NOT use:
#   --proj_dir results/models
#
# because src_proj.pt and tgt_proj.pt are inside checkpoint subfolders.
# ============================================================

PROJ_DIR="results/models/b2_xlmr_50ep"
ALIGN_DIR="results/alignment/alignment_B2_xlmr_10K_50ep"
SRC_MODEL="xlm-roberta-base"
TGT_MODEL="xlm-roberta-base"

echo "Checking required files..."

test -f "$PROJ_DIR/src_proj.pt" || { echo "Missing $PROJ_DIR/src_proj.pt"; exit 1; }
test -f "$PROJ_DIR/tgt_proj.pt" || { echo "Missing $PROJ_DIR/tgt_proj.pt"; exit 1; }
test -f "$ALIGN_DIR/R.npy" || { echo "Missing $ALIGN_DIR/R.npy"; exit 1; }
test -f "$ALIGN_DIR/t.npy" || { echo "Missing $ALIGN_DIR/t.npy"; exit 1; }

echo "All required files found."
echo


# -------------------------
# 1. Main previous pipeline:
#    XLM-R 50ep + 10K Kabsch + cosine
# -------------------------
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir "$PROJ_DIR" \
  --alignment_dir "$ALIGN_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_mean \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_cosine \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 2. Main previous pipeline:
#    XLM-R 50ep + 10K Kabsch + CSLS
# -------------------------
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir "$PROJ_DIR" \
  --alignment_dir "$ALIGN_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_mean \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_csls \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 3. Ablation:
#    XLM-R 50ep + no Kabsch + cosine
# -------------------------
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir "$PROJ_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_mean \
  --no_kabsch \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_cosine \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 4. Optional ablation:
#    XLM-R 50ep + no Kabsch + CSLS
# -------------------------
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir "$PROJ_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_mean \
  --no_kabsch \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_csls \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 5. Optional pooling ablation:
#    XLM-R 50ep + 10K Kabsch + token-IDF + cosine
# -------------------------
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir "$PROJ_DIR" \
  --alignment_dir "$ALIGN_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_idf \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_cosine \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 6. Optional pooling ablation:
#    XLM-R 50ep + 10K Kabsch + token-IDF + CSLS
# -------------------------
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir "$PROJ_DIR" \
  --alignment_dir "$ALIGN_DIR" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_idf \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_csls \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10

echo
echo "Finished Baseline 6: Previous pipeline / Projection + Procrustes."
