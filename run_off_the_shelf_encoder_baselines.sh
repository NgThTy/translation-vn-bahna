#!/usr/bin/env bash
set -e

# ============================================================
# Baseline 5: Off-the-shelf multilingual encoders
# ============================================================
#
# This script evaluates pretrained multilingual sentence/word encoders
# without any fine-tuning.
#
# Models:
# 1. mBERT
# 2. XLM-R
# 3. multilingual MiniLM
# 4. LaBSE
#
# Retrieval:
# - cosine similarity
# - optional CSLS variants for sentence-transformer models
#
# Note:
# LASER is not included here because it usually requires an additional
# external LASER/Fairseq installation and model download. LaBSE is included
# as the strongest easy-to-run multilingual sentence encoder baseline.
# ============================================================


# -------------------------
# 1. mBERT, raw
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name bert-base-multilingual-cased \
  --encoder_name mbert \
  --backend transformers \
  --output_dir results/baselines/offtheshelf_mbert_cosine_raw \
  --batch_size 8 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 2. mBERT, normalized
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name bert-base-multilingual-cased \
  --encoder_name mbert \
  --backend transformers \
  --output_dir results/baselines/offtheshelf_mbert_cosine_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --batch_size 8 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 3. XLM-R base, raw
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name xlm-roberta-base \
  --encoder_name xlmr_base \
  --backend transformers \
  --output_dir results/baselines/offtheshelf_xlmr_base_cosine_raw \
  --batch_size 8 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 4. XLM-R base, normalized
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name xlm-roberta-base \
  --encoder_name xlmr_base \
  --backend transformers \
  --output_dir results/baselines/offtheshelf_xlmr_base_cosine_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --batch_size 8 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 5. multilingual MiniLM, raw
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --encoder_name multilingual_minilm \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_multilingual_minilm_cosine_raw \
  --batch_size 32 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 6. multilingual MiniLM, normalized
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --encoder_name multilingual_minilm \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_multilingual_minilm_cosine_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --batch_size 32 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 7. multilingual MiniLM, CSLS normalized
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --encoder_name multilingual_minilm \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_multilingual_minilm_csls_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --use_csls \
  --csls_k 10 \
  --batch_size 32 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 8. LaBSE, raw
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/LaBSE \
  --encoder_name labse \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_labse_cosine_raw \
  --batch_size 16 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 9. LaBSE, normalized
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/LaBSE \
  --encoder_name labse \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_labse_cosine_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --batch_size 16 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10


# -------------------------
# 10. LaBSE, CSLS normalized
# -------------------------
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/LaBSE \
  --encoder_name labse \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_labse_csls_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --use_csls \
  --csls_k 10 \
  --batch_size 16 \
  --max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10
