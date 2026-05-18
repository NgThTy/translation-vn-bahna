#!/usr/bin/env bash
set -e

# ============================================================
# Reference: original-style raw baseline
# phrase-level lexicon supervision, mean pooling, cosine
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_phrase_mean_cosine_raw \
  --align_unit phrase \
  --pooling mean \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ============================================================
# Previous best-style setting
# phrase-level lexicon supervision, mean pooling, VecMap + CSLS
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_phrase_mean_vecmap_csls_strip_accents_no_punct \
  --align_unit phrase \
  --pooling mean \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize \
  --use_csls \
  --csls_k 10 \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ============================================================
# Better sentence aggregation
# phrase-level lexicon supervision, IDF pooling
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_phrase_idf_vecmap_csls_strip_accents_no_punct \
  --align_unit phrase \
  --pooling idf \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize \
  --use_csls \
  --csls_k 10 \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ============================================================
# Cleaner Procrustes supervision
# token-level one-to-one lexicon supervision, mean pooling
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_token_mean_vecmap_csls_strip_accents_no_punct \
  --align_unit token \
  --pooling mean \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize \
  --use_csls \
  --csls_k 10 \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ============================================================
# Proposed best variant
# token-level Procrustes, IDF pooling, map after pooling
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_token_idf_vecmap_csls_strip_accents_no_punct \
  --align_unit token \
  --pooling idf \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize \
  --use_csls \
  --csls_k 10 \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ============================================================
# Explicit phrase/token mapping variant
# map each Bahnaric token before pooling
# This should be similar to map-after-pooling for mean pooling,
# but it is useful to report/check.
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_token_idf_map_before_pool_vecmap_csls_strip_accents_no_punct \
  --align_unit token \
  --pooling idf \
  --map_before_pool \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize \
  --use_csls \
  --csls_k 10 \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

# ============================================================
# Keep accents, remove punctuation
# Previous results showed strip_accents alone hurt, so check this.
# ============================================================
python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_token_idf_vecmap_csls_no_punct_keep_accents \
  --align_unit token \
  --pooling idf \
  --remove_punct \
  --vecmap_normalize \
  --use_csls \
  --csls_k 10 \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10
