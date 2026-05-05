#!/usr/bin/env bash
set -e

python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_cosine_raw \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_cosine_strip_accents \
  --strip_accents \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_vecmap_cosine_strip_accents \
  --strip_accents \
  --vecmap_normalize \
  --vector_size 100 \
  --window 5 \
  --min_count 1 \
  --epochs 20 \
  --sg 1 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_vecmap_csls_strip_accents \
  --strip_accents \
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

python src/fasttext_procrustes_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --lexicon_train_csv data/lexicon_train.csv \
  --output_dir results/baselines/fasttext_procrustes_vecmap_csls_strip_accents_no_punct \
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
