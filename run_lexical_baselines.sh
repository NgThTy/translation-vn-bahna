#!/usr/bin/env bash
set -e

python src/lexical_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method tfidf_char \
  --output_dir results/baselines/tfidf_char_2_5 \
  --ngram_min 2 \
  --ngram_max 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/lexical_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method bm25_char \
  --output_dir results/baselines/bm25_char_2_5 \
  --ngram_min 2 \
  --ngram_max 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/lexical_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method tfidf_char \
  --output_dir results/baselines/tfidf_char_3_6 \
  --ngram_min 3 \
  --ngram_max 6 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/lexical_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method bm25_char \
  --output_dir results/baselines/bm25_char_3_6 \
  --ngram_min 3 \
  --ngram_max 6 \
  --topk_eval 10 \
  --eval_ks 1 5 10
