#!/usr/bin/env bash
set -e

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_ratio \
  --output_dir results/baselines/levenshtein_ratio_raw \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_ratio \
  --output_dir results/baselines/levenshtein_ratio_strip_accents \
  --strip_accents \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_ratio \
  --output_dir results/baselines/levenshtein_ratio_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_ratio \
  --output_dir results/baselines/levenshtein_ratio_strip_accents_no_punct_no_spaces \
  --strip_accents \
  --remove_punct \
  --remove_spaces \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_distance \
  --output_dir results/baselines/levenshtein_distance_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method token_jaccard \
  --output_dir results/baselines/token_jaccard_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --topk_eval 10 \
  --eval_ks 1 5 10
