#!/usr/bin/env bash
set -e

python src/word_alignment_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --method ibm1 \
  --output_dir results/baselines/ibm1_raw \
  --ibm_iters 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --export_fastalign_input

python src/word_alignment_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --method ibm1 \
  --output_dir results/baselines/ibm1_strip_accents \
  --strip_accents \
  --ibm_iters 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/word_alignment_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --method ibm1 \
  --output_dir results/baselines/ibm1_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --ibm_iters 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/word_alignment_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --method ibm1_sym \
  --output_dir results/baselines/ibm1_sym_strip_accents_no_punct \
  --strip_accents \
  --remove_punct \
  --ibm_iters 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/word_alignment_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --method ibm1_sym \
  --output_dir results/baselines/ibm1_sym_strip_accents_no_punct_lenpen \
  --strip_accents \
  --remove_punct \
  --length_penalty 0.1 \
  --ibm_iters 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10
