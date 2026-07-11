# #!/usr/bin/env bash
# set -e

# python src/lexical_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method tfidf_char \
#   --output_dir results/baselines/tfidf_char_2_5 \
#   --ngram_min 2 \
#   --ngram_max 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/lexical_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method bm25_char \
#   --output_dir results/baselines/bm25_char_2_5 \
#   --ngram_min 2 \
#   --ngram_max 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/lexical_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method tfidf_char \
#   --output_dir results/baselines/tfidf_char_3_6 \
#   --ngram_min 3 \
#   --ngram_max 6 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/lexical_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method bm25_char \
#   --output_dir results/baselines/bm25_char_3_6 \
#   --ngram_min 3 \
#   --ngram_max 6 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
DEV_ROOT="${DEV_ROOT:-results/dev/lexical}"
SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"

if [[ ! -f "$DEV_CSV" ]]; then
  echo "Development split not found; creating data/train_fit.csv and data/dev.csv with seed 42."
  "$PYTHON" src/prepare_train_dev_split.py \
    --input_csv data/train.csv \
    --train_output data/train_fit.csv \
    --dev_output "$DEV_CSV" \
    --manifest_output data/train_dev_split_manifest.json \
    --dev_ratio 0.10 \
    --seed 42
fi

run_variant() {
  local name="$1"
  shift
  echo "=== DEV lexical configuration: $name ==="
  "$PYTHON" src/lexical_retrieval_baseline.py \
    --input_csv "$DEV_CSV" \
    --split_name dev \
    --configuration_name "$name" \
    --output_dir "$DEV_ROOT/$name" \
    --topk_eval 10 \
    --eval_ks 1 5 10 \
    "$@"
}

# All method, character n-gram, and preprocessing choices are compared only on dev.
for method in tfidf_char bm25_char; do
  for ngrams in "2 5" "3 6"; do
    read -r nmin nmax <<< "$ngrams"

    run_variant "${method}_${nmin}_${nmax}_raw" \
      --method "$method" --ngram_min "$nmin" --ngram_max "$nmax"

    run_variant "${method}_${nmin}_${nmax}_strip_accents" \
      --method "$method" --ngram_min "$nmin" --ngram_max "$nmax" \
      --strip_accents

    run_variant "${method}_${nmin}_${nmax}_strip_accents_no_punct" \
      --method "$method" --ngram_min "$nmin" --ngram_max "$nmax" \
      --strip_accents --remove_punct

    run_variant "${method}_${nmin}_${nmax}_strip_accents_no_punct_no_spaces" \
      --method "$method" --ngram_min "$nmin" --ngram_max "$nmax" \
      --strip_accents --remove_punct --remove_spaces
  done
done

# This command writes the dev-selection manifest and evaluates exactly one lexical
# configuration on the held-out test pool.
"$PYTHON" src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root "$SELECTION_ROOT" \
  --test_csv "$TEST_CSV" \
  --family lexical \
  --evaluate_test
