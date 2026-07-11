# #!/usr/bin/env bash
# set -e

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_ratio \
#   --output_dir results/baselines/levenshtein_ratio_raw \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_ratio \
#   --output_dir results/baselines/levenshtein_ratio_strip_accents \
#   --strip_accents \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_ratio \
#   --output_dir results/baselines/levenshtein_ratio_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_ratio \
#   --output_dir results/baselines/levenshtein_ratio_strip_accents_no_punct_no_spaces \
#   --strip_accents \
#   --remove_punct \
#   --remove_spaces \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_distance \
#   --output_dir results/baselines/levenshtein_distance_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method token_jaccard \
#   --output_dir results/baselines/token_jaccard_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
DEV_ROOT="${DEV_ROOT:-results/dev/edit_distance}"
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
  echo "=== DEV edit-distance configuration: $name ==="
  "$PYTHON" src/edit_distance_retrieval_baseline.py \
    --input_csv "$DEV_CSV" \
    --split_name dev \
    --configuration_name "$name" \
    --output_dir "$DEV_ROOT/$name" \
    --topk_eval 10 \
    --eval_ks 1 5 10 \
    "$@"
}

# These are the original method/preprocessing variants. Every variant is run on
# dev; the test set is evaluated once using only the dev-selected winner.
run_variant "levenshtein_ratio_raw" \
  --method levenshtein_ratio

run_variant "levenshtein_ratio_strip_accents" \
  --method levenshtein_ratio \
  --strip_accents

run_variant "levenshtein_ratio_strip_accents_no_punct" \
  --method levenshtein_ratio \
  --strip_accents --remove_punct

run_variant "levenshtein_ratio_strip_accents_no_punct_no_spaces" \
  --method levenshtein_ratio \
  --strip_accents --remove_punct --remove_spaces

run_variant "levenshtein_distance_strip_accents_no_punct" \
  --method levenshtein_distance \
  --strip_accents --remove_punct

run_variant "token_jaccard_strip_accents_no_punct" \
  --method token_jaccard \
  --strip_accents --remove_punct

"$PYTHON" src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root "$SELECTION_ROOT" \
  --test_csv "$TEST_CSV" \
  --family edit_distance \
  --evaluate_test
