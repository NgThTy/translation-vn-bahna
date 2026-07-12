# #!/usr/bin/env bash
# set -e

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1 \
#   --output_dir results/baselines/ibm1_raw \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10 \
#   --export_fastalign_input

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1 \
#   --output_dir results/baselines/ibm1_strip_accents \
#   --strip_accents \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1 \
#   --output_dir results/baselines/ibm1_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1_sym \
#   --output_dir results/baselines/ibm1_sym_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1_sym \
#   --output_dir results/baselines/ibm1_sym_strip_accents_no_punct_lenpen \
#   --strip_accents \
#   --remove_punct \
#   --length_penalty 0.1 \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="data/train_fit.csv"
DEV_CSV="data/dev.csv"
TEST_CSV="data/test.csv"
DEV_ROOT="results/dev/ibm1"
TEST_ROOT="results/test/ibm1"
SELECTION_ROOT="results/dev_selection"

for required_file in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

# Remove stale IBM1-family outputs so only the five declared variants
# participate in this development selection run.
rm -rf "$DEV_ROOT" "$TEST_ROOT"
mkdir -p "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT"

COMMON_ARGS=(
  --train_csv "$TRAIN_CSV"
  --input_csv "$DEV_CSV"
  --split_name dev
  --ibm_iters 5
  --topk_eval 10
  --eval_ks 1 5 10
)

python src/word_alignment_baseline.py \
  "${COMMON_ARGS[@]}" \
  --method ibm1 \
  --configuration_name ibm1_raw \
  --output_dir "$DEV_ROOT/ibm1_raw" \
  --export_fastalign_input

python src/word_alignment_baseline.py \
  "${COMMON_ARGS[@]}" \
  --method ibm1 \
  --strip_accents \
  --configuration_name ibm1_strip_accents \
  --output_dir "$DEV_ROOT/ibm1_strip_accents"

python src/word_alignment_baseline.py \
  "${COMMON_ARGS[@]}" \
  --method ibm1 \
  --strip_accents \
  --remove_punct \
  --configuration_name ibm1_strip_accents_no_punct \
  --output_dir "$DEV_ROOT/ibm1_strip_accents_no_punct"

python src/word_alignment_baseline.py \
  "${COMMON_ARGS[@]}" \
  --method ibm1_sym \
  --strip_accents \
  --remove_punct \
  --configuration_name ibm1_sym_strip_accents_no_punct \
  --output_dir "$DEV_ROOT/ibm1_sym_strip_accents_no_punct"

python src/word_alignment_baseline.py \
  "${COMMON_ARGS[@]}" \
  --method ibm1_sym \
  --strip_accents \
  --remove_punct \
  --length_penalty 0.1 \
  --configuration_name ibm1_sym_strip_accents_no_punct_lenpen \
  --output_dir "$DEV_ROOT/ibm1_sym_strip_accents_no_punct_lenpen"

# The central selector scans every family under results/dev, rewrites the shared
# consolidated manifest, and evaluates only the dev-selected IBM1 variant on test.
python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root "$SELECTION_ROOT" \
  --test_csv "$TEST_CSV" \
  --evaluate_test \
  --family ibm1
