# #!/usr/bin/env bash
# set -e

# # ============================================================
# # Reference: original-style raw baseline
# # phrase-level lexicon supervision, mean pooling, cosine
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_phrase_mean_cosine_raw \
#   --align_unit phrase \
#   --pooling mean \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # ============================================================
# # Previous best-style setting
# # phrase-level lexicon supervision, mean pooling, VecMap + CSLS
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_phrase_mean_vecmap_csls_strip_accents_no_punct \
#   --align_unit phrase \
#   --pooling mean \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10 \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # ============================================================
# # Better sentence aggregation
# # phrase-level lexicon supervision, IDF pooling
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_phrase_idf_vecmap_csls_strip_accents_no_punct \
#   --align_unit phrase \
#   --pooling idf \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10 \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # ============================================================
# # Cleaner Procrustes supervision
# # token-level one-to-one lexicon supervision, mean pooling
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_token_mean_vecmap_csls_strip_accents_no_punct \
#   --align_unit token \
#   --pooling mean \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10 \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # ============================================================
# # Proposed best variant
# # token-level Procrustes, IDF pooling, map after pooling
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_token_idf_vecmap_csls_strip_accents_no_punct \
#   --align_unit token \
#   --pooling idf \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10 \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # ============================================================
# # Explicit phrase/token mapping variant
# # map each Bahnaric token before pooling
# # This should be similar to map-after-pooling for mean pooling,
# # but it is useful to report/check.
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_token_idf_map_before_pool_vecmap_csls_strip_accents_no_punct \
#   --align_unit token \
#   --pooling idf \
#   --map_before_pool \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10 \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # ============================================================
# # Keep accents, remove punctuation
# # Previous results showed strip_accents alone hurt, so check this.
# # ============================================================
# python src/fasttext_procrustes_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --lexicon_train_csv data/lexicon_train.csv \
#   --output_dir results/baselines/fasttext_procrustes_token_idf_vecmap_csls_no_punct_keep_accents \
#   --align_unit token \
#   --pooling idf \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10 \
#   --vector_size 100 \
#   --window 5 \
#   --min_count 1 \
#   --epochs 20 \
#   --sg 1 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# #!/usr/bin/env bash
# set -euo pipefail

# PYTHON_BIN="${PYTHON_BIN:-python}"
# TRAIN_CSV="data/train_fit.csv"
# DEV_CSV="data/dev.csv"
# FAMILY="fasttext_procrustes"
# DEV_ROOT="results/dev/${FAMILY}"

# for required_file in "$TRAIN_CSV" "$DEV_CSV"; do
#   if [[ ! -f "$required_file" ]]; then
#     echo "Missing required file: $required_file" >&2
#     echo "Create the deterministic split first with src/prepare_train_dev_split.py." >&2
#     exit 1
#   fi
# done

# run_dev_variant() {
#   local configuration_name="$1"
#   shift

#   echo
#   echo "============================================================"
#   echo "Development evaluation: ${configuration_name}"
#   echo "============================================================"

#   "$PYTHON_BIN" src/fasttext_procrustes_baseline.py \
#     --train_csv "$TRAIN_CSV" \
#     --alignment_csv "$TRAIN_CSV" \
#     --input_csv "$DEV_CSV" \
#     --split_name dev \
#     --configuration_name "$configuration_name" \
#     --output_dir "${DEV_ROOT}/${configuration_name}" \
#     --vector_size 100 \
#     --window 5 \
#     --min_count 1 \
#     --epochs 20 \
#     --sg 1 \
#     --workers 1 \
#     --seed 42 \
#     --topk_eval 10 \
#     --eval_ks 1 5 10 \
#     "$@"
# }

# # Original-style raw baseline: phrase alignment, mean pooling, cosine.
# run_dev_variant \
#   fasttext_procrustes_phrase_mean_cosine_raw \
#   --align_unit phrase \
#   --pooling mean

# # Phrase alignment, mean pooling, VecMap normalization, CSLS.
# run_dev_variant \
#   fasttext_procrustes_phrase_mean_vecmap_csls_strip_accents_no_punct \
#   --align_unit phrase \
#   --pooling mean \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10

# # Phrase alignment with IDF pooling.
# run_dev_variant \
#   fasttext_procrustes_phrase_idf_vecmap_csls_strip_accents_no_punct \
#   --align_unit phrase \
#   --pooling idf \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10

# # Token-level one-to-one alignment with mean pooling.
# run_dev_variant \
#   fasttext_procrustes_token_mean_vecmap_csls_strip_accents_no_punct \
#   --align_unit token \
#   --pooling mean \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10

# # Token-level alignment with IDF pooling and mapping after pooling.
# run_dev_variant \
#   fasttext_procrustes_token_idf_vecmap_csls_strip_accents_no_punct \
#   --align_unit token \
#   --pooling idf \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10

# # Token-level alignment with mapping before pooling.
# run_dev_variant \
#   fasttext_procrustes_token_idf_map_before_pool_vecmap_csls_strip_accents_no_punct \
#   --align_unit token \
#   --pooling idf \
#   --map_before_pool \
#   --strip_accents \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10

# # Keep accents while removing punctuation.
# run_dev_variant \
#   fasttext_procrustes_token_idf_vecmap_csls_no_punct_keep_accents \
#   --align_unit token \
#   --pooling idf \
#   --remove_punct \
#   --vecmap_normalize \
#   --use_csls \
#   --csls_k 10

# # Select exactly one FastText/Procrustes configuration on development data,
# # then retrain that same configuration on train_fit only and evaluate it once
# # on the untouched held-out test set.
# "$PYTHON_BIN" src/select_dev_configs_and_evaluate_test.py \
#   --dev_root results/dev \
#   --test_root results/test \
#   --output_root results/dev_selection \
#   --test_csv data/test.csv \
#   --selection_metric Top1_acc \
#   --tie_breakers MRR Recall@5 \
#   --family "$FAMILY" \
#   --evaluate_test

# echo
# echo "Selected FastText/Procrustes configuration:"
# "$PYTHON_BIN" - <<'PY'
# import json
# from pathlib import Path

# path = Path("results/dev_selection/selected_configs.json")
# selected = json.loads(path.read_text(encoding="utf-8"))["fasttext_procrustes"]
# print(json.dumps(selected, ensure_ascii=False, indent=2))
# PY













#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
FAMILY="fasttext_procrustes"
DEV_ROOT="${DEV_ROOT:-results/dev/${FAMILY}}"
SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"
NEIGHBORHOOD_K="${NEIGHBORHOOD_K:-10}"
RETRIEVALS=(cosine csls margin_ratio)

for required_file in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    echo "Create the deterministic split first with src/prepare_train_dev_split.py." >&2
    exit 1
  fi
done

mkdir -p "$DEV_ROOT" "$SELECTION_ROOT"

run_dev_variant() {
  local configuration_name="$1"
  local retrieval="$2"
  shift 2

  local output_dir="${DEV_ROOT}/${configuration_name}"
  local metrics_file="${output_dir}/metrics.json"

  if [[ -s "$metrics_file" ]]; then
    echo "[SKIP] Completed variant: ${configuration_name}"
    return
  fi

  if [[ -d "$output_dir" ]]; then
    echo "[CLEAN] Removing incomplete output: $output_dir"
    rm -rf "$output_dir"
  fi

  echo
  echo "============================================================"
  echo "Development evaluation: ${configuration_name}"
  echo "Retrieval criterion: ${retrieval}; neighborhood k=${NEIGHBORHOOD_K}"
  echo "============================================================"

  "$PYTHON_BIN" src/fasttext_procrustes_baseline.py \
    --train_csv "$TRAIN_CSV" \
    --alignment_csv "$TRAIN_CSV" \
    --input_csv "$DEV_CSV" \
    --split_name dev \
    --configuration_name "$configuration_name" \
    --output_dir "$output_dir" \
    --vector_size 100 \
    --window 5 \
    --min_count 1 \
    --epochs 20 \
    --sg 1 \
    --workers 1 \
    --seed 42 \
    --topk_eval 10 \
    --eval_ks 1 5 10 \
    --retrieval "$retrieval" \
    --neighborhood_k "$NEIGHBORHOOD_K" \
    "$@"
}

run_retrieval_triplet() {
  local prefix="$1"
  local suffix="$2"
  shift 2

  local retrieval configuration_name
  for retrieval in "${RETRIEVALS[@]}"; do
    configuration_name="${prefix}_${retrieval}_${suffix}"
    run_dev_variant "$configuration_name" "$retrieval" "$@"
  done
}

# For every non-retrieval configuration, compare cosine, CSLS, and ratio margin
# on the complete development candidate pool. This isolates the retrieval
# criterion while keeping training, alignment, pooling, and preprocessing fixed.

# Original-style raw baseline: phrase alignment and mean pooling.
run_retrieval_triplet \
  fasttext_procrustes_phrase_mean \
  raw \
  --align_unit phrase \
  --pooling mean

# Phrase alignment, mean pooling, and VecMap normalization.
run_retrieval_triplet \
  fasttext_procrustes_phrase_mean_vecmap \
  strip_accents_no_punct \
  --align_unit phrase \
  --pooling mean \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize

# Phrase alignment with IDF pooling.
run_retrieval_triplet \
  fasttext_procrustes_phrase_idf_vecmap \
  strip_accents_no_punct \
  --align_unit phrase \
  --pooling idf \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize

# Token-level one-to-one alignment with mean pooling.
run_retrieval_triplet \
  fasttext_procrustes_token_mean_vecmap \
  strip_accents_no_punct \
  --align_unit token \
  --pooling mean \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize

# Token-level alignment with IDF pooling and mapping after pooling.
run_retrieval_triplet \
  fasttext_procrustes_token_idf_vecmap \
  strip_accents_no_punct \
  --align_unit token \
  --pooling idf \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize

# Token-level alignment with mapping before pooling.
run_retrieval_triplet \
  fasttext_procrustes_token_idf_map_before_pool_vecmap \
  strip_accents_no_punct \
  --align_unit token \
  --pooling idf \
  --map_before_pool \
  --strip_accents \
  --remove_punct \
  --vecmap_normalize

# Keep accents while removing punctuation.
run_retrieval_triplet \
  fasttext_procrustes_token_idf_vecmap \
  no_punct_keep_accents \
  --align_unit token \
  --pooling idf \
  --remove_punct \
  --vecmap_normalize

# Select exactly one FastText/Procrustes configuration on development data and
# evaluate only that configuration once on the untouched test pool.
"$PYTHON_BIN" src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root "$SELECTION_ROOT" \
  --test_csv "$TEST_CSV" \
  --selection_metric Top1_acc \
  --tie_breakers MRR Recall@5 \
  --family "$FAMILY" \
  --evaluate_test

echo
echo "Selected FastText/Procrustes configuration:"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("results/dev_selection/selected_configs.json")
selected = json.loads(path.read_text(encoding="utf-8"))["fasttext_procrustes"]
print(json.dumps(selected, ensure_ascii=False, indent=2))
PY
