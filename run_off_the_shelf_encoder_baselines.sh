# #!/usr/bin/env bash
# set -e

# # ============================================================
# # Baseline 5: Off-the-shelf multilingual encoders
# # ============================================================
# #
# # This script evaluates pretrained multilingual sentence/word encoders
# # without any fine-tuning.
# #
# # Models:
# # 1. mBERT
# # 2. XLM-R
# # 3. multilingual MiniLM
# # 4. LaBSE
# #
# # Retrieval:
# # - cosine similarity
# # - optional CSLS variants for sentence-transformer models
# #
# # Note:
# # LASER is not included here because it usually requires an additional
# # external LASER/Fairseq installation and model download. LaBSE is included
# # as the strongest easy-to-run multilingual sentence encoder baseline.
# # ============================================================


# # -------------------------
# # 1. mBERT, raw
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name bert-base-multilingual-cased \
#   --encoder_name mbert \
#   --backend transformers \
#   --output_dir results/baselines/offtheshelf_mbert_cosine_raw \
#   --batch_size 8 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 2. mBERT, normalized
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name bert-base-multilingual-cased \
#   --encoder_name mbert \
#   --backend transformers \
#   --output_dir results/baselines/offtheshelf_mbert_cosine_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --batch_size 8 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 3. XLM-R base, raw
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name xlm-roberta-base \
#   --encoder_name xlmr_base \
#   --backend transformers \
#   --output_dir results/baselines/offtheshelf_xlmr_base_cosine_raw \
#   --batch_size 8 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 4. XLM-R base, normalized
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name xlm-roberta-base \
#   --encoder_name xlmr_base \
#   --backend transformers \
#   --output_dir results/baselines/offtheshelf_xlmr_base_cosine_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --batch_size 8 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 5. multilingual MiniLM, raw
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
#   --encoder_name multilingual_minilm \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_multilingual_minilm_cosine_raw \
#   --batch_size 32 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 6. multilingual MiniLM, normalized
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
#   --encoder_name multilingual_minilm \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_multilingual_minilm_cosine_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --batch_size 32 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 7. multilingual MiniLM, CSLS normalized
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
#   --encoder_name multilingual_minilm \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_multilingual_minilm_csls_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --use_csls \
#   --csls_k 10 \
#   --batch_size 32 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 8. LaBSE, raw
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/LaBSE \
#   --encoder_name labse \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_labse_cosine_raw \
#   --batch_size 16 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 9. LaBSE, normalized
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/LaBSE \
#   --encoder_name labse \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_labse_cosine_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --batch_size 16 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # -------------------------
# # 10. LaBSE, CSLS normalized
# # -------------------------
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/LaBSE \
#   --encoder_name labse \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_labse_csls_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --use_csls \
#   --csls_k 10 \
#   --batch_size 16 \
#   --max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

#################################################
#!/usr/bin/env bash
set -euo pipefail

DEV_CSV="data/dev.csv"
TEST_CSV="data/test.csv"
DEV_ROOT="results/dev/off_the_shelf"
TEST_ROOT="results/test/off_the_shelf"
SELECTION_ROOT="results/dev_selection"
CACHE_ROOT="results/cache/off_the_shelf_embeddings"

VARIANTS=(
  offtheshelf_mbert_cosine_raw
  offtheshelf_mbert_cosine_strip_accents_no_punct
  offtheshelf_xlmr_base_cosine_raw
  offtheshelf_xlmr_base_cosine_strip_accents_no_punct
  offtheshelf_multilingual_minilm_cosine_raw
  offtheshelf_multilingual_minilm_cosine_strip_accents_no_punct
  offtheshelf_multilingual_minilm_csls_strip_accents_no_punct
  offtheshelf_labse_cosine_raw
  offtheshelf_labse_cosine_strip_accents_no_punct
  offtheshelf_labse_csls_strip_accents_no_punct
)

for required_file in "$DEV_CSV" "$TEST_CSV"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

# Preserve completed development runs so the script can be resumed across
# multiple short server allocations.
mkdir -p "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT" "$CACHE_ROOT"

COMMON_ARGS=(
  --input_csv "$DEV_CSV"
  --split_name dev
  --topk_eval 10
  --eval_ks 1 5 10
  --csls_k 10
  --embedding_cache_dir "$CACHE_ROOT"
)

# Set OFFSHELF_NO_CUDA=1 when a CPU-only run is required.
RUNTIME_ARGS=()
if [[ "${OFFSHELF_NO_CUDA:-0}" == "1" ]]; then
  RUNTIME_ARGS+=(--no_cuda)
fi
if [[ "${OFFSHELF_NO_MPS:-0}" == "1" ]]; then
  RUNTIME_ARGS+=(--no_mps)
fi

run_variant() {
  local name="$1"
  shift
  local output_dir="$DEV_ROOT/$name"
  local metrics_file="$output_dir/metrics.json"

  if [[ -s "$metrics_file" ]]; then
    echo "[SKIP] Completed variant: $name"
    return
  fi

  if [[ -d "$output_dir" ]]; then
    echo "[CLEAN] Removing incomplete output: $output_dir"
    rm -rf "$output_dir"
  fi

  echo "[RUN] $name"
  python src/multilingual_encoder_baseline.py \
    "${COMMON_ARGS[@]}" \
    "${RUNTIME_ARGS[@]}" \
    --configuration_name "$name" \
    --output_dir "$output_dir" \
    "$@"
}

run_named_variant() {
  local name="$1"
  case "$name" in
    offtheshelf_mbert_cosine_raw)
      run_variant "$name" \
        --model_name bert-base-multilingual-cased \
        --encoder_name mbert \
        --backend transformers \
        --batch_size 8 \
        --max_len 256
      ;;
    offtheshelf_mbert_cosine_strip_accents_no_punct)
      run_variant "$name" \
        --model_name bert-base-multilingual-cased \
        --encoder_name mbert \
        --backend transformers \
        --strip_accents \
        --remove_punct \
        --batch_size 8 \
        --max_len 256
      ;;
    offtheshelf_xlmr_base_cosine_raw)
      run_variant "$name" \
        --model_name xlm-roberta-base \
        --encoder_name xlmr_base \
        --backend transformers \
        --batch_size 8 \
        --max_len 256
      ;;
    offtheshelf_xlmr_base_cosine_strip_accents_no_punct)
      run_variant "$name" \
        --model_name xlm-roberta-base \
        --encoder_name xlmr_base \
        --backend transformers \
        --strip_accents \
        --remove_punct \
        --batch_size 8 \
        --max_len 256
      ;;
    offtheshelf_multilingual_minilm_cosine_raw)
      run_variant "$name" \
        --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --encoder_name multilingual_minilm \
        --backend sentence_transformers \
        --batch_size 32 \
        --max_len 256
      ;;
    offtheshelf_multilingual_minilm_cosine_strip_accents_no_punct)
      run_variant "$name" \
        --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --encoder_name multilingual_minilm \
        --backend sentence_transformers \
        --strip_accents \
        --remove_punct \
        --batch_size 32 \
        --max_len 256
      ;;
    offtheshelf_multilingual_minilm_csls_strip_accents_no_punct)
      run_variant "$name" \
        --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --encoder_name multilingual_minilm \
        --backend sentence_transformers \
        --strip_accents \
        --remove_punct \
        --use_csls \
        --batch_size 32 \
        --max_len 256
      ;;
    offtheshelf_labse_cosine_raw)
      run_variant "$name" \
        --model_name sentence-transformers/LaBSE \
        --encoder_name labse \
        --backend sentence_transformers \
        --batch_size 16 \
        --max_len 256
      ;;
    offtheshelf_labse_cosine_strip_accents_no_punct)
      run_variant "$name" \
        --model_name sentence-transformers/LaBSE \
        --encoder_name labse \
        --backend sentence_transformers \
        --strip_accents \
        --remove_punct \
        --batch_size 16 \
        --max_len 256
      ;;
    offtheshelf_labse_csls_strip_accents_no_punct)
      run_variant "$name" \
        --model_name sentence-transformers/LaBSE \
        --encoder_name labse \
        --backend sentence_transformers \
        --strip_accents \
        --remove_punct \
        --use_csls \
        --batch_size 16 \
        --max_len 256
      ;;
    *)
      echo "Unknown off-the-shelf variant: $name" >&2
      exit 1
      ;;
  esac
}

# Run exactly one development variant in a short allocation, for example:
# OFFSHELF_VARIANT=offtheshelf_labse_cosine_raw \
#   bash run_off_the_shelf_encoder_baselines.sh
if [[ -n "${OFFSHELF_VARIANT:-}" ]]; then
  run_named_variant "$OFFSHELF_VARIANT"
  exit 0
fi

# Otherwise resume the full development sweep, skipping completed variants.
for variant in "${VARIANTS[@]}"; do
  run_named_variant "$variant"
done

# Do not select from an incomplete set.
missing=0
for variant in "${VARIANTS[@]}"; do
  metrics_file="$DEV_ROOT/$variant/metrics.json"
  if [[ ! -s "$metrics_file" ]]; then
    echo "Missing completed result: $metrics_file" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Off-the-shelf development experiments are incomplete." >&2
  echo "Selection and test evaluation will not run." >&2
  exit 2
fi

# Keep exactly one current dev-selected test result for this family.
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"

python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root "$SELECTION_ROOT" \
  --test_csv "$TEST_CSV" \
  --evaluate_test \
  --family off_the_shelf
