#!/usr/bin/env bash
set -euo pipefail

DEV_ROOT="${DEV_ROOT:-results/dev/xlmr_lora_projection}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"

required_variants=(
  previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_cosine_lora
  previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_csls_lora
  previous_pipeline_xlmr_50ep_no_kabsch_token_mean_cosine_lora
  previous_pipeline_xlmr_50ep_no_kabsch_token_mean_csls_lora
  previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_cosine_lora
  previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_csls_lora
  previous_pipeline_xlmr_50ep_no_kabsch_token_idf_cosine_lora
  previous_pipeline_xlmr_50ep_no_kabsch_token_idf_csls_lora
  previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_cosine_lora
  previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_csls_lora
  previous_pipeline_xlmr_50ep_no_kabsch_sentence_mean_cosine_lora
  previous_pipeline_xlmr_50ep_no_kabsch_sentence_mean_csls_lora
  previous_pipeline_xlmr_50ep_10K_token_kabsch_token_mean_csls_lora
  previous_pipeline_xlmr_50ep_10K_token_kabsch_token_idf_csls_lora
)

missing=0
for variant in "${required_variants[@]}"; do
  metrics="$DEV_ROOT/$variant/metrics.json"
  if [[ ! -s "$metrics" ]]; then
    echo "Missing completed development result: $metrics" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "XLM-R LoRA development sweep is incomplete." >&2
  echo "Selection and test evaluation were not run." >&2
  exit 2
fi

if [[ ! -f "$TEST_CSV" ]]; then
  echo "Missing held-out test CSV: $TEST_CSV" >&2
  exit 1
fi

selector_args=(
  --dev_root results/dev
  --test_root results/test
  --output_root "$SELECTION_ROOT"
  --test_csv "$TEST_CSV"
  --evaluate_test
  --family xlmr_lora_projection
)

if [[ "${FORCE_TEST:-0}" == "1" ]]; then
  selector_args+=(--force_test)
fi

python src/select_dev_configs_and_evaluate_test.py "${selector_args[@]}"
