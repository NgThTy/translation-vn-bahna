#!/usr/bin/env bash
set -euo pipefail

# Core development variants. Every fixed checkpoint/pooling/Kabsch setup is
# compared under cosine, CSLS, and Artetxe-Schwenk ratio-margin retrieval.
# This script remains dev-only and never touches test.csv.
#
# Run one named variant with:
#   XLMR_VARIANT=previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_margin_ratio_lora \
#     bash run_previous_pipeline_baselines.sh

PROJ_DIR="${PROJ_DIR:-results/models/reviewer_xlmr_50ep/checkpoint_final}"
ALIGN_DIR="${ALIGN_DIR:-results/alignment/reviewer_xlmr_10K_50ep_train_fit}"
NEIGHBORHOOD_K="${NEIGHBORHOOD_K:-10}"

RETRIEVALS=(cosine csls margin_ratio)

BASE_VARIANTS=(
  previous_pipeline_xlmr_50ep_10K_kabsch_token_mean
  previous_pipeline_xlmr_50ep_no_kabsch_token_mean
  previous_pipeline_xlmr_50ep_10K_kabsch_token_idf
)

if [[ ! -f "$PROJ_DIR/training_manifest.json" ]]; then
  echo "Missing reviewer-compliant checkpoint manifest:" >&2
  echo "  $PROJ_DIR/training_manifest.json" >&2
  echo "Set PROJ_DIR to the verified checkpoint directory." >&2
  exit 1
fi

run_variant() {
  local variant="$1"
  local require_alignment

  case "$variant" in
    previous_pipeline_xlmr_50ep_no_kabsch_*)
      require_alignment=0
      ;;

    previous_pipeline_xlmr_50ep_10K_kabsch_* | \
    previous_pipeline_xlmr_50ep_10K_token_kabsch_*)
      require_alignment=1
      ;;

    *)
      echo "Cannot determine alignment requirement for variant:" >&2
      echo "  $variant" >&2
      exit 2
      ;;
  esac

  PROJ_DIR="$PROJ_DIR" \
  ALIGN_DIR="$ALIGN_DIR" \
  REQUIRE_ALIGNMENT_MANIFEST="$require_alignment" \
  NEIGHBORHOOD_K="$NEIGHBORHOOD_K" \
    bash run_previous_pipeline_variant.sh "$variant"
}

if [[ -n "${XLMR_VARIANT:-}" ]]; then
  run_variant "$XLMR_VARIANT"
  exit 0
fi

for base_variant in "${BASE_VARIANTS[@]}"; do
  for retrieval in "${RETRIEVALS[@]}"; do
    run_variant "${base_variant}_${retrieval}_lora"
  done
done
