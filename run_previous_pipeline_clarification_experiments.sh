# #!/usr/bin/env bash
# set -e

# # ============================================================
# # Baseline 6 clarification experiments
# # ============================================================
# #
# # Purpose:
# # Clarify whether the rejected-paper previous pipeline used:
# #
# #   1. encoder -> mean pooling -> projection
# #      implemented as: --pooling sentence_mean
# #
# # or:
# #
# #   2. encoder -> projection per token -> mean pooling
# #      implemented as: --pooling token_mean
# #
# # Also clarify whether the main retrieval number should be reported
# # with cosine or CSLS.
# #
# # IMPORTANT:
# # The XLM-R 50ep checkpoint was trained with LoRA + projection heads.
# # Therefore every run must include --use_lora.
# # ============================================================

# PROJ_DIR="results/models/b2_xlmr_50ep"
# ALIGN_DIR="results/alignment/alignment_B2_xlmr_10K_50ep"
# SRC_MODEL="xlm-roberta-base"
# TGT_MODEL="xlm-roberta-base"
# TEST_CSV="data/test.csv"

# echo "Checking required files..."

# test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

# test -f "$PROJ_DIR/src_proj.pt" || { echo "Missing $PROJ_DIR/src_proj.pt"; exit 1; }
# test -f "$PROJ_DIR/tgt_proj.pt" || { echo "Missing $PROJ_DIR/tgt_proj.pt"; exit 1; }

# test -d "$PROJ_DIR/src_adapters" || { echo "Missing $PROJ_DIR/src_adapters"; exit 1; }
# test -d "$PROJ_DIR/tgt_adapters" || { echo "Missing $PROJ_DIR/tgt_adapters"; exit 1; }

# test -f "$PROJ_DIR/src_adapters/adapter_config.json" || { echo "Missing $PROJ_DIR/src_adapters/adapter_config.json"; exit 1; }
# test -f "$PROJ_DIR/tgt_adapters/adapter_config.json" || { echo "Missing $PROJ_DIR/tgt_adapters/adapter_config.json"; exit 1; }

# test -f "$ALIGN_DIR/R.npy" || { echo "Missing $ALIGN_DIR/R.npy"; exit 1; }
# test -f "$ALIGN_DIR/t.npy" || { echo "Missing $ALIGN_DIR/t.npy"; exit 1; }

# echo "All required files found."
# echo


# # ============================================================
# # 1. Paper-style pooling + cosine
# # encoder -> mean pool -> projection -> Kabsch -> cosine
# # ============================================================
# python src/previous_pipeline_baseline.py \
#   --test_csv "$TEST_CSV" \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling sentence_mean \
#   --use_lora \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_cosine_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # ============================================================
# # 2. Paper-style pooling + CSLS
# # encoder -> mean pool -> projection -> Kabsch -> CSLS
# # ============================================================
# python src/previous_pipeline_baseline.py \
#   --test_csv "$TEST_CSV" \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling sentence_mean \
#   --use_lora \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # ============================================================
# # 3. Current Baseline-6 pooling + cosine
# # encoder -> projection per token -> mean pool -> Kabsch -> cosine
# # ============================================================
# python src/previous_pipeline_baseline.py \
#   --test_csv "$TEST_CSV" \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --use_lora \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_cosine_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # ============================================================
# # 4. Current Baseline-6 pooling + CSLS
# # encoder -> projection per token -> mean pool -> Kabsch -> CSLS
# # ============================================================
# python src/previous_pipeline_baseline.py \
#   --test_csv "$TEST_CSV" \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --use_lora \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# echo
# echo "Finished Baseline 6 clarification experiments."

#####################
# #!/usr/bin/env bash
# set -euo pipefail

# # Sentence-mean Kabsch clarification variants, evaluated on dev only.
# # The token-mean Kabsch variants are already part of run_previous_pipeline_baselines.sh
# # and are intentionally not duplicated here.

# if [[ -n "${XLMR_VARIANT:-}" ]]; then
#   bash run_previous_pipeline_variant.sh "$XLMR_VARIANT"
#   exit 0
# fi

# variants=(
#   previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_cosine_lora
#   previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_csls_lora
# )

# for variant in "${variants[@]}"; do
#   bash run_previous_pipeline_variant.sh "$variant"
# done



















# #!/usr/bin/env bash
# set -euo pipefail

# # Sentence-mean Kabsch clarification variants, evaluated on dev only.
# # The token-mean Kabsch variants are part of run_previous_pipeline_baselines.sh.

# NEIGHBORHOOD_K="${NEIGHBORHOOD_K:-10}"
# RETRIEVALS=(cosine csls margin_ratio)
# BASE_VARIANT="previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean"

# run_variant() {
#   local variant="$1"
#   NEIGHBORHOOD_K="$NEIGHBORHOOD_K" \
#     bash run_previous_pipeline_variant.sh "$variant"
# }

# if [[ -n "${XLMR_VARIANT:-}" ]]; then
#   run_variant "$XLMR_VARIANT"
#   exit 0
# fi

# for retrieval in "${RETRIEVALS[@]}"; do
#   run_variant "${BASE_VARIANT}_${retrieval}_lora"
# done
















#!/usr/bin/env bash
set -euo pipefail

# Sentence-mean Kabsch clarification variants, evaluated on dev only.
# The token-mean Kabsch variants are part of run_previous_pipeline_baselines.sh.

PROJ_DIR="${PROJ_DIR:-results/models/reviewer_xlmr_50ep/checkpoint_final}"
ALIGN_DIR="${ALIGN_DIR:-results/alignment/reviewer_xlmr_10K_50ep_train_fit}"
NEIGHBORHOOD_K="${NEIGHBORHOOD_K:-10}"

RETRIEVALS=(cosine csls margin_ratio)
BASE_VARIANT="previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean"

if [[ ! -f "$PROJ_DIR/training_manifest.json" ]]; then
  echo "Missing reviewer-compliant checkpoint manifest:" >&2
  echo "  $PROJ_DIR/training_manifest.json" >&2
  echo "Set PROJ_DIR to the directory containing the verified checkpoint." >&2
  exit 1
fi

run_variant() {
  local variant="$1"

  PROJ_DIR="$PROJ_DIR" \
  ALIGN_DIR="$ALIGN_DIR" \
  REQUIRE_ALIGNMENT_MANIFEST=1 \
  NEIGHBORHOOD_K="$NEIGHBORHOOD_K" \
    bash run_previous_pipeline_variant.sh "$variant"
}

if [[ -n "${XLMR_VARIANT:-}" ]]; then
  run_variant "$XLMR_VARIANT"
  exit 0
fi

for retrieval in "${RETRIEVALS[@]}"; do
  run_variant "${BASE_VARIANT}_${retrieval}_lora"
done




