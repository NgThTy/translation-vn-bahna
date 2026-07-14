# #!/usr/bin/env bash
# set -e

# # ============================================================
# # Baseline 6: Previous pipeline / Projection + Procrustes
# # ============================================================
# #
# # Purpose:
# # Reproduce the old/current system:
# #
# #   encoder -> pooling -> projection head -> optional Kabsch -> retrieval
# #
# # IMPORTANT FIX:
# # The XLM-R B2 checkpoints were trained with LoRA + projection heads.
# # Therefore, all XLM-R B2 runs below MUST use --use_lora.
# #
# # Without --use_lora, the script evaluates vanilla XLM-R with projection
# # heads trained on LoRA-adapted features, which causes near-random results.
# #
# # Main setting:
# #   Backbone: XLM-R
# #   Epochs: 50
# #   Kabsch alignment: 10K
# #   Projection dir: results/models/b2_xlmr_50ep
# #   Alignment dir: results/alignment/alignment_B2_xlmr_10K_50ep
# # ============================================================

# PROJ_DIR="results/models/b2_xlmr_50ep"
# ALIGN_DIR="results/alignment/alignment_B2_xlmr_10K_50ep"
# SRC_MODEL="xlm-roberta-base"
# TGT_MODEL="xlm-roberta-base"

# echo "Checking required projection, alignment, and LoRA files..."

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

# # -------------------------
# # 1. Main previous pipeline:
# #    XLM-R 50ep + LoRA + 10K Kabsch + cosine
# # -------------------------
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --use_lora \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_cosine_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # -------------------------
# # 2. Main previous pipeline:
# #    XLM-R 50ep + LoRA + 10K Kabsch + CSLS
# # -------------------------
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --use_lora \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # -------------------------
# # 3. Ablation:
# #    XLM-R 50ep + LoRA + no Kabsch + cosine
# # -------------------------
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir "$PROJ_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --use_lora \
#   --no_kabsch \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_cosine_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # -------------------------
# # 4. Ablation:
# #    XLM-R 50ep + LoRA + no Kabsch + CSLS
# # -------------------------
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir "$PROJ_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --use_lora \
#   --no_kabsch \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # -------------------------
# # 5. Optional pooling ablation:
# #    XLM-R 50ep + LoRA + 10K Kabsch + token-IDF + cosine
# # -------------------------
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_idf \
#   --use_lora \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_cosine_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# # -------------------------
# # 6. Optional pooling ablation:
# #    XLM-R 50ep + LoRA + 10K Kabsch + token-IDF + CSLS
# # -------------------------
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_idf \
#   --use_lora \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# echo
# echo "Finished Baseline 6 with LoRA-enabled previous pipeline."

##########################################
# #!/usr/bin/env bash
# set -euo pipefail

# # Core six development variants. This script is resumable and never touches test.csv.
# # Run one named variant with:
# #   XLMR_VARIANT=previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_csls_lora \
# #     bash run_previous_pipeline_baselines.sh

# if [[ -n "${XLMR_VARIANT:-}" ]]; then
#   bash run_previous_pipeline_variant.sh "$XLMR_VARIANT"
#   exit 0
# fi

# variants=(
#   previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_cosine_lora
#   previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_csls_lora
#   previous_pipeline_xlmr_50ep_no_kabsch_token_mean_cosine_lora
#   previous_pipeline_xlmr_50ep_no_kabsch_token_mean_csls_lora
#   previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_cosine_lora
#   previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_csls_lora
# )

# for variant in "${variants[@]}"; do
#   bash run_previous_pipeline_variant.sh "$variant"
# done



















# #!/usr/bin/env bash
# set -euo pipefail

# # Core development variants. Every fixed checkpoint/pooling/Kabsch setup is
# # compared under cosine, CSLS, and Artetxe-Schwenk ratio-margin retrieval.
# # This script remains dev-only and never touches test.csv.
# #
# # Run one named variant with:
# #   XLMR_VARIANT=previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_margin_ratio_lora \
# #     bash run_previous_pipeline_baselines.sh

# NEIGHBORHOOD_K="${NEIGHBORHOOD_K:-10}"
# RETRIEVALS=(cosine csls margin_ratio)
# BASE_VARIANTS=(
#   previous_pipeline_xlmr_50ep_10K_kabsch_token_mean
#   previous_pipeline_xlmr_50ep_no_kabsch_token_mean
#   previous_pipeline_xlmr_50ep_10K_kabsch_token_idf
# )

# run_variant() {
#   local variant="$1"
#   NEIGHBORHOOD_K="$NEIGHBORHOOD_K" \
#     bash run_previous_pipeline_variant.sh "$variant"
# }

# if [[ -n "${XLMR_VARIANT:-}" ]]; then
#   run_variant "$XLMR_VARIANT"
#   exit 0
# fi

# for base_variant in "${BASE_VARIANTS[@]}"; do
#   for retrieval in "${RETRIEVALS[@]}"; do
#     run_variant "${base_variant}_${retrieval}_lora"
#   done
# done










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
  echo "Set PROJ_DIR to the directory containing the verified checkpoint." >&2
  exit 1
fi

run_variant() {
  local variant="$1"
  local require_alignment=0

  if [[ "$variant" == *"_kabsch_"* ]]; then
    require_alignment=1
  fi

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














