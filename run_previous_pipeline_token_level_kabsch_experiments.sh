# #!/usr/bin/env bash
# set -e

# # ============================================================
# # Baseline 6: token-level Kabsch before pooling
# # ============================================================
# #
# # Purpose:
# # Test a more theoretically appropriate use of word-derived Kabsch.
# #
# # Instead of:
# #   encoder -> projection/pooling -> sentence embedding -> Kabsch
# #
# # This tests:
# #   encoder -> project each token -> Kabsch each projected token -> pool -> CSLS
# #
# # Experiments:
# #   1. token_mean + token-level Kabsch + CSLS
# #   2. token_idf  + token-level Kabsch + CSLS
# #
# # IMPORTANT:
# # The XLM-R B2 checkpoint was trained with LoRA + projection heads.
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
# # 1. token_mean + token-level Kabsch + CSLS
# # encoder -> project each token -> Kabsch tokens -> mean pool -> CSLS
# # ============================================================
# python src/previous_pipeline_baseline.py \
#   --test_csv "$TEST_CSV" \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --kabsch_stage token \
#   --use_lora \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_token_kabsch_token_mean_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10


# # ============================================================
# # 2. token_idf + token-level Kabsch + CSLS
# # encoder -> project each token -> Kabsch tokens -> IDF pool -> CSLS
# # ============================================================
# python src/previous_pipeline_baseline.py \
#   --test_csv "$TEST_CSV" \
#   --proj_dir "$PROJ_DIR" \
#   --alignment_dir "$ALIGN_DIR" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_idf \
#   --kabsch_stage token \
#   --use_lora \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_token_kabsch_token_idf_csls_lora \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# echo
# echo "Finished Baseline 6 token-level Kabsch experiments."

#!/usr/bin/env bash
set -euo pipefail

# Token-level Kabsch variants, evaluated on dev only.

if [[ -n "${XLMR_VARIANT:-}" ]]; then
  bash run_previous_pipeline_variant.sh "$XLMR_VARIANT"
  exit 0
fi

variants=(
  previous_pipeline_xlmr_50ep_10K_token_kabsch_token_mean_csls_lora
  previous_pipeline_xlmr_50ep_10K_token_kabsch_token_idf_csls_lora
)

for variant in "${variants[@]}"; do
  bash run_previous_pipeline_variant.sh "$variant"
done
