#!/usr/bin/env bash
set -e

# ============================================================
# Reviewer response:
# Evaluation with other low-resource language pairs
# ============================================================
#
# Pairs:
#   Khmer -> Vietnamese
#   Lao   -> Vietnamese
#
# Experiments:
#   1. Frozen XLM-R retrieval
#   2. Baseline-6-style LoRA + projection + no Kabsch + CSLS
#
# Do not run this on the head node. Submit it to a GPU compute node.
# ============================================================

SRC_MODEL="xlm-roberta-base"
TGT_MODEL="xlm-roberta-base"

PAIRS=(
  "khmkhmr_vielatn"
  "laolaoo_vielatn"
)

echo "Preparing FLORES data..."
python src/prepare_flores_generalization_pairs.py

for PAIR in "${PAIRS[@]}"; do
  TRAIN_CSV="data/flores_generalization/${PAIR}/train.csv"
  TEST_CSV="data/flores_generalization/${PAIR}/test.csv"

  test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
  test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

  echo "============================================================"
  echo "Pair: $PAIR"
  echo "============================================================"

  # ------------------------------------------------------------
  # Experiment 1A: Frozen XLM-R + cosine
  # ------------------------------------------------------------
  python src/frozen_xlmr_flores_retrieval.py \
    --test_csv "$TEST_CSV" \
    --output_dir "results/baselines/flores_${PAIR}_frozen_xlmr_cosine" \
    --model_name "$SRC_MODEL" \
    --max_len 256 \
    --batch_size 16 \
    --topk_eval 10 \
    --eval_ks 1 5 10

  # ------------------------------------------------------------
  # Experiment 1B: Frozen XLM-R + CSLS
  # ------------------------------------------------------------
  python src/frozen_xlmr_flores_retrieval.py \
    --test_csv "$TEST_CSV" \
    --output_dir "results/baselines/flores_${PAIR}_frozen_xlmr_csls" \
    --model_name "$SRC_MODEL" \
    --max_len 256 \
    --batch_size 16 \
    --use_csls \
    --csls_k 10 \
    --topk_eval 10 \
    --eval_ks 1 5 10

  # ------------------------------------------------------------
  # Experiment 2: Baseline-6-style LoRA + projection
  # no Kabsch + CSLS
  #
  # This is the cross-language equivalent of your best Baseline 6:
  # XLM-R + LoRA + projection + no Kabsch + CSLS.
  # ------------------------------------------------------------
  python src/lora_projection_generalization_train_eval.py \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --src_model "$SRC_MODEL" \
    --tgt_model "$TGT_MODEL" \
    --pooling token_mean \
    --projection_dim 256 \
    --proj_dropout 0.1 \
    --epochs 50 \
    --batch_size 16 \
    --grad_accum_steps 2 \
    --eval_batch_size 16 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --temperature 0.07 \
    --warmup_ratio 0.06 \
    --src_max_len 256 \
    --tgt_max_len 256 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --eval_retrievals both \
    --csls_k 10 \
    --topk_eval 10 \
    --eval_ks 1 5 10 \
    --num_workers 0 \
    --output_dir "results/baselines/flores_${PAIR}_b6style_lora_projection_50ep_token_mean_both"

done

echo
echo "Finished FLORES reviewer-response generalization experiments."
