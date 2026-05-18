#!/usr/bin/env bash
set -e

# ============================================================
# Baseline 7: Full encoder contrastive fine-tuning
# Improved version
# ============================================================
#
# Improvements:
# - Add projection heads: XLM-R -> 256-d retrieval space.
# - Lower learning rate to 1e-5.
# - Increase effective batch size to 64:
#     batch_size 4 x grad_accum_steps 16.
# - Train once, evaluate same checkpoint with cosine and CSLS.
# - Optional validation retrieval if data/valid.csv exists.
#
# Run this on a GPU compute node.
# ============================================================

SRC_MODEL="xlm-roberta-base"
TGT_MODEL="xlm-roberta-base"

TRAIN_CSV="data/train.csv"
TEST_CSV="data/test.csv"
VALID_CSV="data/valid.csv"

echo "Checking required data files..."

test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

VALID_ARGS=""
if [ -f "$VALID_CSV" ]; then
  echo "Found validation file: $VALID_CSV"
  VALID_ARGS="--valid_csv $VALID_CSV --early_stop_patience 2"
else
  echo "No validation file found at $VALID_CSV. Training without validation early stopping."
fi

echo "All required data files found."
echo

echo "GPU status:"
nvidia-smi || true
echo


# ============================================================
# 1. Main improved run:
#    full XLM-R + projection head + sentence_mean + InfoNCE
#    train once, evaluate cosine + CSLS
# ============================================================
python src/full_encoder_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  $VALID_ARGS \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling sentence_mean \
  --projection_dim 256 \
  --proj_dropout 0.1 \
  --output_dir results/baselines/full_encoder_xlmr_proj_infonce_5ep_sentence_mean_both \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum_steps 16 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --temperature 0.07 \
  --warmup_ratio 0.06 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --eval_batch_size 16 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --eval_retrievals both \
  --csls_k 10 \
  --fp16


# ============================================================
# 2. Pooling ablation:
#    full XLM-R + projection per token + token_mean + InfoNCE
#    train once, evaluate cosine + CSLS
# ============================================================
python src/full_encoder_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  $VALID_ARGS \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling token_mean \
  --projection_dim 256 \
  --proj_dropout 0.1 \
  --output_dir results/baselines/full_encoder_xlmr_proj_infonce_5ep_token_mean_both \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum_steps 16 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --temperature 0.07 \
  --warmup_ratio 0.06 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --eval_batch_size 16 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --eval_retrievals both \
  --csls_k 10 \
  --fp16


# ============================================================
# 3. Longer run:
#    full XLM-R + projection head + sentence_mean + InfoNCE
#    10 epochs, train once, evaluate cosine + CSLS
# ============================================================
python src/full_encoder_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  $VALID_ARGS \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --pooling sentence_mean \
  --projection_dim 256 \
  --proj_dropout 0.1 \
  --output_dir results/baselines/full_encoder_xlmr_proj_infonce_10ep_sentence_mean_both \
  --epochs 10 \
  --batch_size 4 \
  --grad_accum_steps 16 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --temperature 0.07 \
  --warmup_ratio 0.06 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --eval_batch_size 16 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --eval_retrievals both \
  --csls_k 10 \
  --fp16

echo
echo "Finished improved Baseline 7: full encoder + projection-head contrastive fine-tuning."
