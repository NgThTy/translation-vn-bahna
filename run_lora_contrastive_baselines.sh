#!/usr/bin/env bash
set -e

# ============================================================
# Baseline 7: LoRA contrastive fine-tuning
# ============================================================
#
# This baseline trains a new retrieval model:
#
#   XLM-R encoder + LoRA + pooling + projection head + InfoNCE
#
# Then it evaluates Bahnaric -> Vietnamese sentence retrieval on data/test.csv.
#
# This baseline does NOT use Kabsch/Procrustes alignment.
# Kabsch belongs to Baseline 6, the previous-pipeline baseline.
# ============================================================

SRC_MODEL="xlm-roberta-base"
TGT_MODEL="xlm-roberta-base"

TRAIN_CSV="data/train.csv"
TEST_CSV="data/test.csv"

echo "Checking required data files..."

test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

echo "All required data files found."
echo


# -------------------------
# 1. Main LoRA contrastive baseline:
#    XLM-R + LoRA + InfoNCE + cosine
# -------------------------
python src/lora_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --use_lora \
  --pooling sentence_mean \
  --loss_type infonce \
  --temperature 0.07 \
  --projection_dim 256 \
  --epochs 10 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/lora_contrastive_xlmr_10ep_cosine


# -------------------------
# 2. Main LoRA contrastive baseline:
#    XLM-R + LoRA + InfoNCE + CSLS
# -------------------------
python src/lora_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --use_lora \
  --pooling sentence_mean \
  --loss_type infonce \
  --temperature 0.07 \
  --projection_dim 256 \
  --epochs 10 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --use_csls \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/lora_contrastive_xlmr_10ep_csls


# -------------------------
# 3. Longer run:
#    XLM-R + LoRA + InfoNCE + cosine, 20 epochs
# -------------------------
python src/lora_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --use_lora \
  --pooling sentence_mean \
  --loss_type infonce \
  --temperature 0.07 \
  --projection_dim 256 \
  --epochs 20 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/lora_contrastive_xlmr_20ep_cosine


# -------------------------
# 4. Longer run:
#    XLM-R + LoRA + InfoNCE + CSLS, 20 epochs
# -------------------------
python src/lora_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --use_lora \
  --pooling sentence_mean \
  --loss_type infonce \
  --temperature 0.07 \
  --projection_dim 256 \
  --epochs 20 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --use_csls \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/lora_contrastive_xlmr_20ep_csls


# -------------------------
# 5. Pooling ablation:
#    XLM-R + LoRA + InfoNCE + token_mean + CSLS
# -------------------------
python src/lora_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --use_lora \
  --pooling token_mean \
  --loss_type infonce \
  --temperature 0.07 \
  --projection_dim 256 \
  --epochs 10 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --use_csls \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/lora_contrastive_xlmr_10ep_token_mean_csls


# -------------------------
# 6. Normalization ablation:
#    XLM-R + LoRA + InfoNCE + strip accents + remove punctuation + CSLS
# -------------------------
python src/lora_contrastive_finetune_baseline.py \
  --train_csv "$TRAIN_CSV" \
  --test_csv "$TEST_CSV" \
  --src_model "$SRC_MODEL" \
  --tgt_model "$TGT_MODEL" \
  --use_lora \
  --pooling sentence_mean \
  --loss_type infonce \
  --temperature 0.07 \
  --projection_dim 256 \
  --epochs 10 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --strip_accents \
  --remove_punct \
  --use_csls \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/lora_contrastive_xlmr_10ep_strip_accents_no_punct_csls


echo
echo "Finished Baseline 7: LoRA contrastive fine-tuning."
