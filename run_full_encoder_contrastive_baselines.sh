# #!/usr/bin/env bash
# set -e

# # ============================================================
# # Baseline 7: Full encoder contrastive fine-tuning
# # Improved version
# # ============================================================
# #
# # Improvements:
# # - Add projection heads: XLM-R -> 256-d retrieval space.
# # - Lower learning rate to 1e-5.
# # - Increase effective batch size to 64:
# #     batch_size 4 x grad_accum_steps 16.
# # - Train once, evaluate same checkpoint with cosine and CSLS.
# # - Optional validation retrieval if data/valid.csv exists.
# #
# # Run this on a GPU compute node.
# # ============================================================

# SRC_MODEL="xlm-roberta-base"
# TGT_MODEL="xlm-roberta-base"

# TRAIN_CSV="data/train.csv"
# TEST_CSV="data/test.csv"
# VALID_CSV="data/valid.csv"

# echo "Checking required data files..."

# test -f "$TRAIN_CSV" || { echo "Missing $TRAIN_CSV"; exit 1; }
# test -f "$TEST_CSV" || { echo "Missing $TEST_CSV"; exit 1; }

# VALID_ARGS=""
# if [ -f "$VALID_CSV" ]; then
#   echo "Found validation file: $VALID_CSV"
#   VALID_ARGS="--valid_csv $VALID_CSV --early_stop_patience 2"
# else
#   echo "No validation file found at $VALID_CSV. Training without validation early stopping."
# fi

# echo "All required data files found."
# echo

# echo "GPU status:"
# nvidia-smi || true
# echo


# # ============================================================
# # 1. Main improved run:
# #    full XLM-R + projection head + sentence_mean + InfoNCE
# #    train once, evaluate cosine + CSLS
# # ============================================================
# python src/full_encoder_contrastive_finetune_baseline.py \
#   --train_csv "$TRAIN_CSV" \
#   $VALID_ARGS \
#   --test_csv "$TEST_CSV" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling sentence_mean \
#   --projection_dim 256 \
#   --proj_dropout 0.1 \
#   --output_dir results/baselines/full_encoder_xlmr_proj_infonce_5ep_sentence_mean_both \
#   --epochs 5 \
#   --batch_size 4 \
#   --grad_accum_steps 16 \
#   --lr 1e-5 \
#   --weight_decay 0.01 \
#   --temperature 0.07 \
#   --warmup_ratio 0.06 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --eval_batch_size 16 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10 \
#   --eval_retrievals both \
#   --csls_k 10 \
#   --fp16


# # ============================================================
# # 2. Pooling ablation:
# #    full XLM-R + projection per token + token_mean + InfoNCE
# #    train once, evaluate cosine + CSLS
# # ============================================================
# python src/full_encoder_contrastive_finetune_baseline.py \
#   --train_csv "$TRAIN_CSV" \
#   $VALID_ARGS \
#   --test_csv "$TEST_CSV" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling token_mean \
#   --projection_dim 256 \
#   --proj_dropout 0.1 \
#   --output_dir results/baselines/full_encoder_xlmr_proj_infonce_5ep_token_mean_both \
#   --epochs 5 \
#   --batch_size 4 \
#   --grad_accum_steps 16 \
#   --lr 1e-5 \
#   --weight_decay 0.01 \
#   --temperature 0.07 \
#   --warmup_ratio 0.06 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --eval_batch_size 16 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10 \
#   --eval_retrievals both \
#   --csls_k 10 \
#   --fp16


# # ============================================================
# # 3. Longer run:
# #    full XLM-R + projection head + sentence_mean + InfoNCE
# #    10 epochs, train once, evaluate cosine + CSLS
# # ============================================================
# python src/full_encoder_contrastive_finetune_baseline.py \
#   --train_csv "$TRAIN_CSV" \
#   $VALID_ARGS \
#   --test_csv "$TEST_CSV" \
#   --src_model "$SRC_MODEL" \
#   --tgt_model "$TGT_MODEL" \
#   --pooling sentence_mean \
#   --projection_dim 256 \
#   --proj_dropout 0.1 \
#   --output_dir results/baselines/full_encoder_xlmr_proj_infonce_10ep_sentence_mean_both \
#   --epochs 10 \
#   --batch_size 4 \
#   --grad_accum_steps 16 \
#   --lr 1e-5 \
#   --weight_decay 0.01 \
#   --temperature 0.07 \
#   --warmup_ratio 0.06 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --eval_batch_size 16 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10 \
#   --eval_retrievals both \
#   --csls_k 10 \
#   --fp16

# echo
# echo "Finished improved Baseline 7: full encoder + projection-head contrastive fine-tuning."

##############################
# #!/usr/bin/env bash
# set -euo pipefail

# TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
# DEV_CSV="${DEV_CSV:-data/dev.csv}"
# TEST_CSV="${TEST_CSV:-data/test.csv}"
# MODEL_ROOT="${MODEL_ROOT:-results/models/full_encoder}"
# DEV_ROOT="${DEV_ROOT:-results/dev/full_encoder}"
# TEST_ROOT="${TEST_ROOT:-results/test/full_encoder}"
# SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"
# MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-105}"

# for required in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"; do
#   if [[ ! -f "$required" ]]; then
#     echo "Missing required file: $required" >&2
#     exit 1
#   fi
# done

# mkdir -p "$MODEL_ROOT" "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT" logs

# # Six declared training configurations. Each run is trained to epoch 10 and
# # creates six dev variants: epochs 3/5/10 x cosine/CSLS.
# CONFIG_NAMES=(
#   "sent_mlp256_sym_t007_lr1e5"
#   "token_mlp256_sym_t007_lr1e5"
#   "sent_linear256_sym_t007_lr1e5"
#   "token_identity_sym_t007_lr1e5"
#   "token_mlp256_sym_t005_lr1e5"
#   "token_mlp128_oneway_t007_lr5e6"
# )

# config_args() {
#   case "$1" in
#     sent_mlp256_sym_t007_lr1e5)
#       printf '%s\n' --pooling sentence_mean --projection_type mlp --projection_dim 256 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
#       ;;
#     token_mlp256_sym_t007_lr1e5)
#       printf '%s\n' --pooling token_mean --projection_type mlp --projection_dim 256 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
#       ;;
#     sent_linear256_sym_t007_lr1e5)
#       printf '%s\n' --pooling sentence_mean --projection_type linear --projection_dim 256 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
#       ;;
#     token_identity_sym_t007_lr1e5)
#       printf '%s\n' --pooling token_mean --projection_type identity --projection_dim 768 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
#       ;;
#     token_mlp256_sym_t005_lr1e5)
#       printf '%s\n' --pooling token_mean --projection_type mlp --projection_dim 256 --loss_type symmetric_infonce --temperature 0.05 --lr 1e-5
#       ;;
#     token_mlp128_oneway_t007_lr5e6)
#       printf '%s\n' --pooling token_mean --projection_type mlp --projection_dim 128 --loss_type source_to_target_infonce --temperature 0.07 --lr 5e-6
#       ;;
#     *)
#       echo "Unknown full-encoder configuration: $1" >&2
#       exit 1
#       ;;
#   esac
# }

# variant_complete() {
#   local prefix="$1"
#   local epoch retrieval
#   for epoch in 3 5 10; do
#     for retrieval in cosine csls; do
#       if [[ ! -s "$DEV_ROOT/${prefix}_${epoch}ep_${retrieval}/metrics.json" ]]; then
#         return 1
#       fi
#     done
#   done
#   return 0
# }

# run_config() {
#   local name="$1"
#   if variant_complete "$name"; then
#     echo "[SKIP] All six dev variants already exist for $name"
#     return 0
#   fi

#   mapfile -t EXTRA_ARGS < <(config_args "$name")
#   echo "[RUN/RESUME] $name"
#   python src/full_encoder_contrastive_finetune_baseline.py \
#     --mode train \
#     --train_csv "$TRAIN_CSV" \
#     --dev_csv "$DEV_CSV" \
#     --output_dir "$MODEL_ROOT/$name" \
#     --dev_output_root "$DEV_ROOT" \
#     --configuration_prefix "$name" \
#     --src_model xlm-roberta-base \
#     --tgt_model xlm-roberta-base \
#     --epochs 10 \
#     --checkpoint_epochs 3 5 10 \
#     --batch_size "${BATCH_SIZE:-4}" \
#     --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
#     --grad_accum_steps "${GRAD_ACCUM_STEPS:-16}" \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.06 \
#     --proj_dropout 0.1 \
#     --src_max_len 256 \
#     --tgt_max_len 256 \
#     --eval_retrievals both \
#     --csls_k 10 \
#     --topk_eval 10 \
#     --eval_ks 1 5 10 \
#     --bf16 \
#     --auto_resume \
#     --save_every_steps "${SAVE_EVERY_STEPS:-200}" \
#     --max_runtime_minutes "$MAX_RUNTIME_MINUTES" \
#     "${EXTRA_ARGS[@]}"
# }

# select_and_test() {
#   local missing=0 prefix
#   for prefix in "${CONFIG_NAMES[@]}"; do
#     if ! variant_complete "$prefix"; then
#       echo "Missing dev results for: $prefix" >&2
#       missing=1
#     fi
#   done
#   if [[ "$missing" -ne 0 ]]; then
#     echo "Refusing selection from an incomplete full-encoder search space." >&2
#     exit 2
#   fi

#   python src/select_dev_configs_and_evaluate_test.py \
#     --dev_root results/dev \
#     --test_root results/test \
#     --output_root "$SELECTION_ROOT" \
#     --test_csv "$TEST_CSV" \
#     --evaluate_test \
#     --family full_encoder
# }

# ACTION="${FULL_ENCODER_ACTION:-train}"
# if [[ "$ACTION" == "select" ]]; then
#   select_and_test
#   exit 0
# fi

# if [[ -n "${FULL_ENCODER_CONFIG:-}" ]]; then
#   run_config "$FULL_ENCODER_CONFIG"
#   exit 0
# fi

# if [[ -n "${FULL_ENCODER_INDEX:-}" ]]; then
#   if ! [[ "$FULL_ENCODER_INDEX" =~ ^[0-9]+$ ]]; then
#     echo "FULL_ENCODER_INDEX must be an integer" >&2
#     exit 1
#   fi
#   if (( FULL_ENCODER_INDEX < 0 || FULL_ENCODER_INDEX >= ${#CONFIG_NAMES[@]} )); then
#     echo "FULL_ENCODER_INDEX is out of range: $FULL_ENCODER_INDEX" >&2
#     exit 1
#   fi
#   run_config "${CONFIG_NAMES[$FULL_ENCODER_INDEX]}"
#   exit 0
# fi

# if [[ "${FULL_ENCODER_RUN_ALL:-0}" == "1" ]]; then
#   for name in "${CONFIG_NAMES[@]}"; do
#     run_config "$name"
#   done
#   exit 0
# fi

# # Safe default for a two-hour allocation: run or resume only the first
# # incomplete training configuration, then stop.
# for name in "${CONFIG_NAMES[@]}"; do
#   if ! variant_complete "$name"; then
#     run_config "$name"
#     exit 0
#   fi
# done

# echo "All full-encoder development variants are complete."
# echo "Run: FULL_ENCODER_ACTION=select bash run_full_encoder_contrastive_baselines.sh"















#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
MODEL_ROOT="${MODEL_ROOT:-results/models/full_encoder}"
DEV_ROOT="${DEV_ROOT:-results/dev/full_encoder}"
TEST_ROOT="${TEST_ROOT:-results/test/full_encoder}"
SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-105}"
NEIGHBORHOOD_K="${NEIGHBORHOOD_K:-10}"
RETRIEVALS=(cosine csls margin_ratio)

for required in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$MODEL_ROOT" "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT" logs

# Six declared training configurations. Each run is trained to epoch 10 and
# creates nine development variants: epochs 3/5/10 x cosine/CSLS/ratio-margin.
CONFIG_NAMES=(
  "sent_mlp256_sym_t007_lr1e5"
  "token_mlp256_sym_t007_lr1e5"
  "sent_linear256_sym_t007_lr1e5"
  "token_identity_sym_t007_lr1e5"
  "token_mlp256_sym_t005_lr1e5"
  "token_mlp128_oneway_t007_lr5e6"
)

config_args() {
  case "$1" in
    sent_mlp256_sym_t007_lr1e5)
      printf '%s\n' --pooling sentence_mean --projection_type mlp --projection_dim 256 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
      ;;
    token_mlp256_sym_t007_lr1e5)
      printf '%s\n' --pooling token_mean --projection_type mlp --projection_dim 256 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
      ;;
    sent_linear256_sym_t007_lr1e5)
      printf '%s\n' --pooling sentence_mean --projection_type linear --projection_dim 256 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
      ;;
    token_identity_sym_t007_lr1e5)
      printf '%s\n' --pooling token_mean --projection_type identity --projection_dim 768 --loss_type symmetric_infonce --temperature 0.07 --lr 1e-5
      ;;
    token_mlp256_sym_t005_lr1e5)
      printf '%s\n' --pooling token_mean --projection_type mlp --projection_dim 256 --loss_type symmetric_infonce --temperature 0.05 --lr 1e-5
      ;;
    token_mlp128_oneway_t007_lr5e6)
      printf '%s\n' --pooling token_mean --projection_type mlp --projection_dim 128 --loss_type source_to_target_infonce --temperature 0.07 --lr 5e-6
      ;;
    *)
      echo "Unknown full-encoder configuration: $1" >&2
      exit 1
      ;;
  esac
}

variant_complete() {
  local prefix="$1"
  local epoch retrieval
  for epoch in 3 5 10; do
    for retrieval in "${RETRIEVALS[@]}"; do
      if [[ ! -s "$DEV_ROOT/${prefix}_${epoch}ep_${retrieval}/metrics.json" ]]; then
        return 1
      fi
    done
  done
  return 0
}

run_config() {
  local name="$1"
  if variant_complete "$name"; then
    echo "[SKIP] All nine dev variants already exist for $name"
    return 0
  fi

  mapfile -t EXTRA_ARGS < <(config_args "$name")
  echo "[RUN/RESUME] $name"
  "$PYTHON_BIN" src/full_encoder_contrastive_finetune_baseline.py \
    --mode train \
    --train_csv "$TRAIN_CSV" \
    --dev_csv "$DEV_CSV" \
    --output_dir "$MODEL_ROOT/$name" \
    --dev_output_root "$DEV_ROOT" \
    --configuration_prefix "$name" \
    --src_model xlm-roberta-base \
    --tgt_model xlm-roberta-base \
    --epochs 10 \
    --checkpoint_epochs 3 5 10 \
    --batch_size "${BATCH_SIZE:-4}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS:-16}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --proj_dropout 0.1 \
    --src_max_len 256 \
    --tgt_max_len 256 \
    --eval_retrievals all \
    --neighborhood_k "$NEIGHBORHOOD_K" \
    --topk_eval 10 \
    --eval_ks 1 5 10 \
    --bf16 \
    --auto_resume \
    --save_every_steps "${SAVE_EVERY_STEPS:-200}" \
    --max_runtime_minutes "$MAX_RUNTIME_MINUTES" \
    "${EXTRA_ARGS[@]}"
}

select_and_test() {
  local missing=0 prefix
  for prefix in "${CONFIG_NAMES[@]}"; do
    if ! variant_complete "$prefix"; then
      echo "Missing dev results for: $prefix" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Refusing selection from an incomplete full-encoder search space." >&2
    exit 2
  fi

  "$PYTHON_BIN" src/select_dev_configs_and_evaluate_test.py \
    --dev_root results/dev \
    --test_root results/test \
    --output_root "$SELECTION_ROOT" \
    --test_csv "$TEST_CSV" \
    --evaluate_test \
    --family full_encoder
}

ACTION="${FULL_ENCODER_ACTION:-train}"
if [[ "$ACTION" == "select" ]]; then
  select_and_test
  exit 0
fi

if [[ -n "${FULL_ENCODER_CONFIG:-}" ]]; then
  run_config "$FULL_ENCODER_CONFIG"
  exit 0
fi

if [[ -n "${FULL_ENCODER_INDEX:-}" ]]; then
  if ! [[ "$FULL_ENCODER_INDEX" =~ ^[0-9]+$ ]]; then
    echo "FULL_ENCODER_INDEX must be an integer" >&2
    exit 1
  fi
  if (( FULL_ENCODER_INDEX < 0 || FULL_ENCODER_INDEX >= ${#CONFIG_NAMES[@]} )); then
    echo "FULL_ENCODER_INDEX is out of range: $FULL_ENCODER_INDEX" >&2
    exit 1
  fi
  run_config "${CONFIG_NAMES[$FULL_ENCODER_INDEX]}"
  exit 0
fi

if [[ "${FULL_ENCODER_RUN_ALL:-0}" == "1" ]]; then
  for name in "${CONFIG_NAMES[@]}"; do
    run_config "$name"
  done
  exit 0
fi

# Safe default for a two-hour allocation: run or resume only the first
# incomplete training configuration, then stop.
for name in "${CONFIG_NAMES[@]}"; do
  if ! variant_complete "$name"; then
    run_config "$name"
    exit 0
  fi
done

echo "All full-encoder development variants are complete."
echo "Run: FULL_ENCODER_ACTION=select bash run_full_encoder_contrastive_baselines.sh"
