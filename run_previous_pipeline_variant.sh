#!/usr/bin/env bash
set -euo pipefail

# Shared resumable runner for the 14 unique XLM-R LoRA projection variants.
# Usage:
#   bash run_previous_pipeline_variant.sh <configuration_name>

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <configuration_name>" >&2
  exit 2
fi

CONFIGURATION_NAME="$1"
TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
DEV_ROOT="${DEV_ROOT:-results/dev/xlmr_lora_projection}"
PROJ_DIR="${PROJ_DIR:-results/models/b2_xlmr_50ep}"
ALIGN_DIR="${ALIGN_DIR:-results/alignment/alignment_B2_xlmr_10K_50ep}"
SRC_MODEL="${SRC_MODEL:-xlm-roberta-base}"
TGT_MODEL="${TGT_MODEL:-xlm-roberta-base}"
CACHE_DIR="${CACHE_DIR:-results/cache/xlmr_lora_projection_embeddings}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-$PROJ_DIR/training_manifest.json}"
ALIGNMENT_MANIFEST="${ALIGNMENT_MANIFEST:-$ALIGN_DIR/alignment_manifest.json}"
ALLOW_UNVERIFIED_CHECKPOINT="${ALLOW_UNVERIFIED_CHECKPOINT:-0}"
REQUIRE_ALIGNMENT_MANIFEST="${REQUIRE_ALIGNMENT_MANIFEST:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for required_file in "$TRAIN_CSV" "$DEV_CSV"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

for required_file in "$PROJ_DIR/src_proj.pt" "$PROJ_DIR/tgt_proj.pt"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required projection file: $required_file" >&2
    exit 1
  fi
done

for required_dir in "$PROJ_DIR/src_adapters" "$PROJ_DIR/tgt_adapters"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Missing required LoRA adapter directory: $required_dir" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --input_csv "$DEV_CSV"
  --split_name dev
  --configuration_name "$CONFIGURATION_NAME"
  --output_dir "$DEV_ROOT/$CONFIGURATION_NAME"
  --proj_dir "$PROJ_DIR"
  --src_model "$SRC_MODEL"
  --tgt_model "$TGT_MODEL"
  --use_lora
  --batch_size "$BATCH_SIZE"
  --src_max_len 256
  --tgt_max_len 256
  --topk_eval 10
  --eval_ks 1 5 10
  --csls_k 10
  --embedding_cache_dir "$CACHE_DIR"
)

if [[ -f "$CHECKPOINT_MANIFEST" ]]; then
  COMMON_ARGS+=(--checkpoint_manifest "$CHECKPOINT_MANIFEST")
elif [[ "$ALLOW_UNVERIFIED_CHECKPOINT" == "1" ]]; then
  echo "WARNING: using a legacy checkpoint without verified train_fit provenance." >&2
  echo "Do not report this run as reviewer-compliant." >&2
  COMMON_ARGS+=(--allow_unverified_checkpoint)
else
  echo "Missing reviewer-compliant checkpoint manifest: $CHECKPOINT_MANIFEST" >&2
  echo "Retrain with src/lora_projection_generalization_train_eval.py, or set" >&2
  echo "ALLOW_UNVERIFIED_CHECKPOINT=1 only for legacy reproduction." >&2
  exit 1
fi

alignment_args=()
if [[ -f "$ALIGNMENT_MANIFEST" ]]; then
  alignment_args+=(--alignment_manifest "$ALIGNMENT_MANIFEST")
elif [[ "$REQUIRE_ALIGNMENT_MANIFEST" == "1" ]]; then
  echo "Missing alignment provenance manifest: $ALIGNMENT_MANIFEST" >&2
  exit 1
else
  echo "WARNING: alignment manifest is absent; R.npy/t.npy hashes will be recorded," >&2
  echo "but train/dev provenance cannot be automatically verified." >&2
fi

run_variant() {
  local output_dir="$DEV_ROOT/$CONFIGURATION_NAME"
  local metrics_file="$output_dir/metrics.json"
  shift 0

  if [[ -s "$metrics_file" ]]; then
    echo "[SKIP] Completed variant: $CONFIGURATION_NAME"
    return 0
  fi

  if [[ -d "$output_dir" ]]; then
    echo "[CLEAN] Removing incomplete output: $output_dir"
    rm -rf "$output_dir"
  fi
  mkdir -p "$DEV_ROOT" "$CACHE_DIR"

  echo "[RUN] $CONFIGURATION_NAME"
  python src/previous_pipeline_baseline.py \
    "${COMMON_ARGS[@]}" \
    "$@"
}

case "$CONFIGURATION_NAME" in
  previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_cosine_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling token_mean \
      --kabsch_stage sentence
    ;;

  previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_csls_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling token_mean \
      --kabsch_stage sentence \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_no_kabsch_token_mean_cosine_lora)
    run_variant \
      --pooling token_mean \
      --no_kabsch
    ;;

  previous_pipeline_xlmr_50ep_no_kabsch_token_mean_csls_lora)
    run_variant \
      --pooling token_mean \
      --no_kabsch \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_cosine_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling token_idf \
      --idf_csv "$TRAIN_CSV" \
      --kabsch_stage sentence
    ;;

  previous_pipeline_xlmr_50ep_10K_kabsch_token_idf_csls_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling token_idf \
      --idf_csv "$TRAIN_CSV" \
      --kabsch_stage sentence \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_no_kabsch_token_idf_cosine_lora)
    run_variant \
      --pooling token_idf \
      --idf_csv "$TRAIN_CSV" \
      --no_kabsch
    ;;

  previous_pipeline_xlmr_50ep_no_kabsch_token_idf_csls_lora)
    run_variant \
      --pooling token_idf \
      --idf_csv "$TRAIN_CSV" \
      --no_kabsch \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_cosine_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling sentence_mean \
      --kabsch_stage sentence
    ;;

  previous_pipeline_xlmr_50ep_10K_kabsch_sentence_mean_csls_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling sentence_mean \
      --kabsch_stage sentence \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_no_kabsch_sentence_mean_cosine_lora)
    run_variant \
      --pooling sentence_mean \
      --no_kabsch
    ;;

  previous_pipeline_xlmr_50ep_no_kabsch_sentence_mean_csls_lora)
    run_variant \
      --pooling sentence_mean \
      --no_kabsch \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_10K_token_kabsch_token_mean_csls_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling token_mean \
      --kabsch_stage token \
      --use_csls
    ;;

  previous_pipeline_xlmr_50ep_10K_token_kabsch_token_idf_csls_lora)
    test -f "$ALIGN_DIR/R.npy" && test -f "$ALIGN_DIR/t.npy"
    run_variant \
      --alignment_dir "$ALIGN_DIR" \
      "${alignment_args[@]}" \
      --pooling token_idf \
      --idf_csv "$TRAIN_CSV" \
      --kabsch_stage token \
      --use_csls
    ;;

  *)
    echo "Unknown XLM-R LoRA projection configuration: $CONFIGURATION_NAME" >&2
    exit 2
    ;;
esac
