#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS=(
    "token_mlp256_sym_t007_lr1e5"
    "sent_mlp256_sym_t007_lr1e5"
    "sent_linear256_sym_t007_lr1e5"
    "token_identity_sym_t007_lr1e5"
    "token_mlp256_sym_t005_lr1e5"
    "token_mlp128_oneway_t007_lr5e6"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "Submitting job for config: $CONFIG"

    qsub -v FULL_ENCODER_CONFIG="$CONFIG" "$SCRIPT_DIR/run_full_encoder_contrastive_baselines.pbs"

    echo "Submitted job for config: $CONFIG"
    echo "------------------------------------------"

    sleep 1
done
