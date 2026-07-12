#!/usr/bin/env bash
#SBATCH --job-name=bahna_full_encoder_select
#SBATCH --output=logs/full_encoder_select_%j.out
#SBATCH --error=logs/full_encoder_select_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/translation-vn-bahna}"
cd "$PROJECT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-bahnaric311}"

FULL_ENCODER_ACTION=select bash run_full_encoder_contrastive_baselines.sh
