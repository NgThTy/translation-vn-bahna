#!/usr/bin/env bash
#SBATCH --job-name=bahna_full_encoder
#SBATCH --output=logs/full_encoder_%j.out
#SBATCH --error=logs/full_encoder_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -e

cd "$HOME/projects/translation-vn-bahna"

mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo

nvidia-smi
echo

# Activate your conda environment.
# Change bahnaric311 if you use a different environment name on the GPU server.
source "$HOME/miniconda3/etc/profile.d/conda.sh" || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate bahnaric311

echo "Python:"
which python
python --version
echo

./run_full_encoder_contrastive_baselines.sh

echo
echo "Job finished at: $(date)"
