#!/usr/bin/env bash
#SBATCH --job-name=bahna_lora_contrastive
#SBATCH --output=logs/lora_contrastive_%j.out
#SBATCH --error=logs/lora_contrastive_%j.err
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

# Activate conda environment.
source "$HOME/miniconda3/etc/profile.d/conda.sh" || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate bahnaric311

echo "Python:"
which python
python --version
echo

echo "Git commit:"
git rev-parse --short HEAD
echo

chmod +x run_lora_contrastive_baselines.sh

./run_lora_contrastive_baselines.sh

echo
echo "Job finished at: $(date)"
