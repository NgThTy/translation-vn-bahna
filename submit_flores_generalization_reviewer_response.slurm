#!/usr/bin/env bash
#SBATCH --job-name=flores_b6_gen
#SBATCH --output=logs/flores_b6_gen_%j.out
#SBATCH --error=logs/flores_b6_gen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00

set -e

# Change this path to your GPU server project path if needed.
cd "$HOME/projects/translation-vn-bahna"

mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo

source "$HOME/miniconda3/etc/profile.d/conda.sh" || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate bahnaric311

echo "Python:"
which python
python --version
echo

echo "GPU:"
nvidia-smi || true
echo

python -m pip install -U datasets peft

./run_flores_generalization_reviewer_response.sh

echo
echo "Job finished at: $(date)"
