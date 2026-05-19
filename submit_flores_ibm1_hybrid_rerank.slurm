#!/usr/bin/env bash
#SBATCH --job-name=flores_ibm1_hybrid
#SBATCH --output=logs/flores_ibm1_hybrid_%j.out
#SBATCH --error=logs/flores_ibm1_hybrid_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -e

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

python -m pip install -U pandas numpy

./run_flores_ibm1_hybrid_rerank.sh

echo
echo "Job finished at: $(date)"
