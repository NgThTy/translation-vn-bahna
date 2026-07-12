# #!/usr/bin/env bash
# #SBATCH --job-name=bahna_full_encoder
# #SBATCH --output=logs/full_encoder_%j.out
# #SBATCH --error=logs/full_encoder_%j.err
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G
# #SBATCH --time=24:00:00

# set -e

# cd "$HOME/projects/translation-vn-bahna"

# mkdir -p logs

# echo "Job started at: $(date)"
# echo "Running on node: $(hostname)"
# echo "Current directory: $(pwd)"
# echo

# nvidia-smi
# echo

# # Activate your conda environment.
# # Change bahnaric311 if you use a different environment name on the GPU server.
# source "$HOME/miniconda3/etc/profile.d/conda.sh" || source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate bahnaric311

# echo "Python:"
# which python
# python --version
# echo

# ./run_full_encoder_contrastive_baselines.sh

# echo
# echo "Job finished at: $(date)"

#################################
#!/usr/bin/env bash
#SBATCH --job-name=bahna_full_encoder
#SBATCH --output=logs/full_encoder_%A_%a.out
#SBATCH --error=logs/full_encoder_%A_%a.err
#SBATCH --array=0-5%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/translation-vn-bahna}"
cd "$PROJECT_DIR"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-bahnaric311}"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# Each array task owns one GPU and one training configuration. The %2 array
# throttle lets at most two configurations run concurrently. Re-submit the
# same array job to resume unfinished configurations; completed configurations
# are skipped automatically.
FULL_ENCODER_INDEX="$SLURM_ARRAY_TASK_ID" \
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-105}" \
bash run_full_encoder_contrastive_baselines.sh

echo "Job finished: $(date)"
