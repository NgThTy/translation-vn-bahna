#!/usr/bin/env bash
#SBATCH --job-name=zh_za_lora
#SBATCH --output=logs/zh_za_lora_%j.out
#SBATCH --error=logs/zh_za_lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00

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

echo "GPU:"
nvidia-smi || true
echo

echo "Checking input files..."
test -f data/other_low_resource_generalization/zhuang_chinese/train.csv
test -f data/other_low_resource_generalization/zhuang_chinese/test.csv

python src/lora_projection_generalization_train_eval.py \
  --train_csv data/other_low_resource_generalization/zhuang_chinese/train.csv \
  --test_csv data/other_low_resource_generalization/zhuang_chinese/test.csv \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --pooling token_mean \
  --projection_dim 256 \
  --epochs 10 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --eval_retrievals both \
  --csls_k 10 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --output_dir results/baselines/other_zhuang_chinese_b6style_lora_projection_10ep_token_mean_both

echo
echo "Job finished at: $(date)"
