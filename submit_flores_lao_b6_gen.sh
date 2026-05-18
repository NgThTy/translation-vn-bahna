#!/usr/bin/env bash
set -e

sbatch \
  --job-name=flores_lao_b6_gen \
  submit_flores_generalization_reviewer_response.slurm \
  laolaoo_vielatn
