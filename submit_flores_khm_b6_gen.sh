#!/usr/bin/env bash
set -e

sbatch \
  --job-name=flores_khm_b6_gen \
  submit_flores_generalization_reviewer_response.slurm \
  khmkhmr_vielatn
