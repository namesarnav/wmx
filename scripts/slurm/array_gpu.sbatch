#!/bin/bash
# One SLURM array task = one run of a sweep. Submit via: python scripts/sweep.py SWEEP.yaml --slurm
#SBATCH --job-name=wmx-gpu
#SBATCH --output=results/slurm/%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# VERIFY: add the NYU partition/account lines (e.g. #SBATCH --account=..., #SBATCH --partition=...).
set -euo pipefail
poetry run python scripts/sweep.py "$1" --index "$SLURM_ARRAY_TASK_ID" "${@:2}"  # extra args, e.g. --group 3
