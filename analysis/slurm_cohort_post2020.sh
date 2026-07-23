#!/bin/bash
#SBATCH --job-name=post2020
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/yen/home/users/lenardst/prosocial-status-online-community/analysis/logs/slurm_post2020_%j.log

# ISS-37 (Audit #9): Model A on the post-2020 subsample (pooled + newcomer).
set -euo pipefail
REPO=/yen/home/users/lenardst/prosocial-status-online-community
cd "$REPO/analysis"
export COX_MAX_FIT_ROWS=12000000
"$REPO/.venv/bin/python" cohort_post2020.py
