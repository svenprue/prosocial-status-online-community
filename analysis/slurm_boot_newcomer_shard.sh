#!/bin/bash
#SBATCH --job-name=boot_nc_sh
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=700G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/yen/home/users/lenardst/prosocial-status-online-community/analysis/logs/slurm_boot_nc_shard_%A_%a.log

# ISS-31 (Audit #10): newcomer (< 1 Week) matched-pair bootstrap, 500 reps, sharded
# 4 ways across nodes (each node has its own memory bandwidth). Full-newcomer fits
# (16M cap > ~14M newcomer interval rows => no per-replicate subsampling).
set -euo pipefail
REPO=/yen/home/users/lenardst/prosocial-status-online-community
cd "$REPO/analysis"

export COX_MAX_FIT_ROWS=16000000
export COX_MAX_FIT_WORKERS=24
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

START=$(( SLURM_ARRAY_TASK_ID * 125 + 1 ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * 125 ))
echo "Shard $SLURM_ARRAY_TASK_ID: reps $START..$END"

"$REPO/.venv/bin/python" pair_bootstrap_se.py \
  --scope "< 1 Week" \
  --n-bootstrap 500 \
  --seed 42 \
  --n-jobs 24 \
  --rep-start "$START" \
  --rep-end "$END"
