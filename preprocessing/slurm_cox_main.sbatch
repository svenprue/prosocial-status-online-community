#!/bin/bash
#SBATCH --job-name=cox_main
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_cox_main_%j.out
#SBATCH --error=logs/slurm_cox_main_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} cox_main ==="
cd "$ROOT/analysis"
"$PYTHON" fit_cox_models.py --input ../data/event_history --no-cache
echo "=== $(date -Is) cox_main done ==="
