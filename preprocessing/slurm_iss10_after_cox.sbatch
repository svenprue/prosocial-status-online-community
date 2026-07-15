#!/bin/bash
#SBATCH --job-name=iss10_cohort
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --output=logs/slurm_iss10_cohort_%j.out
#SBATCH --error=logs/slurm_iss10_cohort_%j.err

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} iss10_cohort ==="
cd "$ROOT/analysis"
# use_cache=True so we reuse ModelA_AllData from cox_main when present;
# Pre2020 still fits fresh.
"$PYTHON" -c "
from cohort_robustness import run_cohort_robustness
run_cohort_robustness('../data/event_history', use_cache=True)
print('=== cohort done ===', flush=True)
"
echo "=== $(date -Is) iss10_cohort done ==="
