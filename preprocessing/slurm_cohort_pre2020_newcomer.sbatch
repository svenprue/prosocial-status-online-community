#!/bin/bash
#SBATCH --job-name=cohort_nc
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_cohort_pre2020_newcomer_%j.out
#SBATCH --error=logs/slurm_cohort_pre2020_newcomer_%j.err
#
# ISS-10 extension: add ModelA_Pre2020_Newcomer to cohort robustness,
# regenerate the cohort table, and rebuild manuscript.pdf.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} cohort_pre2020_newcomer start ==="
cd "$ROOT/analysis"
"$PYTHON" cohort_robustness.py --input ../data/event_history
"$PYTHON" create_figures.py
cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
fi
echo "=== $(date -Is) cohort_pre2020_newcomer done ==="
