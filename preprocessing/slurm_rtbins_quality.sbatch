#!/bin/bash
#SBATCH --job-name=rtbin_qual
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_rtbins_quality_%j.out
#SBATCH --error=logs/slurm_rtbins_quality_%j.err
#
# ISS-06 / #27: Model A + answer-quality controls per response-time bin,
# then regenerate LaTeX tables (appendix) and rebuild manuscript.pdf.

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
# Two concurrent bin fits keep peak RAM closer to prior Stage-6 RT-bin jobs.
export COX_MAX_FIT_WORKERS=2

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} rtBins_quality start ==="
cd "$ROOT/analysis"
"$PYTHON" revision_robustness.py --input ../data/event_history --only rt_bins_quality --n-jobs 2
echo "=== $(date -Is) fits done; regenerating tables ==="
"$PYTHON" create_figures.py
cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
fi
echo "=== $(date -Is) rtBins_quality done ==="
