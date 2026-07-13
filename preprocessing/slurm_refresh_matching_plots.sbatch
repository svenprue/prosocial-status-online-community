#!/bin/bash
#SBATCH --job-name=match_plots
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --output=logs/slurm_match_plots_%j.out
#SBATCH --error=logs/slurm_match_plots_%j.err
#
# Refresh appendix common_support.pdf / love_plot.pdf from existing matched parquet
# (no Stage-4 PSM re-run).

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} matching plots ==="
"$PYTHON" matching/refresh_matching_plots.py
ls -lah matching/common_support.pdf matching/love_plot.pdf
echo "=== $(date -Is) matching plots done ==="
