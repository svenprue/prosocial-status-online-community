#!/bin/bash
#SBATCH --job-name=sel_sens
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=logs/slurm_sel_sens_%j.out
#SBATCH --error=logs/slurm_sel_sens_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} selection_sensitivity ==="
cd "$ROOT/analysis"
"$PYTHON" selection_sensitivity.py --input ../data/event_history
echo "=== $(date -Is) sel_sens done ==="
