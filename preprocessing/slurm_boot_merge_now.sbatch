#!/bin/bash
#SBATCH --job-name=boot_now
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_boot_now_%j.out
#SBATCH --error=logs/slurm_boot_now_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

echo "=== $(date -Is) host=$(hostname) merge bootstrap + figures (ISS-04 still running separately) ==="
cd "$ROOT/analysis"
"$PYTHON" pair_bootstrap_se.py --merge --scope all --n-bootstrap 100 --seed 42
"$PYTHON" create_figures.py
"$PYTHON" effect_sizes.py
"$PYTHON" help_rate_over_time.py --input ../data/event_history
cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
fi
echo "=== $(date -Is) boot_now done ==="
