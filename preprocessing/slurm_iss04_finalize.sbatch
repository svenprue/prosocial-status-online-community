#!/bin/bash
#SBATCH --job-name=iss04_fin
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm_iss04_fin_%j.out
#SBATCH --error=logs/slurm_iss04_fin_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

echo "=== $(date -Is) host=$(hostname) post-ISS-04 create_figures + PDF ==="
cd "$ROOT/analysis"
"$PYTHON" create_figures.py
cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
fi
echo "=== $(date -Is) iss04_finalize done ==="
