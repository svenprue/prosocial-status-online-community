#!/bin/bash
#SBATCH --job-name=boot_merge
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_boot_merge_%j.out
#SBATCH --error=logs/slurm_boot_merge_%j.err
#
# After all bootstrap shards finish: merge checkpoints, refresh figures/tables, PDF.
# Usually submitted with --dependency=afterok:<array_jobid> by submit_pair_bootstrap_shards.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
SCOPE="${SCOPE:-all}"
SEED="${SEED:-42}"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} merge+rest ==="

cd "$ROOT/analysis"
echo "=== Merge pair-bootstrap checkpoints ==="
"$PYTHON" pair_bootstrap_se.py --merge --scope "$SCOPE" --n-bootstrap "$N_BOOTSTRAP" --seed "$SEED"

echo "=== $(date -Is) create_figures ==="
"$PYTHON" create_figures.py

echo "=== $(date -Is) effect_sizes ==="
"$PYTHON" effect_sizes.py

echo "=== $(date -Is) help_rate_over_time ==="
"$PYTHON" help_rate_over_time.py --input ../data/event_history

echo "=== $(date -Is) pdflatex ==="
cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
elif [ -f latex_paper/main.tex ]; then
  cd latex_paper
  pdflatex -interaction=nonstopmode main.tex || true
  pdflatex -interaction=nonstopmode main.tex || true
else
  echo "⚠ No manuscript.tex / latex_paper/main.tex; skipping PDF."
fi

echo "=== $(date -Is) All done ==="
