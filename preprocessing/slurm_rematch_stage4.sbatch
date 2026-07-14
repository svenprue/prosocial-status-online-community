#!/bin/bash
#SBATCH --job-name=rematch_s4
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_rematch_s4_%j.out
#SBATCH --error=logs/slurm_rematch_s4_%j.err
#
# Stage 4 only: rematch + inspect. Does not touch Stages 2–3 inputs.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} Stage 4 rematch ==="
echo "Python: $PYTHON"
ls -lh data/study_datasets/question_centered_model_7d_processed.parquet

echo "=== Stage 4: Rematch ==="
"$PYTHON" matching/create_matched_dataset.py

echo "=== Stage 4b: Inspect matched dataset ==="
"$PYTHON" matching/inspect_matched_dataset.py

echo "=== Key outputs ==="
ls -lh data/input/matched_questions.parquet \
  matching/balance_check.tex matching/love_plot.pdf matching/common_support.pdf \
  matching/prematch_*.tex matching/prematch_*.pdf 2>/dev/null || true

echo "=== $(date -Is) Stage 4 done ==="
