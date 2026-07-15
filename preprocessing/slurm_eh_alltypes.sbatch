#!/bin/bash
#SBATCH --job-name=eh_alltypes
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_eh_alltypes_%j.out
#SBATCH --error=logs/slurm_eh_alltypes_%j.err
#
# Build data/event_history_alltypes (include accepts) in parallel with default EH.
# Does NOT clear caches or write model results — that is iss04_fit.

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

OUT="$ROOT/data/event_history_alltypes"
MATCHED="$ROOT/data/input/matched_questions.parquet"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} eh_alltypes ==="
"$PYTHON" preprocessing/create_matched_event_histories.py \
  --matched "$MATCHED" \
  --output "$OUT" \
  --include-accepts
ls -lah "$OUT"
echo "=== $(date -Is) eh_alltypes done ==="
