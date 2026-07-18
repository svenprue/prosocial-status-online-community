#!/bin/bash
#SBATCH --job-name=fd_composite
#SBATCH --partition=normal
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --output=logs/slurm_fd_composite_%j.out
#SBATCH --error=logs/slurm_fd_composite_%j.err
#
# 2026-07-16 full-data refit of the outcome-decomposition table (results_composite_outcome).
# Calls the existing revision_robustness.run_composite_outcome (unchanged) at
# COX_MAX_FIT_ROWS=100M so every help-type outcome (answers/comments/edits/composites)
# is fit on its full frame, not an 8M subsample. This builds the non-answer interval
# caches (comment/edit/answer_comment/answer_comment_edit) under rev2ao on first use,
# so it is the long pole (~day). The answers_only row already matches the full-data
# headline (patched); this job makes the remaining decomposition rows full-data too.

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
export COX_MAX_FIT_ROWS=100000000

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} full-data composite ==="
cd "$ROOT/analysis"
"$PYTHON" -c "from cox_config import MAX_FIT_ROWS; print('MAX_FIT_ROWS', MAX_FIT_ROWS)"
"$PYTHON" -c "
import revision_robustness as r
r.run_composite_outcome('../data/event_history', use_cache=True)
print('=== composite full-data done ===', flush=True)
"
echo "=== $(date -Is) full-data composite complete ==="
