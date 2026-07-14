#!/bin/bash
#SBATCH --job-name=eh_default
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_eh_default_%j.out
#SBATCH --error=logs/slurm_eh_default_%j.err
#
# After rematch: helping metrics → default event histories → clear caches →
# warm primary interval cache (so parallel Cox/revision/bootstrap share it).

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} eh_default ==="

echo "=== Helping metrics (optional) ==="
"$PYTHON" preprocessing/calculate_matched_questions_helping.py || true

echo "=== Event histories (composite help) ==="
"$PYTHON" preprocessing/create_matched_event_histories.py

echo "=== Clear stale caches (solo barrier) ==="
"$PYTHON" analysis/clear_caches.py

echo "=== Warm primary interval cache ==="
cd "$ROOT/analysis"
"$PYTHON" -c "
from cox_config import PRIMARY_HELP_TYPES
from cox_data import load_and_prepare
load_and_prepare('../data/event_history', event_help_types=PRIMARY_HELP_TYPES)
print('=== interval cache warm done ===', flush=True)
"
ls -lh "$ROOT/analysis/data_cache"/intervals_full_answer_comment*.parquet

echo "=== $(date -Is) eh_default done ==="
