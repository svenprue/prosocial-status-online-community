#!/bin/bash
#SBATCH --job-name=pipe_s5_7
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_pipe_s5_7_%j.out
#SBATCH --error=logs/slurm_pipe_s5_7_%j.err
#
# Stages 5→7 after rematch: helping → event histories → clear_caches →
# Cox → revision robustness → selection → figures → cohort / question-loss.
# Ends by rebuilding the interval cache so bootstrap shards start warm.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} Stages 5→7 ==="

echo "=== Stage 5a: Helping metrics (optional) ==="
"$PYTHON" preprocessing/calculate_matched_questions_helping.py || true

echo "=== Stage 5b: Event histories (composite help events) ==="
"$PYTHON" preprocessing/create_matched_event_histories.py

echo "=== Stage 6: Clear stale caches ==="
"$PYTHON" analysis/clear_caches.py

echo "=== Stage 6: Main Cox models ==="
cd "$ROOT/analysis"
"$PYTHON" fit_cox_models.py --input ../data/event_history --no-cache

echo "=== Stage 6: Revision robustness ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache

echo "=== Stage 6b: Selection sensitivity ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history || true

echo "=== Stage 7: Tables and figures ==="
"$PYTHON" create_figures.py

echo "=== Paper-facing extras that read matched_questions.parquet ==="
cd "$ROOT"
"$PYTHON" analysis/cohort_robustness.py || true
"$PYTHON" preprocessing/investigate_question_loss.py || true

echo "=== Rebuild interval cache for bootstrap shards ==="
cd "$ROOT/analysis"
"$PYTHON" -c "
from cox_config import PRIMARY_HELP_TYPES
from cox_data import load_and_prepare
load_and_prepare('../data/event_history', event_help_types=PRIMARY_HELP_TYPES)
print('=== interval cache build done ===', flush=True)
"
ls -lh "$ROOT/analysis/data_cache/intervals_full_answer_comment.parquet"

echo "=== $(date -Is) Stages 5→7 done ==="
