#!/bin/bash
#SBATCH --job-name=rev_iss
#SBATCH --partition=normal
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_rev_%x_%j.out
#SBATCH --error=logs/slurm_rev_%x_%j.err
#
# One revision robustness step. Set REV_STEP via --export:
#   observable | quality | placebo | newcomer

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

STEP="${REV_STEP:?REV_STEP unset (observable|quality|placebo|newcomer)}"
INPUT="${INPUT:-$ROOT/data/event_history}"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} rev step=$STEP ==="
cd "$ROOT/analysis"
"$PYTHON" -c "
import revision_robustness as r
step = '$STEP'
inp = '$INPUT'
use_cache = False
if step == 'observable':
    r.run_observable_controls(inp, use_cache=use_cache)
elif step == 'quality':
    r.run_answer_quality(inp, use_cache=use_cache)
elif step == 'placebo':
    r.run_viewcount_placebo(inp, use_cache=use_cache)
elif step == 'newcomer':
    r.run_newcomer_bucket_checks(inp, use_cache=use_cache)
else:
    raise SystemExit(f'unknown REV_STEP={step!r}')
print(f'=== rev step {step} done ===', flush=True)
"
echo "=== $(date -Is) rev_$STEP done ==="
