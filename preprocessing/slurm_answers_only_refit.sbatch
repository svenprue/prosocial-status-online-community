#!/bin/bash
#SBATCH --job-name=ao_refit
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_ao_refit_%j.out
#SBATCH --error=logs/slurm_ao_refit_%j.err
#
# 2026-07-15 answers-only default-outcome switch (reviews/answers_only_default_spec.md).
# PRIMARY_HELP_TYPES=["answer"] and DATA_VERSION bumped to rev2ao in cox_config.py, so
# every fit_cox_cached cache_name misses and must refit. Runs, in order:
#   1. fit_cox_models.py     — pooled Model A/B, 7-bucket Model A/B, RT bins (Model A x10)
#   2. revision_robustness.py --only observable  — baseline + observable-controls specs
#   3. revision_robustness.py --only quality     — 4 speed-quality specs
#   4. revision_robustness.py --only newcomer    — 6 newcomer-bucket specs
#   5. revision_robustness.py --only rt_bins_quality — RT bins x10 with quality controls
#   6. cohort_robustness.py  — pooled/newcomer x full-sample/pre-2020
# Does NOT touch run_composite_outcome (results_composite_outcome.csv) — per spec T6,
# those answers_only/comments_only/... fits already exist and are outcome-explicit
# (independent of PRIMARY_HELP_TYPES), so re-fitting them here would be redundant.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} answers-only refit ==="

cd "$ROOT/analysis"
"$PYTHON" -c "from cox_config import PRIMARY_HELP_TYPES, DATA_VERSION; print('PRIMARY_HELP_TYPES=', PRIMARY_HELP_TYPES, 'DATA_VERSION=', DATA_VERSION)"

echo "=== $(date -Is) [1/6] fit_cox_models.py ==="
"$PYTHON" fit_cox_models.py --input ../data/event_history --no-cache

echo "=== $(date -Is) [2/6] revision_robustness.py --only observable ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache --only observable

echo "=== $(date -Is) [3/6] revision_robustness.py --only quality ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache --only quality

echo "=== $(date -Is) [4/6] revision_robustness.py --only newcomer ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache --only newcomer

echo "=== $(date -Is) [5/6] revision_robustness.py --only rt_bins_quality ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache --only rt_bins_quality

echo "=== $(date -Is) [6/6] cohort_robustness.py ==="
"$PYTHON" cohort_robustness.py --input ../data/event_history --no-cache

echo "=== $(date -Is) answers-only refit done ==="
