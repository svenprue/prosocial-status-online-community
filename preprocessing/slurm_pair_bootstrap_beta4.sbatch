#!/bin/bash
#SBATCH --job-name=boot_ao_beta4
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_boot_ao_beta4_%j.out
#SBATCH --error=logs/slurm_boot_ao_beta4_%j.err
#
# 2026-07-15 answers-only default-outcome switch: bootstrap beta_4 (is_treated_active)
# alone, on the answers-only outcome (PRIMARY_HELP_TYPES=["answer"], DATA_VERSION=
# rev2ao). New checkpoint namespace (pair_bootstrap_checkpoint_all_ao_b4*) so this
# cannot merge with prior beta_4-on-composite-outcome or summed-stat checkpoints.
# Single job, full 100 replicates, Model A, pooled scope.
#
# n_jobs=4 (not 8): an earlier n_jobs=8 attempt (job 377043) OOM'd within ~6 minutes
# under the same 350G/8-cpu allocation — each worker holds its own copy of the scoped
# ~64M-row frame before its own MAX_FIT_ROWS subsample, so 8 concurrent workers
# exceeded available memory. n_jobs=4 matches the template that succeeded on
# 2026-07-13 (jobs 370659/372148/371128) under the same allocation.

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

N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
N_JOBS="${N_JOBS:-4}"
SCOPE="${SCOPE:-all}"
SEED="${SEED:-42}"

CACHE="$ROOT/analysis/data_cache/intervals_full_answer_rev2ao.parquet"
if [ ! -f "$CACHE" ]; then
  echo "ERROR: missing interval cache ($CACHE). Run slurm_answers_only_refit.sbatch first."
  exit 1
fi
echo "Using interval cache: $CACHE"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} ==="
echo "answers-only beta_4 bootstrap: n_bootstrap=${N_BOOTSTRAP} n_jobs=${N_JOBS} scope=${SCOPE} seed=${SEED}"
echo "Python: $PYTHON"
"$PYTHON" -c "import lifelines, pandas, numpy; print('lifelines', lifelines.__version__)"
"$PYTHON" -c "from cox_config import PRIMARY_HELP_TYPES, DATA_VERSION; print('PRIMARY_HELP_TYPES=', PRIMARY_HELP_TYPES, 'DATA_VERSION=', DATA_VERSION)" || true

cd "$ROOT/analysis"
"$PYTHON" pair_bootstrap_se.py \
  --scope "$SCOPE" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED" \
  --n-jobs "$N_JOBS"

echo "=== $(date -Is) answers-only beta_4 bootstrap done ==="
