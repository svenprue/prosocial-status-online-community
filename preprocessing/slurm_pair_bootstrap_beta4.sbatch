#!/bin/bash
#SBATCH --job-name=boot_beta4
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_boot_beta4_%j.out
#SBATCH --error=logs/slurm_boot_beta4_%j.err
#
# 2026-07-14 estimand decision: re-target the matched-pair bootstrap onto beta_4
# (is_treated_active) alone, the DiD treatment effect, instead of the summed
# treated_post_question + is_treated_active contrast. New checkpoint namespace
# (pair_bootstrap_checkpoint_all_b4*) so this cannot merge with the old summed-stat
# checkpoints. Single job, full 100 replicates, Model A, pooled scope.

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
N_JOBS="${N_JOBS:-8}"
SCOPE="${SCOPE:-all}"
SEED="${SEED:-42}"

CACHE="$ROOT/analysis/data_cache/intervals_full_answer_comment_rev1.parquet"
if [ ! -f "$CACHE" ]; then
  CACHE="$ROOT/analysis/data_cache/intervals_full_answer_comment.parquet"
fi
if [ ! -f "$CACHE" ]; then
  echo "ERROR: missing interval cache (intervals_full_answer_comment_rev1.parquet)."
  exit 1
fi
echo "Using interval cache: $CACHE"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} ==="
echo "beta_4 bootstrap: n_bootstrap=${N_BOOTSTRAP} n_jobs=${N_JOBS} scope=${SCOPE} seed=${SEED}"
echo "Python: $PYTHON"
"$PYTHON" -c "import lifelines, pandas, numpy; print('lifelines', lifelines.__version__)"

cd "$ROOT/analysis"
"$PYTHON" pair_bootstrap_se.py \
  --scope "$SCOPE" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED" \
  --n-jobs "$N_JOBS"

echo "=== $(date -Is) beta_4 bootstrap done ==="
