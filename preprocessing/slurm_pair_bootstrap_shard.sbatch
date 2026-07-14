#!/bin/bash
#SBATCH --job-name=boot_shard
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_boot_shard_%A_%a.out
#SBATCH --error=logs/slurm_boot_shard_%A_%a.err
#SBATCH --array=0-3
#
# One array task per replicate shard of the 100x matched-pair bootstrap.
# Shards (1-based inclusive): 21-40, 41-60, 61-80, 81-100.
# Replicates 1-20 already live in pair_bootstrap_checkpoint_all.csv.
#
# Submit via: preprocessing/submit_pair_bootstrap_shards.sh

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

# Task 0 → 21-40, 1 → 41-60, … (skip 1-20 which are already done)
STARTS=(21 41 61 81)
ENDS=(40 60 80 100)
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset}"
REP_START="${STARTS[$TASK]}"
REP_END="${ENDS[$TASK]}"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} task=$TASK ==="
echo "Shard reps ${REP_START}-${REP_END} of ${N_BOOTSTRAP}; n_jobs=${N_JOBS}"
echo "Python: $PYTHON"
"$PYTHON" -c "import lifelines, pandas, numpy; print('lifelines', lifelines.__version__)"

cd "$ROOT/analysis"
"$PYTHON" pair_bootstrap_se.py \
  --scope "$SCOPE" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED" \
  --rep-start "$REP_START" \
  --rep-end "$REP_END" \
  --n-jobs "$N_JOBS"

echo "=== $(date -Is) shard ${REP_START}-${REP_END} done ==="
