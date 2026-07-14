#!/bin/bash
#SBATCH --job-name=boot_shard
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_boot_shard_%A_%a.out
#SBATCH --error=logs/slurm_boot_shard_%A_%a.err
#SBATCH --array=0-4
#
# Full 100× matched-pair bootstrap after rematch (caches cleared).
# Covers reps 1–20, 21–40, 41–60, 61–80, 81–100 with --no-resume.

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

STARTS=(1 21 41 61 81)
ENDS=(20 40 60 80 100)
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset}"
REP_START="${STARTS[$TASK]}"
REP_END="${ENDS[$TASK]}"

# Prefer DATA_VERSION-tagged cache (rev1); fall back to untagged legacy name.
CACHE="$ROOT/analysis/data_cache/intervals_full_answer_comment_rev1.parquet"
if [ ! -f "$CACHE" ]; then
  CACHE="$ROOT/analysis/data_cache/intervals_full_answer_comment.parquet"
fi
if [ ! -f "$CACHE" ]; then
  echo "ERROR: missing interval cache (intervals_full_answer_comment_rev1.parquet) — eh_default must finish first."
  exit 1
fi
echo "Using interval cache: $CACHE"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} task=$TASK ==="
echo "Shard reps ${REP_START}-${REP_END} of ${N_BOOTSTRAP}; n_jobs=${N_JOBS} (--no-resume)"
echo "Python: $PYTHON"
"$PYTHON" -c "import lifelines, pandas, numpy; print('lifelines', lifelines.__version__)"

cd "$ROOT/analysis"
"$PYTHON" pair_bootstrap_se.py \
  --scope "$SCOPE" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED" \
  --rep-start "$REP_START" \
  --rep-end "$REP_END" \
  --n-jobs "$N_JOBS" \
  --no-resume

echo "=== $(date -Is) shard ${REP_START}-${REP_END} done ==="
