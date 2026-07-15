#!/bin/bash
#SBATCH --job-name=match_shard
#SBATCH --partition=normal
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_match_shard_%A_%a.out
#SBATCH --error=logs/slurm_match_shard_%A_%a.err
#SBATCH --array=0-5
#
# Stage 4 match shard: years where year_index % N_SHARDS == task id.
# Uses shared global z-stats from --prepare so propensity scaling matches
# the single-node pipeline.

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

N_SHARDS="${N_SHARDS:-6}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset}"
# Don't inherit bootstrap's N_JOBS=4 — use all CPUs on this node for matching.
N_JOBS="${MATCH_N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} match_shard task=$TASK/$N_SHARDS ==="
"$PYTHON" matching/create_matched_dataset.py \
  --match-shard "$TASK" \
  --n-shards "$N_SHARDS" \
  --n-jobs "$N_JOBS"
echo "=== $(date -Is) match_shard $TASK done ==="
