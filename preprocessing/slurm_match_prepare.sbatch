#!/bin/bash
#SBATCH --job-name=match_prep
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_match_prep_%j.out
#SBATCH --error=logs/slurm_match_prep_%j.err
#
# Stage 4 prepare: load processed parquet, build covariates, write per-year
# prematch parquets + global z-stats (no matching yet).

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} match_prep ==="
"$PYTHON" matching/create_matched_dataset.py --prepare
ls -lh data/input/prematch_meta.json data/input/prematch_years/ | head -40
echo "=== $(date -Is) match_prep done ==="
