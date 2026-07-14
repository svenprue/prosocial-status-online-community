#!/bin/bash
#SBATCH --job-name=match_merge
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_match_merge_%j.out
#SBATCH --error=logs/slurm_match_merge_%j.err
#
# Stage 4 merge: concat match shards → matched_questions.parquet + balance /
# love / common-support diagnostics + inspect.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} match_merge ==="
"$PYTHON" matching/create_matched_dataset.py --merge-shards --n-shards "$N_SHARDS"
"$PYTHON" matching/inspect_matched_dataset.py
ls -lh data/input/matched_questions.parquet matching/balance_check.tex matching/love_plot.pdf matching/common_support.pdf
echo "=== $(date -Is) match_merge done ==="
