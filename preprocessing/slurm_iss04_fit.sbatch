#!/bin/bash
#SBATCH --job-name=iss04_fit
#SBATCH --partition=normal
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_iss04_fit_%j.out
#SBATCH --error=logs/slurm_iss04_fit_%j.err
#
# Composite outcome on pre-built event_history_alltypes (after clear barrier).

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

OUT="$ROOT/data/event_history_alltypes"
echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} iss04_fit ==="
test -d "$OUT"
ls -lah "$OUT"
cd "$ROOT/analysis"
"$PYTHON" -c "
import revision_robustness as r
r.run_composite_outcome('$OUT', use_cache=False)
print('=== ISS-04 composite done ===', flush=True)
"
echo "=== $(date -Is) iss04_fit done ==="
