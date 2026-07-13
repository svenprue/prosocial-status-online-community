#!/bin/bash
#SBATCH --job-name=iss04_comp
#SBATCH --partition=normal
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_iss04_%j.out
#SBATCH --error=logs/slurm_iss04_%j.err
#
# ISS-04: build all-types event history (incl. accepts) + outcome decomposition.
# Independent of bootstrap shards; merge job should wait on this too.

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
MATCHED="$ROOT/data/input/matched_questions.parquet"

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} ISS-04 ==="

echo "=== Build event_history_alltypes (include accepts) ==="
"$PYTHON" preprocessing/create_matched_event_histories.py \
  --matched "$MATCHED" \
  --output "$OUT" \
  --include-accepts
ls -lah "$OUT"

echo "=== run_composite_outcome on alltypes ==="
cd "$ROOT/analysis"
"$PYTHON" -c "
import revision_robustness as r
r.run_composite_outcome('$OUT', use_cache=False)
print('=== ISS-04 composite done ===', flush=True)
"

echo "=== $(date -Is) ISS-04 complete ==="
