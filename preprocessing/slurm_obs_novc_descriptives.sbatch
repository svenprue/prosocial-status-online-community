#!/bin/bash
#SBATCH --job-name=obs_novc
#SBATCH --partition=normal
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --output=logs/slurm_obs_novc_%j.out
#SBATCH --error=logs/slurm_obs_novc_%j.err
#
# ISS brief A+B+E: drop viewCount from observables, fix PRIMARY_HELP_TYPES,
# refresh descriptives for answers+comments, regenerate tables + PDF.

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

echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local} obs_novc start ==="
cd "$ROOT/analysis"

echo "=== Refresh descriptives (PRIMARY_HELP_TYPES) ==="
"$PYTHON" - <<'PY'
import os, pickle
from cox_config import CACHE_DIR, DATA_CACHE_DIR, PRIMARY_HELP_TYPES
from cox_data import load_and_prepare
# Force descriptives recompute on the interval-cache path (filter fix in cox_data).
desc_cache = os.path.join(
    DATA_CACHE_DIR, "descriptives_full_answer_comment_rev1.pkl"
)
if os.path.exists(desc_cache):
    os.remove(desc_cache)
    print(f"removed {desc_cache}")
_, desc = load_and_prepare("../data/event_history", event_help_types=PRIMARY_HELP_TYPES)
os.makedirs(CACHE_DIR, exist_ok=True)
out = os.path.join(CACHE_DIR, "descriptives.pkl")
with open(out, "wb") as f:
    pickle.dump(desc, f)
print(
    f"help_events mean={desc['help_events_mean']:.4f} "
    f"std={desc['help_events_std']:.4f} med={desc['help_events_median']:.4f}"
)
print(f"✓ Wrote {out}")
PY

echo "=== Observable controls (A+B) ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --only observable
echo "=== Newcomer robustness (A observable without viewCount) ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --only newcomer

echo "=== Tables + figures ==="
"$PYTHON" create_figures.py

cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
fi
echo "=== $(date -Is) obs_novc done ==="
