#!/usr/bin/env bash
# Resume missing revision robustness (ISS-04 + placebo/newcomer/cohort/selection).
# Does NOT run pair_bootstrap — that is already in progress under
# preprocessing/run_revision_rerun_steps2c_5.sh (PID parent tree from 03:08).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
INPUT="$ROOT/data/event_history"
cd "$ROOT/analysis"

echo "=== $(date -Is) ISS-04: composite outcome ==="
"$PYTHON" -c "
import gc
import revision_robustness as r
r.run_composite_outcome('${INPUT}', use_cache=True)
gc.collect()
print('=== ISS-04 done ===', flush=True)
"

echo "=== $(date -Is) ISS-16 + newcomer + cohort ==="
"$PYTHON" -c "
import gc
import revision_robustness as r
r.run_viewcount_placebo('${INPUT}', use_cache=True)
gc.collect()
r.run_newcomer_bucket_checks('${INPUT}', use_cache=True)
gc.collect()
try:
    from cohort_robustness import run_cohort_robustness
    run_cohort_robustness('${INPUT}', use_cache=True)
except Exception as e:
    print(f'⚠ Cohort robustness skipped: {e}', flush=True)
print('=== placebo/newcomer/cohort done ===', flush=True)
"

echo "=== $(date -Is) selection_sensitivity ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history || true

echo "=== $(date -Is) create_figures (refresh tables) ==="
"$PYTHON" create_figures.py

echo "=== $(date -Is) All resume steps done (bootstrap left to existing run) ==="
