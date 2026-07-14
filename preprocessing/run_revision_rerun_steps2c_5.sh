#!/usr/bin/env bash
# Resume revision re-run from Stage 2c (after revision_robustness core specs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOG="${LOG:-$ROOT/logs/revision_rerun_steps2c_5_$(date +%Y%m%d_%H%M%S).log}"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 2a: finish remaining robustness (placebo, cohort) ==="
cd analysis
INPUT="../data/event_history"
"$PYTHON" -c "
import revision_robustness as r
r.run_viewcount_placebo('${INPUT}', use_cache=True)
r.run_newcomer_bucket_checks('${INPUT}', use_cache=True)
try:
    from cohort_robustness import run_cohort_robustness
    run_cohort_robustness('${INPUT}', use_cache=True)
except Exception as e:
    print(f'⚠ Cohort robustness skipped: {e}')
"

echo "=== Stage 2b: selection_sensitivity ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history

echo "=== Stage 2c: pair_bootstrap (ISS-01) ==="
"$PYTHON" pair_bootstrap_se.py --scope all --n-bootstrap 200

echo "=== Stage 3: ISS-04 all-types decomposition ==="
cd "$ROOT"
"$PYTHON" preprocessing/create_matched_event_histories.py \
  --output data/event_history_alltypes --include-accepts
"$PYTHON" -c "
import sys
sys.path.insert(0, 'analysis')
import revision_robustness as r
r.run_composite_outcome('data/event_history_alltypes', use_cache=False)
"

echo "=== Stage 4: create_figures ==="
cd analysis && "$PYTHON" create_figures.py

echo "=== Stage 4b: effect_sizes ==="
"$PYTHON" effect_sizes.py

echo "=== Stage 4c: help_rate_over_time ==="
"$PYTHON" help_rate_over_time.py --input ../data/event_history

echo "=== Stage 5: pdflatex ==="
cd "$ROOT"
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex

echo "=== Done ==="
