#!/usr/bin/env bash
# Steps 2-5 of the corrected revision re-run (after fit_cox_models completes).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOG="${LOG:-$ROOT/logs/revision_rerun_steps2_5_$(date +%Y%m%d_%H%M%S).log}"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 2: revision_robustness ==="
cd analysis && "$PYTHON" revision_robustness.py

echo "=== Stage 2b: selection_sensitivity ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history

echo "=== Stage 2c: pair_bootstrap (ISS-01) ==="
"$PYTHON" pair_bootstrap_se.py --scope all --n-bootstrap 200

echo "=== Stage 3 (optional): ISS-04 all-types decomposition ==="
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
