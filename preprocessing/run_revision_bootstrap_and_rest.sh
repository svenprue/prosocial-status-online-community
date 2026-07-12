#!/usr/bin/env bash
# After placebo/figures fix: run checkpointed pair bootstrap, then remaining stages.
# Safe to re-run — bootstrap resumes from pair_bootstrap_checkpoint_*.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT/analysis"

echo "=== $(date -Is) Pair bootstrap (200x, checkpointed) ==="
"$PYTHON" pair_bootstrap_se.py --scope all --n-bootstrap 200

echo "=== $(date -Is) ISS-04 all-types decomposition ==="
cd "$ROOT"
"$PYTHON" preprocessing/create_matched_event_histories.py \
  --output data/event_history_alltypes --include-accepts
"$PYTHON" -c "
import sys
sys.path.insert(0, 'analysis')
import revision_robustness as r
r.run_composite_outcome('data/event_history_alltypes', use_cache=False)
"

echo "=== $(date -Is) create_figures (refresh) ==="
cd "$ROOT/analysis"
"$PYTHON" create_figures.py

echo "=== $(date -Is) effect_sizes ==="
"$PYTHON" effect_sizes.py

echo "=== $(date -Is) help_rate_over_time ==="
"$PYTHON" help_rate_over_time.py --input ../data/event_history

echo "=== $(date -Is) pdflatex ==="
cd "$ROOT"
if [ -f manuscript.tex ]; then
  pdflatex -interaction=nonstopmode manuscript.tex || true
  pdflatex -interaction=nonstopmode manuscript.tex || true
elif [ -f latex_paper/main.tex ]; then
  cd latex_paper
  pdflatex -interaction=nonstopmode main.tex || true
  pdflatex -interaction=nonstopmode main.tex || true
else
  echo "⚠ No manuscript.tex / latex_paper/main.tex found; skipping PDF."
fi

echo "=== $(date -Is) All done ==="
