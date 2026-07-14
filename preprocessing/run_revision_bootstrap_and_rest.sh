#!/usr/bin/env bash
# Build answer+comment interval cache, then resume checkpointed 100x bootstrap + rest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT/analysis"
CACHE="$ROOT/analysis/data_cache/intervals_full_answer_comment.parquet"

if [ -f "$CACHE" ]; then
  echo "=== $(date -Is) Interval cache already present: $CACHE ==="
  ls -lh "$CACHE"
else
  echo "=== $(date -Is) Building intervals_full_answer_comment cache (once) ==="
  "$PYTHON" -c "
from cox_config import PRIMARY_HELP_TYPES
from cox_data import load_and_prepare
load_and_prepare('../data/event_history', event_help_types=PRIMARY_HELP_TYPES)
print('=== interval cache build done ===', flush=True)
"
  ls -lh "$CACHE"
fi

echo "=== $(date -Is) Pair bootstrap (100x, checkpointed) ==="
"$PYTHON" pair_bootstrap_se.py --scope all --n-bootstrap 100

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
