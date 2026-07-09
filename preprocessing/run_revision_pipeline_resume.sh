#!/usr/bin/env bash
# Resume revision pipeline from Stage 3 (after Stage 2 raw build completed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "=== Stage 3: Processed reciprocity dataset ==="
"$PYTHON" preprocessing/processing_reciprocity_dataset.py

echo "=== Stage 4: Matching ==="
"$PYTHON" matching/create_matched_dataset.py

echo "=== Stage 5a: Helping metrics (optional) ==="
"$PYTHON" preprocessing/calculate_matched_questions_helping.py || true

echo "=== Stage 5b: Event histories (composite help events) ==="
"$PYTHON" preprocessing/create_matched_event_histories.py

echo "=== Stage 6: Clear stale caches ==="
"$PYTHON" analysis/clear_caches.py

echo "=== Stage 6: Main Cox models ==="
cd analysis && "$PYTHON" fit_cox_models.py --input ../data/event_history --no-cache

echo "=== Stage 6: Revision robustness ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache

echo "=== Stage 6b: Selection sensitivity ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history || true

echo "=== Stage 7: Tables and figures ==="
"$PYTHON" create_figures.py

echo "=== Done ==="
