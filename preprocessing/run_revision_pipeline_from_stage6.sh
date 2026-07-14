#!/usr/bin/env bash
# Resume revision pipeline from Stage 6 (event histories already complete).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONUNBUFFERED=1
cd "$ROOT"

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
