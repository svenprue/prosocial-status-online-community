#!/usr/bin/env bash
# Resume revision pipeline after main Cox fits (RT bins + robustness + tables).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "=== Stage 6 (resume): Finish Cox models (cached tenure/all-data; RT bins sequential) ==="
cd analysis && "$PYTHON" fit_cox_models.py --input ../data/event_history --n-jobs 1

echo "=== Stage 6: Revision robustness ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache

echo "=== Stage 6b: Selection sensitivity ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history || true

echo "=== Stage 7: Tables and figures ==="
"$PYTHON" create_figures.py

echo "=== Done ==="
