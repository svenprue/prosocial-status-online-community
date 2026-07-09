#!/usr/bin/env bash
# Rebuild study pipeline after revision data wiring.
# Run from repo root: bash preprocessing/run_revision_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "=== Stage 2: Raw reciprocity dataset (FULL — may take hours) ==="
"$PYTHON" preprocessing/creating_raw_reciprocity_dataset.py

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

echo "=== Stage 6: Revision robustness (ISS-02, 04, 06, 10, 16) ==="
"$PYTHON" revision_robustness.py --input ../data/event_history --no-cache

echo "=== Stage 6b: Selection sensitivity (ISS-03) ==="
"$PYTHON" selection_sensitivity.py --input ../data/event_history || true

echo "=== Stage 7: Regenerate tables and figures ==="
"$PYTHON" create_figures.py

echo "=== Done ==="
