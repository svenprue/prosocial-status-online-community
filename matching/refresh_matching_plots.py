#!/usr/bin/env python3
"""Regenerate common_support.pdf / love_plot.pdf from existing matched data.

Does NOT re-run propensity-score matching (Stage 4). Loads:
  - data/input/prematch_pool.parquet        (slim pre-match pool written by create_matched_dataset.py)
  - data/input/prematch_covariates.json     (exact covariate list matching used)
  - data/input/matched_questions.parquet    (matched pairs)
and rewrites matching/common_support.pdf and matching/love_plot.pdf.

The pre-match pool + covariate list are persisted by create_matched_dataset.py so the
diagnostics are built from exactly the data and covariates matching used (the raw
processed parquet lacks the derived covariates and the treatment recode/filters).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from create_matched_dataset import (  # noqa: E402
    OUTPUT_MATCHED_PATH,
    PLOT_COMMON_SUPPORT,
    PLOT_LOVE,
    PREMATCH_COVS_PATH,
    PREMATCH_POOL_PATH,
    generate_love_plot,
    plot_psm_diagnostics_phase1,
)


def main() -> None:
    if not PREMATCH_POOL_PATH.exists() or not PREMATCH_COVS_PATH.exists():
        raise FileNotFoundError(
            f"Pre-match pool/covariate list not found ({PREMATCH_POOL_PATH}). "
            "Run create_matched_dataset.py first — it writes the pool and covariate "
            "list these plots must be built from."
        )
    if not OUTPUT_MATCHED_PATH.exists():
        raise FileNotFoundError(f"Matched data not found: {OUTPUT_MATCHED_PATH}")

    print(f"Loading pre-match pool: {PREMATCH_POOL_PATH}")
    unmatched = pd.read_parquet(PREMATCH_POOL_PATH)
    print(f"  {len(unmatched):,} rows")

    print(f"Loading matched: {OUTPUT_MATCHED_PATH}")
    matched = pd.read_parquet(OUTPUT_MATCHED_PATH)
    print(f"  {len(matched):,} rows, {matched['match_id'].nunique():,} pairs")

    with open(PREMATCH_COVS_PATH) as f:
        covs_all = json.load(f)["continuous_covariates"]
    covs = [c for c in covs_all if c in unmatched.columns and c in matched.columns]
    missing = [c for c in covs_all if c not in covs]
    if missing:
        print(f"  Note: {len(missing)} covariate(s) absent from pool/matched, skipped: {missing}")
    if not covs:
        raise RuntimeError("No overlapping covariates between pre-match pool and matched data")
    print(f"Covariates for plots ({len(covs)}): {covs}")

    print(f"Writing {PLOT_COMMON_SUPPORT}")
    plot_psm_diagnostics_phase1(unmatched, matched, "hasAnswer", covs)
    print(f"Writing {PLOT_LOVE}")
    generate_love_plot(unmatched, matched, "hasAnswer", covs, str(PLOT_LOVE))
    print("Done.")


if __name__ == "__main__":
    main()
