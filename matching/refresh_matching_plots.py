#!/usr/bin/env python3
"""Regenerate common_support.pdf / love_plot.pdf from existing matched + processed data.

Does NOT re-run propensity-score matching (Stage 4). Loads:
  - data/study_datasets/question_centered_model_7d_processed.parquet  (unmatched pool)
  - data/input/matched_questions.parquet                             (matched pairs)
and rewrites matching/common_support.pdf and matching/love_plot.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from create_matched_dataset import (  # noqa: E402
    CONTINUOUS_COVS,
    DATA_PATH,
    OUTPUT_MATCHED_PATH,
    PLOT_COMMON_SUPPORT,
    PLOT_LOVE,
    generate_love_plot,
    plot_psm_diagnostics_phase1,
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Unmatched processed data not found: {DATA_PATH}")
    if not OUTPUT_MATCHED_PATH.exists():
        raise FileNotFoundError(f"Matched data not found: {OUTPUT_MATCHED_PATH}")

    print(f"Loading unmatched: {DATA_PATH}")
    unmatched = pd.read_parquet(DATA_PATH)
    print(f"  {len(unmatched):,} rows")

    print(f"Loading matched: {OUTPUT_MATCHED_PATH}")
    matched = pd.read_parquet(OUTPUT_MATCHED_PATH)
    print(f"  {len(matched):,} rows, {matched['match_id'].nunique():,} pairs")

    covs = [c for c in CONTINUOUS_COVS if c in unmatched.columns and c in matched.columns]
    for extra in ("tag_accept_share_avg", "tag_accept_share_max"):
        if extra in unmatched.columns and extra in matched.columns:
            covs.append(extra)
    if not covs:
        raise RuntimeError("No overlapping continuous covariates between unmatched and matched")
    print(f"Covariates for plots ({len(covs)}): {covs}")

    print(f"Writing {PLOT_COMMON_SUPPORT}")
    plot_psm_diagnostics_phase1(unmatched, matched, "hasAnswer", covs)
    print(f"Writing {PLOT_LOVE}")
    generate_love_plot(unmatched, matched, "hasAnswer", covs, str(PLOT_LOVE))
    print("Done.")


if __name__ == "__main__":
    main()
