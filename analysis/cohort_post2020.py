"""ISS-37 (Audit #9, R2-6): estimate Model A on the POST-2020 subsample separately.

Complements the pre-2020 cohort check in cohort_robustness.py. The pre-2020 check
shares ~79% of rows / ~91% of events with the full sample (a compositional check);
the post-2020 remainder (~5.16M rows / ~291,818 events, event rate ~0.057 vs ~0.164
pre-2020) is the genuinely complementary, near-independent replication.

Fits ONLY the two new post-2020 models (pooled + newcomer) so the expensive cached
pooled/pre-2020 fits are never touched. Writes results to a SEPARATE csv
(results_cohort_post2020.csv) so the manuscript tables stay byte-identical until the
numbers are reviewed and integrated by hand.

Run (on a compute node):
  COX_MAX_FIT_ROWS=12000000 python cohort_post2020.py
"""
import os
import sys

import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import CACHE_DIR, PRIMARY_HELP_TYPES
from cox_data import load_and_prepare
from cohort_robustness import (
    _cohort_series_from_matched_questions,
    _cohort_series_from_model_df,
    _cohort_series_from_timelines,
    _default_input_folder,
    _fit_main_model,
)


def run_post2020(input_folder: str, use_cache: bool = True) -> pd.DataFrame:
    model_df, _ = load_and_prepare(
        input_folder, sample_size=None, event_help_types=PRIMARY_HELP_TYPES
    )
    os.makedirs(CACHE_DIR, exist_ok=True)

    cohort_series, cohort_source = _cohort_series_from_model_df(model_df)
    if cohort_series is None:
        cohort_series, cohort_source = _cohort_series_from_timelines(model_df, input_folder)
    if cohort_series is None:
        cohort_series, cohort_source = _cohort_series_from_matched_questions(model_df, input_folder)
    if cohort_series is None:
        raise RuntimeError("No usable question-year/cohort field found; cannot fit post-2020 subset.")

    cohort_df = model_df.copy()
    cohort_df["cohort_year"] = cohort_series
    post_2020 = cohort_df[cohort_df["cohort_year"].notna() & (cohort_df["cohort_year"] >= 2020)].copy()
    print(
        f"✓ Cohort field from {cohort_source}; post-2020 pooled: "
        f"{int(post_2020['unique_id'].nunique()):,} questions / "
        f"{len(post_2020):,} interval rows / "
        f"{int(post_2020['event_occurred'].sum()):,} events "
        f"(event rate {post_2020['event_occurred'].sum() / max(len(post_2020), 1):.4f})",
        flush=True,
    )

    rows = []
    pooled = _fit_main_model(post_2020, "ModelA_Post2020", use_cache=use_cache)
    if pooled is not None:
        rows.append(pooled)

    if "tenure_bucket" in post_2020.columns:
        post_nc = post_2020[post_2020["tenure_bucket"] == "< 1 Week"].copy()
        print(
            f"  Post-2020 newcomer (< 1 Week): "
            f"{int(post_nc['unique_id'].nunique()):,} questions / "
            f"{int(post_nc['event_occurred'].sum()):,} events",
            flush=True,
        )
        nc = _fit_main_model(post_nc, "ModelA_Post2020_Newcomer", use_cache=use_cache)
        if nc is not None:
            rows.append(nc)

    results = pd.DataFrame(rows)
    out_path = os.path.join(CACHE_DIR, "results_cohort_post2020.csv")
    results.to_csv(out_path, index=False)
    print(f"✓ Saved post-2020 results to {out_path}")
    if not results.empty:
        for _, r in results.iterrows():
            print(
                f"  {r['model']}: arrival HR {r['HR_increment_only']:.4f} "
                f"[{r['arrival_ci_lo']:.4f}, {r['arrival_ci_hi']:.4f}] "
                f"(p={r['arrival_p']:.3g}, events={int(r['events'])})",
                flush=True,
            )
    return results


if __name__ == "__main__":
    run_post2020(_default_input_folder(), use_cache=True)
