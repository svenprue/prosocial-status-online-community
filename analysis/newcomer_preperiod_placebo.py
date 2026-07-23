"""ISS-38 (Audit #8, OPTIONAL): pre-period (T_Q - 1 day) placebo for the newcomer
(< 1 Week) stratum, Model A.

Motivation
----------
The observable-controls spec inflates the newcomer waiting-period term
beta_2 (treated_post_question) from +0.050 (baseline Model A) to +0.122. beta_2 is the
treated-vs-control gap in the POST-question / pre-answer waiting window, relative to the
pre-question baseline. If treated and control questions already diverge BEFORE the
question is posted (a pre-existing trend), the inflated beta_2 is partly a
parallel-trends artefact. If the pre-question window is FLAT, the +0.122 is a
post-question exposure phenomenon (a much better story).

Design
------
Timelines are in HOURS relative to question posting: t_start == -48, t_question == 0 for
every question, so a uniform 48h (2-day) pre-question observation window exists and it
carries real helping events (newcomer answers-only: ~188k pre-question events; ~110k in
the final day [-24, 0), ~78k in [-48, -24)).

We insert a cutpoint at t = -24 (T_Q - 1 day) and add a placebo covariate

    treated_pre_period = 1  iff  hasAnswer == 1  AND  -24 <= start < 0

i.e. the treated indicator switched on ONLY in the final pre-question day. With hasAnswer
(the always-on treated main effect) still in the model, the reference phase for hasAnswer
becomes the earliest pre-window [-48, -24). The fitted coefficients are then all measured
relative to that same baseline:

    hasAnswer            : treated-vs-control gap in [-48, -24)   (earliest pre-window)
    treated_pre_period   : ADDITIONAL gap in [-24, 0)            <-- THE PLACEBO (beta_pre)
    treated_post_question: ADDITIONAL gap in the waiting window  (beta_2)
    is_treated_active    : ADDITIONAL gap after answer arrival   (beta_4)

So treated_pre_period is constructed exactly like beta_2 -- a treated x phase interaction
relative to the earliest pre-window -- and is directly comparable to it. Flat pre-trend =>
beta_pre ~ 0 while beta_2 = +0.050/+0.122 => the jump is post-question (exposure). Non-flat
(beta_pre comparably positive) => a pre-existing trend.

We report the placebo for BOTH the baseline Model A (COVARIATES_MAIN) and the
observable-controls spec (COVARIATES_MAIN_OBSERVABLE), since the +0.122 concern lives in
the latter. Fit settings mirror the tabled newcomer robustness fits (revision_robustness):
robust=False (model-based SEs; matched-pair bootstrap supplies clustered uncertainty
elsewhere), 1-hour time rounding, full-data (COX_MAX_FIT_ROWS raised so no subsampling).

Results are PRINTED to stdout (captured in the slurm log). No caches or result files are
written (save_cache=False), per the issue's file-creation constraint.

Usage
-----
    COX_MAX_FIT_ROWS=16000000 python newcomer_preperiod_placebo.py \
        --input ../data/event_history
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cox_config import (  # noqa: E402
    COVARIATES_MAIN,
    COVARIATES_MAIN_OBSERVABLE,
    DATA_CACHE_DIR,
    DATA_VERSION,
    PRIMARY_HELP_TYPES,
    MAX_FIT_ROWS,
)
from cox_fit import fit_cox_cached  # noqa: E402

PRE_BREAK = -24.0          # T_Q - 1 day, in hours (t_question == 0, t_start == -48)
PLACEBO_TERM = "treated_pre_period"
NEWCOMER_BUCKET = "< 1 Week"


def load_newcomer(input_folder: str) -> pd.DataFrame:
    """Return the prepared newcomer (< 1 Week) model frame.

    Reads the cached interval parquet directly when present -- it is the exact output of
    cox_data.load_and_prepare(_build_covariates) and reusing it avoids re-writing the
    shared descriptives cache while other jobs may be running. Falls back to
    load_and_prepare() if the cache is absent.
    """
    cache = os.path.join(DATA_CACHE_DIR, f"intervals_full_answer_{DATA_VERSION}.parquet")
    if os.path.exists(cache):
        print(f"Loading prepared intervals from cache: {cache}")
        df = pd.read_parquet(cache)
    else:
        print("Interval cache absent; calling cox_data.load_and_prepare ...")
        from cox_data import load_and_prepare
        df, _ = load_and_prepare(input_folder, event_help_types=PRIMARY_HELP_TYPES)
    nc = df[df["tenure_bucket"] == NEWCOMER_BUCKET].copy()
    print(f"Newcomer ({NEWCOMER_BUCKET}) interval rows: {len(nc):,}")
    return nc


def add_preperiod_placebo(nc: pd.DataFrame) -> pd.DataFrame:
    """Insert a cutpoint at t = PRE_BREAK and add the treated_pre_period placebo covariate.

    Intervals straddling PRE_BREAK (start < -24 < stop -- every question carries one, since
    the standard first interval runs [-48, first-break) with no boundary at -24) are split
    into contiguous, non-overlapping [start, -24) and [-24, stop) pieces sharing the same
    unique_id. The event flag (which marks an event at the interval's STOP boundary) stays
    with the second piece; the first piece gets event_occurred = 0 (its new stop, -24, is a
    synthetic cutpoint with no event). All pre-question intervals have stop <= 0 (t_question
    = 0 is itself a boundary), so the second piece never crosses into the post-question
    period. Every other covariate is copied unchanged (all are time-invariant per question
    OR are already 0 throughout the pre-question window).
    """
    n_before = len(nc)
    straddle = (nc["start"] < PRE_BREAK) & (nc["stop"] > PRE_BREAK)
    n_straddle = int(straddle.sum())
    keep = nc[~straddle].copy()
    strad = nc[straddle]

    first = strad.copy()
    first["stop"] = PRE_BREAK
    first["event_occurred"] = 0.0
    # Guard against any (rounding) degeneracy where start already >= -24 within a straddler.
    first = first[first["start"] < first["stop"]]

    second = strad.copy()
    second["start"] = PRE_BREAK

    out = pd.concat([keep, first, second], ignore_index=True)
    out[PLACEBO_TERM] = (
        (out["hasAnswer"] == 1)
        & (out["start"] >= PRE_BREAK)
        & (out["start"] < 0.0)
    ).astype(int)

    print(
        f"Cutpoint at t={PRE_BREAK}h: {n_straddle:,} straddling intervals split; "
        f"rows {n_before:,} -> {len(out):,}."
    )
    pre_rows = int(((out["start"] < 0.0)).sum())
    lastday_rows = int(((out["start"] >= PRE_BREAK) & (out["start"] < 0.0)).sum())
    print(f"  pre-question interval rows (start<0): {pre_rows:,}")
    print(f"  final-day interval rows [-24,0):      {lastday_rows:,}")
    # Event counts by pre-window (events flagged at stop)
    ev = out[out["event_occurred"] == 1]
    n_ev_last = int(((ev["stop"] > PRE_BREAK) & (ev["stop"] <= 0.0)).sum())
    n_ev_first = int(((ev["stop"] > -48.0) & (ev["stop"] <= PRE_BREAK)).sum())
    print(f"  events in final day (-24,0]:   {n_ev_last:,}")
    print(f"  events in [-48,-24]:           {n_ev_first:,}")
    placebo_treated_rows = int((out[PLACEBO_TERM] == 1).sum())
    print(f"  treated_pre_period == 1 rows:  {placebo_treated_rows:,}")
    return out


def _fmt(res, term):
    s = res.summary_df
    if term not in s.index:
        return f"    {term:<40s}  ABSENT"
    coef = float(s.loc[term, "coef"])
    se = float(s.loc[term, "se(coef)"])
    p = float(s.loc[term, "p"])
    hr = float(np.exp(coef))
    lo = float(np.exp(s.loc[term, "coef lower 95%"]))
    hi = float(np.exp(s.loc[term, "coef upper 95%"]))
    return (
        f"    {term:<40s}  coef={coef:+.5f}  se={se:.5f}  "
        f"HR={hr:.4f} [{lo:.4f}, {hi:.4f}]  p={p:.3e}"
    )


def run_spec(nc_pl: pd.DataFrame, base_covs: list, model_name: str):
    covs = list(base_covs) + [PLACEBO_TERM]
    drop_cols = [
        c for c in ["tenure_bucket", "response_time_hours", "response_time_bin",
                    "treated_bin2", "treated_bin3", "help_type"]
        if c in nc_pl.columns
    ]
    fit_df = nc_pl.drop(columns=drop_cols, errors="ignore").copy()
    missing = [c for c in covs if c not in fit_df.columns]
    if missing:
        print(f"  SKIP {model_name}: missing covariates {missing}")
        return
    print("\n" + "=" * 78)
    print(f"  {model_name}")
    print(f"  covariates = {covs}")
    print("=" * 78)
    res = fit_cox_cached(
        fit_df,
        model_name,
        covs,
        use_cache=False,      # never read a possibly-stale pickle for this diagnostic
        robust=False,         # match the tabled newcomer robustness fits (model-based SEs)
        save_cache=False,     # do not write a pickle (file-creation constraint)
    )
    if res is None:
        print(f"  {model_name}: fit returned None (too few events / convergence).")
        return
    print(f"  N analysis rows={res.meta.get('n_rows'):,}  "
          f"events={res.meta.get('n_events'):,}  subsampled={res.meta.get('subsampled')}")
    print("  Coefficients (log-HR relative to the [-48,-24) pre-window baseline):")
    print(_fmt(res, PLACEBO_TERM) + "   <-- PLACEBO (beta_pre)")
    print(_fmt(res, "treated_post_question") + "   (beta_2, waiting period)")
    print(_fmt(res, "is_treated_active") + "   (beta_4, arrival increment)")
    print(_fmt(res, "hasAnswer") + "   (treated main effect, [-48,-24) ref)")


def main():
    ap = argparse.ArgumentParser(description="Newcomer pre-period (T_Q-1D) placebo")
    ap.add_argument("--input", default="../data/event_history",
                    help="folder with study_timelines.parquet / study_events.parquet")
    args = ap.parse_args()

    print(f"COX_MAX_FIT_ROWS = {MAX_FIT_ROWS:,}")
    nc = load_newcomer(args.input)
    nc_pl = add_preperiod_placebo(nc)

    run_spec(nc_pl, COVARIATES_MAIN, "ModelA_Newcomer_PrePeriodPlacebo")
    run_spec(nc_pl, COVARIATES_MAIN_OBSERVABLE, "ModelA_Newcomer_PrePeriodPlacebo_Observable")

    print("\nDone.")


if __name__ == "__main__":
    main()
