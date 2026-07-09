"""Revision-data robustness analyses (ISS-02, ISS-04 partial, ISS-06, ISS-10, ISS-16).

Run after regenerating the pipeline with revision inputs:
  python revision_robustness.py --input ../data/event_history
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import (
    CACHE_DIR,
    COVARIATES_MAIN,
    COVARIATES_MAIN_OBSERVABLE,
    COVARIATES_SPEED,
    COVARIATES_SPEED_QUALITY,
    BUCKET_ORDER,
)
from cox_data import load_and_prepare, create_tenure_buckets
from cox_fit import fit_cox_cached, _linear_combo, DID_TERMS


def _default_input_folder() -> str:
    return os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))


def _extract_treatment_row(result, model_name: str, n_questions: int) -> dict | None:
    if result is None:
        return None
    summary = result.summary_df
    if "is_treated_active" not in summary.index:
        return None
    # Report the SUMMED DiD (treated_post_question + is_treated_active) — the treatment
    # effect (post-answer vs. pre-question baseline, treated vs. control) — not the raw
    # is_treated_active coefficient, which is only the post-answer increment over the
    # (potentially large) waiting-period term.
    combo = _linear_combo(result, DID_TERMS)
    inc = summary.loc["is_treated_active"]
    if combo is None:
        combo = {
            "hr": float(np.exp(inc["coef"])),
            "ci_lo": float(np.exp(inc["coef lower 95%"])),
            "ci_hi": float(np.exp(inc["coef upper 95%"])),
            "se": float(inc["se(coef)"]),
            "p": float(inc["p"]),
        }
    return {
        "model": model_name,
        "N_questions": int(n_questions),
        "events": int(result.meta.get("n_events", 0)),
        "HR": combo["hr"],
        "CI_low": combo["ci_lo"],
        "CI_high": combo["ci_hi"],
        "SE": combo["se"],
        "p": combo["p"],
        "HR_increment_only": float(np.exp(inc["coef"])),  # is_treated_active alone, reference
    }


def _fit_subset(model_df: pd.DataFrame, model_name: str, covariates: list, use_cache: bool) -> dict | None:
    drop_cols = ["tenure_bucket", "response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3", "help_type"]
    fit_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns], errors="ignore").copy()
    missing = [c for c in covariates if c not in fit_df.columns]
    if missing:
        print(f"⚠ {model_name}: skipping — missing covariates {missing}")
        return None
    n_q = int(fit_df["unique_id"].nunique())
    res = fit_cox_cached(fit_df, model_name, covariates, use_cache=use_cache, robust=True)
    return _extract_treatment_row(res, model_name, n_q)


def run_observable_controls(input_folder: str, use_cache: bool) -> pd.DataFrame:
    print("\n=== ISS-02: Observable selection controls ===")
    model_df, _ = load_and_prepare(input_folder)
    rows = []
    base = _fit_subset(model_df, "ModelA_AllData_Baseline", COVARIATES_MAIN, use_cache)
    if base:
        rows.append({**base, "spec": "baseline"})
    ext = _fit_subset(model_df, "ModelA_AllData_ObservableControls", COVARIATES_MAIN_OBSERVABLE, use_cache)
    if ext:
        rows.append({**ext, "spec": "observable_controls"})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_observable_controls.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_answer_quality(input_folder: str, use_cache: bool) -> pd.DataFrame:
    print("\n=== ISS-06: Answer-quality robustness (speed spec) ===")
    model_df, _ = load_and_prepare(input_folder)
    rows = []
    base = _fit_subset(model_df, "ModelB_AllData_Baseline", COVARIATES_SPEED, use_cache)
    if base:
        rows.append({**base, "spec": "speed_baseline"})
    qual = _fit_subset(model_df, "ModelB_AllData_QualityControls", COVARIATES_SPEED_QUALITY, use_cache)
    if qual:
        rows.append({**qual, "spec": "speed_quality_controls"})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_answer_quality.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_composite_outcome(input_folder: str, use_cache: bool) -> pd.DataFrame:
    print("\n=== ISS-04 partial: Composite vs answers-only outcome ===")
    rows = []
    for label, types in [("answers_only", ["answer"]), ("composite_acc", ["answer", "comment", "accept"])]:
        model_df, _ = load_and_prepare(input_folder, event_help_types=types)
        row = _fit_subset(model_df, f"ModelA_AllData_{label}", COVARIATES_MAIN, use_cache)
        if row:
            rows.append({**row, "outcome": label})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_composite_outcome.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_viewcount_placebo(input_folder: str, use_cache: bool) -> pd.DataFrame:
    """ISS-16: among no-answer questions, high vs low ViewCount post-pseudo-window helping."""
    print("\n=== ISS-16: ViewCount exposure placebo (no-answer questions) ===")
    timelines = pd.read_parquet(os.path.join(input_folder, "study_timelines.parquet"))
    events = pd.read_parquet(os.path.join(input_folder, "study_events.parquet"))
    if "viewCount" not in timelines.columns:
        print("⚠ viewCount not in study_timelines; skipping placebo.")
        return pd.DataFrame()

    controls = timelines[timelines["hasAnswer"] == 0].copy()
    if controls.empty:
        print("⚠ No control timelines; skipping.")
        return pd.DataFrame()

    median_vc = controls["viewCount"].median()
    controls["high_view"] = (controls["viewCount"] >= median_vc).astype(int)
    print(f"  No-answer questions: {len(controls):,}; viewCount median split = {median_vc:.0f}")

    # Build minimal placebo intervals: post-pseudo phase only
    records = []
    for _, row in controls.iterrows():
        mid, qid = row["match_id"], row["question_id"]
        q_events = events[(events["match_id"] == mid) & (events["question_id"] == qid)]
        t_q, t_a, t_end = row["t_question"], row["t_answer"], row["t_end"]
        # pre pseudo-answer
        records.append({
            "match_id": mid, "question_id": qid, "unique_id": f"{mid}_{qid}",
            "start": row["t_start"], "stop": t_a, "event_occurred": 0,
            "high_view": row["high_view"], "phase_post": 0,
        })
        # post pseudo-answer
        n_ev = len(q_events[(q_events["t_event"] >= t_a) & (q_events["t_event"] <= t_end)])
        records.append({
            "match_id": mid, "question_id": qid, "unique_id": f"{mid}_{qid}",
            "start": t_a, "stop": t_end, "event_occurred": int(n_ev > 0),
            "high_view": row["high_view"], "phase_post": 1,
        })

    placebo_df = pd.DataFrame(records)
    placebo_df["high_view_post"] = placebo_df["high_view"] * placebo_df["phase_post"]
    covariates = ["phase_post", "high_view", "high_view_post"]
    res = fit_cox_cached(
        placebo_df,
        "ViewCountPlacebo_NoAnswer",
        covariates,
        use_cache=use_cache,
        robust=True,
    )
    rows = []
    if res is not None:
        for term in ["high_view_post", "phase_post"]:
            if term in res.summary_df.index:
                r = res.summary_df.loc[term]
                rows.append({
                    "term": term,
                    "HR": float(np.exp(r["coef"])),
                    "CI_low": float(np.exp(r["coef lower 95%"])),
                    "CI_high": float(np.exp(r["coef upper 95%"])),
                    "p": float(r["p"]),
                    "median_viewCount": median_vc,
                    "n_questions": len(controls),
                })
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_viewcount_placebo.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_newcomer_bucket_checks(input_folder: str, use_cache: bool) -> pd.DataFrame:
    """ISS-02/06: <1 Week bucket HR under extended specs."""
    print("\n=== Newcomer bucket robustness (< 1 Week) ===")
    model_df, _ = load_and_prepare(input_folder)
    bucket = "< 1 Week"
    sub = model_df[model_df["tenure_bucket"] == bucket].copy()
    rows = []
    for name, covs in [
        ("ModelA_Newcomer_Baseline", COVARIATES_MAIN),
        ("ModelA_Newcomer_Observable", COVARIATES_MAIN_OBSERVABLE),
        ("ModelB_Newcomer_Quality", COVARIATES_SPEED_QUALITY),
    ]:
        row = _fit_subset(sub, name, covs, use_cache)
        if row:
            rows.append({**row, "tenure_bucket": bucket})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_newcomer_robustness.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Revision-data robustness analyses")
    parser.add_argument("--input", default=_default_input_folder())
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    use_cache = not args.no_cache

    run_observable_controls(args.input, use_cache)
    run_answer_quality(args.input, use_cache)
    run_composite_outcome(args.input, use_cache)
    run_viewcount_placebo(args.input, use_cache)
    run_newcomer_bucket_checks(args.input, use_cache)

    try:
        from cohort_robustness import run_cohort_robustness
        run_cohort_robustness(args.input, use_cache=use_cache)
    except Exception as e:
        print(f"⚠ Cohort robustness skipped: {e}")

    print("\nDone. Results in analysis/model_cache/results_*.csv")


if __name__ == "__main__":
    main()
