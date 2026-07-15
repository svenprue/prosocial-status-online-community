"""Revision-data robustness analyses (ISS-02, ISS-04 partial, ISS-06, ISS-10, ISS-16).

Run after regenerating the pipeline with revision inputs:
  python revision_robustness.py --input ../data/event_history
"""
import argparse
import gc
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
    PRIMARY_HELP_TYPES,
)
from cox_data import load_and_prepare, create_tenure_buckets
from cox_fit import fit_cox_cached, fit_response_time_bin_quality_models, _linear_combo, DID_TERMS


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
    # (potentially large) waiting-period term. We also return the decomposition:
    # beta_2 (waiting-period / "anticipatory engagement" pre-trend) and beta_4 (the
    # increment at answer arrival, net of that pre-trend).
    inc = summary.loc["is_treated_active"]                       # beta_4
    combo = _linear_combo(result, DID_TERMS)                     # beta_2 + beta_4
    if combo is None:
        combo = {
            "coef": float(inc["coef"]),
            "hr": float(np.exp(inc["coef"])),
            "ci_lo": float(np.exp(inc["coef lower 95%"])),
            "ci_hi": float(np.exp(inc["coef upper 95%"])),
            "se": float(inc["se(coef)"]),
            "p": float(inc["p"]),
        }
    waiting_coef = (
        float(summary.loc["treated_post_question", "coef"])
        if "treated_post_question" in summary.index else float("nan")
    )
    return {
        "model": model_name,
        "N_questions": int(n_questions),
        "events": int(result.meta.get("n_events", 0)),
        "HR": combo["hr"],
        "CI_low": combo["ci_lo"],
        "CI_high": combo["ci_hi"],
        "SE": combo["se"],
        "p": combo["p"],
        "did_coef": combo.get("coef", float("nan")),      # beta_2 + beta_4
        "waiting_coef": waiting_coef,                       # beta_2 (pre-trend)
        "HR_waiting": float(np.exp(waiting_coef)) if np.isfinite(waiting_coef) else float("nan"),
        "arrival_coef": float(inc["coef"]),                 # beta_4 (arrival increment)
        "HR_increment_only": float(np.exp(inc["coef"])),   # exp(beta_4), reference
        # ISS-24 re-headline: full arrival uncertainty so create_figures can lead with the
        # arrival increment (primary) and show the summed DiD as a labeled secondary column.
        "arrival_se": float(inc["se(coef)"]),
        "arrival_p": float(inc["p"]),
        "arrival_ci_lo": float(np.exp(inc["coef lower 95%"])),
        "arrival_ci_hi": float(np.exp(inc["coef upper 95%"])),
    }


def _fit_subset(model_df: pd.DataFrame, model_name: str, covariates: list, use_cache: bool) -> dict | None:
    drop_cols = ["tenure_bucket", "response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3", "help_type"]
    fit_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns], errors="ignore").copy()
    missing = [c for c in covariates if c not in fit_df.columns]
    if missing:
        print(f"⚠ {model_name}: skipping — missing covariates {missing}")
        return None
    n_q = int(fit_df["unique_id"].nunique())
    # Point estimates use model-based SEs; pair_bootstrap_se.py supplies clustered uncertainty (ISS-01).
    res = fit_cox_cached(fit_df, model_name, covariates, use_cache=use_cache, robust=False)
    return _extract_treatment_row(res, model_name, n_q)


def run_observable_controls(input_folder: str, use_cache: bool) -> pd.DataFrame:
    print("\n=== ISS-02: Observable selection controls ===")
    model_df, _ = load_and_prepare(input_folder, event_help_types=PRIMARY_HELP_TYPES)
    rows = []
    # Cache name must stay tied to PRIMARY_HELP_TYPES; the old ModelA_AllData_Baseline
    # pickle can be from answers+comments+edits and would corrupt this comparison.
    base = _fit_subset(model_df, "ModelA_AllData_Baseline_primary", COVARIATES_MAIN, use_cache)
    if base:
        rows.append({**base, "model": "ModelA_AllData_Baseline", "spec": "baseline"})
    # Cache name bumps when the observable covariate list changes (e.g. viewCount drop).
    ext = _fit_subset(
        model_df, "ModelA_AllData_ObservableControls_novc", COVARIATES_MAIN_OBSERVABLE, use_cache
    )
    if ext:
        rows.append({**ext, "spec": "observable_controls"})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_observable_controls.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_answer_quality(input_folder: str, use_cache: bool) -> pd.DataFrame:
    print("\n=== ISS-06: Answer-quality robustness (speed spec) ===")
    model_df, _ = load_and_prepare(input_folder, event_help_types=PRIMARY_HELP_TYPES)
    rows = []
    base = _fit_subset(model_df, "ModelB_AllData_Baseline", COVARIATES_SPEED, use_cache)
    if base:
        rows.append({**base, "spec": "speed_baseline"})
    # Length-only control: the least-endogenous of the quality signals (fixed by the
    # answerer at composition, unlike acceptance/score) and the reviewer's exact concern
    # (terse vs. detailed). Isolates how much of the flip is length vs. acceptance+score.
    length = _fit_subset(model_df, "ModelB_AllData_LengthOnly",
                         COVARIATES_SPEED + ["firstAnswerBodyLenChars"], use_cache)
    if length:
        rows.append({**length, "spec": "speed_length_only"})
    # Score ≈ upvote count (Posts.Score corr with upvotes = 0.9998; VoteTypeId=2 has no
    # UserId in the dump, so asker votes cannot be stripped). Isolates community approval
    # without the asker-return acceptance control.
    score = _fit_subset(model_df, "ModelB_AllData_ScoreOnly",
                        COVARIATES_SPEED + ["firstAnswerScore"], use_cache)
    if score:
        rows.append({**score, "spec": "speed_score_only"})
    qual = _fit_subset(model_df, "ModelB_AllData_QualityControls", COVARIATES_SPEED_QUALITY, use_cache)
    if qual:
        rows.append({**qual, "spec": "speed_quality_controls"})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_answer_quality.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_composite_outcome(input_folder: str, use_cache: bool) -> pd.DataFrame:
    """ISS-04: decompose the reciprocity outcome by help type.

    Fits the same Model A on each help-type subset so the treatment effect can be
    read component-by-component. Each row reports the summed DiD (HR) together with
    its decomposition into the waiting-period coefficient (beta_2, "anticipatory
    engagement" pre-trend) and the answer-arrival increment (beta_4). This directly
    answers R2's point 4: whether newcomers' low-effort actions (comments) carry a
    reciprocity signal, or whether that signal is pre-answer activity selection.

    Note on inputs: the help-type filter can only surface types that are present in
    ``study_events.parquet``. The default pipeline excludes ``accept`` events (see
    create_matched_event_histories.py: include_accept_help=False), so the
    ``accepts_only`` and ``composite_all`` rows will be skipped unless ``input_folder``
    points to an event history generated with ``include_accept_help=True``. Comments
    and edits are counted only on *other users'* posts (self-comments/edits excluded).
    ``accepts`` (accepting an answer on one's own question) measure direct
    reciprocity to the helper, not the generalized reciprocity this study models --- we
    report them separately and labeled, never blended into the primary outcome.
    """
    print("\n=== ISS-04: Outcome decomposition by help type ===")
    outcomes = [
        ("answers_only", ["answer"]),
        ("comments_only", ["comment"]),
        ("edits_only", ["edit"]),
        ("accepts_only", ["accept"]),                       # direct (dyadic) reciprocity; treated-only by construction
        ("answers_comments", ["answer", "comment"]),        # generalized-reciprocity composite, no accepts/edits
        ("answers_comments_edits", ["answer", "comment", "edit"]),
        ("composite_all", ["answer", "comment", "edit", "accept"]),
    ]
    rows = []
    for label, types in outcomes:
        model_df, _ = load_and_prepare(input_folder, event_help_types=types)
        if {"event_occurred", "hasAnswer"}.issubset(model_df.columns):
            control_events = int(model_df.loc[model_df["hasAnswer"] == 0, "event_occurred"].sum())
            treated_events = int(model_df.loc[model_df["hasAnswer"] == 1, "event_occurred"].sum())
        else:
            control_events = treated_events = -1
        if control_events == 0:
            print(
                f"  ⚠ {label}: control group has zero events (treated-only outcome by "
                "construction); the treatment contrast is degenerate and is reported for "
                "reference only, flagged via control_events=0."
            )
        row = _fit_subset(model_df, f"ModelA_AllData_{label}", COVARIATES_MAIN, use_cache)
        del model_df
        gc.collect()
        if row:
            rows.append({
                **row,
                "outcome": label,
                "help_types": "+".join(types),
                "control_events": control_events,
                "treated_events": treated_events,
            })
        else:
            print(f"  ⚠ {label}: no result (type absent from study_events, or too few events) — skipped.")
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_composite_outcome.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_viewcount_placebo(input_folder: str, use_cache: bool) -> pd.DataFrame:
    """ISS-16 (deprecated): ViewCount high-vs-low helping among unanswered questions.

    Cumulative dump ``ViewCount`` conflates age, topic popularity, and years of
    post-window search traffic with contemporaneous exposure, so this is *not* a
    valid identification check. Kept as an opt-in diagnostic (``--only placebo``);
    it is no longer part of the default revision pipeline or manuscript package.
    """
    print("\n=== ISS-16: ViewCount exposure placebo (no-answer questions) [OPT-IN ONLY] ===")
    print(
        "  ⚠ Cumulative dump ViewCount is not identified for a contemporaneous "
        "exposure placebo; this run is diagnostic-only and not manuscript output."
    )
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

    # Build minimal placebo intervals: pre + post pseudo-answer phase per control.
    # Post-window event counts are computed with a single merge+groupby instead
    # of an O(controls x events) per-row scan of the full events frame.
    ctrl_keys = controls[["match_id", "question_id", "t_answer", "t_end"]]
    ev = events.merge(ctrl_keys, on=["match_id", "question_id"], how="inner")
    ev = ev[(ev["t_event"] >= ev["t_answer"]) & (ev["t_event"] <= ev["t_end"])]
    post_counts = (
        ev.groupby(["match_id", "question_id"]).size().rename("n_ev").reset_index()
    )
    controls = controls.merge(post_counts, on=["match_id", "question_id"], how="left")
    controls["n_ev"] = controls["n_ev"].fillna(0)

    uid = controls["match_id"].astype(str) + "_" + controls["question_id"].astype(str)
    pre = pd.DataFrame({
        "match_id": controls["match_id"].to_numpy(),
        "question_id": controls["question_id"].to_numpy(),
        "unique_id": uid.to_numpy(),
        "start": controls["t_start"].to_numpy(),
        "stop": controls["t_answer"].to_numpy(),
        "event_occurred": 0,
        "high_view": controls["high_view"].to_numpy(),
        "phase_post": 0,
    })
    post = pd.DataFrame({
        "match_id": controls["match_id"].to_numpy(),
        "question_id": controls["question_id"].to_numpy(),
        "unique_id": uid.to_numpy(),
        "start": controls["t_answer"].to_numpy(),
        "stop": controls["t_end"].to_numpy(),
        "event_occurred": (controls["n_ev"] > 0).astype(int).to_numpy(),
        "high_view": controls["high_view"].to_numpy(),
        "phase_post": 1,
    })
    placebo_df = pd.concat([pre, post], ignore_index=True)
    placebo_df["high_view_post"] = placebo_df["high_view"] * placebo_df["phase_post"]
    covariates = ["phase_post", "high_view", "high_view_post"]
    # Prefer robust SEs; fall back to model-based if this lifelines build lacks
    # CoxTimeVarying robust variance (common: NotImplementedError → fit returns None).
    res = fit_cox_cached(
        placebo_df,
        "ViewCountPlacebo_NoAnswer",
        covariates,
        use_cache=use_cache,
        robust=True,
    )
    se_note = "robust"
    if res is None:
        print(
            "  ⚠ Robust placebo fit unavailable; retrying with model-based SEs "
            "(pair_bootstrap_se.py remains the clustered-uncertainty source)."
        )
        res = fit_cox_cached(
            placebo_df,
            "ViewCountPlacebo_NoAnswer",
            covariates,
            use_cache=False,
            robust=False,
        )
        se_note = "model_based"
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
                    "se_type": se_note,
                })
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_viewcount_placebo.csv")
    if out.empty:
        # Avoid writing a 0-byte/headerless file that breaks create_figures.
        if os.path.exists(path):
            os.remove(path)
        print(f"⚠ Placebo produced no rows; removed {path} if present.")
    else:
        out.to_csv(path, index=False)
        print(f"✓ Saved {path} ({len(out)} rows, SEs={se_note})")
    return out


def run_newcomer_bucket_checks(input_folder: str, use_cache: bool) -> pd.DataFrame:
    """ISS-02/06: <1 Week bucket HR under extended specs."""
    print("\n=== Newcomer bucket robustness (< 1 Week) ===")
    model_df, _ = load_and_prepare(input_folder, event_help_types=PRIMARY_HELP_TYPES)
    bucket = "< 1 Week"
    sub = model_df[model_df["tenure_bucket"] == bucket].copy()
    rows = []
    for name, covs in [
        ("ModelA_Newcomer_Baseline", COVARIATES_MAIN),
        ("ModelA_Newcomer_Observable_novc", COVARIATES_MAIN_OBSERVABLE),
        ("ModelB_Newcomer_Baseline", COVARIATES_SPEED),
        ("ModelB_Newcomer_LengthOnly", COVARIATES_SPEED + ["firstAnswerBodyLenChars"]),
        ("ModelB_Newcomer_ScoreOnly", COVARIATES_SPEED + ["firstAnswerScore"]),
        ("ModelB_Newcomer_Quality", COVARIATES_SPEED_QUALITY),
    ]:
        row = _fit_subset(sub, name, covs, use_cache)
        if row:
            # Display name without the cache-busting suffix.
            display = name.replace("_novc", "")
            rows.append({**row, "model": display, "tenure_bucket": bucket})
    out = pd.DataFrame(rows)
    path = os.path.join(CACHE_DIR, "results_newcomer_robustness.csv")
    out.to_csv(path, index=False)
    print(f"✓ Saved {path}")
    return out


def run_rt_bins_quality(
    input_folder: str, use_cache: bool, n_jobs: int | None = None
) -> pd.DataFrame:
    """ISS-06 / #27: Model A + answer-quality controls per response-time bin."""
    print("\n=== ISS-06/#27: Response-time bins with answer-quality controls ===")
    model_df, _ = load_and_prepare(input_folder, event_help_types=PRIMARY_HELP_TYPES)
    out = fit_response_time_bin_quality_models(
        model_df, use_cache=use_cache, n_jobs=n_jobs
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Revision-data robustness analyses")
    parser.add_argument("--input", default=_default_input_folder())
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--only",
        choices=[
            "observable",
            "quality",
            "composite",
            "placebo",
            "newcomer",
            "rt_bins_quality",
            "all",
        ],
        default="all",
        help=(
            "Run a single step. Default 'all' skips the ViewCount placebo "
            "(opt-in via --only placebo; not manuscript-ready)."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    use_cache = not args.no_cache

    if args.only in ("observable", "all"):
        run_observable_controls(args.input, use_cache)
    if args.only in ("quality", "all"):
        run_answer_quality(args.input, use_cache)
    if args.only in ("composite", "all"):
        run_composite_outcome(args.input, use_cache)
    if args.only == "placebo":
        run_viewcount_placebo(args.input, use_cache)
    if args.only in ("newcomer", "all"):
        run_newcomer_bucket_checks(args.input, use_cache)
    if args.only == "rt_bins_quality":
        run_rt_bins_quality(args.input, use_cache, n_jobs=args.n_jobs)

    if args.only == "all":
        try:
            from cohort_robustness import run_cohort_robustness
            run_cohort_robustness(args.input, use_cache=use_cache)
        except Exception as e:
            print(f"⚠ Cohort robustness skipped: {e}")

    print("\nDone. Results in analysis/model_cache/results_*.csv")


if __name__ == "__main__":
    main()
