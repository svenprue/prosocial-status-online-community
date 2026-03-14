"""
Inspect why n_questions_control is 0 in results_main.csv.

Traces the pipeline: timelines -> intervals -> _build_covariates (pre-dropna and post-dropna)
and reports counts of control vs treated unique questions and NaN per column for control rows.

Usage:
    python inspect_control_questions.py [--input <path>] [--use-cache]
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_data import (
    create_tenure_buckets,
    _build_covariates,
    load_and_prepare,
    _compute_descriptives,
)
from cox_config import DATA_CACHE_DIR


def _build_covariates_debug(intervals: pd.DataFrame, timelines: pd.DataFrame):
    """
    Same as cox_data._build_covariates but returns (full_df_before_dropna, full_df_after_dropna)
    so we can inspect which rows/columns drop control.
    """
    full_df = intervals.merge(
        timelines[["match_id", "question_id", "hasAnswer", "t_question", "t_answer"]],
        on=["match_id", "question_id"],
        how="inner",
    )
    full_df["hasAnswer"] = full_df["hasAnswer"].astype(int)
    full_df["unique_id"] = (
        full_df["match_id"].astype(str) + "_" + full_df["question_id"].astype(str)
    )
    full_df["phase_post_question"] = (full_df["start"] >= full_df["t_question"]).astype(int)
    full_df["phase_post"] = (full_df["start"] >= full_df["t_answer"]).astype(int)
    full_df["treated_post_question"] = (
        (full_df["hasAnswer"] == 1) & (full_df["phase_post_question"] == 1)
    ).astype(int)
    full_df["is_treated_active"] = (
        (full_df["hasAnswer"] == 1) & (full_df["phase_post"] == 1)
    ).astype(int)

    full_df["log_response_time"] = np.where(
        full_df["hasAnswer"] == 1,
        np.log1p(full_df["t_answer"]),
        0.0,
    )
    full_df["hasAnswer_response_time_interaction"] = np.where(
        full_df["hasAnswer"] == 1, full_df["log_response_time"], 0.0
    )
    full_df["treated_response_time_interaction"] = (
        full_df["is_treated_active"] * full_df["log_response_time"]
    )
    full_df["treated_post_question_response_time_interaction"] = (
        full_df["treated_post_question"] * full_df["log_response_time"]
    )

    full_df["response_time_hours"] = np.where(
        full_df["hasAnswer"] == 1,
        full_df["t_answer"] - full_df["t_question"],
        np.nan,
    )
    treated_rt = full_df.loc[full_df["hasAnswer"] == 1, "response_time_hours"].dropna()
    if len(treated_rt) >= 3:
        tertile_edges = treated_rt.quantile([1 / 3, 2 / 3]).values
        full_df["response_time_bin"] = 0
        mask = full_df["hasAnswer"] == 1
        full_df.loc[mask & (full_df["response_time_hours"] <= tertile_edges[0]), "response_time_bin"] = 1
        full_df.loc[mask & (full_df["response_time_hours"] > tertile_edges[0]) & (full_df["response_time_hours"] <= tertile_edges[1]), "response_time_bin"] = 2
        full_df.loc[mask & (full_df["response_time_hours"] > tertile_edges[1]), "response_time_bin"] = 3
    else:
        full_df["response_time_bin"] = 0
    full_df["treated_bin2"] = ((full_df["response_time_bin"] == 2) & (full_df["is_treated_active"] == 1)).astype(int)
    full_df["treated_bin3"] = ((full_df["response_time_bin"] == 3) & (full_df["is_treated_active"] == 1)).astype(int)
    full_df["response_time_hours"] = full_df["response_time_hours"].fillna(0.0)

    cols = [
        "match_id", "unique_id", "start", "stop", "event_occurred",
        "hasAnswer", "phase_post_question", "treated_post_question",
        "phase_post", "is_treated_active",
        "hasAnswer_response_time_interaction",
        "treated_post_question_response_time_interaction",
        "treated_response_time_interaction",
        "tenure_bucket",
        "response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3",
    ]
    before = full_df[cols].replace([np.inf, -np.inf], np.nan)
    after = before.dropna()
    return before, after


def main():
    default_input = os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))
    parser = argparse.ArgumentParser(description="Inspect control vs treated in Cox pipeline")
    parser.add_argument("--input", default=default_input, help="Input data folder (timelines/events)")
    parser.add_argument("--use-cache", action="store_true", help="Use cached intervals if present (intervals_full.parquet)")
    args = parser.parse_args()

    input_folder = args.input
    timelines_path = os.path.join(input_folder, "study_timelines.parquet")
    events_path = os.path.join(input_folder, "study_events.parquet")

    if not os.path.exists(timelines_path):
        print(f"ERROR: Timelines not found at {timelines_path}")
        print("  Use --input to point to the folder containing study_timelines.parquet and study_events.parquet")
        sys.exit(1)
    has_events = os.path.exists(events_path)
    if not has_events:
        print(f"WARNING: Events not found at {events_path}. Will only report timeline stats and cached model_df (if --use-cache).")

    print("=" * 60)
    print("1. TIMELINES (raw)")
    print("=" * 60)
    timelines = pd.read_parquet(timelines_path)
    for col in ["t_start", "t_question", "t_answer", "t_end", "user_tenure_days"]:
        if col in timelines.columns:
            timelines[col] = timelines[col].astype(float)
    # Unique questions by treatment
    q = timelines[["match_id", "question_id", "hasAnswer"]].drop_duplicates()
    n_control_tl = (q["hasAnswer"] == 0).sum()
    n_treated_tl = (q["hasAnswer"] == 1).sum()
    print(f"  Unique (match_id, question_id): {len(q)}")
    print(f"  Control (hasAnswer==0): {n_control_tl}")
    print(f"  Treated (hasAnswer==1): {n_treated_tl}")
    print(f"  hasAnswer value_counts:\n{timelines['hasAnswer'].value_counts(dropna=False)}")
    if "t_answer" in timelines.columns:
        na_answer = timelines["t_answer"].isna()
        print(f"  t_answer: NaN count = {na_answer.sum()}, non-NaN = {(~na_answer).sum()} (by row)")
        # Among unique questions
        q_answer = timelines.groupby(["match_id", "question_id"]).agg({"t_answer": "first", "hasAnswer": "first"})
        print(f"  Among unique questions: t_answer NaN = {q_answer['t_answer'].isna().sum()}, hasAnswer==0 = {(q_answer['hasAnswer']==0).sum()}")

    intervals = None
    interval_cache = os.path.join(DATA_CACHE_DIR, "intervals_full.parquet")
    # Note: interval_cache stores model_df (output of _build_covariates), not raw intervals.
    if args.use_cache and os.path.exists(interval_cache):
        cached = pd.read_parquet(interval_cache)
        print("\n" + "=" * 60)
        print("2a. Cached model_df (intervals_full.parquet = pipeline output)")
        print("=" * 60)
        if "hasAnswer" in cached.columns:
            c, t = (cached["hasAnswer"] == 0).sum(), (cached["hasAnswer"] == 1).sum()
            uq_c = cached.loc[cached["hasAnswer"] == 0, "unique_id"].nunique()
            uq_t = cached.loc[cached["hasAnswer"] == 1, "unique_id"].nunique()
            print(f"  Rows: control = {c}, treated = {t}")
            print(f"  Unique unique_id: control = {uq_c}, treated = {uq_t}")

    if has_events:
        print("\n" + "=" * 60)
        print("2. INTERVALS (built from timelines + events)")
        print("=" * 60)
        timelines = create_tenure_buckets(timelines)
        events = pd.read_parquet(events_path)
        if "t_event" in events.columns:
            events["t_event"] = events["t_event"].astype(float)
        boundaries = timelines.melt(
            id_vars=["match_id", "question_id", "hasAnswer", "t_answer", "tenure_bucket"],
            value_vars=["t_start", "t_question", "t_end"],
            value_name="time",
        )[["match_id", "question_id", "tenure_bucket", "time"]]
        answer_boundaries = timelines[["match_id", "question_id", "tenure_bucket", "t_answer"]].rename(columns={"t_answer": "time"})
        event_times = events[["match_id", "question_id", "t_event"]].rename(columns={"t_event": "time"})
        event_times["is_event"] = 1
        all_times = pd.concat([boundaries, answer_boundaries, event_times], ignore_index=True)
        all_times["is_event"] = all_times["is_event"].fillna(0)
        all_times = all_times.sort_values(["match_id", "question_id", "time"])
        all_times = all_times.groupby(["match_id", "question_id", "time"], as_index=False).agg(
            {"is_event": "max", "tenure_bucket": "first"}
        )
        all_times["start"] = all_times["time"]
        all_times["stop"] = all_times.groupby(["match_id", "question_id"])["time"].shift(-1)
        intervals = all_times.dropna(subset=["stop"]).copy()
        intervals = intervals[intervals["stop"] > intervals["start"]]
        intervals["event_occurred"] = (
            all_times.groupby(["match_id", "question_id"])["is_event"]
            .shift(-1)
            .reindex(intervals.index)
        )
        tl_keys = timelines[["match_id", "question_id", "hasAnswer"]].drop_duplicates()
        inter_keys = intervals[["match_id", "question_id"]].drop_duplicates()
        merged = inter_keys.merge(tl_keys, on=["match_id", "question_id"], how="left")
        print(f"  Interval rows: {len(intervals)}")
        print(f"  Unique (match_id, question_id) in intervals: {len(inter_keys)}")
        print(f"  After merge with timelines: hasAnswer 0 = {(merged['hasAnswer']==0).sum()}, 1 = {(merged['hasAnswer']==1).sum()}, NaN = {merged['hasAnswer'].isna().sum()}")

    if intervals is None:
        print("\n  Skipping steps 3-4 (no intervals; need events to build them).")
    else:
        if "tenure_bucket" not in intervals.columns and "tenure_bucket" in timelines.columns:
            timelines = create_tenure_buckets(timelines)

        print("\n" + "=" * 60)
        print("3. COVARIATES: before dropna()")
        print("=" * 60)
        before, after = _build_covariates_debug(intervals, timelines)
        control = before["hasAnswer"] == 0
        treated = before["hasAnswer"] == 1
        print(f"  Rows: total = {len(before)}, control = {control.sum()}, treated = {treated.sum()}")
        n_uq_control = before.loc[control, "unique_id"].nunique()
        n_uq_treated = before.loc[treated, "unique_id"].nunique()
        print(f"  Unique unique_id: control = {n_uq_control}, treated = {n_uq_treated}")

        # NaN per column among control rows
        print("\n  NaN counts (control rows only) per column:")
        for c in before.columns:
            n_na = before.loc[control, c].isna().sum()
            if n_na > 0:
                print(f"    {c}: {n_na} NaN (of {control.sum()} control rows)")

        print("\n" + "=" * 60)
        print("4. COVARIATES: after dropna()")
        print("=" * 60)
        control_after = after["hasAnswer"] == 0
        treated_after = after["hasAnswer"] == 1
        print(f"  Rows: total = {len(after)}, control = {control_after.sum()}, treated = {treated_after.sum()}")
        n_uq_control_after = after.loc[control_after, "unique_id"].nunique() if control_after.any() else 0
        n_uq_treated_after = after.loc[treated_after, "unique_id"].nunique()
        print(f"  Unique unique_id: control = {n_uq_control_after}, treated = {n_uq_treated_after}")
        if n_uq_control_after == 0 and n_uq_control > 0:
            print("\n  >>> PROBLEM: All control questions were dropped by dropna().")
            print("  Rows dropped: control rows with NaN in at least one column (see above).")

    print("\n" + "=" * 60)
    print("5. MODEL_DF from load_and_prepare (sanity check)")
    print("=" * 60)
    try:
        model_df, _ = load_and_prepare(input_folder, sample_size=None)
        c = (model_df["hasAnswer"] == 0).sum()
        t = (model_df["hasAnswer"] == 1).sum()
        uq_c = model_df.loc[model_df["hasAnswer"] == 0, "unique_id"].nunique()
        uq_t = model_df.loc[model_df["hasAnswer"] == 1, "unique_id"].nunique()
        print(f"  model_df rows: control = {c}, treated = {t}")
        print(f"  model_df unique_id: control = {uq_c}, treated = {uq_t}")
    except Exception as e:
        print(f"  load_and_prepare failed: {e}")


if __name__ == "__main__":
    main()
