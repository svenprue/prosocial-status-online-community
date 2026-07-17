"""Load timelines/events, build survival intervals and covariates, compute descriptives."""
import os
import pickle
import pandas as pd
import numpy as np

from cox_config import BUCKET_ORDER, DATA_CACHE_DIR, DATA_VERSION


def create_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign tenure_bucket from the treated question's tenure per match_id, so control
    and treated of the same match_id always share the same bucket.
    """
    bins = [-np.inf, 7, 30, 180, 365, 1095, 2190, np.inf]
    treated_tenure = (
        df.loc[df["hasAnswer"] == 1, ["match_id", "user_tenure_days"]]
        .drop_duplicates(subset=["match_id"])
    )
    treated_tenure["tenure_bucket"] = pd.cut(
        treated_tenure["user_tenure_days"],
        bins=bins,
        labels=BUCKET_ORDER,
        right=True,
    )
    df = df.drop(columns=["tenure_bucket"], errors="ignore").merge(
        treated_tenure[["match_id", "tenure_bucket"]],
        on="match_id",
        how="left",
    )
    return df


def _compute_descriptives(timelines: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Descriptive statistics for tables (N = unique questions)."""
    desc = {}
    desc["n_questions"] = int(timelines[["match_id", "question_id"]].drop_duplicates().shape[0])
    desc["n_unique_users"] = timelines["question_id"].nunique()
    if "user_id" in timelines.columns:
        desc["n_unique_users"] = timelines["user_id"].nunique()
    desc["pct_has_answer"] = timelines["hasAnswer"].mean() * 100

    event_counts = (
        events.groupby(["match_id", "question_id"]).size().reset_index(name="n_help_events")
    )
    merged = timelines.merge(event_counts, on=["match_id", "question_id"], how="left")
    merged["n_help_events"] = merged["n_help_events"].fillna(0)
    desc["help_events_mean"] = merged["n_help_events"].mean()
    desc["help_events_std"] = merged["n_help_events"].std()
    desc["help_events_median"] = merged["n_help_events"].median()

    desc["tenure_mean"] = timelines["user_tenure_days"].mean()
    desc["tenure_std"] = timelines["user_tenure_days"].std()
    desc["tenure_median"] = timelines["user_tenure_days"].median()

    treated = timelines[timelines["hasAnswer"] == 1].copy()
    if "t_answer" in treated.columns and "t_question" in treated.columns:
        rt = treated["t_answer"] - treated["t_question"]
        rt = rt[rt > 0]
        desc["response_time_mean_hours"] = rt.mean()
        desc["response_time_std_hours"] = rt.std()
        desc["response_time_median_hours"] = rt.median()

    timelines_bucketed = create_tenure_buckets(timelines.copy())
    desc["tenure_bucket_counts"] = (
        timelines_bucketed.dropna(subset=["tenure_bucket"])
        .groupby("tenure_bucket", observed=True)
        .apply(lambda g: g[["match_id", "question_id"]].drop_duplicates().shape[0], include_groups=False)
        .to_dict()
    )

    desc["response_time_by_tenure_bucket"] = {}
    treated_bucketed = timelines_bucketed[timelines_bucketed["hasAnswer"] == 1].copy()
    if "t_answer" in treated_bucketed.columns:
        treated_rt = treated_bucketed.copy()
        treated_rt["rt_hours"] = treated_rt["t_answer"] - treated_rt["t_question"]
        treated_rt = treated_rt[treated_rt["rt_hours"] > 0]
        for bucket in BUCKET_ORDER:
            b = treated_rt[treated_rt["tenure_bucket"] == bucket]
            if len(b) > 0:
                desc["response_time_by_tenure_bucket"][bucket] = {
                    "mean": float(b["rt_hours"].mean()),
                    "std": float(b["rt_hours"].std()) if len(b) > 1 else np.nan,
                    "median": float(b["rt_hours"].median()),
                    "n": len(b),
                }
            else:
                desc["response_time_by_tenure_bucket"][bucket] = {
                    "mean": np.nan, "std": np.nan, "median": np.nan, "n": 0,
                }
    return desc


def _build_covariates(intervals: pd.DataFrame, timelines: pd.DataFrame) -> pd.DataFrame:
    """Merge intervals with timelines and add phase/response-time covariates."""
    timeline_cols = ["match_id", "question_id", "hasAnswer", "t_question", "t_answer"]
    optional_cols = [
        "user_id", "question_year",
        "hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars", "viewCount",
        "postHour", "postDayOfWeek", "numTags", "bodyLenChars", "titleLenChars", "ownerReputation",
    ]
    for optional_col in optional_cols:
        if optional_col in timelines.columns:
            timeline_cols.append(optional_col)
    full_df = intervals.merge(
        timelines[timeline_cols],
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

    # fix(scale): standardize log-RT ONCE, on the treated rows with a positive response
    # time, and build ALL THREE RT interaction terms from that single standardized log-RT.
    # Previously each interaction was z-scored downstream by its OWN nonzero-row SD (in
    # fit_cox_cached), so gamma (treated_response_time_interaction) and delta
    # (treated_post_question_response_time_interaction) lived on different scales and their
    # sum gamma+delta was not a valid net moderation. A single common scale fixes that;
    # fit_cox_cached now skips re-scaling these (see RT_INTERACTION_TERMS).
    full_df["log_response_time"] = np.where(
        full_df["hasAnswer"] == 1,
        np.log1p(np.maximum(full_df["t_answer"] - full_df["t_question"], 0)),
        0.0,
    )
    rt_std_mask = (full_df["hasAnswer"] == 1) & (
        (full_df["t_answer"] - full_df["t_question"]) > 0
    )
    rt_ref = full_df.loc[rt_std_mask, "log_response_time"]
    rt_mu = float(rt_ref.mean()) if len(rt_ref) else 0.0
    rt_sd = float(rt_ref.std()) if len(rt_ref) > 1 else 0.0
    if not (rt_sd and rt_sd > 0):
        rt_sd = 1.0  # degenerate/absent RT: fall back to unit scale (no division blow-up)
    # Standardized log-RT is defined only on treated rows; control rows keep 0 so the
    # indicator-multiplied interactions stay 0 for controls (as before).
    log_rt_std = np.where(
        full_df["hasAnswer"] == 1,
        (full_df["log_response_time"] - rt_mu) / rt_sd,
        0.0,
    )
    full_df["hasAnswer_response_time_interaction"] = np.where(
        full_df["hasAnswer"] == 1, log_rt_std, 0.0
    )
    full_df["treated_response_time_interaction"] = (
        full_df["is_treated_active"] * log_rt_std
    )
    full_df["treated_post_question_response_time_interaction"] = (
        full_df["treated_post_question"] * log_rt_std
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

    # Answer-derived quality covariates (ISS-06) are undefined for control (unanswered)
    # questions: a control has no answer, hence no accepted answer and zero answer
    # score/length. Set them to 0 for control rows so those rows survive the
    # answer-quality spec's per-model dropna in fit_cox_cached. Otherwise every control
    # would be dropped from that spec, silently reducing it to a treated-only fit and
    # destroying the treated-vs-control DiD contrast.
    for acol in ["hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars"]:
        if acol in full_df.columns:
            control_mask = full_df["hasAnswer"] == 0
            full_df.loc[control_mask, acol] = full_df.loc[control_mask, acol].fillna(0.0)

    # fix(fillna): guard asymmetric attrition. We just filled answer-quality NaNs to 0 for
    # CONTROL rows only (a control has no answer, so 0 is the correct absence value). A
    # TREATED row with a NaN in these answer-derived columns is NOT filled here (0 is a
    # real score/length, not "missing"), so it is later dropped by the answer-quality
    # spec's per-model dropna in fit_cox_cached — while its matched control survives. That
    # asymmetric attrition biases the treated-vs-control DiD. We do not silently fill; we
    # warn so the count is visible and can be triaged.
    quality_cols = [
        c for c in ["hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars"]
        if c in full_df.columns
    ]
    if quality_cols:
        treated_nan_mask = (full_df["hasAnswer"] == 1) & full_df[quality_cols].isna().any(axis=1)
        n_treated_nan_rows = int(treated_nan_mask.sum())
        if n_treated_nan_rows > 0:
            n_affected_questions = int(full_df.loc[treated_nan_mask, "question_id"].nunique())
            print(
                f"  ⚠ fix(fillna): {n_treated_nan_rows:,} TREATED interval rows "
                f"({n_affected_questions:,} questions) carry NaN in answer-quality covariates "
                f"{quality_cols}. These treated rows (not their matched controls) will be "
                "dropped by the answer-quality spec's dropna → asymmetric attrition. NOT "
                "auto-filling to 0 (0 is a real score/length)."
            )

    # Columns that must be present on every retained row. The optional selection/quality
    # covariates below are deliberately NOT part of this list: each Cox spec drops its own
    # covariate-specific NaNs in fit_cox_cached (dropna(subset=keep)). Dropping on them
    # here would remove control rows (which lack answer-level covariates by construction)
    # from the shared baseline model_df and break every spec, not just the ones using them.
    required_cols = [
        "match_id", "question_id", "unique_id", "start", "stop", "event_occurred",
        "hasAnswer", "phase_post_question", "treated_post_question",
        "phase_post", "is_treated_active",
        "hasAnswer_response_time_interaction",
        "treated_post_question_response_time_interaction",
        "treated_response_time_interaction",
        "tenure_bucket",
        "response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3",
    ]
    cols = list(required_cols)
    for optional_col in ["user_id", "question_year",
                         "hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars", "viewCount",
                         "postHour", "postDayOfWeek", "numTags", "bodyLenChars", "titleLenChars", "ownerReputation"]:
        if optional_col in full_df.columns:
            cols.append(optional_col)
    return full_df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)


def load_and_prepare(input_folder: str, sample_size: int = None, event_help_types: list = None):
    """Load parquet, optionally subsample, build intervals and model_df."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    help_tag = "" if not event_help_types else "_" + "_".join(event_help_types)
    cache_tag = f"sample_{sample_size}" if sample_size else "full"
    # fix(cache): tag the interval cache with DATA_VERSION so a data-construction bump
    # (e.g. the fix(scale) RT re-standardization) does not silently reuse stale intervals.
    cache_tag = f"{cache_tag}{help_tag}_{DATA_VERSION}"
    interval_cache = os.path.join(DATA_CACHE_DIR, f"intervals_{cache_tag}.parquet")
    desc_cache = os.path.join(DATA_CACHE_DIR, f"descriptives_{cache_tag}.pkl")

    if os.path.exists(interval_cache):
        print(f"✓ Loading cached intervals from {interval_cache}")
        model_df = pd.read_parquet(interval_cache)
        print("Recomputing descriptives from timelines …")
        timelines = pd.read_parquet(f"{input_folder}/study_timelines.parquet")
        events = pd.read_parquet(f"{input_folder}/study_events.parquet")
        for col in ["t_start", "t_question", "t_answer", "t_end", "user_tenure_days", "question_year"]:
            if col in timelines.columns:
                timelines[col] = timelines[col].astype(float)
        if "t_event" in events.columns:
            events["t_event"] = events["t_event"].astype(float)
        timelines = create_tenure_buckets(timelines)
        if sample_size and sample_size < timelines["match_id"].nunique():
            ids = np.random.choice(timelines["match_id"].unique(), sample_size, replace=False)
            timelines = timelines[timelines["match_id"].isin(ids)].copy()
            events = events[events["match_id"].isin(ids)].copy()
        # Match the analysis outcome: filter before descriptives when a help-type
        # subset was requested (interval cache path previously counted all types).
        if event_help_types and "help_type" in events.columns:
            events = events[events["help_type"].isin(event_help_types)].copy()
            print(f"Filtered events to help_type in {event_help_types}: {len(events):,} rows")
        descriptives = _compute_descriptives(timelines, events)
        # Atomic write: concurrent full-data stage jobs all call load_and_prepare and
        # would otherwise write this same path simultaneously (torn-write risk).
        _tmp = f"{desc_cache}.tmp.{os.getpid()}"
        with open(_tmp, "wb") as f:
            pickle.dump(descriptives, f)
        os.replace(_tmp, desc_cache)
        print(f"✓ Saved {desc_cache}")
        return model_df, descriptives

    print("=== Loading raw data ===")
    timelines = pd.read_parquet(f"{input_folder}/study_timelines.parquet")
    events = pd.read_parquet(f"{input_folder}/study_events.parquet")
    if event_help_types and "help_type" in events.columns:
        events = events[events["help_type"].isin(event_help_types)].copy()
        print(f"Filtered events to help_type in {event_help_types}: {len(events):,} rows")
    print(f"Loaded {len(timelines):,} timelines, {len(events):,} events")
    for col in ["t_start", "t_question", "t_answer", "t_end", "user_tenure_days", "question_year"]:
        if col in timelines.columns:
            timelines[col] = timelines[col].astype(float)
    if "t_event" in events.columns:
        events["t_event"] = events["t_event"].astype(float)
    timelines = create_tenure_buckets(timelines)

    if sample_size and sample_size < timelines["match_id"].nunique():
        print(f"Subsampling {sample_size:,} matched pairs …")
        ids = np.random.choice(timelines["match_id"].unique(), sample_size, replace=False)
        timelines = timelines[timelines["match_id"].isin(ids)].copy()
        events = events[events["match_id"].isin(ids)].copy()

    descriptives = _compute_descriptives(timelines, events)

    print("Constructing event intervals …")
    boundary_id_vars = ["match_id", "question_id", "hasAnswer", "t_answer", "tenure_bucket"]
    if "question_year" in timelines.columns:
        boundary_id_vars.append("question_year")
    boundaries = timelines.melt(
        id_vars=boundary_id_vars,
        value_vars=["t_start", "t_question", "t_end"],
        value_name="time",
    )[["match_id", "question_id", "tenure_bucket", "time"] + (["question_year"] if "question_year" in timelines.columns else [])]
    answer_cols = ["match_id", "question_id", "tenure_bucket", "t_answer"]
    if "question_year" in timelines.columns:
        answer_cols.append("question_year")
    answer_boundaries = timelines[answer_cols].rename(columns={"t_answer": "time"})
    event_times = events[["match_id", "question_id", "t_event"]].rename(columns={"t_event": "time"})
    event_times["is_event"] = 1
    all_times = pd.concat([boundaries, answer_boundaries, event_times], ignore_index=True)
    all_times["is_event"] = all_times["is_event"].fillna(0)
    all_times = all_times.sort_values(["match_id", "question_id", "time"])
    agg_cols = {"is_event": "max", "tenure_bucket": "first"}
    if "question_year" in all_times.columns:
        agg_cols["question_year"] = "first"
    all_times = all_times.groupby(["match_id", "question_id", "time"], as_index=False).agg(
        agg_cols
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
    # Event-only rows had no tenure_bucket in groupby; propagate from same (match_id, question_id)
    intervals["tenure_bucket"] = (
        intervals.groupby(["match_id", "question_id"], group_keys=False)["tenure_bucket"]
        .apply(lambda s: s.ffill().bfill())
    )
    if intervals["tenure_bucket"].isna().any():
        n_bad = intervals["tenure_bucket"].isna().sum()
        intervals = intervals.dropna(subset=["tenure_bucket"])
        print(f"  Dropped {n_bad} interval rows with no tenure_bucket (no boundaries for that question).")
    if "question_year" in intervals.columns:
        intervals["question_year"] = (
            intervals.groupby(["match_id", "question_id"], group_keys=False)["question_year"]
            .apply(lambda s: s.ffill().bfill())
        )

    print("Computing covariates …")
    model_df = _build_covariates(intervals, timelines)
    # Free peak memory before the parquet write (OOM was killing jobs here).
    del intervals, all_times, boundaries, answer_boundaries, event_times
    import gc
    gc.collect()
    model_df.to_parquet(interval_cache)
    _tmp = f"{desc_cache}.tmp.{os.getpid()}"
    with open(_tmp, "wb") as f:
        pickle.dump(descriptives, f)
    os.replace(_tmp, desc_cache)
    print(f"✓ Cached intervals to {interval_cache}")
    return model_df, descriptives
