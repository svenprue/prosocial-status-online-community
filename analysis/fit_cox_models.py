"""
fit_cox_models.py
=================
Loads Stack Overflow event-history data, constructs survival intervals,
fits Cox time-varying models (no regularization), and caches results to disk.

Per tenure bucket:
  Model A – Main effect (no response-time interaction)
  Model B – With response-time × treatment interaction

Additional analyses (pooled all data only):
  - All data: one reciprocity model (Model A), one with speed interaction (Model B).
  - Model C: same as B plus quadratic log(response time) and all relevant interactions.

Outputs: model_cache/*.csv (and *.pkl)

Usage:
    python fit_cox_models.py [--input <path>] [--sample 200000]
    (default input: data/event_history relative to project root)
"""

import os
import pickle
import argparse
import hashlib
import json
import time as timer
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceError
import warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore")

# =====================================================================
# Configuration
# =====================================================================
CACHE_DIR = "model_cache"
DATA_CACHE_DIR = "data_cache"
FIGURE_DIR = "output_figures"
BUCKET_ORDER = [
    "< 1 Week", "1 Week - 1 Month", "1 - 6 Months",
    "6 - 12 Months", "1 - 3 Years", "3 - 6 Years", "> 6 Years",
]
# Interval boundaries rounded to this many hours (phase/treatment from original times).
# Use 1h to reduce ties and Hessian issues; 6h is faster but can trigger nan/inf in lifelines.
ROUND_TO_HOURS = 1
# Covariates with continuous scale: standardize and clip to stabilize Cox fit (avoid nan/inf in Hessian)
CONTINUOUS_COVARIATES = [
    "hasAnswer_response_time_interaction",
    "treated_response_time_interaction",
    "treated_post_question_response_time_interaction",
    "hasAnswer_response_time_sq_interaction",
    "treated_response_time_sq_interaction",
    "treated_post_question_response_time_sq_interaction",
]
# Max rows passed to Cox fitter; if exceeded, stratified subsample to avoid nan/inf in large risk sets
MAX_FIT_ROWS = 8_000_000
SUBSAMPLE_SEED = 42

# =====================================================================
# Caching helpers
# =====================================================================

class CachedCoxResult:
    """Lightweight proxy for a fitted CoxTimeVaryingFitter."""

    def __init__(self, ctv: CoxTimeVaryingFitter, meta: dict = None):
        self.summary_df = ctv.summary.copy()
        self.params_ = ctv.params_.copy()
        self.confidence_intervals_ = ctv.confidence_intervals_.copy()
        self.variance_matrix_ = ctv.variance_matrix_.copy()
        self.log_likelihood_ = ctv.log_likelihood_
        self.meta = meta or {}

    @property
    def summary(self):
        return self.summary_df


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _model_path(model_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{_safe_name(model_name)}.pkl")


def _data_hash(df: pd.DataFrame, cols: list) -> str:
    """Quick hash of subset shape + a sample of values for cache invalidation."""
    info = f"{len(df)}_{df[cols].iloc[:5].to_json()}"
    return hashlib.md5(info.encode()).hexdigest()[:12]


def fit_cox_cached(
    subset_df: pd.DataFrame,
    model_name: str,
    covariates: list,
    penalizer: float = 0.0,
    use_cache: bool = True,
    round_to_hours: float = None,
    initial_point=None,
) -> CachedCoxResult:
    """
    Fit a CoxTimeVaryingFitter on *subset_df* with the given covariates.
    Uses a small L2 penalizer for numerical stability (stronger when round_to_hours >= 1).
    Phase/treatment indicators are taken from the dataframe (computed from original
    start/t_answer); only start/stop are rounded, so pre-treatment is never moved
    past treatment. Continuous covariates (see CONTINUOUS_COVARIATES) are clipped
    to 0.01/0.99 quantiles and z-scored to avoid nan/inf in the Hessian; their
    coefficients are then per standard deviation. Optional warm start via initial_point.
    Returns a CachedCoxResult (loaded from disk if available).
    """
    if round_to_hours is None:
        round_to_hours = ROUND_TO_HOURS
    base_suffix = f"_{round_to_hours}h" if round_to_hours and round_to_hours != 0.1 else ""
    cache_name = f"{model_name}{base_suffix}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _model_path(cache_name)

    # --- Check cache ---
    if use_cache and os.path.exists(path):
        print(f"  ✓ Cache hit: '{cache_name}'")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  ⚠ Cache load failed ({e}), refitting…")

    # --- Prepare data ---
    keep = ["unique_id", "start", "stop", "event_occurred"] + covariates
    extra = ["match_id"] if "match_id" in subset_df.columns else []
    fit_df = subset_df[[c for c in keep + extra if c in subset_df.columns]].copy()
    # Round only interval boundaries; phase/treatment already from original start (no reclassification)
    if round_to_hours and round_to_hours > 0:
        fit_df["start"] = (fit_df["start"] / round_to_hours).round() * round_to_hours
        fit_df["stop"] = (fit_df["stop"] / round_to_hours).round() * round_to_hours
    else:
        fit_df["start"] = fit_df["start"].round(1)
        fit_df["stop"] = fit_df["stop"].round(1)
    fit_df = fit_df[fit_df["start"] < fit_df["stop"]]
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan)
    fit_df = fit_df.dropna(subset=keep)

    # Subsample by match_id when over cap to preserve matched pairs and full trajectories
    subsampled = False
    n_rows_original = len(fit_df)
    if len(fit_df) > MAX_FIT_ROWS:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        if "match_id" in fit_df.columns:
            match_ids = fit_df["match_id"].unique()
            n_matches = len(match_ids)
            target_n_matches = max(1, int(MAX_FIT_ROWS * n_matches / len(fit_df)))
            target_n_matches = min(target_n_matches, n_matches)
            sampled_matches = rng.choice(match_ids, size=target_n_matches, replace=False)
            fit_df = fit_df[fit_df["match_id"].isin(sampled_matches)].copy()
            fit_df = fit_df.drop(columns=["match_id"])
            print(f"  (subsampled to {len(fit_df):,} rows from {target_n_matches:,} matches for numerical stability)")
        else:
            unique_ids = fit_df["unique_id"].unique()
            n_ids = len(unique_ids)
            target_n_ids = max(1, int(MAX_FIT_ROWS * n_ids / len(fit_df)))
            target_n_ids = min(target_n_ids, n_ids)
            sampled_ids = rng.choice(unique_ids, size=target_n_ids, replace=False)
            fit_df = fit_df[fit_df["unique_id"].isin(sampled_ids)].copy()
            print(f"  (subsampled to {len(fit_df):,} rows from {target_n_ids:,} questions for numerical stability)")
        subsampled = True

    # Clip and standardize continuous covariates to avoid nan/inf in Hessian/gradient
    for col in covariates:
        if col not in fit_df.columns or col not in CONTINUOUS_COVARIATES:
            continue
        q05, q95 = fit_df[col].quantile([0.05, 0.95])
        fit_df[col] = fit_df[col].clip(lower=q05, upper=q95)
        mu, sigma = fit_df[col].mean(), fit_df[col].std()
        if sigma > 0:
            fit_df[col] = (fit_df[col] - mu) / sigma

    # Center all covariates (mean 0) so linear predictor stays bounded in large risk sets
    for col in covariates:
        if col in fit_df.columns:
            fit_df[col] = fit_df[col] - fit_df[col].mean()

    # Jitter zero-variance covariates to avoid singular Hessian (keep same model for result extraction)
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    for col in covariates:
        if col in fit_df.columns and fit_df[col].std() < 1e-10:
            fit_df[col] = fit_df[col] + rng.standard_normal(len(fit_df)) * 1e-8

    n_events = int(fit_df["event_occurred"].sum())
    print(f"  Fitting '{cache_name}' — {len(fit_df):,} rows, {n_events:,} events …")

    if n_events < 10:
        print("  ⚠ Too few events, skipping.")
        return None

    # Build initial_point array in covariate order for warm start
    if initial_point is not None:
        if hasattr(initial_point, "get"):
            init_arr = np.array([float(initial_point.get(c, 0.0)) for c in covariates], dtype=float)
        else:
            init_arr = np.asarray(initial_point, dtype=float)
        if len(init_arr) != len(covariates):
            init_arr = None
    else:
        init_arr = None

    t0 = timer.time()
    has_continuous = any(c in covariates for c in CONTINUOUS_COVARIATES)
    if penalizer > 0:
        effective_penalizer = penalizer
    elif has_continuous:
        effective_penalizer = 1e-2  # stronger when response-time interactions in model
    elif round_to_hours and round_to_hours >= 1:
        effective_penalizer = 5e-3
    else:
        effective_penalizer = 1e-6
    ctv = CoxTimeVaryingFitter(penalizer=effective_penalizer)
    if round_to_hours and round_to_hours >= 1:
        step = 0.25 if has_continuous else 0.5
        fit_opts = {"step_size": step, "max_steps": 1000}
    else:
        fit_opts = None
    try:
        ctv.fit(
            fit_df,
            id_col="unique_id",
            event_col="event_occurred",
            start_col="start",
            stop_col="stop",
            show_progress=False,
            initial_point=init_arr,
            fit_options=fit_opts,
            robust=False,  # avoid sandwich estimator nan/inf with many ties
        )
    except ConvergenceError as e:
        print(f"  ⚠ Convergence failed: {e}. Skipping '{cache_name}'.")
        return None
    elapsed = timer.time() - t0
    print(f"  ✓ Fit in {elapsed:.1f}s")

    meta = {"n_events": n_events, "n_rows": len(fit_df)}
    if subsampled:
        meta["n_rows_original"] = n_rows_original
    result = CachedCoxResult(ctv, meta=meta)

    try:
        with open(path, "wb") as f:
            pickle.dump(result, f)
        print(f"  ✓ Cached to {path}")
    except Exception as e:
        print(f"  ⚠ Cache save failed: {e}")

    return result


# =====================================================================
# Data loading & interval construction
# =====================================================================

def create_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 7, 30, 180, 365, 1095, 2190, np.inf]
    df["tenure_bucket"] = pd.cut(
        df["user_tenure_days"], bins=bins, labels=BUCKET_ORDER, right=True
    )
    return df


def load_and_prepare(input_folder: str, sample_size: int = None):
    """Load parquet files, optionally subsample, build interval dataframe."""

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    cache_version = "v3"  # bump when adding columns (v4: quadratic log_response_time terms for Model C)
    cache_tag = f"sample_{sample_size}" if sample_size else "full"
    interval_cache = os.path.join(DATA_CACHE_DIR, f"intervals_{cache_tag}_{cache_version}.parquet")
    desc_cache = os.path.join(DATA_CACHE_DIR, f"descriptives_{cache_tag}_{cache_version}.pkl")

    if os.path.exists(interval_cache):
        print(f"✓ Loading cached intervals from {interval_cache}")
        model_df = pd.read_parquet(interval_cache)
        descriptives = None
        if os.path.exists(desc_cache):
            with open(desc_cache, "rb") as f:
                descriptives = pickle.load(f)
        return model_df, descriptives

    # --- Raw load ---
    print("=== Loading raw data ===")
    timelines = pd.read_parquet(f"{input_folder}/study_timelines.parquet")
    events = pd.read_parquet(f"{input_folder}/study_events.parquet")
    print(f"Loaded {len(timelines):,} timelines, {len(events):,} events")

    for col in ["t_start", "t_question", "t_answer", "t_end", "user_tenure_days"]:
        if col in timelines.columns:
            timelines[col] = timelines[col].astype(float)
    if "t_event" in events.columns:
        events["t_event"] = events["t_event"].astype(float)

    timelines = create_tenure_buckets(timelines)

    # --- Subsample ---
    if sample_size and sample_size < timelines["match_id"].nunique():
        print(f"Subsampling {sample_size:,} matched pairs …")
        ids = np.random.choice(
            timelines["match_id"].unique(), sample_size, replace=False
        )
        timelines = timelines[timelines["match_id"].isin(ids)].copy()
        events = events[events["match_id"].isin(ids)].copy()

    # --- Descriptive statistics (pre-interval) ---
    descriptives = _compute_descriptives(timelines, events)

    # --- Construct intervals ---
    print("Constructing event intervals …")

    boundaries = timelines.melt(
        id_vars=["match_id", "question_id", "hasAnswer", "t_answer", "tenure_bucket"],
        value_vars=["t_start", "t_question", "t_end"],
        value_name="time",
    )[["match_id", "question_id", "tenure_bucket", "time"]]

    answer_boundaries = timelines[
        ["match_id", "question_id", "tenure_bucket", "t_answer"]
    ].rename(columns={"t_answer": "time"})

    event_times = events[["match_id", "question_id", "t_event"]].rename(
        columns={"t_event": "time"}
    )
    event_times["is_event"] = 1

    all_times = pd.concat([boundaries, answer_boundaries, event_times], ignore_index=True)
    all_times["is_event"] = all_times["is_event"].fillna(0)
    all_times = all_times.sort_values(["match_id", "question_id", "time"])

    all_times = all_times.groupby(
        ["match_id", "question_id", "time"], as_index=False
    ).agg({"is_event": "max", "tenure_bucket": "first"})

    all_times["start"] = all_times["time"]
    all_times["stop"] = all_times.groupby(["match_id", "question_id"])["time"].shift(-1)

    intervals = all_times.dropna(subset=["stop"]).copy()
    intervals = intervals[intervals["stop"] > intervals["start"]]
    intervals["event_occurred"] = (
        all_times.groupby(["match_id", "question_id"])["is_event"]
        .shift(-1)
        .reindex(intervals.index)
    )

    # --- Covariates ---
    print("Computing covariates …")
    full_df = intervals.merge(
        timelines[["match_id", "question_id", "hasAnswer", "t_question", "t_answer"]],
        on=["match_id", "question_id"],
        how="inner",
    )
    full_df["hasAnswer"] = full_df["hasAnswer"].astype(int)
    full_df["unique_id"] = (
        full_df["match_id"].astype(str) + "_" + full_df["question_id"].astype(str)
    )

    # Phases
    full_df["phase_post_question"] = (full_df["start"] >= full_df["t_question"]).astype(int)
    full_df["phase_post"] = (full_df["start"] >= full_df["t_answer"]).astype(int)

    # Interactions
    full_df["treated_post_question"] = (
        (full_df["hasAnswer"] == 1) & (full_df["phase_post_question"] == 1)
    ).astype(int)
    full_df["is_treated_active"] = (
        (full_df["hasAnswer"] == 1) & (full_df["phase_post"] == 1)
    ).astype(int)

    # Response-time interaction
    full_df["log_response_time"] = np.log1p(full_df["t_answer"])
    full_df["hasAnswer_response_time_interaction"] = np.where(
        full_df["hasAnswer"] == 1,
        full_df["log_response_time"],
        0.0,
    )
    full_df["treated_response_time_interaction"] = (
        full_df["is_treated_active"] * full_df["log_response_time"]
    )
    full_df["treated_post_question_response_time_interaction"] = (
        full_df["treated_post_question"] * full_df["log_response_time"]
    )
    # Quadratic (log response time)² and interactions for Model C
    full_df["log_response_time_sq"] = full_df["log_response_time"] ** 2
    full_df["hasAnswer_response_time_sq_interaction"] = np.where(
        full_df["hasAnswer"] == 1,
        full_df["log_response_time_sq"],
        0.0,
    )
    full_df["treated_response_time_sq_interaction"] = (
        full_df["is_treated_active"] * full_df["log_response_time_sq"]
    )
    full_df["treated_post_question_response_time_sq_interaction"] = (
        full_df["treated_post_question"] * full_df["log_response_time_sq"]
    )

    # Response time (hours from question to answer) and tertile bin for non-linearity analysis
    full_df["response_time_hours"] = np.where(
        full_df["hasAnswer"] == 1,
        full_df["t_answer"] - full_df["t_question"],
        np.nan,
    )
    treated_for_tertiles = full_df.loc[full_df["hasAnswer"] == 1, "response_time_hours"].dropna()
    if len(treated_for_tertiles) >= 3:
        tertile_edges = treated_for_tertiles.quantile([1 / 3, 2 / 3]).values
        full_df["response_time_bin"] = 0  # control
        mask_treated = full_df["hasAnswer"] == 1
        full_df.loc[mask_treated & (full_df["response_time_hours"] <= tertile_edges[0]), "response_time_bin"] = 1
        full_df.loc[mask_treated & (full_df["response_time_hours"] > tertile_edges[0]) & (full_df["response_time_hours"] <= tertile_edges[1]), "response_time_bin"] = 2
        full_df.loc[mask_treated & (full_df["response_time_hours"] > tertile_edges[1]), "response_time_bin"] = 3
    else:
        full_df["response_time_bin"] = 0
    full_df["treated_bin2"] = ((full_df["response_time_bin"] == 2) & (full_df["is_treated_active"] == 1)).astype(int)
    full_df["treated_bin3"] = ((full_df["response_time_bin"] == 3) & (full_df["is_treated_active"] == 1)).astype(int)

    cols_to_keep = [
        "match_id", "unique_id", "start", "stop", "event_occurred",
        "hasAnswer",
        "phase_post_question", "treated_post_question",
        "phase_post", "is_treated_active",
        "hasAnswer_response_time_interaction",
        "treated_post_question_response_time_interaction",
        "treated_response_time_interaction",
        "hasAnswer_response_time_sq_interaction",
        "treated_post_question_response_time_sq_interaction",
        "treated_response_time_sq_interaction",
        "tenure_bucket",
        "response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3",
    ]
    model_df = full_df[cols_to_keep].replace([np.inf, -np.inf], np.nan).dropna()

    # --- Cache ---
    model_df.to_parquet(interval_cache)
    with open(desc_cache, "wb") as f:
        pickle.dump(descriptives, f)
    print(f"✓ Cached intervals to {interval_cache}")

    return model_df, descriptives


def _compute_descriptives(timelines: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Compute descriptive statistics from raw data for Table 2."""
    desc = {}

    desc["n_questions"] = len(timelines)
    desc["n_unique_users"] = timelines["question_id"].nunique()  # approximate
    if "user_id" in timelines.columns:
        desc["n_unique_users"] = timelines["user_id"].nunique()

    desc["pct_has_answer"] = timelines["hasAnswer"].mean() * 100

    # Help events per observation window
    events_per_q = events.groupby(["match_id", "question_id"]).size()
    # Merge back to get zeros
    all_q = timelines[["match_id", "question_id"]].copy()
    all_q["n_help_events"] = (
        all_q.set_index(["match_id", "question_id"])
        .index.map(events_per_q)
    )
    # Fallback: simple merge
    event_counts = (
        events.groupby(["match_id", "question_id"])
        .size()
        .reset_index(name="n_help_events")
    )
    merged = timelines.merge(event_counts, on=["match_id", "question_id"], how="left")
    merged["n_help_events"] = merged["n_help_events"].fillna(0)

    desc["help_events_mean"] = merged["n_help_events"].mean()
    desc["help_events_std"] = merged["n_help_events"].std()
    desc["help_events_median"] = merged["n_help_events"].median()

    # Tenure distribution
    desc["tenure_mean"] = timelines["user_tenure_days"].mean()
    desc["tenure_std"] = timelines["user_tenure_days"].std()
    desc["tenure_median"] = timelines["user_tenure_days"].median()

    # Response time (treated only)
    treated = timelines[timelines["hasAnswer"] == 1].copy()
    if "t_answer" in treated.columns and "t_question" in treated.columns:
        rt = treated["t_answer"] - treated["t_question"]
        rt = rt[rt > 0]
        desc["response_time_mean_hours"] = rt.mean()
        desc["response_time_std_hours"] = rt.std()
        desc["response_time_median_hours"] = rt.median()

    # Tenure bucket distribution
    timelines_bucketed = create_tenure_buckets(timelines.copy())
    desc["tenure_bucket_counts"] = (
        timelines_bucketed["tenure_bucket"].value_counts().to_dict()
    )

    # Balance-related columns (store whatever matching covariates exist)
    balance_candidates = [
        "timeSinceFirstActivityDays", "numQuestionsAskedAT",
        "numHelpProvidedAT", "numQuestionsAsked30D", "numHelpProvided30D",
        "numQuestionsAsked7D", "numHelpProvided7D", "year",
    ]
    available_balance_cols = [c for c in balance_candidates if c in timelines.columns]
    if available_balance_cols:
        desc["balance_data"] = {
            "columns": available_balance_cols,
            "treated": timelines[timelines["hasAnswer"] == 1][available_balance_cols]
            .describe()
            .to_dict(),
            "control": timelines[timelines["hasAnswer"] == 0][available_balance_cols]
            .describe()
            .to_dict(),
        }
    else:
        desc["balance_data"] = None

    return desc


# =====================================================================
# Model fitting
# =====================================================================

COVARIATES_MAIN = [
    "hasAnswer",
    "phase_post_question",
    "treated_post_question",
    "phase_post",
    "is_treated_active",
]

COVARIATES_SPEED = COVARIATES_MAIN + [
    "hasAnswer_response_time_interaction",
    "treated_response_time_interaction",
    "treated_post_question_response_time_interaction",
]

# Model C: linear + quadratic log(response time) and all interactions (pooled all-data only)
COVARIATES_QUADRATIC = COVARIATES_SPEED + [
    "hasAnswer_response_time_sq_interaction",
    "treated_response_time_sq_interaction",
]

# Response time bins (hours) for non-parametric moderation: [0, 0.25), [0.25, 0.5), ..., [8, 12)
RT_BIN_EDGES_HOURS = [0, 0.25, 0.5, 1, 2, 4, 8, 12]
RT_BIN_LABELS = [
    "0-15 min", "15-30 min", "30-60 min", "1-2 hr", "2-4 hr", "4-8 hr", "8-12 hr",
]


def _fit_one_rt_bin(args):
    """
    Worker for parallel response-time bin fits.
    args: (label, subset_df, cache_name, use_cache)
    Returns: (label, result_dict or None)
    """
    label, subset, cache_name, use_cache = args
    n_events = int(subset["event_occurred"].sum())
    if len(subset) < 100 or n_events < 10:
        return (label, None)
    res = fit_cox_cached(
        subset, cache_name, COVARIATES_MAIN,
        use_cache=use_cache, round_to_hours=ROUND_TO_HOURS,
    )
    if res is None:
        return (label, None)
    s = res.summary_df
    row = {
        "bucket": label,
        "treat_coef": s.loc["is_treated_active", "coef"],
        "treat_se": s.loc["is_treated_active", "se(coef)"],
        "treat_p": s.loc["is_treated_active", "p"],
        "treat_hr": np.exp(s.loc["is_treated_active", "coef"]),
        "treat_ci_lo": np.exp(s.loc["is_treated_active", "coef lower 95%"]),
        "treat_ci_hi": np.exp(s.loc["is_treated_active", "coef upper 95%"]),
        "n_rows": res.meta.get("n_rows", len(subset)),
        "n_events": res.meta.get("n_events", n_events),
    }
    return (label, row)


def fit_response_time_bin_models(
    model_df: pd.DataFrame, use_cache: bool = True, n_jobs: int = None
) -> pd.DataFrame:
    """
    Fit Model A (main effect) separately for each response-time bin (pooled over tenure).
    Each bin: treated = hasAnswer==1 and response_time_hours in [lo, hi); control = hasAnswer==0.
    Runs in parallel when n_jobs > 1.
    Returns a DataFrame with columns: bucket, treat_coef, treat_se, treat_p, treat_hr, treat_ci_lo, treat_ci_hi, n_rows, n_events.
    """
    if "response_time_hours" not in model_df.columns:
        print("  ⚠ response_time_hours not in model_df; skipping response-time bin models.")
        return pd.DataFrame()
    df_full = model_df.drop(columns=["tenure_bucket"], errors="ignore").copy()
    drop_cols = [c for c in ["response_time_bin", "treated_bin2", "treated_bin3"] if c in df_full.columns]
    n_bins = len(RT_BIN_EDGES_HOURS) - 1
    n_workers = n_jobs if n_jobs is not None else min(cpu_count() or 4, n_bins)
    tasks = []
    for i in range(n_bins):
        lo, hi = RT_BIN_EDGES_HOURS[i], RT_BIN_EDGES_HOURS[i + 1]
        label = RT_BIN_LABELS[i]
        mask = (
            (df_full["hasAnswer"] == 0)
            | (
                (df_full["hasAnswer"] == 1)
                & (df_full["response_time_hours"].notna())
                & (df_full["response_time_hours"] >= lo)
                & (df_full["response_time_hours"] < hi)
            )
        )
        subset = df_full.loc[mask].copy()
        for c in drop_cols:
            if c in subset.columns:
                subset = subset.drop(columns=[c])
        subset = subset.drop(columns=["response_time_hours"], errors="ignore")
        cache_name = f"ModelA_AllData_RTbin_{lo}_{hi}".replace(".", "_")
        tasks.append((label, subset, cache_name, use_cache))
    print("\n" + "=" * 60)
    print("  Response time bin models (Model A per bin)")
    print("=" * 60)
    if n_workers <= 1:
        results = [_fit_one_rt_bin(t) for t in tasks]
    else:
        print(f"  Fitting {n_bins} bins in parallel (n_jobs={n_workers}) …")
        with Pool(n_workers) as pool:
            results = pool.map(_fit_one_rt_bin, tasks)
    results = [r[1] for r in results if r[1] is not None]
    results.sort(key=lambda x: RT_BIN_LABELS.index(x["bucket"]) if x["bucket"] in RT_BIN_LABELS else 999)
    if not results:
        return pd.DataFrame()
    df_bins = pd.DataFrame(results)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_bins.to_csv(os.path.join(CACHE_DIR, "results_response_time_bins.csv"), index=False)
    print(f"✓ Saved results_response_time_bins.csv ({len(df_bins)} bins) to {CACHE_DIR}/")
    return df_bins


def _fit_one_tenure_bucket(args):
    """
    Worker for parallel tenure-bucket fits. Fits Model A then Model B (with warm start from A).
    args: (bucket, subset_df, use_cache, round_to_hours)
    Returns: (bucket, main_row_dict or None, speed_row_dict or None)
    """
    bucket, subset_df, use_cache, round_to_hours = args
    subset = subset_df.copy()
    if "tenure_bucket" in subset.columns:
        subset = subset.drop(columns=["tenure_bucket"])
    n_events = int(subset["event_occurred"].sum())
    if len(subset) < 100 or n_events < 10:
        return (bucket, None, None)

    name_a = f"ModelA_{bucket}"
    res_a = fit_cox_cached(
        subset, name_a, COVARIATES_MAIN,
        use_cache=use_cache, round_to_hours=round_to_hours,
    )
    if res_a is None:
        return (bucket, None, None)
    s_a = res_a.summary_df
    main_row = {
        "bucket": bucket,
        "n_rows": res_a.meta.get("n_rows", len(subset)),
        "n_events": res_a.meta.get("n_events", n_events),
        "treat_coef": s_a.loc["is_treated_active", "coef"],
        "treat_hr": np.exp(s_a.loc["is_treated_active", "coef"]),
        "treat_se": s_a.loc["is_treated_active", "se(coef)"],
        "treat_p": s_a.loc["is_treated_active", "p"],
        "treat_ci_lo": np.exp(s_a.loc["is_treated_active", "coef lower 95%"]),
        "treat_ci_hi": np.exp(s_a.loc["is_treated_active", "coef upper 95%"]),
        "gap_coef": s_a.loc["treated_post_question", "coef"],
        "gap_hr": np.exp(s_a.loc["treated_post_question", "coef"]),
        "gap_p": s_a.loc["treated_post_question", "p"],
        "phase_post_q_coef": s_a.loc["phase_post_question", "coef"],
        "phase_post_coef": s_a.loc["phase_post", "coef"],
        "hasAnswer_coef": s_a.loc["hasAnswer", "coef"],
    }

    name_b = f"ModelB_{bucket}"
    res_b = fit_cox_cached(
        subset, name_b, COVARIATES_SPEED,
        use_cache=use_cache, round_to_hours=round_to_hours,
        initial_point=None,  # cold start to avoid nan/inf with extra covariates
    )
    if res_b is None:
        return (bucket, main_row, None)
    s_b = res_b.summary_df
    speed_row = {
        "bucket": bucket,
        "n_rows": res_b.meta.get("n_rows", len(subset)),
        "n_events": res_b.meta.get("n_events", n_events),
        "treat_coef": s_b.loc["is_treated_active", "coef"],
        "treat_hr": np.exp(s_b.loc["is_treated_active", "coef"]),
        "treat_se": s_b.loc["is_treated_active", "se(coef)"],
        "treat_p": s_b.loc["is_treated_active", "p"],
        "speed_coef": s_b.loc["treated_response_time_interaction", "coef"],
        "speed_se": s_b.loc["treated_response_time_interaction", "se(coef)"],
        "speed_p": s_b.loc["treated_response_time_interaction", "p"],
        "gap_speed_coef": s_b.loc["treated_post_question_response_time_interaction", "coef"],
        "gap_speed_p": s_b.loc["treated_post_question_response_time_interaction", "p"],
    }
    return (bucket, main_row, speed_row)


def fit_all_models(model_df: pd.DataFrame, use_cache: bool = True, n_jobs: int = None):
    """Fit Model A (main) and Model B (speed) for each tenure bucket in parallel with warm start."""
    round_to_hours = ROUND_TO_HOURS
    n_workers = n_jobs if n_jobs is not None else min(cpu_count() or 4, len(BUCKET_ORDER))
    tasks = []
    for bucket in BUCKET_ORDER:
        subset = model_df[model_df["tenure_bucket"] == bucket].copy()
        tasks.append((bucket, subset, use_cache, round_to_hours))

    print(f"\nFitting tenure-bucket models in parallel (n_jobs={n_workers}, {round_to_hours}h windows) …")
    if n_workers <= 1:
        results = [_fit_one_tenure_bucket(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_fit_one_tenure_bucket, tasks)

    results_main = [r[1] for r in results if r[1] is not None]
    results_speed = [r[2] for r in results if r[2] is not None]
    # Preserve bucket order
    results_main.sort(key=lambda x: BUCKET_ORDER.index(x["bucket"]) if x["bucket"] in BUCKET_ORDER else 999)
    results_speed.sort(key=lambda x: BUCKET_ORDER.index(x["bucket"]) if x["bucket"] in BUCKET_ORDER else 999)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_main = pd.DataFrame(results_main)
    df_speed = pd.DataFrame(results_speed)
    df_main.to_csv(os.path.join(CACHE_DIR, "results_main.csv"), index=False)
    df_speed.to_csv(os.path.join(CACHE_DIR, "results_speed.csv"), index=False)
    print(f"✓ Saved results_main.csv, results_speed.csv ({len(df_main)} tenure buckets) to {CACHE_DIR}/")
    return df_main, df_speed


def fit_all_data_models(model_df: pd.DataFrame, use_cache: bool = True):
    """Fit one reciprocity model on all data (not stratified by seniority)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    subset = model_df.drop(columns=["tenure_bucket"], errors="ignore").copy()
    # Drop response_time_bin columns if present (not in COVARIATES_MAIN/SPEED/QUADRATIC)
    for c in ["response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3"]:
        if c in subset.columns:
            subset = subset.drop(columns=[c])
    n_events = int(subset["event_occurred"].sum())
    if len(subset) < 100 or n_events < 10:
        print("  ⚠ All data: too few rows/events, skipping.")
        return None, None, None

    print("\n" + "=" * 60)
    print("  All data (no tenure stratification)")
    print("=" * 60)
    res_a = fit_cox_cached(
        subset, "ModelA_AllData", COVARIATES_MAIN,
        use_cache=use_cache, round_to_hours=ROUND_TO_HOURS,
    )
    res_b = fit_cox_cached(
        subset, "ModelB_AllData", COVARIATES_SPEED,
        use_cache=use_cache, round_to_hours=ROUND_TO_HOURS,
        initial_point=None,  # cold start for speed model stability
    )
    if res_a is None or res_b is None:
        return None, None, None

    s_a = res_a.summary_df
    s_b = res_b.summary_df
    results_main_all = [{
        "model": "AllData_Main",
        "n_rows": res_a.meta.get("n_rows", len(subset)),
        "n_events": res_a.meta.get("n_events", n_events),
        "treat_coef": s_a.loc["is_treated_active", "coef"],
        "treat_hr": np.exp(s_a.loc["is_treated_active", "coef"]),
        "treat_se": s_a.loc["is_treated_active", "se(coef)"],
        "treat_p": s_a.loc["is_treated_active", "p"],
        "treat_ci_lo": np.exp(s_a.loc["is_treated_active", "coef lower 95%"]),
        "treat_ci_hi": np.exp(s_a.loc["is_treated_active", "coef upper 95%"]),
        "gap_coef": s_a.loc["treated_post_question", "coef"],
        "gap_hr": np.exp(s_a.loc["treated_post_question", "coef"]),
        "gap_p": s_a.loc["treated_post_question", "p"],
    }]
    results_speed_all = [{
        "model": "AllData_Speed",
        "n_rows": res_b.meta.get("n_rows", len(subset)),
        "n_events": res_b.meta.get("n_events", n_events),
        "treat_coef": s_b.loc["is_treated_active", "coef"],
        "treat_hr": np.exp(s_b.loc["is_treated_active", "coef"]),
        "treat_se": s_b.loc["is_treated_active", "se(coef)"],
        "treat_p": s_b.loc["is_treated_active", "p"],
        "speed_coef": s_b.loc["treated_response_time_interaction", "coef"],
        "speed_se": s_b.loc["treated_response_time_interaction", "se(coef)"],
        "speed_p": s_b.loc["treated_response_time_interaction", "p"],
    }]
    df_main_all = pd.DataFrame(results_main_all)
    df_speed_all = pd.DataFrame(results_speed_all)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_main_all.to_csv(os.path.join(CACHE_DIR, "results_main_all.csv"), index=False)
    df_speed_all.to_csv(os.path.join(CACHE_DIR, "results_speed_all.csv"), index=False)

    # Model C: pooled all-data with quadratic log(response time) and all interactions
    df_model_c_all = None
    if all(c in subset.columns for c in COVARIATES_QUADRATIC):
        res_c = fit_cox_cached(
            subset, "ModelC_AllData", COVARIATES_QUADRATIC,
            use_cache=use_cache, round_to_hours=ROUND_TO_HOURS,
            initial_point=None,
        )
    else:
        res_c = None
    if res_c is not None:
        s_c = res_c.summary_df
        results_model_c_all = [{
            "model": "AllData_ModelC",
            "n_rows": res_c.meta.get("n_rows", len(subset)),
            "n_events": res_c.meta.get("n_events", n_events),
            "treat_coef": s_c.loc["is_treated_active", "coef"],
            "treat_hr": np.exp(s_c.loc["is_treated_active", "coef"]),
            "treat_se": s_c.loc["is_treated_active", "se(coef)"],
            "treat_p": s_c.loc["is_treated_active", "p"],
            "treat_ci_lo": np.exp(s_c.loc["is_treated_active", "coef lower 95%"]),
            "treat_ci_hi": np.exp(s_c.loc["is_treated_active", "coef upper 95%"]),
            "speed_coef": s_c.loc["treated_response_time_interaction", "coef"],
            "speed_se": s_c.loc["treated_response_time_interaction", "se(coef)"],
            "speed_p": s_c.loc["treated_response_time_interaction", "p"],
            "speed_sq_coef": s_c.loc["treated_response_time_sq_interaction", "coef"],
            "speed_sq_se": s_c.loc["treated_response_time_sq_interaction", "se(coef)"],
            "speed_sq_p": s_c.loc["treated_response_time_sq_interaction", "p"],
            "gap_coef": s_c.loc["treated_post_question", "coef"],
            "gap_p": s_c.loc["treated_post_question", "p"],
            "gap_speed_coef": s_c.loc["treated_post_question_response_time_interaction", "coef"],
            "gap_speed_p": s_c.loc["treated_post_question_response_time_interaction", "p"],
        }]
        df_model_c_all = pd.DataFrame(results_model_c_all)
        df_model_c_all.to_csv(os.path.join(CACHE_DIR, "results_model_c_all.csv"), index=False)
        print(f"✓ Saved results_main_all.csv, results_speed_all.csv, results_model_c_all.csv (All data) to {CACHE_DIR}/")
    else:
        print(f"✓ Saved results_main_all.csv, results_speed_all.csv (All data) to {CACHE_DIR}/")

    return df_main_all, df_speed_all, df_model_c_all


# =====================================================================
# CLI entry point
# =====================================================================

def main():
    # Default input relative to this script so it works from any cwd
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_input = os.path.normpath(os.path.join(_script_dir, "..", "data", "event_history"))
    parser = argparse.ArgumentParser(description="Fit Cox survival models per tenure bucket")
    parser.add_argument("--input", default=_default_input, help="Input data folder")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached models")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for tenure-bucket fits (default: min(cpu_count, 7))")
    args = parser.parse_args()

    model_df, descriptives = load_and_prepare(args.input, sample_size=args.sample)

    # Save descriptives for the output script
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "descriptives.pkl"), "wb") as f:
        pickle.dump(descriptives, f)

    df_main, df_speed = fit_all_models(model_df, use_cache=not args.no_cache, n_jobs=args.n_jobs)

    # 1. One reciprocity model on all data (not stratified by seniority) + Model C (quadratic)
    df_main_all, df_speed_all, df_model_c_all = fit_all_data_models(model_df, use_cache=not args.no_cache)
    if df_main_all is not None:
        print("\n=== All data: Main effect ===")
        print(df_main_all.to_string(index=False))
    if df_speed_all is not None:
        print("\n=== All data: Speed interaction ===")
        print(df_speed_all.to_string(index=False))
    if df_model_c_all is not None:
        print("\n=== All data: Model C (quadratic log response time) ===")
        print(df_model_c_all.to_string(index=False))

    df_rt_bins = fit_response_time_bin_models(
        model_df, use_cache=not args.no_cache, n_jobs=args.n_jobs
    )
    if not df_rt_bins.empty:
        print("\n=== Response time bin models ===")
        print(df_rt_bins.to_string(index=False))

    print("\n=== Main effect (by tenure bucket) ===")
    print(df_main.to_string(index=False))
    print("\n=== Speed interaction (by tenure bucket) ===")
    print(df_speed.to_string(index=False))


if __name__ == "__main__":
    main()