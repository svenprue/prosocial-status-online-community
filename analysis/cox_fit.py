"""Cox model fitting: cached fitter, tenure-bucket models, all-data and RT-bin models."""
import os
import pickle
import time as timer
import inspect
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceError

from cox_config import (
    CACHE_DIR,
    BUCKET_ORDER,
    ROUND_TO_HOURS,
    CONTINUOUS_COVARIATES,
    CONTINUOUS_COVARIATES_EXTENDED,
    MAX_FIT_ROWS,
    SUBSAMPLE_SEED,
    COVARIATES_MAIN,
    COVARIATES_SPEED,
    RT_BIN_EDGES_HOURS,
    RT_BIN_LABELS,
    VARIANCE_ESTIMATOR,
    CLUSTER_COL,
)


class CachedCoxResult:
    """Lightweight proxy for a fitted CoxTimeVaryingFitter."""

    def __init__(self, ctv, meta: dict = None):
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


def fit_cox_cached(
    subset_df: pd.DataFrame,
    model_name: str,
    covariates: list,
    penalizer: float = 0.0,
    use_cache: bool = True,
    round_to_hours: float = None,
    initial_point=None,
    robust: bool = False,
    cluster_col: str = CLUSTER_COL,
    save_cache: bool = True,
):
    """Fit Cox time-varying model; load from cache if present."""
    if round_to_hours is None:
        round_to_hours = ROUND_TO_HOURS
    base_suffix = f"_{round_to_hours}h" if round_to_hours and round_to_hours != 0.1 else ""
    variance_suffix = f"_{VARIANCE_ESTIMATOR}" if robust else ""
    cache_name = f"{model_name}{base_suffix}{variance_suffix}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _model_path(cache_name)

    if use_cache and os.path.exists(path):
        print(f"  ✓ Cache hit: '{cache_name}'")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  ⚠ Cache load failed ({e}), refitting…")

    keep = ["unique_id", "start", "stop", "event_occurred"] + covariates
    extra = [cluster_col] if robust and cluster_col and cluster_col in subset_df.columns else []
    fit_df = subset_df[[c for c in keep + extra if c in subset_df.columns]].copy()
    if round_to_hours and round_to_hours > 0:
        fit_df["start"] = (fit_df["start"] / round_to_hours).round() * round_to_hours
        fit_df["stop"] = (fit_df["stop"] / round_to_hours).round() * round_to_hours
    else:
        fit_df["start"] = fit_df["start"].round(1)
        fit_df["stop"] = fit_df["stop"].round(1)
    fit_df = fit_df[fit_df["start"] < fit_df["stop"]]
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep)

    subsampled = False
    n_rows_original = len(fit_df)
    if len(fit_df) > MAX_FIT_ROWS:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        if "match_id" in fit_df.columns:
            match_ids = fit_df["match_id"].unique()
            target_n = max(1, int(MAX_FIT_ROWS * len(match_ids) / len(fit_df)))
            target_n = min(target_n, len(match_ids))
            sampled = rng.choice(match_ids, size=target_n, replace=False)
            fit_df = fit_df[fit_df["match_id"].isin(sampled)].copy()
        else:
            unique_ids = fit_df["unique_id"].unique()
            target_n = max(1, int(MAX_FIT_ROWS * len(unique_ids) / len(fit_df)))
            target_n = min(target_n, len(unique_ids))
            sampled = rng.choice(unique_ids, size=target_n, replace=False)
            fit_df = fit_df[fit_df["unique_id"].isin(sampled)].copy()
        print(f"  (subsampled to {len(fit_df):,} rows for numerical stability)")
        subsampled = True

    for col in covariates:
        if col not in fit_df.columns or col not in CONTINUOUS_COVARIATES_EXTENDED:
            continue
        q05, q95 = fit_df[col].quantile([0.05, 0.95])
        fit_df[col] = fit_df[col].clip(lower=q05, upper=q95)
        mu, sigma = fit_df[col].mean(), fit_df[col].std()
        if sigma > 0:
            fit_df[col] = (fit_df[col] - mu) / sigma
    for col in covariates:
        if col in fit_df.columns:
            fit_df[col] = fit_df[col] - fit_df[col].mean()
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    for col in covariates:
        if col in fit_df.columns and fit_df[col].std() < 1e-10:
            fit_df[col] = fit_df[col] + rng.standard_normal(len(fit_df)) * 1e-8

    n_events = int(fit_df["event_occurred"].sum())
    cluster_for_fit = cluster_col if robust and cluster_col and cluster_col in fit_df.columns else None
    if robust and cluster_col and cluster_for_fit is None:
        raise ValueError(
            f"Requested clustered Cox SEs with cluster_col='{cluster_col}', but that column is absent. "
            "Regenerate the model dataframe or call fit_cox_cached(..., robust=False)."
        )
    variance_label = f"clustered by {cluster_for_fit}" if cluster_for_fit else "model-based"
    print(f"  Fitting '{cache_name}' — {len(fit_df):,} rows, {n_events:,} events, SEs {variance_label} …")
    if n_events < 10:
        print("  ⚠ Too few events, skipping.")
        return None

    init_arr = None
    if initial_point is not None:
        if hasattr(initial_point, "get"):
            init_arr = np.array([float(initial_point.get(c, 0.0)) for c in covariates], dtype=float)
        else:
            init_arr = np.asarray(initial_point, dtype=float)
        if len(init_arr) != len(covariates):
            init_arr = None

    has_continuous = any(c in covariates for c in CONTINUOUS_COVARIATES_EXTENDED)
    effective_penalizer = 1e-2 if has_continuous else (5e-3 if round_to_hours and round_to_hours >= 1 else 1e-6)
    if penalizer > 0:
        effective_penalizer = penalizer
    ctv = CoxTimeVaryingFitter(penalizer=effective_penalizer)
    fit_opts = {"step_size": 0.25 if has_continuous else 0.5, "max_steps": 1000} if round_to_hours and round_to_hours >= 1 else None
    t0 = timer.time()
    fit_kwargs = {
        "df": fit_df,
        "id_col": "unique_id",
        "event_col": "event_occurred",
        "start_col": "start",
        "stop_col": "stop",
        "show_progress": False,
        "initial_point": init_arr,
        "fit_options": fit_opts,
    }
    fit_signature = inspect.signature(ctv.fit)
    if "robust" in fit_signature.parameters:
        fit_kwargs["robust"] = bool(cluster_for_fit)
    if cluster_for_fit and "cluster_col" in fit_signature.parameters:
        fit_kwargs["cluster_col"] = cluster_for_fit
    elif cluster_for_fit and "robust" in fit_signature.parameters:
        fit_kwargs["df"] = fit_df.drop(columns=[cluster_for_fit])
        print(
            "  ⚠ lifelines CoxTimeVaryingFitter.fit does not expose cluster_col; "
            "using its robust=True sandwich variance without an explicit cluster column."
        )
    elif cluster_for_fit:
        raise RuntimeError(
            "Installed lifelines CoxTimeVaryingFitter.fit exposes neither cluster_col nor robust; "
            "use a lifelines version with robust variance support or run a matched-pair bootstrap."
        )
    try:
        ctv.fit(**fit_kwargs)
    except NotImplementedError as e:
        print(
            f"  ⚠ Robust CoxTimeVaryingFitter variance is unavailable in this lifelines version: {e}. "
            "Use analysis/pair_bootstrap_se.py for matched-pair bootstrap uncertainty."
        )
        return None
    except ConvergenceError as e:
        print(f"  ⚠ Convergence failed: {e}. Skipping '{cache_name}'.")
        return None
    print(f"  ✓ Fit in {timer.time() - t0:.1f}s")
    meta = {
        "n_events": n_events,
        "n_rows": len(fit_df),
        "variance_estimator": VARIANCE_ESTIMATOR if cluster_for_fit else "model_based",
        "cluster_col": cluster_for_fit,
    }
    if subsampled:
        meta["n_rows_original"] = n_rows_original
    result = CachedCoxResult(ctv, meta=meta)
    if not save_cache:
        return result
    try:
        with open(path, "wb") as f:
            pickle.dump(result, f)
        print(f"  ✓ Cached to {path}")
    except Exception as e:
        print(f"  ⚠ Cache save failed: {e}")
    return result


def _fit_one_rt_bin(args):
    label, subset, cache_name, use_cache = args
    n_events = int(subset["event_occurred"].sum())
    n_questions = int(subset["unique_id"].nunique())
    if len(subset) < 100 or n_events < 10:
        return (label, None)
    res = fit_cox_cached(subset, cache_name, COVARIATES_MAIN, use_cache=use_cache, round_to_hours=ROUND_TO_HOURS)
    if res is None:
        return (label, None)
    s = res.summary_df
    return (label, {
        "bucket": label,
        "treat_coef": s.loc["is_treated_active", "coef"],
        "treat_se": s.loc["is_treated_active", "se(coef)"],
        "treat_p": s.loc["is_treated_active", "p"],
        "treat_hr": np.exp(s.loc["is_treated_active", "coef"]),
        "treat_ci_lo": np.exp(s.loc["is_treated_active", "coef lower 95%"]),
        "treat_ci_hi": np.exp(s.loc["is_treated_active", "coef upper 95%"]),
        "n_rows": res.meta.get("n_rows", len(subset)),
        "n_questions": n_questions,
        "n_events": res.meta.get("n_events", n_events),
    })


def fit_response_time_bin_models(model_df: pd.DataFrame, use_cache: bool = True, n_jobs: int = None):
    """Fit Model A per response-time bin (pooled)."""
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
            | ((df_full["hasAnswer"] == 1) & (df_full["response_time_hours"].notna())
               & (df_full["response_time_hours"] >= lo) & (df_full["response_time_hours"] < hi))
        )
        subset = df_full.loc[mask].copy()
        for c in drop_cols:
            if c in subset.columns:
                subset = subset.drop(columns=[c])
        subset = subset.drop(columns=["response_time_hours"], errors="ignore")
        tasks.append((label, subset, f"ModelA_AllData_RTbin_{lo}_{hi}".replace(".", "_"), use_cache))
    print("\n" + "=" * 60 + "\n  Response time bin models (Model A per bin)\n" + "=" * 60)
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_fit_one_rt_bin, tasks)
    else:
        results = [_fit_one_rt_bin(t) for t in tasks]
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
    bucket, subset_df, use_cache, round_to_hours = args
    subset = subset_df.copy()
    if "tenure_bucket" in subset.columns:
        subset = subset.drop(columns=["tenure_bucket"])
    n_events = int(subset["event_occurred"].sum())
    n_questions = int(subset["unique_id"].nunique())
    n_treated = int(subset.loc[subset["hasAnswer"] == 1, "unique_id"].nunique())
    n_control = int(subset.loc[subset["hasAnswer"] == 0, "unique_id"].nunique())
    assert n_questions == n_treated + n_control
    # Per-bucket n_treated vs n_control can differ: tenure is assigned per question, so stratifying by tenure_bucket does not preserve the global 1:1 match within each bucket.
    if len(subset) < 100 or n_events < 10:
        return (bucket, None, None)

    res_a = fit_cox_cached(subset, f"ModelA_{bucket}", COVARIATES_MAIN, use_cache=use_cache, round_to_hours=round_to_hours)
    if res_a is None:
        return (bucket, None, None)
    s_a = res_a.summary_df
    main_row = {
        "bucket": bucket,
        "n_rows": res_a.meta.get("n_rows", len(subset)),
        "n_questions": n_questions,
        "n_questions_treated": n_treated,
        "n_questions_control": n_control,
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

    res_b = fit_cox_cached(subset, f"ModelB_{bucket}", COVARIATES_SPEED, use_cache=use_cache, round_to_hours=round_to_hours, initial_point=None)
    if res_b is None:
        return (bucket, main_row, None)
    s_b = res_b.summary_df
    speed_row = {
        "bucket": bucket,
        "n_rows": res_b.meta.get("n_rows", len(subset)),
        "n_questions": n_questions,
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
    """Fit Model A and B per tenure bucket."""
    n_workers = n_jobs if n_jobs is not None else min(cpu_count() or 4, len(BUCKET_ORDER))
    tasks = [(b, model_df[model_df["tenure_bucket"] == b].copy(), use_cache, ROUND_TO_HOURS) for b in BUCKET_ORDER]
    print(f"\nFitting tenure-bucket models in parallel (n_jobs={n_workers}, {ROUND_TO_HOURS}h windows) …")
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_fit_one_tenure_bucket, tasks)
    else:
        results = [_fit_one_tenure_bucket(t) for t in tasks]
    results_main = [r[1] for r in results if r[1] is not None]
    results_speed = [r[2] for r in results if r[2] is not None]
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
    """Fit Model A and B on all data (no tenure stratification)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    subset = model_df.drop(columns=["tenure_bucket"], errors="ignore").copy()
    for c in ["response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3"]:
        if c in subset.columns:
            subset = subset.drop(columns=[c])
    n_events = int(subset["event_occurred"].sum())
    n_questions = int(subset["unique_id"].nunique())
    if len(subset) < 100 or n_events < 10:
        print("  ⚠ All data: too few rows/events, skipping.")
        return None, None
    print("\n" + "=" * 60 + "\n  All data (no tenure stratification)\n" + "=" * 60)
    res_a = fit_cox_cached(subset, "ModelA_AllData", COVARIATES_MAIN, use_cache=use_cache, round_to_hours=ROUND_TO_HOURS)
    res_b = fit_cox_cached(subset, "ModelB_AllData", COVARIATES_SPEED, use_cache=use_cache, round_to_hours=ROUND_TO_HOURS, initial_point=None)
    if res_a is None or res_b is None:
        return None, None
    s_a, s_b = res_a.summary_df, res_b.summary_df
    results_main_all = [{
        "model": "AllData_Main",
        "n_rows": res_a.meta.get("n_rows", len(subset)),
        "n_questions": n_questions,
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
        "n_questions": n_questions,
        "n_events": res_b.meta.get("n_events", n_events),
        "treat_coef": s_b.loc["is_treated_active", "coef"],
        "treat_hr": np.exp(s_b.loc["is_treated_active", "coef"]),
        "treat_se": s_b.loc["is_treated_active", "se(coef)"],
        "treat_ci_lo": np.exp(s_b.loc["is_treated_active", "coef lower 95%"]),
        "treat_ci_hi": np.exp(s_b.loc["is_treated_active", "coef upper 95%"]),
        "treat_p": s_b.loc["is_treated_active", "p"],
        "gap_coef": s_b.loc["treated_post_question", "coef"],
        "gap_p": s_b.loc["treated_post_question", "p"],
        "speed_coef": s_b.loc["treated_response_time_interaction", "coef"],
        "speed_se": s_b.loc["treated_response_time_interaction", "se(coef)"],
        "speed_p": s_b.loc["treated_response_time_interaction", "p"],
    }]
    df_main_all = pd.DataFrame(results_main_all)
    df_speed_all = pd.DataFrame(results_speed_all)
    df_main_all.to_csv(os.path.join(CACHE_DIR, "results_main_all.csv"), index=False)
    df_speed_all.to_csv(os.path.join(CACHE_DIR, "results_speed_all.csv"), index=False)
    print(f"✓ Saved results_main_all.csv, results_speed_all.csv (All data) to {CACHE_DIR}/")
    return df_main_all, df_speed_all
