"""Cox model fitting: cached fitter, tenure-bucket models, all-data and RT-bin models."""
import os
import pickle
import hashlib
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
    RT_INTERACTION_TERMS,
    MAX_FIT_ROWS,
    MAX_FIT_WORKERS,
    SUBSAMPLE_SEED,
    COVARIATES_MAIN,
    COVARIATES_MAIN_QUALITY,
    COVARIATES_SPEED,
    RT_BIN_EDGES_HOURS,
    RT_BIN_LABELS,
    VARIANCE_ESTIMATOR,
    CLUSTER_COL,
    DATA_VERSION,
)


def _parallel_workers(n_units: int, n_jobs: int | None) -> int:
    """Bound Pool size by CPU, unit count, and memory-aware MAX_FIT_WORKERS."""
    if n_jobs is not None:
        return max(1, min(int(n_jobs), n_units))
    return max(1, min(cpu_count() or 4, n_units, MAX_FIT_WORKERS))

# The two treated indicators are nested: `treated_post_question` switches on when
# the question is posted and STAYS on through the post-answer phase; `is_treated_active`
# adds on only after the answer arrives. Per the 2026-07-14 estimand decision,
# `is_treated_active` (beta_4) ALONE is the difference-in-differences treatment effect
# (the answer-arrival increment); `treated_post_question` (beta_2) is reported separately
# as a parallel-trends diagnostic (it has no sign guarantee, so summing it into beta_4
# would not bound anything). The SUM of the two coefficients (_linear_combo / DID_TERMS)
# is still computed into `did_*` CSV fields for internal diagnostics, but must never be
# emitted into a .tex table/caption/footnote — see cox_config.HEADLINE_ESTIMAND.
DID_TERMS = ["treated_post_question", "is_treated_active"]
# Speed spec: the CSV-diagnostic-only summed RT moderation is the sum of the two RT interactions.
DID_SPEED_TERMS = [
    "treated_post_question_response_time_interaction",
    "treated_response_time_interaction",
]


def _linear_combo(res, terms):
    """Point estimate, SE, HR, and 95% CI for a linear combination (sum) of fitted
    coefficients.

    Uses the fitted parameter vector and its covariance matrix, so the SE correctly
    accounts for the covariance between the terms — essential here because the two
    nested treated indicators are strongly (negatively) correlated, which is exactly
    why their sum is the well-identified quantity even when each term alone is not.

    Returns a dict with keys coef/se/z/p/hr/ci_lo/ci_hi, or None if no term is present.
    """
    from scipy.stats import norm

    params = getattr(res, "params_", None)
    vcov = getattr(res, "variance_matrix_", None)
    if params is None or vcov is None:
        return None
    present = [t for t in terms if t in params.index]
    if not present:
        return None
    coef = float(params.loc[present].sum())
    try:
        var = float(np.asarray(vcov.loc[present, present].values).sum())
    except Exception:
        # lifelines sometimes stores variance_matrix with integer column labels
        try:
            idx = [params.index.get_loc(t) for t in present]
            sub = np.asarray(vcov.values)[np.ix_(idx, idx)]
            var = float(sub.sum())
        except Exception:
            return None
    se = float(np.sqrt(var)) if var and var > 0 else float("nan")
    z = coef / se if se and np.isfinite(se) and se > 0 else float("nan")
    p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {
        "coef": coef,
        "se": se,
        "z": z,
        "p": p,
        "hr": float(np.exp(coef)),
        "ci_lo": float(np.exp(coef - 1.96 * se)) if np.isfinite(se) else float("nan"),
        "ci_hi": float(np.exp(coef + 1.96 * se)) if np.isfinite(se) else float("nan"),
    }


def _did_fields(res, terms=DID_TERMS, prefix="did"):
    """Return a dict of {prefix}_coef/se/p/hr/ci_lo/ci_hi for the summed contrast
    (CSV diagnostic only — never rendered into a .tex table; see DID_TERMS comment),
    or all-NaN fields (so downstream CSV columns stay consistent) if unavailable."""
    combo = _linear_combo(res, terms)
    keys = ["coef", "se", "p", "hr", "ci_lo", "ci_hi"]
    if combo is None:
        return {f"{prefix}_{k}": float("nan") for k in keys}
    return {f"{prefix}_{k}": combo[k] for k in keys}


def _arrival_fields(summary, term="is_treated_active", prefix="arrival"):
    """Emit the ARRIVAL increment (a single fitted coefficient) with a normal-approx 95%
    CI (exp(coef ± 1.96·se)) so every results CSV carries {prefix}_coef/se/p/hr/ci_lo/ci_hi.
    This is THE DiD treatment effect (beta_4); the summed did_* fields alongside it are
    CSV-only diagnostics that create_figures.py must never render into a .tex table.

    For the headline treatment effect pass term='is_treated_active' (beta_4). For the RT
    moderation pass term='treated_response_time_interaction' (gamma) with prefix='arr_speed'.
    Returns all-NaN fields if the term is absent (keeps CSV columns consistent)."""
    keys = ["coef", "se", "p", "hr", "ci_lo", "ci_hi"]
    if term not in summary.index:
        return {f"{prefix}_{k}": float("nan") for k in keys}
    coef = float(summary.loc[term, "coef"])
    se = float(summary.loc[term, "se(coef)"])
    p = float(summary.loc[term, "p"])
    return {
        f"{prefix}_coef": coef,
        f"{prefix}_se": se,
        f"{prefix}_p": p,
        f"{prefix}_hr": float(np.exp(coef)),
        f"{prefix}_ci_lo": float(np.exp(coef - 1.96 * se)),
        f"{prefix}_ci_hi": float(np.exp(coef + 1.96 * se)),
    }


class CachedCoxResult:
    """Lightweight proxy for a fitted CoxTimeVaryingFitter."""

    def __init__(self, ctv, meta: dict = None):
        self.summary_df = ctv.summary.copy()
        self.params_ = ctv.params_.copy()
        self.confidence_intervals_ = ctv.confidence_intervals_.copy()
        vcov = ctv.variance_matrix_.copy()
        if not vcov.index.equals(vcov.columns):
            vcov.columns = vcov.index
        self.variance_matrix_ = vcov
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
    # fix(cache): key on DATA_VERSION and a short hash of the (sorted) covariate list, so
    # two specs that share a model_name but differ in covariates — or a rerun after a data
    # rebuild — get distinct cache files instead of colliding. Backward-safe (old names miss).
    cov_hash = hashlib.md5(",".join(sorted(covariates)).encode("utf-8")).hexdigest()[:8]
    # fix(cache): tag the cache name with MAX_FIT_ROWS whenever it differs from the 8M
    # default, so a full-data (COX_MAX_FIT_ROWS-raised) point-estimate fit never collides
    # with — or is mistaken for — the 8M-subsample pickle of the same model_name. Default
    # (8M) runs keep the untagged name, so existing caches stay valid and the matched-pair
    # bootstrap (which runs at the 8M default) is unaffected.
    rows_tag = "" if MAX_FIT_ROWS == 8_000_000 else f"_r{MAX_FIT_ROWS // 1_000_000}M"
    cache_name = f"{model_name}{base_suffix}{variance_suffix}_{DATA_VERSION}{rows_tag}_{cov_hash}"
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
    # fix(subsample): retain match_id (when present) through the subsample step for BOTH
    # robust and non-robust fits, so the >MAX_FIT_ROWS block below takes its match_id branch
    # and keeps matched pairs together (the "matched-pair subsample" the footnote promises),
    # instead of falling through to the unique_id branch. match_id is dropped before ctv.fit
    # (below) whenever it is not the active cluster column, so it is never fit as a covariate.
    extra = [cluster_col] if cluster_col and cluster_col in subset_df.columns else []
    fit_df = subset_df[[c for c in keep + extra if c in subset_df.columns]].copy()
    if round_to_hours and round_to_hours > 0:
        fit_df["start"] = (fit_df["start"] / round_to_hours).round() * round_to_hours
        fit_df["stop"] = (fit_df["stop"] / round_to_hours).round() * round_to_hours
    else:
        fit_df["start"] = fit_df["start"].round(1)
        fit_df["stop"] = fit_df["stop"].round(1)
    fit_df = fit_df[fit_df["start"] < fit_df["stop"]]
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep)

    # Analysis-sample sizes (post time-rounding, pre MAX_FIT_ROWS subsample). Tables must
    # report these so pooled Events match the sum of tenure-bucket Events / full N.
    n_rows_analysis = len(fit_df)
    n_events_analysis = int(fit_df["event_occurred"].sum())

    subsampled = False
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
        print(
            f"  (subsampled to {len(fit_df):,} rows for numerical stability; "
            f"analysis sample had {n_rows_analysis:,} rows / {n_events_analysis:,} events)"
        )
        subsampled = True

    for col in covariates:
        if col not in fit_df.columns or col not in CONTINUOUS_COVARIATES_EXTENDED:
            continue
        # fix(scale): the RT interaction terms are pre-standardized in cox_data on a single
        # common (treated-row) scale. Re-winsorizing/z-scoring them here — each by its own
        # nonzero-row SD — would re-break the common scale and make gamma+delta an invalid
        # sum of differently-scaled coefficients. Skip them, but keep them counted for
        # has_continuous / penalizer / step_size (they stay in CONTINUOUS_COVARIATES_EXTENDED).
        if col in RT_INTERACTION_TERMS:
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

    n_events_fit = int(fit_df["event_occurred"].sum())
    cluster_for_fit = cluster_col if robust and cluster_col and cluster_col in fit_df.columns else None
    if robust and cluster_col and cluster_for_fit is None:
        raise ValueError(
            f"Requested clustered Cox SEs with cluster_col='{cluster_col}', but that column is absent. "
            "Regenerate the model dataframe or call fit_cox_cached(..., robust=False)."
        )

    # fix(subsample): match_id was retained above only so the matched-pair subsample keeps
    # pairs together. Drop it (and any other non-covariate id column) now — before the fit —
    # so lifelines does not treat it as a covariate. Keep it ONLY when it is the active
    # cluster column (the robust/cluster path passes it via cluster_col / robust below).
    # Derive the drop list from the retained id columns (`extra`), not a hardcoded
    # ["match_id"], so a non-default cluster_col passed with robust=False is also dropped
    # rather than silently fit as a covariate.
    drop_before_fit = [
        c for c in extra
        if c in fit_df.columns and c not in covariates and c != cluster_for_fit
    ]
    if drop_before_fit:
        fit_df = fit_df.drop(columns=drop_before_fit)
    variance_label = f"clustered by {cluster_for_fit}" if cluster_for_fit else "model-based"
    print(
        f"  Fitting '{cache_name}' — {len(fit_df):,} rows, {n_events_fit:,} fit events"
        f" ({n_events_analysis:,} analysis-sample events), SEs {variance_label} …"
    )
    if n_events_fit < 10:
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
        # n_events / n_rows = analysis sample (for tables); *_fit = estimation subsample.
        "n_events": n_events_analysis,
        "n_rows": n_rows_analysis,
        "n_events_fit": n_events_fit,
        "n_rows_fit": len(fit_df),
        "subsampled": subsampled,
        "variance_estimator": VARIANCE_ESTIMATOR if cluster_for_fit else "model_based",
        "cluster_col": cluster_for_fit,
    }
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
    label, subset, cache_name, use_cache, covariates = args
    n_events = int(subset["event_occurred"].sum())
    n_questions = int(subset["unique_id"].nunique())
    if len(subset) < 100 or n_events < 10:
        return (label, None)
    missing = [c for c in covariates if c not in subset.columns]
    if missing:
        print(f"  ⚠ {cache_name}: skipping — missing covariates {missing}")
        return (label, None)
    res = fit_cox_cached(
        subset, cache_name, covariates, use_cache=use_cache, round_to_hours=ROUND_TO_HOURS
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
        "gap_coef": s.loc["treated_post_question", "coef"],
        "gap_p": s.loc["treated_post_question", "p"],
        "n_rows": res.meta.get("n_rows", len(subset)),
        "n_questions": n_questions,
        "n_events": res.meta.get("n_events", n_events),
    }
    # Summed contrast (post- vs pre-question baseline, treated vs control): CSV diagnostic
    # only, comparable across bins in a way `treat_*` alone is not (the waiting-period term
    # varies with RT) — but never rendered into a .tex table (see DID_TERMS comment).
    row.update(_did_fields(res))
    # ISS-24 re-headline: also carry the arrival increment (beta_4) with its own CI so the
    # per-bin table can lead with arrival when HEADLINE_ESTIMAND=="arrival".
    row.update(_arrival_fields(s))
    return (label, row)


def _rt_bin_task(
    model_df: pd.DataFrame,
    bin_idx: int,
    use_cache: bool,
    covariates: list | None = None,
    cache_prefix: str = "ModelA_AllData_RTbin",
):
    lo, hi = RT_BIN_EDGES_HOURS[bin_idx], RT_BIN_EDGES_HOURS[bin_idx + 1]
    label = RT_BIN_LABELS[bin_idx]
    covariates = list(covariates) if covariates is not None else list(COVARIATES_MAIN)
    drop_cols = [
        c for c in ["tenure_bucket", "response_time_bin", "treated_bin2", "treated_bin3"]
        if c in model_df.columns
    ]
    mask = (
        (model_df["hasAnswer"] == 0)
        | (
            (model_df["hasAnswer"] == 1)
            & (model_df["response_time_hours"].notna())
            & (model_df["response_time_hours"] >= lo)
            & (model_df["response_time_hours"] < hi)
        )
    )
    subset = model_df.loc[mask].drop(columns=drop_cols + ["response_time_hours"], errors="ignore").copy()
    cache_name = f"{cache_prefix}_{lo}_{hi}".replace(".", "_")
    return (label, subset, cache_name, use_cache, covariates)


def fit_response_time_bin_models(
    model_df: pd.DataFrame,
    use_cache: bool = True,
    n_jobs: int = None,
    covariates: list | None = None,
    cache_prefix: str = "ModelA_AllData_RTbin",
    results_csv: str = "results_response_time_bins.csv",
    title: str = "Response time bin models (Model A per bin)",
):
    """Fit Cox per response-time bin (pooled controls + treated in that RT window)."""
    if "response_time_hours" not in model_df.columns:
        print("  ⚠ response_time_hours not in model_df; skipping response-time bin models.")
        return pd.DataFrame()
    covariates = list(covariates) if covariates is not None else list(COVARIATES_MAIN)
    n_bins = len(RT_BIN_EDGES_HOURS) - 1
    n_workers = _parallel_workers(n_bins, n_jobs)
    print("\n" + "=" * 60 + f"\n  {title}\n" + "=" * 60)
    print(f"  covariates={covariates}")
    print(f"  (n_jobs={n_workers}, MAX_FIT_WORKERS={MAX_FIT_WORKERS})")
    if n_workers > 1:
        tasks = [
            _rt_bin_task(model_df, i, use_cache, covariates=covariates, cache_prefix=cache_prefix)
            for i in range(n_bins)
        ]
        with Pool(n_workers) as pool:
            results = pool.map(_fit_one_rt_bin, tasks)
    else:
        import gc
        results = []
        for i in range(n_bins):
            task = _rt_bin_task(
                model_df, i, use_cache, covariates=covariates, cache_prefix=cache_prefix
            )
            results.append(_fit_one_rt_bin(task))
            del task
            gc.collect()
    results = [r[1] for r in results if r[1] is not None]
    results.sort(key=lambda x: RT_BIN_LABELS.index(x["bucket"]) if x["bucket"] in RT_BIN_LABELS else 999)
    if not results:
        return pd.DataFrame()
    df_bins = pd.DataFrame(results)
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, results_csv)
    df_bins.to_csv(out_path, index=False)
    print(f"✓ Saved {results_csv} ({len(df_bins)} bins) to {CACHE_DIR}/")
    return df_bins


def fit_response_time_bin_quality_models(
    model_df: pd.DataFrame, use_cache: bool = True, n_jobs: int = None
) -> pd.DataFrame:
    """ISS-06 extension (#27): Model A + answer-quality controls per RT bin."""
    return fit_response_time_bin_models(
        model_df,
        use_cache=use_cache,
        n_jobs=n_jobs,
        covariates=COVARIATES_MAIN_QUALITY,
        cache_prefix="ModelA_AllData_RTbinQuality",
        results_csv="results_response_time_bins_quality.csv",
        title="Response time bin models with answer-quality controls",
    )


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
        "gap_se": s_a.loc["treated_post_question", "se(coef)"],
        "gap_p": s_a.loc["treated_post_question", "p"],
        "phase_post_q_coef": s_a.loc["phase_post_question", "coef"],
        "phase_post_coef": s_a.loc["phase_post", "coef"],
        "hasAnswer_coef": s_a.loc["hasAnswer", "coef"],
    }
    # Summed contrast (post- vs pre-question baseline, treated vs control): CSV diagnostic only.
    main_row.update(_did_fields(res_a))
    # ISS-24 re-headline: arrival increment (beta_4) with its own CI, primary under "arrival".
    main_row.update(_arrival_fields(s_a))

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
        "gap_speed_se": s_b.loc["treated_post_question_response_time_interaction", "se(coef)"],
        "gap_speed_p": s_b.loc["treated_post_question_response_time_interaction", "p"],
    }
    # Covariates are mean-centred, so the RT interactions are zero at their mean. The
    # summed base contrast (treated_post_question + is_treated_active) and the summed RT
    # moderation (`did_speed_*`, the sum of the two RT interaction terms) are CSV-only
    # diagnostics — never rendered into a .tex table.
    speed_row.update(_did_fields(res_b, prefix="did"))
    speed_row.update(_did_fields(res_b, terms=DID_SPEED_TERMS, prefix="did_speed"))
    # ISS-24 re-headline: arrival base increment (beta_4) and the arrival RT moderation
    # (gamma = treated_response_time_interaction) with their own CIs, primary under "arrival".
    speed_row.update(_arrival_fields(s_b))
    speed_row.update(_arrival_fields(s_b, term="treated_response_time_interaction", prefix="arr_speed"))
    return (bucket, main_row, speed_row)


def fit_all_models(model_df: pd.DataFrame, use_cache: bool = True, n_jobs: int = None):
    """Fit Model A and B per tenure bucket."""
    n_workers = _parallel_workers(len(BUCKET_ORDER), n_jobs)
    tasks = [(b, model_df[model_df["tenure_bucket"] == b].copy(), use_cache, ROUND_TO_HOURS) for b in BUCKET_ORDER]
    print(
        f"\nFitting tenure-bucket models in parallel "
        f"(n_jobs={n_workers}, MAX_FIT_WORKERS={MAX_FIT_WORKERS}, {ROUND_TO_HOURS}h windows) …"
    )
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
        "gap_se": s_a.loc["treated_post_question", "se(coef)"],
        "gap_p": s_a.loc["treated_post_question", "p"],
        **_did_fields(res_a),
        # ISS-24 re-headline: arrival increment (beta_4) with its own CI.
        **_arrival_fields(s_a),
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
        "gap_se": s_b.loc["treated_post_question", "se(coef)"],
        "gap_p": s_b.loc["treated_post_question", "p"],
        "speed_coef": s_b.loc["treated_response_time_interaction", "coef"],
        "speed_se": s_b.loc["treated_response_time_interaction", "se(coef)"],
        "speed_p": s_b.loc["treated_response_time_interaction", "p"],
        "gap_speed_coef": s_b.loc["treated_post_question_response_time_interaction", "coef"],
        "gap_speed_se": s_b.loc["treated_post_question_response_time_interaction", "se(coef)"],
        "gap_speed_p": s_b.loc["treated_post_question_response_time_interaction", "p"],
        **_did_fields(res_b),
        **_did_fields(res_b, terms=DID_SPEED_TERMS, prefix="did_speed"),
        # ISS-24 re-headline: arrival base increment (beta_4) and arrival RT moderation
        # (gamma = treated_response_time_interaction), each with its own CI.
        **_arrival_fields(s_b),
        **_arrival_fields(s_b, term="treated_response_time_interaction", prefix="arr_speed"),
    }]
    df_main_all = pd.DataFrame(results_main_all)
    df_speed_all = pd.DataFrame(results_speed_all)
    df_main_all.to_csv(os.path.join(CACHE_DIR, "results_main_all.csv"), index=False)
    df_speed_all.to_csv(os.path.join(CACHE_DIR, "results_speed_all.csv"), index=False)
    print(f"✓ Saved results_main_all.csv, results_speed_all.csv (All data) to {CACHE_DIR}/")
    return df_main_all, df_speed_all
