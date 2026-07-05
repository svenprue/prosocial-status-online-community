"""
Matched-pair bootstrap uncertainty for Cox time-varying models.

lifelines 0.27.x exposes ``robust=True`` on CoxTimeVaryingFitter but raises
NotImplementedError, and it does not wire a public ``cluster_col`` argument for
time-varying fits. This script implements the reviewer-requested matched-pair
alternative by resampling whole match_id pairs and refitting Model A.

Outputs:
  analysis/model_cache/results_pair_bootstrap.csv

Usage examples:
  python pair_bootstrap_se.py --scope all --n-bootstrap 200
  python pair_bootstrap_se.py --scope all --sample 10000 --max-pairs 5000 --n-bootstrap 20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import BUCKET_ORDER, CACHE_DIR, COVARIATES_MAIN
from cox_data import load_and_prepare


DROP_FOR_MODEL_A = [
    "tenure_bucket",
    "response_time_hours",
    "response_time_bin",
    "treated_bin2",
    "treated_bin3",
]


def _scope_subset(model_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return model_df.copy()
    if scope not in BUCKET_ORDER:
        raise ValueError(f"Unknown scope '{scope}'. Use 'all' or one of: {', '.join(BUCKET_ORDER)}")
    return model_df.loc[model_df["tenure_bucket"] == scope].copy()


def _model_a_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_FOR_MODEL_A, errors="ignore").copy()


def _bootstrap_sample_by_match(
    df: pd.DataFrame,
    rng: np.random.Generator,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    match_ids = pd.Series(df["match_id"].unique())
    n_pairs = len(match_ids) if max_pairs is None else min(int(max_pairs), len(match_ids))
    sampled = rng.choice(match_ids.to_numpy(), size=n_pairs, replace=True)
    draws = pd.DataFrame({"match_id": sampled, "_boot_draw": np.arange(n_pairs, dtype=np.int64)})
    boot = draws.merge(df, on="match_id", how="left", sort=False)
    draw_prefix = boot["_boot_draw"].astype(str)
    boot["match_id"] = draw_prefix + "_" + boot["match_id"].astype(str)
    boot["unique_id"] = draw_prefix + "_" + boot["unique_id"].astype(str)
    return boot.drop(columns=["_boot_draw"])


def _coef_from_result(result) -> float | None:
    if result is None or "is_treated_active" not in result.summary_df.index:
        return None
    return float(result.summary_df.loc["is_treated_active", "coef"])


def run_pair_bootstrap(
    input_folder: str,
    scope: str,
    n_bootstrap: int,
    sample_size: int | None = None,
    max_pairs: int | None = None,
    seed: int = 42,
    use_cache: bool = True,
) -> pd.DataFrame:
    try:
        from cox_fit import fit_cox_cached
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cox fitting dependencies are unavailable in this environment; "
            "install the project requirements, including lifelines, before running pair bootstrap fits."
        ) from exc

    model_df, _ = load_and_prepare(input_folder, sample_size=sample_size)
    scoped = _model_a_frame(_scope_subset(model_df, scope))
    n_questions = int(scoped["unique_id"].nunique())
    n_events = int(scoped["event_occurred"].sum())
    if len(scoped) < 100 or n_events < 10:
        raise ValueError(f"Scope '{scope}' has too few rows/events for bootstrap fitting")

    base = fit_cox_cached(
        scoped,
        f"ModelA_BootstrapBase_{scope}",
        COVARIATES_MAIN,
        use_cache=use_cache,
        save_cache=use_cache,
    )
    base_coef = _coef_from_result(base)
    if base_coef is None:
        raise RuntimeError(f"Base fit failed for scope '{scope}'")

    rng = np.random.default_rng(seed)
    boot_coefs = []
    for i in range(int(n_bootstrap)):
        boot_df = _bootstrap_sample_by_match(scoped, rng, max_pairs=max_pairs)
        result = fit_cox_cached(
            boot_df,
            f"ModelA_Bootstrap_{scope}_{i + 1}",
            COVARIATES_MAIN,
            use_cache=False,
            save_cache=False,
        )
        coef = _coef_from_result(result)
        if coef is not None and np.isfinite(coef):
            boot_coefs.append(coef)
        print(f"  Bootstrap {i + 1}/{n_bootstrap}: {'ok' if coef is not None else 'skipped'}")

    if not boot_coefs:
        raise RuntimeError(f"No successful bootstrap fits for scope '{scope}'")

    coefs = np.asarray(boot_coefs, dtype=float)
    row = {
        "scope": scope,
        "n_questions": n_questions,
        "n_events": n_events,
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_success": int(len(coefs)),
        "max_pairs_per_replicate": max_pairs if max_pairs is not None else "",
        "base_coef": base_coef,
        "base_hr": float(np.exp(base_coef)),
        "bootstrap_se_coef": float(coefs.std(ddof=1)) if len(coefs) > 1 else np.nan,
        "bootstrap_coef_ci_lo": float(np.percentile(coefs, 2.5)),
        "bootstrap_coef_ci_hi": float(np.percentile(coefs, 97.5)),
        "bootstrap_hr_ci_lo": float(np.exp(np.percentile(coefs, 2.5))),
        "bootstrap_hr_ci_hi": float(np.exp(np.percentile(coefs, 97.5))),
    }
    return pd.DataFrame([row])


def main():
    default_input = os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))
    parser = argparse.ArgumentParser(description="Matched-pair bootstrap SEs for Model A")
    parser.add_argument("--input", default=default_input, help="Folder with study_timelines.parquet and study_events.parquet")
    parser.add_argument("--scope", default="all", help="Scope to fit: 'all' or an exact tenure bucket label")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs before fitting")
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap pairs resampled per bootstrap replicate")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached base model")
    args = parser.parse_args()

    result = run_pair_bootstrap(
        input_folder=args.input,
        scope=args.scope,
        n_bootstrap=args.n_bootstrap,
        sample_size=args.sample,
        max_pairs=args.max_pairs,
        seed=args.seed,
        use_cache=not args.no_cache,
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, "results_pair_bootstrap.csv")
    if os.path.exists(out):
        existing = pd.read_csv(out)
        result = pd.concat([existing, result], ignore_index=True)
        result = result.drop_duplicates(subset=["scope", "n_bootstrap_requested", "max_pairs_per_replicate"], keep="last")
    result.to_csv(out, index=False)
    print(f"✓ Wrote {out}")


if __name__ == "__main__":
    main()
