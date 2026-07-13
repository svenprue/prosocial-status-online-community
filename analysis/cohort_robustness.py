"""Cohort heterogeneity robustness checks for the Cox pipeline.

This script reuses the existing cached Cox fitter and the standard event-history
loader. It tries to recover a question-year / cohort field from the prepared
intervals first, then from the raw timelines as a fallback. If no usable cohort
field is available, it emits a warning and writes the pooled all-data result
only.
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import CACHE_DIR
from cox_data import load_and_prepare
from cox_config import COVARIATES_MAIN


COHORT_CANDIDATE_COLUMNS = (
    "question_year",
    "question_cohort_year",
    "cohort_year",
    "year",
    "created_year",
    "question_created_year",
)


def _default_input_folder() -> str:
    return os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))


def _normalize_year_series(series: pd.Series) -> pd.Series:
    """Return a best-effort integer year series.

    Accepts already-numeric year columns as well as date-like values.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year.astype("Int64")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_non_na = numeric.dropna()
    if len(numeric_non_na) > 0:
        # If values look like years, keep them as years.
        if numeric_non_na.between(1000, 3000).mean() >= 0.8:
            return numeric.round().astype("Int64")

    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.year.astype("Int64")


def _find_cohort_column(df: pd.DataFrame) -> str | None:
    for column in COHORT_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    return None


def _cohort_series_from_model_df(model_df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    column = _find_cohort_column(model_df)
    if column is None:
        return None, None
    years = _normalize_year_series(model_df[column])
    return years, f"prepared intervals column '{column}'"


def _cohort_series_from_timelines(model_df: pd.DataFrame, input_folder: str) -> tuple[pd.Series | None, str | None]:
    timelines_path = os.path.join(input_folder, "study_timelines.parquet")
    if not os.path.exists(timelines_path):
        return None, None

    timelines = pd.read_parquet(timelines_path)
    column = _find_cohort_column(timelines)
    if column is None:
        return None, None

    timelines = timelines[timelines["match_id"].isin(model_df["match_id"].unique())].copy()
    years = _normalize_year_series(timelines[column])
    timelines = timelines.assign(_cohort_year=years)
    years_by_match = timelines.groupby("match_id")["_cohort_year"].nunique(dropna=True)
    ambiguous_match_ids = years_by_match[years_by_match > 1]
    if len(ambiguous_match_ids) > 0:
        print(
            "⚠ Cohort field varies within at least one match_id in the raw timelines; "
            "precise question-level cohort checks require preserving question_year in the interval data."
        )
        return None, None

    cohort_map = (
        timelines.groupby("match_id")["_cohort_year"]
        .first()
        .rename("cohort_year")
    )
    cohort_series = model_df["match_id"].map(cohort_map)
    return cohort_series.astype("Int64"), f"raw timelines column '{column}'"


def _cohort_series_from_matched_questions(model_df: pd.DataFrame, input_folder: str) -> tuple[pd.Series | None, str | None]:
    matched_path = os.path.normpath(os.path.join(input_folder, "..", "input", "matched_questions.parquet"))
    if not os.path.exists(matched_path):
        return None, None

    matched = pd.read_parquet(matched_path, columns=["match_id", "questionId", "year"])
    matched = matched.rename(columns={"questionId": "question_id", "year": "cohort_year"})
    matched["cohort_year"] = _normalize_year_series(matched["cohort_year"])

    keys = ["match_id", "question_id"]
    if "question_id" in model_df.columns:
        keyed = model_df[keys].drop_duplicates().merge(matched[keys + ["cohort_year"]], on=keys, how="left")
        cohort_map = keyed.set_index(keys)["cohort_year"]
        index = pd.MultiIndex.from_frame(model_df[keys])
        return pd.Series(index.map(cohort_map), index=model_df.index, dtype="Int64"), "data/input/matched_questions.parquet column 'year'"

    parsed = model_df[["match_id", "unique_id"]].drop_duplicates().copy()
    parsed["question_id"] = pd.to_numeric(
        parsed["unique_id"].astype(str).str.rsplit("_", n=1).str[-1],
        errors="coerce",
    ).astype("Int64")
    keyed = parsed.merge(matched[keys + ["cohort_year"]], on=keys, how="left")
    cohort_map = keyed.set_index("unique_id")["cohort_year"]
    return model_df["unique_id"].map(cohort_map).astype("Int64"), "data/input/matched_questions.parquet column 'year'"


def _extract_model_row(result, model_name: str, n_questions: int) -> dict | None:
    if result is None:
        return None

    summary = result.summary_df
    if "is_treated_active" not in summary.index:
        print(f"⚠ {model_name}: missing is_treated_active in fitted summary; skipping row.")
        return None

    # Report the SUMMED DiD (treated_post_question + is_treated_active), the treatment
    # effect, rather than the is_treated_active increment over the waiting-period term.
    try:
        from cox_fit import _linear_combo, DID_TERMS
        combo = _linear_combo(result, DID_TERMS)
    except Exception:
        combo = None
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
        "N": int(n_questions),
        "events": int(result.meta.get("n_events", 0)),
        "HR": combo["hr"],
        "CI_low": combo["ci_lo"],
        "CI_high": combo["ci_hi"],
        "SE": combo["se"],
        "p": combo["p"],
        "HR_increment_only": float(np.exp(inc["coef"])),
        # ISS-24 re-headline: arrival increment (beta_4) uncertainty for the arrival-primary
        # column in create_figures.generate_revision_robustness_table.
        "arrival_coef": float(inc["coef"]),
        "arrival_se": float(inc["se(coef)"]),
        "arrival_p": float(inc["p"]),
        "arrival_ci_lo": float(np.exp(inc["coef lower 95%"])),
        "arrival_ci_hi": float(np.exp(inc["coef upper 95%"])),
    }


def _fit_main_model(subset: pd.DataFrame, model_name: str, use_cache: bool) -> dict | None:
    try:
        from cox_fit import fit_cox_cached
    except ModuleNotFoundError:
        print(
            "Cox fitting dependencies are unavailable in this environment; "
            "install the project requirements (including lifelines) to run cohort robustness fits."
        )
        return None

    fit_df = subset.drop(
        columns=["tenure_bucket", "response_time_hours", "response_time_bin", "treated_bin2", "treated_bin3"],
        errors="ignore",
    ).copy()
    n_questions = int(fit_df["unique_id"].nunique())
    n_events = int(fit_df["event_occurred"].sum())
    if len(fit_df) < 100 or n_events < 10:
        print(f"⚠ {model_name}: too few rows/events for a stable fit, skipping.")
        return None

    result = fit_cox_cached(
        fit_df,
        model_name,
        COVARIATES_MAIN,
        use_cache=use_cache,
    )
    if result is None:
        print(f"⚠ {model_name}: fit failed or was skipped.")
        return None
    return _extract_model_row(result, model_name, n_questions)


def run_cohort_robustness(input_folder: str, sample_size: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    from cox_config import PRIMARY_HELP_TYPES

    model_df, _ = load_and_prepare(
        input_folder, sample_size=sample_size, event_help_types=PRIMARY_HELP_TYPES
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []

    pooled = _fit_main_model(model_df, "ModelA_AllData", use_cache=use_cache)
    if pooled is not None:
        rows.append(pooled)

    cohort_series, cohort_source = _cohort_series_from_model_df(model_df)
    if cohort_series is None:
        cohort_series, cohort_source = _cohort_series_from_timelines(model_df, input_folder)
    if cohort_series is None:
        cohort_series, cohort_source = _cohort_series_from_matched_questions(model_df, input_folder)

    if cohort_series is None:
        print(
            "⚠ No usable question-year/cohort field was found in the prepared intervals or raw timelines; "
            "skipping cohort-specific robustness checks."
        )
    else:
        cohort_df = model_df.copy()
        cohort_df["cohort_year"] = cohort_series
        pre_2020 = cohort_df[cohort_df["cohort_year"].notna() & (cohort_df["cohort_year"] < 2020)].copy()
        if len(pre_2020) == 0:
            print(f"⚠ Cohort field found from {cohort_source}, but there are no pre-2020 rows to fit.")
        else:
            print(f"✓ Using cohort field from {cohort_source}; fitting pre-2020 robustness subset.")
            pre_row = _fit_main_model(pre_2020, "ModelA_Pre2020", use_cache=use_cache)
            if pre_row is not None:
                rows.append(pre_row)

    results = pd.DataFrame(rows, columns=[
        "model", "N", "events", "HR", "CI_low", "CI_high", "SE", "p", "HR_increment_only",
        "arrival_coef", "arrival_se", "arrival_p", "arrival_ci_lo", "arrival_ci_hi",
    ])
    out_path = os.path.join(CACHE_DIR, "results_cohort_robustness.csv")
    results.to_csv(out_path, index=False)
    print(f"✓ Saved cohort robustness results to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit cohort heterogeneity robustness checks for Cox models")
    parser.add_argument("--input", default=_default_input_folder(), help="Input data folder")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached models")
    args = parser.parse_args()

    run_cohort_robustness(args.input, sample_size=args.sample, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
