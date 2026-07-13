"""
Selection/churn sensitivity summaries for the reciprocity Cox models.

This script quantifies retained questions whose askers have zero observed
post-question help events in the event-history window, then combines that
fraction with existing Cox outputs. It does not refit the Cox model.

Outputs in analysis/model_cache/:
  - results_selection_churn_summary.csv
  - results_selection_bounds.csv

Usage:
    python selection_sensitivity.py [--input ../data/event_history]
"""
import argparse
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cox_config import BUCKET_ORDER, CACHE_DIR, HEADLINE_ESTIMAND
from cox_data import create_tenure_buckets


DEFAULT_ASSUMED_CONTROL_EVENT_FRACS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


def _parse_fracs(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val < 0 or val > 1:
            raise argparse.ArgumentTypeError("assumed fractions must be in [0, 1]")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("provide at least one fraction")
    return vals


def _read_timelines(input_folder: str) -> pd.DataFrame:
    path = os.path.join(input_folder, "study_timelines.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    cols = ["match_id", "question_id", "hasAnswer", "user_tenure_days"]
    return pd.read_parquet(path, columns=cols)


def _read_events(input_folder: str) -> pd.DataFrame:
    path = os.path.join(input_folder, "study_events.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path, columns=["match_id", "question_id", "t_event"])


def compute_churn_summary(input_folder: str) -> pd.DataFrame:
    """Return zero-post-question-help summary by treatment and tenure bucket."""
    timelines = _read_timelines(input_folder)
    events = _read_events(input_folder)

    timelines = create_tenure_buckets(timelines)
    post_counts = (
        events.loc[events["t_event"] >= 0]
        .groupby(["match_id", "question_id"])
        .size()
        .reset_index(name="n_post_help_events")
    )
    merged = timelines.merge(post_counts, on=["match_id", "question_id"], how="left")
    merged["n_post_help_events"] = merged["n_post_help_events"].fillna(0).astype(int)
    merged["zero_post_help"] = merged["n_post_help_events"].eq(0)

    rows = []
    for label, group in [("All", merged)] + [
        (bucket, merged.loc[merged["tenure_bucket"] == bucket]) for bucket in BUCKET_ORDER
    ]:
        if group.empty:
            continue
        for has_answer, g in group.groupby("hasAnswer"):
            n_questions = int(len(g))
            n_zero = int(g["zero_post_help"].sum())
            rows.append({
                "scope": label,
                "hasAnswer": int(has_answer),
                "n_questions": n_questions,
                "n_zero_post_help": n_zero,
                "zero_post_help_share": n_zero / max(1, n_questions),
                "n_post_help_events": int(g["n_post_help_events"].sum()),
                "post_help_event_rate": float(g["n_post_help_events"].sum() / max(1, n_questions)),
            })
    return pd.DataFrame(rows)


def _base_hr_fields(r: pd.Series) -> dict:
    """ISS-24 re-headline: the PRIMARY base HR is the answer-arrival increment (treat_* =
    is_treated_active, beta_4) when HEADLINE_ESTIMAND=="arrival"; the summed DiD (did_*) is
    carried alongside as base_hr_summed/base_ci_*_summed. Under "summed" the primary reverts
    to the summed DiD. Both are always emitted so the bounds table can show both."""
    summed = {
        "base_hr_summed": r.get("did_hr", np.nan),
        "base_ci_lo_summed": r.get("did_ci_lo", np.nan),
        "base_ci_hi_summed": r.get("did_ci_hi", np.nan),
    }
    arrival = {
        "base_hr": r.get("treat_hr", np.nan),
        "base_ci_lo": r.get("treat_ci_lo", np.nan),
        "base_ci_hi": r.get("treat_ci_hi", np.nan),
        "base_se": r.get("treat_se", np.nan),
        "base_p": r.get("treat_p", np.nan),
    }
    summed_primary = {
        "base_hr": r.get("did_hr", np.nan),
        "base_ci_lo": r.get("did_ci_lo", np.nan),
        "base_ci_hi": r.get("did_ci_hi", np.nan),
        "base_se": r.get("did_se", np.nan),
        "base_p": r.get("did_p", np.nan),
    }
    if HEADLINE_ESTIMAND == "arrival" and pd.notna(r.get("treat_hr", np.nan)):
        return {**arrival, **summed}
    if pd.notna(r.get("did_hr", np.nan)):
        return {**summed_primary, **summed}
    return {**arrival, **summed}


def _base_model_rows(cache_dir: str) -> pd.DataFrame:
    rows = []
    main_all_path = os.path.join(cache_dir, "results_main_all.csv")
    if os.path.exists(main_all_path):
        df = pd.read_csv(main_all_path)
        if not df.empty:
            r = df.iloc[0]
            rows.append({"scope": "All", "model": r.get("model", "AllData_Main"), **_base_hr_fields(r)})

    main_path = os.path.join(cache_dir, "results_main.csv")
    if os.path.exists(main_path):
        df = pd.read_csv(main_path)
        for _, r in df.iterrows():
            rows.append({"scope": r.get("bucket"), "model": f"ModelA_{r.get('bucket')}", **_base_hr_fields(r)})
    return pd.DataFrame(rows)


def compute_sensitivity_bounds(
    churn_summary: pd.DataFrame,
    base_models: pd.DataFrame,
    assumed_control_event_fracs: Iterable[float] = DEFAULT_ASSUMED_CONTROL_EVENT_FRACS,
) -> pd.DataFrame:
    """Combine churn counts with a transparent event-rate sensitivity grid."""
    if churn_summary.empty or base_models.empty:
        return pd.DataFrame()

    rows = []
    for _, model in base_models.iterrows():
        scope = model["scope"]
        scoped = churn_summary.loc[churn_summary["scope"] == scope]
        if scoped.empty:
            continue
        treated = scoped.loc[scoped["hasAnswer"] == 1]
        control = scoped.loc[scoped["hasAnswer"] == 0]
        if treated.empty or control.empty:
            continue
        treated = treated.iloc[0]
        control = control.iloc[0]
        treated_rate = treated["post_help_event_rate"]
        control_rate = control["post_help_event_rate"]
        observed_rr_proxy = treated_rate / control_rate if control_rate > 0 else np.nan

        for frac in assumed_control_event_fracs:
            added_control_events = frac * control["n_zero_post_help"]
            adjusted_control_rate = (
                control["n_post_help_events"] + added_control_events
            ) / max(1, control["n_questions"])
            adjusted_rr_proxy = (
                treated_rate / adjusted_control_rate
                if adjusted_control_rate > 0
                else np.nan
            )
            rows.append({
                "scope": scope,
                "model": model["model"],
                "base_hr": model["base_hr"],
                "base_ci_lo": model["base_ci_lo"],
                "base_ci_hi": model["base_ci_hi"],
                "base_se": model["base_se"],
                "base_p": model["base_p"],
                # ISS-24: summed DiD base HR carried as a labeled secondary (upper bound).
                "base_hr_summed": model.get("base_hr_summed", np.nan),
                "base_ci_lo_summed": model.get("base_ci_lo_summed", np.nan),
                "base_ci_hi_summed": model.get("base_ci_hi_summed", np.nan),
                "treated_n": int(treated["n_questions"]),
                "control_n": int(control["n_questions"]),
                "treated_post_help_event_rate": treated_rate,
                "control_post_help_event_rate": control_rate,
                "control_zero_post_help_share": control["zero_post_help_share"],
                "assumed_control_zero_post_event_fraction": frac,
                "added_control_events_proxy": added_control_events,
                "adjusted_control_post_help_event_rate_proxy": adjusted_control_rate,
                "observed_event_rate_rr_proxy": observed_rr_proxy,
                "adjusted_event_rate_rr_proxy": adjusted_rr_proxy,
                "event_rate_rr_attenuation_proxy": (
                    adjusted_rr_proxy / observed_rr_proxy
                    if observed_rr_proxy and not np.isnan(observed_rr_proxy)
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


def main():
    default_input = os.path.normpath(os.path.join(_ANALYSIS_DIR, "..", "data", "event_history"))
    parser = argparse.ArgumentParser(description="Selection/churn sensitivity summaries")
    parser.add_argument("--input", default=default_input, help="Folder with study_timelines.parquet and study_events.parquet")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Directory containing Cox model CSV caches")
    parser.add_argument(
        "--assumed-control-event-fracs",
        type=_parse_fracs,
        default=DEFAULT_ASSUMED_CONTROL_EVENT_FRACS,
        help="Comma-separated sensitivity grid for the fraction of zero-post-help controls assigned one latent help event",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    churn = compute_churn_summary(args.input)
    churn_out = os.path.join(args.cache_dir, "results_selection_churn_summary.csv")
    churn.to_csv(churn_out, index=False)
    print(f"✓ Wrote {churn_out}")

    base_models = _base_model_rows(args.cache_dir)
    if base_models.empty:
        print("⚠ No Cox result CSVs found; wrote churn summary only.")
        return
    bounds = compute_sensitivity_bounds(churn, base_models, args.assumed_control_event_fracs)
    bounds_out = os.path.join(args.cache_dir, "results_selection_bounds.csv")
    bounds.to_csv(bounds_out, index=False)
    print(f"✓ Wrote {bounds_out}")


if __name__ == "__main__":
    main()
