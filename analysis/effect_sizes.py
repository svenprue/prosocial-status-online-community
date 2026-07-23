"""
effect_sizes.py
===============

Compute conservative absolute-effect summaries from cached Cox model outputs.

This script reads cached model result CSVs from analysis/model_cache/, derives a
baseline event-risk proxy, absolute risk difference proxy, and NNT, then writes
the combined machine-readable output to:

    analysis/model_cache/results_absolute_effects.csv

The script is import-safe and can also be used as a module:

    from effect_sizes import build_absolute_effects, generate_absolute_effects_latex_table

All paths are resolved relative to this file so the script works from any cwd.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from cox_config import (  # noqa: E402
    BUCKET_ORDER,
    CACHE_DIR,
    DATA_CACHE_DIR,
    DATA_VERSION,
    HEADLINE_ESTIMAND,
    PRIMARY_HELP_TYPES,
    ROUND_TO_HOURS,
)

MAIN_PATH = os.path.join(CACHE_DIR, "results_main.csv")
MAIN_ALL_PATH = os.path.join(CACHE_DIR, "results_main_all.csv")
DESC_PATH = os.path.join(CACHE_DIR, "descriptives.pkl")
OUT_PATH = os.path.join(CACHE_DIR, "results_absolute_effects.csv")
TABLE_DIR = os.path.join(_SCRIPT_DIR, "output_tables")
TEX_PATH = os.path.join(TABLE_DIR, "absolute_effects.tex")

# Sentinel key for the pooled (all-tenure) control-arm rate.
_POOLED_KEY = "__pooled__"

# The interval-level model frame (same universe the Cox fits use) lets us split
# helping events by arm. Its filename mirrors cox_data.load_and_prepare's cache tag:
#   intervals_full{_<help_types>}_<DATA_VERSION>.parquet
_HELP_TAG = ("_" + "_".join(PRIMARY_HELP_TYPES)) if PRIMARY_HELP_TYPES else ""
INTERVAL_CACHE_PATH = os.path.join(
    DATA_CACHE_DIR, f"intervals_full{_HELP_TAG}_{DATA_VERSION}.parquet"
)


def _fmt_num(value: object, decimals: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "—"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def _latex_escape(value: object) -> str:
    text = str(value)
    return text.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}").replace("<", r"$<$").replace(">", r"$>$")


def _load_descriptives(path: str = DESC_PATH) -> dict:
    if not os.path.exists(path):
        print(f"WARNING: {os.path.abspath(path)} not found; using result CSVs only.")
        return {}
    with open(path, "rb") as fh:
        descriptives = pickle.load(fh)
    print(f"Loaded descriptives from: {os.path.abspath(path)}")
    return descriptives if isinstance(descriptives, dict) else {}


def _read_result_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"WARNING: {os.path.abspath(path)} not found.")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {os.path.basename(path)} from: {os.path.abspath(path)}")
    return df


def _row_label(row: pd.Series, source_name: str) -> str:
    if "bucket" in row and pd.notna(row["bucket"]):
        return str(row["bucket"])
    if "model" in row and pd.notna(row["model"]):
        return str(row["model"])
    return source_name


def _control_arm_event_rates(
    interval_path: str = INTERVAL_CACHE_PATH,
    round_to_hours: float = ROUND_TO_HOURS,
) -> dict:
    """Control-arm helping-event rate per tenure bucket and pooled.

    The base quantity for ARD must be the CONTROL arm (matched, *unanswered*
    questions, hasAnswer == 0), not the treated+control pool. We read the same
    interval-level model frame the Cox fits consume and count helping events per
    control question, replicating the fit-time interval rounding (start/stop
    snapped to a `round_to_hours` grid, intervals that collapse are dropped) so
    the counted universe matches each Table row's ``n_events``.

    Returns {tenure_bucket: rate, _POOLED_KEY: rate}. Rate = control-arm helping
    events / control-arm questions = a mean *count* of recurrent events per
    observation window (NOT a probability). Returns {} if the frame or the duckdb
    dependency is unavailable, so callers can fall back to the pooled proxy.
    """
    if not os.path.exists(interval_path):
        print(
            f"WARNING: interval frame {os.path.abspath(interval_path)} not found; "
            "cannot compute control-arm rates — falling back to pooled Events/N proxy."
        )
        return {}
    try:
        import duckdb  # noqa: E402
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"WARNING: duckdb unavailable ({exc}); falling back to pooled Events/N proxy.")
        return {}

    rt = float(round_to_hours) if round_to_hours and round_to_hours > 0 else 1.0
    # Snap start/stop to the fit-time grid and drop intervals that collapse, exactly
    # as fit_cox_cached does before summing event_occurred.
    keep = f"round(start / {rt}) * {rt} < round(stop / {rt}) * {rt}"
    con = duckdb.connect()
    try:
        per_bucket = con.execute(
            f"""
            SELECT tenure_bucket AS bucket,
                   SUM(CASE WHEN hasAnswer = 0 THEN event_occurred ELSE 0 END) AS ev_ctrl,
                   COUNT(DISTINCT CASE WHEN hasAnswer = 0
                                       THEN match_id || '_' || question_id END) AS q_ctrl
            FROM (SELECT * FROM '{interval_path}')
            WHERE {keep}
            GROUP BY tenure_bucket
            """
        ).df()
        pooled = con.execute(
            f"""
            SELECT SUM(CASE WHEN hasAnswer = 0 THEN event_occurred ELSE 0 END) AS ev_ctrl,
                   COUNT(DISTINCT CASE WHEN hasAnswer = 0
                                       THEN match_id || '_' || question_id END) AS q_ctrl
            FROM (SELECT * FROM '{interval_path}')
            WHERE {keep}
            """
        ).df()
    finally:
        con.close()

    rates: dict = {}
    for _, r in per_bucket.iterrows():
        q = float(r["q_ctrl"])
        if q > 0:
            rates[str(r["bucket"])] = float(r["ev_ctrl"]) / q
    q_pool = float(pooled["q_ctrl"].iloc[0])
    if q_pool > 0:
        rates[_POOLED_KEY] = float(pooled["ev_ctrl"].iloc[0]) / q_pool
    print(
        f"Computed control-arm helping-event rates from {os.path.basename(interval_path)}: "
        f"pooled={rates.get(_POOLED_KEY)}, {len(per_bucket)} tenure buckets."
    )
    return rates


def _baseline_event_rate(
    row: pd.Series, descriptives: dict, control_rates: Optional[dict] = None
) -> float:
    """Base rate for ARD = the CONTROL-ARM helping-event rate for this row's universe.

    Falls back to the pooled (treated+control) Events/N proxy only if control-arm
    rates could not be computed (interval frame / duckdb absent).
    """
    control_rates = control_rates or {}
    bucket = row.get("bucket")
    if pd.notna(bucket) and str(bucket) in control_rates:
        return control_rates[str(bucket)]
    if _POOLED_KEY in control_rates:
        return control_rates[_POOLED_KEY]

    # Fallback: pooled Events/N (both arms) — flagged upstream via the warnings above.
    n_questions = row.get("n_questions")
    if pd.isna(n_questions) or float(n_questions) == 0.0:
        n_questions = descriptives.get("n_questions")
    n_events = row.get("n_events")
    if pd.isna(n_questions) or pd.isna(n_events) or float(n_questions) == 0.0:
        return np.nan
    return float(n_events) / float(n_questions)


def _ard_nnt(baseline: float, hr, hr_ci_lo, hr_ci_hi) -> dict:
    """Absolute-risk-difference proxy and NNT (with CI) for a hazard ratio."""
    ard = baseline * (float(hr) - 1.0) if pd.notna(baseline) and pd.notna(hr) else np.nan
    ard_ci_lo = baseline * (float(hr_ci_lo) - 1.0) if pd.notna(baseline) and pd.notna(hr_ci_lo) else np.nan
    ard_ci_hi = baseline * (float(hr_ci_hi) - 1.0) if pd.notna(baseline) and pd.notna(hr_ci_hi) else np.nan
    nnt = 1.0 / ard if pd.notna(ard) and ard > 0 else np.nan
    if pd.notna(ard_ci_lo) and pd.notna(ard_ci_hi) and ard_ci_lo > 0 and ard_ci_hi > 0:
        nnt_ci_lo = 1.0 / ard_ci_hi
        nnt_ci_hi = 1.0 / ard_ci_lo
    else:
        nnt_ci_lo = np.nan
        nnt_ci_hi = np.nan
    return {
        "ard": ard, "ard_ci_lo": ard_ci_lo, "ard_ci_hi": ard_ci_hi,
        "nnt": nnt, "nnt_ci_lo": nnt_ci_lo, "nnt_ci_hi": nnt_ci_hi,
    }


def _summarize_row(
    row: pd.Series,
    source_name: str,
    descriptives: dict,
    control_rates: Optional[dict] = None,
) -> dict:
    baseline = _baseline_event_rate(row, descriptives, control_rates)

    # The PRIMARY absolute effect (HR/ARD/NNT) is based on the answer-arrival increment
    # (beta_4 = is_treated_active), the DiD treatment effect, when HEADLINE_ESTIMAND==
    # "arrival". The summed contrast (exp(beta_2+beta_4)) below is a CSV-only diagnostic —
    # per the 2026-07-14 estimand decision it is never rendered into the .tex table.
    summed_hr = (
        float(row["did_hr"]) if pd.notna(row.get("did_hr"))
        else (float(np.exp(float(row["did_coef"]))) if pd.notna(row.get("did_coef")) else np.nan)
    )
    summed_ci_lo, summed_ci_hi = row.get("did_ci_lo"), row.get("did_ci_hi")
    summed_coef = row.get("did_coef")

    arrival_hr = row.get("treat_hr")
    if pd.isna(arrival_hr) and pd.notna(row.get("treat_coef")):
        arrival_hr = float(np.exp(float(row["treat_coef"])))
    arrival_ci_lo = row.get("arrival_ci_lo", row.get("treat_ci_lo"))
    arrival_ci_hi = row.get("arrival_ci_hi", row.get("treat_ci_hi"))
    arrival_coef = row.get("treat_coef")

    use_arrival = HEADLINE_ESTIMAND == "arrival" and pd.notna(arrival_hr)
    if use_arrival:
        hr, hr_ci_lo, hr_ci_hi = arrival_hr, arrival_ci_lo, arrival_ci_hi
        coef_used, hr_source = arrival_coef, "is_treated_active (arrival)"
    elif pd.notna(summed_hr):
        hr, hr_ci_lo, hr_ci_hi = summed_hr, summed_ci_lo, summed_ci_hi
        coef_used, hr_source = summed_coef, "did_sum"
    else:
        hr, hr_ci_lo, hr_ci_hi = arrival_hr, arrival_ci_lo, arrival_ci_hi
        coef_used, hr_source = arrival_coef, "is_treated_active"

    primary = _ard_nnt(baseline, hr, hr_ci_lo, hr_ci_hi)
    summed = _ard_nnt(baseline, summed_hr, summed_ci_lo, summed_ci_hi)

    return {
        "source_file": source_name,
        "row_label": _row_label(row, source_name),
        "model": row.get("model", row.get("bucket", source_name)),
        "bucket": row.get("bucket"),
        "n_rows": row.get("n_rows"),
        "n_questions": row.get("n_questions", descriptives.get("n_questions", np.nan)),
        "n_events": row.get("n_events"),
        "baseline_event_rate_proxy": baseline,
        "hr_source": hr_source,
        "treat_coef": coef_used,
        "treat_hr": hr,
        "treat_ci_lo": hr_ci_lo,
        "treat_ci_hi": hr_ci_hi,
        "absolute_risk_difference_proxy": primary["ard"],
        "ard_ci_lo": primary["ard_ci_lo"],
        "ard_ci_hi": primary["ard_ci_hi"],
        # Reviewer R1's framing: additional helping events per 1,000 answered questions.
        "additional_events_per_1000": (
            1000.0 * primary["ard"] if pd.notna(primary["ard"]) else np.nan
        ),
        "nnt": primary["nnt"],
        "nnt_ci_lo": primary["nnt_ci_lo"],
        "nnt_ci_hi": primary["nnt_ci_hi"],
        # CSV-only diagnostic (summed contrast) — never rendered into the .tex table.
        "summed_hr": summed_hr,
        "summed_ci_lo": summed_ci_lo,
        "summed_ci_hi": summed_ci_hi,
        "summed_ard": summed["ard"],
        "summed_nnt": summed["nnt"],
    }


def build_absolute_effects(
    main_path: str = MAIN_PATH,
    main_all_path: str = MAIN_ALL_PATH,
    descriptives_path: str = DESC_PATH,
) -> pd.DataFrame:
    """
    Build a combined absolute-effects table from the cached Cox result CSVs.

    Returns an empty DataFrame if no usable cache files are present.
    """
    descriptives = _load_descriptives(descriptives_path)
    control_rates = _control_arm_event_rates()

    sources = [
        ("results_main.csv", _read_result_csv(main_path)),
        ("results_main_all.csv", _read_result_csv(main_all_path)),
    ]

    records = []
    for source_name, df in sources:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            records.append(_summarize_row(row, source_name, descriptives, control_rates))

    if not records:
        print(
            "ERROR: No usable cached result CSVs found. "
            "Expected analysis/model_cache/results_main.csv and/or results_main_all.csv."
        )
        return pd.DataFrame()

    out = pd.DataFrame.from_records(records)
    out["source_order"] = out["source_file"].map({"results_main.csv": 0, "results_main_all.csv": 1}).fillna(99)
    out["bucket_order"] = pd.Categorical(out["bucket"], categories=BUCKET_ORDER, ordered=True)
    out = out.sort_values(["source_order", "bucket_order", "row_label"], kind="stable").drop(columns=["source_order", "bucket_order"])

    # Keep the CSV machine-readable; round only on write if desired by callers.
    return out.reset_index(drop=True)


def generate_absolute_effects_latex_table(df: pd.DataFrame, caption: str = "Absolute Effect Sizes and NNT") -> str:
    """
    Return a compact LaTeX table summarizing the derived absolute-effect metrics.
    """
    if df is None or df.empty:
        return ""

    lines = [
        r"\begin{table}[H]",
        rf"\caption{{{caption}}}",
        r"\label{tab:absolute_effects}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Row} & \textbf{Control rate} & \textbf{HR} & \textbf{ARD} & "
        r"\textbf{Events/1k} & \textbf{NNT} & \textbf{NNT CI} \\",
        r"\midrule",
    ]

    # HR/ARD/NNT are based on the answer-arrival increment (beta_4, the DiD treatment
    # effect) when HEADLINE_ESTIMAND=="arrival". The base is now the CONTROL-arm helping-
    # event rate (unanswered matched questions), not the treated+control pool.
    for _, row in df.iterrows():
        nnt_ci = "—"
        if pd.notna(row.get("nnt_ci_lo")) and pd.notna(row.get("nnt_ci_hi")):
            nnt_ci = f"{_fmt_num(row['nnt_ci_lo'], 1)}--{_fmt_num(row['nnt_ci_hi'], 1)}"
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.get("row_label", "—")),
                    _fmt_num(row.get("baseline_event_rate_proxy"), 4),
                    _fmt_num(row.get("treat_hr"), 3),
                    _fmt_num(row.get("absolute_risk_difference_proxy"), 5),
                    _fmt_num(row.get("additional_events_per_1000"), 2),
                    _fmt_num(row.get("nnt"), 1),
                    nnt_ci,
                ]
            )
            + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.35em}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\raggedright",
        r"\textbf{Notes:} HR/ARD/NNT are based on the answer-arrival increment "
        r"($\beta_4$), the primary (headline) estimand. \emph{Control rate} is the mean "
        r"number of helping events per observation window among the matched \emph{control} "
        r"(unanswered) questions in each row's universe --- i.e.\ a recurrent-event "
        r"\emph{rate}, not a probability or risk. The absolute effect is obtained as "
        r"$\mathrm{ARD} = \text{control rate}\times(\mathrm{HR}-1)$, which treats the "
        r"arrival hazard ratio as a rate (risk) ratio; with a base rate near $0.13$ and a "
        r"recurrent outcome this is a first-order approximation rather than an exact risk "
        r"difference. \emph{Events/1k} $= 1000\times\mathrm{ARD}$ is the implied number of "
        r"additional helping events per $1{,}000$ answered questions. Because the outcome "
        r"is a recurrent-event rate (not a binary risk), $\mathrm{NNT}=1/\mathrm{ARD}$ is "
        r"an approximate ``answers needed per additional helping event'' rather than a "
        r"true number-needed-to-treat.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_absolute_effects_csv(df: pd.DataFrame, out_path: str = OUT_PATH) -> str:
    if df is None or df.empty:
        return ""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {os.path.abspath(out_path)}")
    return out_path


def write_absolute_effects_latex(df: pd.DataFrame, out_path: str = TEX_PATH) -> str:
    if df is None or df.empty:
        return ""
    tex = generate_absolute_effects_latex_table(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(tex)
    print(f"Wrote {os.path.abspath(out_path)}")
    return out_path


def main() -> pd.DataFrame:
    df = build_absolute_effects()
    if df.empty:
        return df
    write_absolute_effects_csv(df)
    write_absolute_effects_latex(df)
    return df


if __name__ == "__main__":
    main()
