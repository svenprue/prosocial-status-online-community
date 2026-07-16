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

from cox_config import BUCKET_ORDER, CACHE_DIR, HEADLINE_ESTIMAND  # noqa: E402

MAIN_PATH = os.path.join(CACHE_DIR, "results_main.csv")
MAIN_ALL_PATH = os.path.join(CACHE_DIR, "results_main_all.csv")
DESC_PATH = os.path.join(CACHE_DIR, "descriptives.pkl")
OUT_PATH = os.path.join(CACHE_DIR, "results_absolute_effects.csv")


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


def _baseline_event_rate(row: pd.Series, descriptives: dict) -> float:
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


def _summarize_row(row: pd.Series, source_name: str, descriptives: dict) -> dict:
    baseline = _baseline_event_rate(row, descriptives)

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

    sources = [
        ("results_main.csv", _read_result_csv(main_path)),
        ("results_main_all.csv", _read_result_csv(main_all_path)),
    ]

    records = []
    for source_name, df in sources:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            records.append(_summarize_row(row, source_name, descriptives))

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
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Row} & \textbf{Base risk} & \textbf{HR} & \textbf{ARD} & \textbf{NNT} & \textbf{NNT CI} \\",
        r"\midrule",
    ]

    # HR/ARD/NNT are based on the answer-arrival increment (beta_4, the DiD treatment
    # effect) when HEADLINE_ESTIMAND=="arrival".
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
                    _fmt_num(row.get("nnt"), 1),
                    nnt_ci,
                ]
            )
            + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{@{}l}{\footnotesize HR/ARD/NNT are based on the answer-arrival increment ($\beta_4$), the primary (headline) estimand.} \\",
        r"\end{tabular}",
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


def main() -> pd.DataFrame:
    df = build_absolute_effects()
    if df.empty:
        return df
    write_absolute_effects_csv(df)
    return df


if __name__ == "__main__":
    main()
