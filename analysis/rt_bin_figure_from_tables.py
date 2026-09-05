"""
rt_bin_figure_from_tables.py
============================
Rebuilds fig:interaction_effect (treatment effect by response-time bin, baseline vs
answer-quality-controlled) without the cluster model cache.

create_figures.py is the authoritative producer of this figure: it reads
model_cache/results_response_time_bins{,_quality}.csv, which are written by
fit_cox_models.py / revision_robustness.py on the cluster and are not kept in the
repository. This script exists so the figure can be regenerated from a clean checkout.
It prefers those CSVs when present and otherwise parses the committed LaTeX tables
output_tables/response_time_bins.tex and response_time_bins_quality.tex, which carry
the same estimates to three decimals. Either way the plotting call is the one in
create_figures.py, so the two routes cannot drift apart.

Usage (from the analysis/ directory):
    python rt_bin_figure_from_tables.py
"""

import os
import re
import sys

import pandas as pd

from cox_config import CACHE_DIR, RT_BIN_LABELS
from create_figures import _ensure_dirs, generate_interaction_effect_figure

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(_SCRIPT_DIR, "output_tables")

# Significance markers as the table generator emits them, mapped to a representative
# p-value. The figure only uses p to choose the star annotation, so a representative
# value inside each band reproduces the annotation exactly.
_STAR_P = [("***", 0.0005), ("**", 0.005), ("*", 0.02), (r"\textdagger", 0.07)]

_NUM = r"([0-9]*\.?[0-9]+)"


def _p_from_stars(cell: str) -> float:
    for marker, p in _STAR_P:
        if marker in cell:
            return p
    return 0.5


def _delatex_bin(label: str) -> str:
    return label.replace(r"$>$", ">").strip()


def _parse_rows(path: str, n_hr_cols: int):
    """
    Parse the HR/CI column families out of a response-time-bin table body.

    n_hr_cols=1 -> response_time_bins.tex   (HR, CI, N, Events)
    n_hr_cols=2 -> response_time_bins_quality.tex (Baseline HR, CI, Quality HR, CI, N)
    """
    with open(path) as fh:
        body = fh.read()

    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or "&" not in line:
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        label = _delatex_bin(cells[0])
        if label not in RT_BIN_LABELS:
            continue

        families = []
        for k in range(n_hr_cols):
            hr_cell, ci_cell = cells[1 + 2 * k], cells[2 + 2 * k]
            hr = re.match(_NUM, hr_cell)
            ci = re.search(rf"\[\s*{_NUM}\s*,\s*{_NUM}\s*\]", ci_cell)
            if not (hr and ci):
                break
            families.append(
                {
                    "treat_hr": float(hr.group(1)),
                    "treat_ci_lo": float(ci.group(1)),
                    "treat_ci_hi": float(ci.group(2)),
                    "treat_p": _p_from_stars(hr_cell),
                }
            )
        if len(families) != n_hr_cols:
            continue
        rows.append((label, families))

    if len(rows) != len(RT_BIN_LABELS):
        raise SystemExit(
            f"{os.path.basename(path)}: parsed {len(rows)} of {len(RT_BIN_LABELS)} "
            "response-time bins; table format may have changed."
        )
    return rows


def _frames_from_tables():
    base_rows = _parse_rows(os.path.join(TABLE_DIR, "response_time_bins.tex"), 1)
    qual_rows = _parse_rows(os.path.join(TABLE_DIR, "response_time_bins_quality.tex"), 2)

    base = pd.DataFrame([dict(bucket=b, **f[0]) for b, f in base_rows])
    qual = pd.DataFrame([dict(bucket=b, **f[1]) for b, f in qual_rows])

    # The baseline family appears in both tables; make sure they agree before plotting.
    qual_base = pd.DataFrame([dict(bucket=b, **f[0]) for b, f in qual_rows])
    merged = base.merge(qual_base, on="bucket", suffixes=("", "_x"))
    drift = merged[(merged["treat_hr"] - merged["treat_hr_x"]).abs() > 1e-9]
    if not drift.empty:
        raise SystemExit(
            "Baseline HRs disagree between response_time_bins.tex and "
            f"response_time_bins_quality.tex for bins: {list(drift['bucket'])}"
        )
    return base, qual


def main():
    _ensure_dirs()

    base_csv = os.path.join(CACHE_DIR, "results_response_time_bins.csv")
    qual_csv = os.path.join(CACHE_DIR, "results_response_time_bins_quality.csv")

    if os.path.exists(base_csv) and os.path.exists(qual_csv):
        print("Source: model_cache CSVs")
        df_base = pd.read_csv(base_csv)
        df_qual = pd.read_csv(qual_csv)
    else:
        print("Source: committed LaTeX tables (model_cache not present)")
        df_base, df_qual = _frames_from_tables()

    generate_interaction_effect_figure(
        df_rt_bins=df_base,
        df_speed_all=None,
        df_rt_bins_quality=df_qual,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
