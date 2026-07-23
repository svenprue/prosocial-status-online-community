"""
create_figures.py
==================
Loads cached model results and descriptive statistics produced by fit_cox_models.py
(from model_cache/*.csv and model_cache/descriptives.pkl), then generates LaTeX tables
and figures for the paper's Results section.

Outputs (written to output_tables/ and output_figures/):
  - desc_stats.tex              Descriptive statistics (tab:desc_stats)
  - regression_all.tex          Pooled Cox regressions for experienced users (tab:pooled_experienced)
  - main_results.tex            Main effect by tenure bucket (tab:main_results)
  - speed_results.tex           Response time moderation by tenure bucket (tab:speed_results)
  - strength_rec.*              Strength of reciprocity across experience (fig:strength_rec)
  - speed_moderation.*          Response time moderation by tenure (appendix)
  - interaction_effect.*       Treatment effect by response time bin, pooled (fig:interaction_effect)

Usage:
    python create_figures.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cox_config import RT_BIN_LABELS, HEADLINE_ESTIMAND
from effect_sizes import (
    build_absolute_effects,
    generate_absolute_effects_latex_table,
    write_absolute_effects_csv,
)

# =====================================================================
# Configuration (paths relative to this script so running from any cwd works)
# =====================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_SCRIPT_DIR, "model_cache")
TABLE_DIR = os.path.join(_SCRIPT_DIR, "output_tables")
FIGURE_DIR = os.path.join(_SCRIPT_DIR, "output_figures")

BUCKET_ORDER = [
    "< 1 Week", "1 Week - 1 Month", "1 - 6 Months",
    "6 - 12 Months", "1 - 3 Years", "3 - 6 Years", "> 6 Years",
]

# Short labels for figures
BUCKET_SHORT = [
    "<1W", "1W–1M", "1–6M", "6–12M", "1–3Y", "3–6Y", ">6Y"
]


def _ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "\\textdagger"
    return ""


def _fmt_coef(val: float, p: float, decimals: int = 4) -> str:
    """Format a coefficient with significance stars."""
    return f"{val:.{decimals}f}{_sig_stars(p)}"


def _fmt_se(val: float, decimals: int = 4) -> str:
    return f"({val:.{decimals}f})"


# Compact column headers for wide tenure-stratified regression tables
TENURE_TABLE_HEADERS = [
    r"\shortstack[c]{$<$ 1\\Week}",
    r"\shortstack[c]{1 Wk--\\1 Mo}",
    r"\shortstack[c]{1--6\\Mo}",
    r"\shortstack[c]{6--12\\Mo}",
    r"\shortstack[c]{1--3\\Yr}",
    r"\shortstack[c]{3--6\\Yr}",
    r"\shortstack[c]{$>$ 6\\Yrs}",
]


def _tenure_table_col_spec(n_buckets: int) -> str:
    # tabularx X columns stretch to fill \linewidth without magnifying the font
    # (a plain `c` column inside \resizebox got scaled up instead).
    return rf"@{{}}>{{\raggedright\arraybackslash}}p{{2.55cm}}*{{{n_buckets}}}{{>{{\centering\arraybackslash}}X}}@{{}}"


def _tenure_table_preamble(caption: str, label: str) -> list[str]:
    """Open a page-width tenure table: tabularx fills \\linewidth at a fixed font
    size, so columns stretch via padding instead of the type being scaled up.
    Footnotes are emitted outside the tabularx by the postamble."""
    return [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
    ]


def _tenure_table_postamble(notes: list[str] | None = None) -> list[str]:
    """Close tabularx, then emit wrapping footnotes at page width."""
    lines = [
        r"\end{tabularx}",
    ]
    if notes:
        lines += _table_notes_block(notes)
    lines.append(r"\end{table}")
    return lines


def _latex_bucket(label: str) -> str:
    """Escape < and > for LaTeX math mode in table labels."""
    return label.replace("<", r"$<$").replace(">", r"$>$")


def _table_notes_block(notes: list[str]) -> list[str]:
    """Footnotes as a wrapping minipage (avoids clipped \multicolumn{@{}l} notes).

    The first note carries a bold ``Notes:'' lead-in (commit e20f0ff note-formatting
    standardization)."""
    lines = [
        r"\vspace{0.35em}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\raggedright",
    ]
    for i, note in enumerate(notes):
        sep = r"\\" if i < len(notes) - 1 else ""
        prefix = r"\textbf{Notes:} " if i == 0 else ""
        lines.append(prefix + note + sep)
    lines.append(r"\end{minipage}")
    return lines


def _standard_error_note_text(bootstrap_available: bool = False) -> str:
    if bootstrap_available:
        return (
            r"Cox-table standard errors are model-based; matched-pair bootstrap "
            r"95\% CIs for the pooled arrival increment ($\beta_4$) are in "
            r"Table~\ref{tab:pair_bootstrap}."
        )
    return (
        r"Cox-table standard errors are model-based; matched-pair bootstrap "
        r"uncertainty is reported separately when generated."
    )


def _estimand_note_text() -> str:
    """Standard footnote sentence (2026-07-14 estimand decision): replaces every
    prior summed-DiD / upper-bound sentence across the output tables."""
    return (
        r"$\beta_4$ (the answer-arrival increment) is the difference-in-differences "
        r"treatment effect; $\beta_2$ (waiting period) is reported as a parallel-trends "
        r"diagnostic and is not added to the effect."
    )


def _events_note_text() -> str:
    return (
        r"Events = helping events in the full analysis sample (after time rounding). "
        r"When interval rows exceed the fit cap, estimation uses a matched-pair "
        r"subsample, but Events still refer to the full sample."
    )


def _conventions_note_text() -> str:
    """Consolidated cross-reference used on every table EXCEPT tab:main_results, which
    defines the estimand ($\\beta_4$/$\\beta_2$), standard-error, and Events conventions
    once (commit e20f0ff). Replaces the three repeated boilerplate note lines."""
    return (
        r"Estimand ($\beta_4$/$\beta_2$), standard-error, and Events conventions "
        r"follow Table~\ref{tab:main_results}."
    )


def _sig_note_text() -> str:
    return r"$^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$; $^{\dagger}p<0.1$"


def _standard_error_note(n_cols: int, bootstrap_available: bool = False) -> str:
    """Legacy in-tabular note; prefer `_table_notes_block` so footnotes wrap."""
    return (
        rf"\multicolumn{{{n_cols}}}{{@{{}}p{{\linewidth}}@{{}}}}{{\footnotesize "
        + _standard_error_note_text(bootstrap_available)
        + r"}} \\"
    )


def _events_note(n_cols: int) -> str:
    """Legacy in-tabular note; prefer `_table_notes_block` so footnotes wrap."""
    return (
        rf"\multicolumn{{{n_cols}}}{{@{{}}p{{\linewidth}}@{{}}}}{{\footnotesize "
        + _events_note_text()
        + r"}} \\"
    )


def _reconcile_pooled_events(df_main: pd.DataFrame, *pooled_dfs: pd.DataFrame) -> bool:
    """Patch pooled n_events when CSVs still store fit-subsample counts.

    Tenure buckets partition the analysis sample, so sum(bucket Events) is the
    analysis-sample event count. Pooled fits that hit MAX_FIT_ROWS previously
    wrote the subsample count (~6× too small). Mutates pooled frames in place.
    Returns True if any frame was patched.
    """
    if df_main is None or df_main.empty or "n_events" not in df_main.columns:
        return False
    bucket_events = int(df_main["n_events"].sum())
    if bucket_events <= 0:
        return False
    patched = False
    for df in pooled_dfs:
        if df is None or df.empty or "n_events" not in df.columns:
            continue
        pooled = int(df["n_events"].iloc[0])
        # Clear discrepancy only when pooled looks like a MAX_FIT_ROWS fit-subsample
        # count (≪ bucket sum). Never overwrite a larger analysis-sample count with
        # the tenure-bucket sum (buckets can omit unbucketed rows).
        if pooled > 0 and bucket_events >= pooled * 2:
            print(
                f"  ⚠ Reconciling pooled Events {pooled:,} → {bucket_events:,} "
                "(analysis-sample; prior CSV stored fit-subsample count)"
            )
            df["n_events"] = bucket_events
            patched = True
        elif pooled > 0 and pooled > bucket_events:
            print(
                f"  Keeping pooled Events {pooled:,} (> tenure-bucket sum "
                f"{bucket_events:,}); buckets may omit unbucketed rows."
            )
    return patched


def _persist_reconciled_pooled_events(
    df_main_all: pd.DataFrame,
    df_speed_all: pd.DataFrame,
) -> None:
    """Write reconciled pooled event counts back to model_cache CSVs."""
    for name, df in (
        ("results_main_all.csv", df_main_all),
        ("results_speed_all.csv", df_speed_all),
    ):
        if df is None or df.empty:
            continue
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path):
            df.to_csv(path, index=False)
            print(f"  ✓ Updated {path} with reconciled n_events")


# =====================================================================
# Table 1: Descriptive Statistics
# =====================================================================
# N = number of unique questions (question_id) throughout all tables.
# =====================================================================

def generate_desc_stats_table(desc: dict, df_main: pd.DataFrame = None) -> str:
    """Generate LaTeX for the descriptive statistics table.
    If df_main is provided, N (total and by bucket) is taken from it (same source as main_results) so the two tables match.
    """
    if df_main is not None and not df_main.empty and "n_questions" in df_main.columns:
        n_q = int(df_main["n_questions"].sum())
    else:
        n_q = desc.get("n_questions", "—")
    n_u = desc.get("n_unique_users", "—")
    pct_ans = desc.get("pct_has_answer", "—")
    he_mean = desc.get("help_events_mean", "—")
    he_std = desc.get("help_events_std", "—")
    he_med = desc.get("help_events_median", "—")
    t_mean = desc.get("tenure_mean", "—")
    t_std = desc.get("tenure_std", "—")
    t_med = desc.get("tenure_median", "—")
    rt_mean = desc.get("response_time_mean_hours", "—")
    rt_std = desc.get("response_time_std_hours", "—")
    rt_med = desc.get("response_time_median_hours", "—")

    def _f(v, d=2):
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "—"
        if isinstance(v, (int, float, np.floating, np.integer)):
            if abs(v) > 100:
                return f"{v:,.0f}"
            return f"{v:.{d}f}"
        return str(v)

    lines = [
        r"\begin{table}[H]",
        r"\caption{Descriptive Statistics}",
        r"\label{tab:desc_stats}",
        r"\centering",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Variable} & \textbf{Mean} & \textbf{SD} & \textbf{Median} & \textbf{N} \\",
        r"\midrule",
        rf"Questions (observations) & & & & {_f(n_q)} \\",
        rf"Unique users & & & & {_f(n_u)} \\",
        rf"\% questions with answer & {_f(pct_ans)}\% & & & \\",
        r"\midrule",
        rf"Help events per window (answers to others) & {_f(he_mean)} & {_f(he_std)} & {_f(he_med)} & \\",
        rf"User tenure (days) & {_f(t_mean)} & {_f(t_std)} & {_f(t_med)} & \\",
        rf"Response time (hours, treated) & {_f(rt_mean)} & {_f(rt_std)} & {_f(rt_med)} & \\",
        r"\midrule",
    ]

    # Response time by tenure bucket
    rt_by_bucket = desc.get("response_time_by_tenure_bucket", {})
    if rt_by_bucket:
        lines.append(r"\multicolumn{5}{@{}l}{\textit{Response time (hours, treated) by Tenure Bucket}} \\")
        for b in BUCKET_ORDER:
            stats = rt_by_bucket.get(b, {})
            mean = stats.get("mean")
            std = stats.get("std")
            med = stats.get("median")
            n = stats.get("n", 0)
            lines.append(rf"\hspace{{1em}} {_latex_bucket(b)} & {_f(mean)} & {_f(std)} & {_f(med)} & {_f(n)} \\")
        lines.append(r"\midrule")

    # Tenure bucket breakdown (N = unique questions per bucket); use df_main so it matches main_results
    if df_main is not None and not df_main.empty and "n_questions" in df_main.columns:
        df_b = df_main.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
        lines.append(r"\multicolumn{5}{@{}l}{\textit{N (unique questions) by Tenure Bucket}} \\")
        for _, r in df_b.iterrows():
            b = r["bucket"]
            ct = int(r["n_questions"]) if pd.notna(r.get("n_questions")) else 0
            lines.append(rf"\hspace{{1em}} {_latex_bucket(b)} & & & & {_f(ct)} \\")
    else:
        bucket_counts = desc.get("tenure_bucket_counts", {})
        if bucket_counts:
            lines.append(r"\multicolumn{5}{@{}l}{\textit{N (unique questions) by Tenure Bucket}} \\")
            for b in BUCKET_ORDER:
                ct = bucket_counts.get(b, 0)
                lines.append(rf"\hspace{{1em}} {_latex_bucket(b)} & & & & {_f(ct)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =====================================================================
# Table: Pooled regressions (regression_all.tex) — Main, Main+Speed
# =====================================================================

def generate_regression_all_table(
    df_main_all: pd.DataFrame,
    df_speed_all: pd.DataFrame = None,
    bootstrap_available: bool = False,
    df_pair_bootstrap: pd.DataFrame = None,
) -> str:
    """
    Generate LaTeX table for pooled Cox regressions (all experience levels).
    Columns: Main; Main + Speed (linear RT interaction).
    """
    if df_main_all.empty or "treat_coef" not in df_main_all.columns:
        return ""
    r = df_main_all.iloc[0]
    has_speed = (
        df_speed_all is not None
        and not df_speed_all.empty
        and "speed_coef" in df_speed_all.columns
    )
    s = df_speed_all.iloc[0] if has_speed else None

    n_cols = 1 + int(has_speed)
    col_spec = "@{}l" + "c" * n_cols + "@{}"
    header_cells = [r"\textbf{Main}"]
    if has_speed:
        header_cells.append(r"\textbf{Main + Speed}")
    header = " & ".join(header_cells) + r" \\"

    lines = [
        r"\begin{table}[H]",
        r"\caption{Pooled Cox Regressions (All Experience Levels)}",
        r"\label{tab:pooled_experienced}",
        r"\centering",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & " + header,
        r"\midrule",
    ]

    # --- Row helpers for the two-column (Main / Main+Speed) layout ------------
    def _cell_coef(rr, kc, kp):
        return _fmt_coef(rr[kc], rr.get(kp, np.nan)) if pd.notna(rr.get(kc)) else "—"

    def _coef_row(label, kc, kp, kse=None):
        cells = [_cell_coef(r, kc, kp)] + ([_cell_coef(s, kc, kp)] if has_speed else [])
        lines.append(rf"{label} & " + " & ".join(cells) + r" \\")
        if kse is not None:
            se_cells = [_fmt_se(r[kse]) if pd.notna(r.get(kse)) else ""]
            if has_speed:
                se_cells.append(_fmt_se(s[kse]) if pd.notna(s.get(kse)) else "")
            lines.append(r" & " + " & ".join(se_cells) + r" \\")

    def _ci_cell(rr, klo, khi):
        if pd.notna(rr.get(klo)) and pd.notna(rr.get(khi)):
            return rf"[{rr[klo]:.3f}, {rr[khi]:.3f}]"
        return "—"

    def _hr_row(label, klo, khi):
        cells = [_ci_cell(r, klo, khi)] + ([_ci_cell(s, klo, khi)] if has_speed else [])
        lines.append(rf"{label} & " + " & ".join(cells) + r" \\[4pt]")

    def _bootstrap_overlay():
        """Matched-pair bootstrap CI for beta_4 (Model A, Main column only)."""
        if (
            df_pair_bootstrap is not None
            and not df_pair_bootstrap.empty
            and "bootstrap_hr_ci_lo" in df_pair_bootstrap.columns
        ):
            boot = df_pair_bootstrap[df_pair_bootstrap["scope"].astype(str) == "all"]
            if boot.empty:
                boot = df_pair_bootstrap
            if not boot.empty:
                br = boot.iloc[0]
                boot_ci = rf"[{br['bootstrap_hr_ci_lo']:.3f}, {br['bootstrap_hr_ci_hi']:.3f}]"
                cells = [boot_ci] + (["—"] if has_speed else [])
                lines.append(
                    r"\hspace{1em} Hazard Ratio [bootstrap 95\% CI] & "
                    + " & ".join(cells)
                    + r" \\[4pt]"
                )

    lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Treatment effect (difference-in-differences): increment at answer arrival ($\beta_4$)}}}} \\")
    _coef_row(r"\hspace{1em} Answer arrival ($\times$ Post-Answer Received)", "treat_coef", "treat_p", "treat_se")
    _hr_row(r"\hspace{1em} Hazard Ratio [95\% CI]", "arrival_ci_lo", "arrival_ci_hi")
    _bootstrap_overlay()
    lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Parallel-trends diagnostic: waiting period ($\beta_2$)}}}} \\")
    _coef_row(r"\hspace{1em} Waiting period ($\beta_2$): Received Answer $\times$ Post-Question", "gap_coef", "gap_p", "gap_se")

    if has_speed:
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Response Time Interaction}}}} \\")
        row = r"\hspace{1em} Answer arrival $\times$ log(RT) ($\gamma$) & —"
        row += rf" & {_fmt_coef(s['speed_coef'], s['speed_p'])}"
        lines.append(row + r" \\")
        row = r" & —"
        row += rf" & {_fmt_se(s['speed_se'])}"
        lines.append(row + r" \\[2pt]")
        if pd.notna(s.get("gap_speed_coef", np.nan)):
            row = r"\hspace{1em} Waiting period $\times$ log(RT) ($\delta$) & —"
            row += rf" & {_fmt_coef(s['gap_speed_coef'], s.get('gap_speed_p', np.nan))}"
            lines.append(row + r" \\")
            row = r" & —"
            row += rf" & {_fmt_se(s['gap_speed_se']) if pd.notna(s.get('gap_speed_se', np.nan)) else ''}"
            lines.append(row + r" \\[4pt]")

    n = int(r.get("n_questions", r["n_rows"]))
    n_events = int(r["n_events"])
    n_interval = int(r["n_rows"]) if pd.notna(r.get("n_rows")) else None
    lines.append(r"\midrule")
    row = rf"N (matched question rows) & {n:,}"
    for _ in range(n_cols - 1):
        row += rf" & {n:,}"
    lines.append(row + r" \\")
    if n_interval is not None:
        row = rf"Person-interval rows & {n_interval:,}"
        for _ in range(n_cols - 1):
            row += rf" & {n_interval:,}"
        lines.append(row + r" \\")
    row = rf"Events & {n_events:,}"
    for _ in range(n_cols - 1):
        row += rf" & {n_events:,}"
    lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        r"Matched-pair bootstrap 95\% CI for the pooled arrival increment is from Table~\ref{tab:pair_bootstrap}.",
        r"N counts matched question-level rows (treated $+$ control); the Cox partial likelihood "
        r"is fit over the person-interval rows expanded from them. The point estimates use the "
        r"full interval set; only the matched-pair bootstrap replicates "
        r"(Table~\ref{tab:pair_bootstrap}) draw an 8M-interval-row subsample per replicate.",
        _conventions_note_text(),
        _sig_note_text(),
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_revision_robustness_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    spec_col: str = "spec",
) -> str:
    """Generic HR table for revision_robustness.py / cohort_robustness.py CSV outputs.

    The PRIMARY HR column is the answer-arrival increment (beta_4, from
    HR_increment_only + arrival_ci_*); the waiting-period coefficient (beta_2, from
    waiting_coef/waiting_p) is shown as a parallel-trends diagnostic column. The summed
    contrast (HR/CI_low/CI_high, beta_2+beta_4) stays in the CSV only — never rendered here."""
    if df.empty or "HR" not in df.columns:
        return ""
    spec_name = spec_col if spec_col in df.columns else ("model" if "model" in df.columns else None)

    have_arrival = "HR_increment_only" in df.columns and df["HR_increment_only"].notna().any()

    def _row_label(r):
        label_txt = str(r[spec_name]) if spec_name else str(r.get("model", ""))
        if "tenure_bucket" in r and pd.notna(r["tenure_bucket"]):
            label_txt = f"{label_txt} ({r['tenure_bucket']})"
        if "outcome" in r and pd.notna(r["outcome"]):
            label_txt = str(r["outcome"])
        return label_txt.replace("_", r"\_")

    if have_arrival:
        lines = [
            r"\begin{table}[H]",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"\textbf{Specification} & \textbf{Arrival HR} & \textbf{95\% CI} & \textbf{Waiting $\beta_2$} & \textbf{N} & \textbf{Events} \\",
            r"\midrule",
        ]
        for _, r in df.iterrows():
            n_col = "N_questions" if "N_questions" in r else "N"
            n_val = int(r.get(n_col, 0))
            arr_hr = r.get("HR_increment_only", np.nan)
            arr_lo, arr_hi = r.get("arrival_ci_lo", np.nan), r.get("arrival_ci_hi", np.nan)
            arr_ci = f"[{arr_lo:.3f}, {arr_hi:.3f}]" if pd.notna(arr_lo) and pd.notna(arr_hi) else "—"
            arr_hr_txt = f"{arr_hr:.3f}" if pd.notna(arr_hr) else "—"
            waiting_coef = r.get("waiting_coef", np.nan)
            waiting_txt = (
                f"{waiting_coef:+.3f}{_sig_stars(r.get('waiting_p', np.nan))}"
                if pd.notna(waiting_coef) else "—"
            )
            lines.append(
                rf"{_row_label(r)} & {arr_hr_txt} & {arr_ci} & {waiting_txt} "
                rf"& {n_val:,} & {int(r.get('events', 0)):,} \\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
        ]
        lines += _table_notes_block([
            _conventions_note_text(),
        ])
        lines.append(r"\end{table}")
        return "\n".join(lines)

    lines = [
        r"\begin{table}[H]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Specification} & \textbf{HR} & \textbf{95\% CI} & \textbf{N} & \textbf{Events} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        n_col = "N_questions" if "N_questions" in r else "N"
        n_val = int(r.get(n_col, 0))
        lines.append(
            rf"{_row_label(r)} & {r['HR']:.3f} "
            rf"& [{r['CI_low']:.3f}, {r['CI_high']:.3f}] "
            rf"& {n_val:,} & {int(r.get('events', 0)):,} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        _conventions_note_text(),
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_score_coding_table(df: pd.DataFrame) -> str:
    """Appendix: firstAnswerScore coding sensitivity (linear / log / bins).

    Long format, grouped by stratum (pooled, newcomer) then specification (baseline,
    +score, +full quality). The arrival HR ($\\beta_4$) carries significance stars
    testing $H_0$: HR $=1$, so a reader can see which estimates are distinguishable from
    one --- the crux of the coding-sensitivity check (e.g. pooled score-only is $>1$ under
    the linear coding but $\\approx 1$ under log/bins). The linear rows and both baselines
    reuse the cache from run_answer_quality / run_newcomer_bucket_checks."""
    if df.empty or "HR_increment_only" not in df.columns:
        return ""
    stratum_order = [("pooled", "Pooled"), ("newcomer_lt1w", r"Newcomer ($<$ 1 Week)")]
    spec_order = [
        ("baseline", "Speed baseline"),
        ("score_only", r"$+$ Vote score"),
        ("quality_controls", r"$+$ Full quality controls"),
    ]
    coding_label = {
        "none": "---",
        "linear": "Linear",
        "log": "Asinh",  # the "log" spec id is asinh(score); see note
        "bins": "Ordinal bins",
    }
    lines = [
        r"\begin{table}[H]",
        r"\caption{Answer-Score Coding Sensitivity}",
        r"\label{tab:score_coding_sensitivity}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"\textbf{Specification} & \textbf{Score coding} & \textbf{Arrival HR} & "
        r"\textbf{95\% CI} & \textbf{Waiting $\beta_2$} & \textbf{N} & \textbf{Events} \\",
        r"\midrule",
    ]
    first_stratum = True
    for skey, slabel in stratum_order:
        sub = df[df["stratum"] == skey] if "stratum" in df.columns else df.iloc[0:0]
        if sub.empty:
            continue
        if not first_stratum:
            lines.append(r"\midrule")
        first_stratum = False
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{slabel}}}}} \\")
        for spkey, splabel in spec_order:
            ss = sub[sub["spec"] == spkey]
            if ss.empty:
                continue
            codings = ["none"] if spkey == "baseline" else ["linear", "log", "bins"]
            spec_shown = False
            for ck in codings:
                rr = ss[ss["coding"] == ck]
                if rr.empty:
                    continue
                r = rr.iloc[0]
                # Label the spec on the first row that actually renders (not keyed on the
                # linear index), so a missing linear row never blanks the Specification cell.
                spec_disp = "" if spec_shown else splabel
                spec_shown = True
                arr_hr = r.get("HR_increment_only", np.nan)
                arr_txt = (
                    f"{arr_hr:.3f}{_sig_stars(r.get('arrival_p', np.nan))}"
                    if pd.notna(arr_hr) else "---"
                )
                lo, hi = r.get("arrival_ci_lo", np.nan), r.get("arrival_ci_hi", np.nan)
                ci = f"[{lo:.3f}, {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else "---"
                wc = r.get("waiting_coef", np.nan)
                wtxt = (
                    f"{wc:+.3f}{_sig_stars(r.get('waiting_p', np.nan))}"
                    if pd.notna(wc) else "---"
                )
                n_val = int(r.get("N_questions", 0))
                lines.append(
                    rf"{spec_disp} & {coding_label.get(ck, ck)} & {arr_txt} & {ci} & "
                    rf"{wtxt} & {n_val:,} & {int(r.get('events', 0)):,} \\"
                )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}"]
    lines += _table_notes_block([
        (
            r"Codings of the answer's vote score, all re-expressing the same control: "
            r"\emph{Linear} is the control tabled in Table~\ref{tab:answer_quality_robustness} "
            r"(raw \emph{Score} winsorized at the 5th/95th percentiles and standardized); "
            r"\emph{Asinh} is the inverse hyperbolic sine "
            r"$\mathrm{asinh}(\mathrm{score})=\ln(\mathrm{score}+\sqrt{\mathrm{score}^2+1})$ "
            r"(a signed log, concave in the score and defined for the negative scores present), "
            r"likewise winsorized and standardized; \emph{Ordinal bins} are indicators for "
            r"1--2, 3--9, and $\ge 10$ votes (reference score $\le 0$). Arrival-HR stars test "
            r"$H_0$: HR $=1$. The bin reference is score $\le 0$ and the concave coding uses "
            r"asinh rather than $\log(1+\mathrm{score})$ because roughly two million answers "
            r"carry net-negative (downvoted) scores, for which $\log(1+\mathrm{score})$ is undefined."
        ),
        _conventions_note_text(),
        _sig_note_text(),
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_outcome_decomposition_table(df: pd.DataFrame) -> str:
    """Decompose the answer-arrival treatment effect (beta_4) by help type, alongside
    the waiting-period (beta_2, parallel-trends) diagnostic. Makes visible where a
    reciprocity signal is cleanly identified (answers) versus dominated by the
    pre-answer activity pre-trend (comments). Accept events are excluded: accepting
    is a self-directed act available only to treated users, so the matched DiD
    contrast is degenerate for it."""
    if df.empty or "HR" not in df.columns:
        return ""
    label_map = {
        "answers_only": "Answers only",
        "comments_only": "Comments only",
        "edits_only": "Edits only",
        "answers_comments": "Answers $+$ comments",
        "answers_comments_edits": "Answers $+$ comments $+$ edits",
    }
    order = {k: i for i, k in enumerate(
        ["answers_only", "comments_only", "edits_only",
         "answers_comments", "answers_comments_edits"]
    )}
    df = df.copy()
    if "outcome" in df.columns:
        # Accepts-inclusive variants (accepts_only, composite_all) are degenerate for a
        # DiD contrast (accepting is treated-only by construction) — drop from the table.
        df = df[df["outcome"].isin(order)]
        df["_o"] = df["outcome"].map(order).fillna(99)
        df = df.sort_values("_o")
    lines = [
        r"\begin{table}[H]",
        r"\caption{Reciprocity Effect Decomposed by Help Type}",
        r"\label{tab:outcome_decomposition}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lccrr@{}}",
        r"\toprule",
        r"\textbf{Outcome} & \textbf{Arrival HR [95\% CI] ($\beta_4$)} & \textbf{Waiting $\beta_2$} & \textbf{N} & \textbf{Events} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        name = label_map.get(str(r.get("outcome", "")), str(r.get("outcome", "")).replace("_", r"\_"))
        b2, b4 = r.get("waiting_coef", np.nan), r.get("arrival_coef", np.nan)
        arr_hr = r.get("HR_increment_only", np.nan)
        if pd.isna(arr_hr) and pd.notna(b4):
            arr_hr = float(np.exp(b4))
        arr_lo, arr_hi = r.get("arrival_ci_lo", np.nan), r.get("arrival_ci_hi", np.nan)
        events = int(r.get("events", 0)) if pd.notna(r.get("events", np.nan)) else 0
        n_val = int(r.get("N_questions", 0)) if pd.notna(r.get("N_questions", np.nan)) else 0
        arr_ci = f"[{arr_lo:.3f}, {arr_hi:.3f}]" if pd.notna(arr_lo) and pd.notna(arr_hi) else ""
        arr_txt = f"{arr_hr:.3f}\\,{arr_ci}" if pd.notna(arr_hr) else "—"
        b2_txt = f"{b2:+.2f}" if pd.notna(b2) else "—"
        lines.append(rf"{name} & {arr_txt} & {b2_txt} & {n_val:,} & {events:,} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        r"Accepting an answer is excluded from the decomposition: it is a self-directed "
        r"act available only to treated users (a control never receives an answer to "
        r"accept), so the matched difference-in-differences contrast is degenerate for it.",
        _conventions_note_text(),
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_viewcount_placebo_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = [
        r"\begin{table}[H]",
        r"\caption{ViewCount Placebo: No-Answer Questions (High vs Low Views)}",
        r"\label{tab:viewcount_placebo}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Term} & \textbf{HR} & \textbf{95\% CI} & \textbf{$p$} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        term_tex = str(r["term"]).replace("_", r"\_")
        lines.append(
            rf"{term_tex} & {r['HR']:.3f} "
            rf"& [{r['CI_low']:.3f}, {r['CI_high']:.3f}] & {r['p']:.3f} \\"
        )
    if "median_viewCount" in df.columns and df["median_viewCount"].notna().any():
        med = df["median_viewCount"].iloc[0]
        nq = int(df["n_questions"].iloc[0]) if "n_questions" in df.columns else 0
        lines += [
            r"\midrule",
            rf"\multicolumn{{4}}{{@{{}}l}}{{\footnotesize Split at median ViewCount = {med:.0f}; n = {nq:,} no-answer questions.}} \\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# =====================================================================
# Table 3: Main effect (by tenure bucket)
# =====================================================================

def generate_main_results_table(df: pd.DataFrame, bootstrap_available: bool = False) -> str:
    """
    Generate LaTeX table for the main effect (Model A) by tenure bucket.
    Each column is one tenure bucket. N = number of unique questions (question_id).
    """
    # Ensure correct order
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["treat_coef"])

    n_buckets = len(df)

    header_labels = " & ".join(TENURE_TABLE_HEADERS[:n_buckets])

    lines = _tenure_table_preamble(
        "Cox Regression Results: Effect of Receiving an Answer on Helping Hazard",
        "tab:main_results",
    )
    lines += [
        rf"\begin{{tabularx}}{{\linewidth}}{{{_tenure_table_col_spec(n_buckets)}}}",
        r"\toprule",
        rf" & {header_labels} \\",
        r"\midrule",
    ]

    # Row helper
    def _row(label, col_coef, col_se, col_p):
        cells = []
        for _, r in df.iterrows():
            coef = r.get(col_coef, np.nan)
            se = r.get(col_se, np.nan)
            p = r.get(col_p, np.nan)
            if pd.notna(coef):
                cells.append(_fmt_coef(coef, p))
            else:
                cells.append("—")
        line1 = rf"{label} & " + " & ".join(cells) + r" \\"

        se_cells = []
        for _, r in df.iterrows():
            se = r.get(col_se, np.nan)
            if pd.notna(se):
                se_cells.append(_fmt_se(se))
            else:
                se_cells.append("")
        line2 = rf" & " + " & ".join(se_cells) + r" \\"
        return line1 + "\n" + line2

    def _section(title):
        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{" + title + r"}} \\")

    def _coef_se_block(label, coef_key, p_key, se_key):
        coef_cells, se_cells = [], []
        for _, r in df.iterrows():
            coef_cells.append(_fmt_coef(r[coef_key], r.get(p_key, np.nan)) if pd.notna(r.get(coef_key)) else "—")
            se_cells.append(_fmt_se(r[se_key]) if (se_key and pd.notna(r.get(se_key))) else "")
        lines.append(rf"{label} & " + " & ".join(coef_cells) + r" \\")
        lines.append(r" & " + " & ".join(se_cells) + r" \\")

    def _hr_ci_block(label, lo_key, hi_key, coef_key=None, se_key=None):
        # Prefer explicit CI columns; fall back to a normal-approx CI from coef/se so old
        # CSVs (or the arrival increment) still render a Hazard Ratio [95% CI] row.
        cells = []
        for _, r in df.iterrows():
            lo, hi = r.get(lo_key, np.nan), r.get(hi_key, np.nan)
            if (pd.isna(lo) or pd.isna(hi)) and coef_key and pd.notna(r.get(coef_key)) and pd.notna(r.get(se_key)):
                lo = float(np.exp(r[coef_key] - 1.96 * r[se_key]))
                hi = float(np.exp(r[coef_key] + 1.96 * r[se_key]))
            cells.append(f"[{lo:.3f}, {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else "—")
        lines.append(rf"{label} & " + " & ".join(cells) + r" \\[4pt]")

    _section(r"Treatment effect: increment at answer arrival ($\beta_4$)")
    _coef_se_block(r"\hspace{1em}Answer arrival ($\times$ Post-Answer Received)", "treat_coef", "treat_p", "treat_se")
    _hr_ci_block(r"\hspace{1em}\textit{Hazard Ratio [95\% CI]}", "arrival_ci_lo", "arrival_ci_hi", "treat_coef", "treat_se")
    _section(r"Waiting period ($\beta_2$, parallel-trends diagnostic)")
    _coef_se_block(r"\hspace{1em}Received Answer $\times$ Post-Question", "gap_coef", "gap_p", "gap_se")

    # N = unique questions; Events = helping events
    lines.append(r"\midrule")
    n_cells = [f"{int(r.get('n_questions', r['n_rows'])):,}" for _, r in df.iterrows()]
    evt_cells = [f"{int(r['n_events']):,}" for _, r in df.iterrows()]
    lines.append(rf"N & " + " & ".join(n_cells) + r" \\")
    lines.append(rf"Events & " + " & ".join(evt_cells) + r" \\")

    lines.append(r"\bottomrule")
    notes = [
        r"N = unique questions (treated + control) in the Cox sample. Within each column, treated vs.\ control counts can differ because tenure is defined per question.",
        _estimand_note_text(),
        _standard_error_note_text(bootstrap_available=bootstrap_available),
        _events_note_text(),
        _sig_note_text(),
    ]
    lines += _tenure_table_postamble(notes)
    return "\n".join(lines)


# =====================================================================
# Table 4: Speed interaction (by tenure bucket)
# =====================================================================

def generate_speed_table(df: pd.DataFrame, bootstrap_available: bool = False) -> str:
    """
    Generate LaTeX table for Model B (speed interaction) by tenure bucket.
    """
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["treat_coef"])

    n_buckets = len(df)
    header_labels = " & ".join(TENURE_TABLE_HEADERS[:n_buckets])

    lines = _tenure_table_preamble(
        "Response Time Moderation of the Reciprocity Effect",
        "tab:speed_results",
    )
    lines += [
        rf"\begin{{tabularx}}{{\linewidth}}{{{_tenure_table_col_spec(n_buckets)}}}",
        r"\toprule",
        rf" & {header_labels} \\",
        r"\midrule",
    ]

    def _section(title):
        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{" + title + r"}} \\")

    def _coef_se_block(label, coef_key, p_key, se_key, trail=r" \\[4pt]"):
        coef_cells, se_cells = [], []
        for _, r in df.iterrows():
            coef_cells.append(_fmt_coef(r[coef_key], r.get(p_key, np.nan)) if pd.notna(r.get(coef_key)) else "—")
            se_cells.append(_fmt_se(r[se_key]) if (se_key and pd.notna(r.get(se_key))) else "")
        lines.append(rf"{label} & " + " & ".join(coef_cells) + r" \\")
        lines.append(r" & " + " & ".join(se_cells) + trail)

    _section(r"Treatment effect: arrival increment ($\beta_4$, at mean response time)")
    _coef_se_block(r"\hspace{1em}Answer arrival (net post-answer)", "treat_coef", "treat_p", "treat_se")

    _section(r"Response-time moderation ($\gamma$ = arrival; $\delta$ = waiting-period diagnostic)")
    _coef_se_block(r"\hspace{1em}Answer arrival $\times$ log(RT) ($\gamma$)", "speed_coef", "speed_p", "speed_se", trail=r" \\[2pt]")
    _coef_se_block(r"\hspace{1em}Waiting period $\times$ log(RT) ($\delta$)", "gap_speed_coef", "gap_speed_p", "gap_speed_se")

    # N = unique questions; Events = helping events
    lines.append(r"\midrule")
    n_cells = [f"{int(r.get('n_questions', r['n_rows'])):,}" for _, r in df.iterrows()]
    evt_cells = [f"{int(r['n_events']):,}" for _, r in df.iterrows()]
    lines.append(rf"N & " + " & ".join(n_cells) + r" \\")
    lines.append(rf"Events & " + " & ".join(evt_cells) + r" \\")

    lines.append(r"\bottomrule")
    notes = [
        r"$\gamma$ and $\delta$ enter response time linearly in standardized log hours; "
        r"the flexible discrete-bin counterpart of this specification is "
        r"Table~\ref{tab:response_time_bins} and Figure~\ref{fig:interaction_effect}.",
        _conventions_note_text(),
        _sig_note_text(),
    ]
    lines += _tenure_table_postamble(notes)
    return "\n".join(lines)


def generate_response_time_bins_table(df: pd.DataFrame, bootstrap_available: bool = False) -> str:
    """Generate LaTeX table for non-parametric response-time-bin treatment effects."""
    if df.empty or "treat_hr" not in df.columns:
        return ""
    df = df.copy()
    df["bucket_order"] = df["bucket"].map(
        {label: i for i, label in enumerate(RT_BIN_LABELS)}
    ).fillna(999)
    df = df.sort_values("bucket_order")

    lines = [
        r"\begin{table}[H]",
        r"\caption{Treatment Effect by Response-Time Bin}",
        r"\label{tab:response_time_bins}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Response time} & \textbf{HR} & \textbf{95\% CI} & \textbf{N} & \textbf{Events} \\",
        r"\midrule",
    ]
    # ISS-24 re-headline: under "arrival" report the per-bin ARRIVAL increment HR as
    # primary (treat_* = is_treated_active, which this CSV already carries); the summed DiD
    # contrast remains available in the pooled regression tables. Under "summed" keep the
    # summed DiD contrast (did_*) primary, which is comparable across bins.
    have_arrival = "treat_hr" in df.columns and df["treat_hr"].notna().any()
    have_did = "did_hr" in df.columns and df["did_hr"].notna().any()
    if HEADLINE_ESTIMAND == "arrival" and have_arrival:
        use_did = False
        hr_key, lo_key, hi_key, p_key = ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    elif have_did:
        use_did = True
        hr_key, lo_key, hi_key, p_key = ("did_hr", "did_ci_lo", "did_ci_hi", "did_p")
    else:
        use_did = False
        hr_key, lo_key, hi_key, p_key = ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    for _, r in df.iterrows():
        p = r.get(p_key, np.nan)
        stars = _sig_stars(p) if pd.notna(p) else ""
        n = int(r.get("n_questions", r.get("n_rows", 0)))
        events = int(r.get("n_events", 0))
        hr = r.get(hr_key, np.nan)
        lo, hi = r.get(lo_key, np.nan), r.get(hi_key, np.nan)
        hr_txt = f"{hr:.3f}{stars}" if pd.notna(hr) else "—"
        ci_txt = f"[{lo:.3f}, {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else "—"
        lines.append(
            rf"{_latex_bucket(str(r['bucket']))} & {hr_txt} "
            rf"& {ci_txt} "
            rf"& {n:,} & {events:,} \\"
        )
    detail_note = (
        r"HR is the summed DiD contrast (post-answer vs.\ pre-question, treated vs.\ control); comparable across bins."
        if use_did
        else r"HR is the answer-arrival increment ($\beta_4$, is\_treated\_active) at answer arrival."
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        r"Each row fits Model~A to treated questions in that response-time bin plus the full no-answer control pool. "
        r"Bins are cut on the actual question-to-answer latency ($T_A - T_Q$); because the observation window closes two days after the answer, every treated question retains a full two-day post-answer phase, so $\beta_4$ is identified in all bins including $>$3 days. "
        + detail_note,
        _conventions_note_text(),
        _sig_note_text(),
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_response_time_bins_quality_table(
    df_base: pd.DataFrame,
    df_qual: pd.DataFrame,
    bootstrap_available: bool = False,
) -> str:
    """Side-by-side baseline vs answer-quality-controlled RT-bin arrival HRs (appendix)."""
    if df_base is None or df_base.empty or df_qual is None or df_qual.empty:
        return ""
    if "treat_hr" not in df_base.columns or "treat_hr" not in df_qual.columns:
        return ""

    order = {label: i for i, label in enumerate(RT_BIN_LABELS)}
    base = df_base.copy()
    qual = df_qual.copy()
    base["bucket_order"] = base["bucket"].map(order).fillna(999)
    qual["bucket_order"] = qual["bucket"].map(order).fillna(999)
    merged = base.merge(
        qual,
        on="bucket",
        how="inner",
        suffixes=("_base", "_qual"),
    )
    if merged.empty:
        return ""
    merged["bucket_order"] = merged["bucket"].map(order).fillna(999)
    merged = merged.sort_values("bucket_order")

    lines = [
        r"\begin{table}[H]",
        r"\caption{Treatment Effect by Response-Time Bin, With and Without Answer-Quality Controls}",
        r"\label{tab:response_time_bins_quality}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Response time} & \textbf{Baseline HR} & \textbf{95\% CI} & \textbf{Quality HR} & \textbf{95\% CI} & \textbf{N} \\",
        r"\midrule",
    ]
    for _, r in merged.iterrows():
        p_b = r.get("treat_p_base", np.nan)
        p_q = r.get("treat_p_qual", np.nan)
        stars_b = _sig_stars(p_b) if pd.notna(p_b) else ""
        stars_q = _sig_stars(p_q) if pd.notna(p_q) else ""
        hr_b, lo_b, hi_b = r.get("treat_hr_base"), r.get("treat_ci_lo_base"), r.get("treat_ci_hi_base")
        hr_q, lo_q, hi_q = r.get("treat_hr_qual"), r.get("treat_ci_lo_qual"), r.get("treat_ci_hi_qual")
        n_val = int(r.get("n_questions_base", r.get("n_questions_qual", 0)))
        lines.append(
            rf"{_latex_bucket(str(r['bucket']))} & "
            rf"{hr_b:.3f}{stars_b} & [{lo_b:.3f}, {hi_b:.3f}] & "
            rf"{hr_q:.3f}{stars_q} & [{lo_q:.3f}, {hi_q:.3f}] & {n_val:,} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        r"Each row fits Model~A to treated questions in that response-time bin plus the full no-answer control pool. "
        r"Bins are cut on the actual question-to-answer latency ($T_A - T_Q$); because the observation window closes two days after the answer, every treated question retains a full two-day post-answer phase, so $\beta_4$ is identified in all bins including $>$3 days. "
        r"HR is the answer-arrival increment ($\beta_4$). Quality columns add acceptance, first-answer score, and length.",
        _conventions_note_text(),
        _sig_note_text(),
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_pair_bootstrap_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for matched-pair bootstrap uncertainty."""
    if df.empty or "bootstrap_hr_ci_lo" not in df.columns:
        return ""
    # 2026-07-14 estimand decision: the bootstrap targets beta_4 (is_treated_active) alone —
    # the DiD treatment effect — not the summed contrast. See pair_bootstrap_se.py.
    lines = [
        r"\begin{table}[H]",
        r"\caption{Matched-Pair Bootstrap Uncertainty for the Answer-Arrival DiD Increment ($\beta_4$; Model A)}",
        r"\label{tab:pair_bootstrap}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Scope} & \textbf{HR} & \textbf{Bootstrap 95\% CI} & \textbf{Replicates} & \textbf{N} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        n_q = r.get("n_questions", np.nan)
        if pd.isna(n_q) or n_q == "":
            n_q_txt = "—"
        else:
            n_q_txt = f"{int(n_q):,}"
        lines.append(
            rf"{_latex_bucket(str(r['scope']))} & {r['base_hr']:.3f} "
            rf"& [{r['bootstrap_hr_ci_lo']:.3f}, {r['bootstrap_hr_ci_hi']:.3f}] "
            rf"& {int(r['n_bootstrap_success'])}/{int(r['n_bootstrap_requested'])} "
            rf"& {n_q_txt} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        r"Replicates resample whole matched pairs with replacement. "
        r"HR is the answer-arrival increment $\exp(\beta_4)$, the difference-in-differences treatment effect.",
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_estimation_samples_table(
    df_main: pd.DataFrame,
    df_main_all: pd.DataFrame,
    n_unique_users: int | None = None,
    n_unique_questions: int | None = None,
) -> str:
    """Appendix mapping table (#34): reconciles the several sample sizes reported across
    the Cox tables, per model / tenure bucket.

    Columns: Users / Questions / Matched rows / Interval rows / Events.
      - Matched rows  = treated $+$ control question-level rows in the Cox sample
                        (= 2 $\\times$ matched pairs; controls counted once per pair they
                        anchor, so a reused control appears in several rows).
      - Interval rows = person-interval rows the Cox partial likelihood is fit over.
      - Users / Questions are dataset-wide DISTINCT counts (from the matched file). They do
        not partition by tenure bucket -- a user (and, via control reuse, a question) can
        contribute to several buckets -- so the per-bucket rows leave them blank.
    """
    def _fmt(v):
        return f"{int(v):,}" if v is not None and pd.notna(v) else "—"

    lines = [
        r"\begin{table}[H]",
        r"\caption{Estimation Sample Sizes by Model}",
        r"\label{tab:estimation_samples}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Users} & \textbf{Questions} & \textbf{Matched rows} & "
        r"\textbf{Interval rows} & \textbf{Events} \\",
        r"\midrule",
    ]

    # Pooled row (Models A and B share the same estimation sample).
    if df_main_all is not None and not df_main_all.empty:
        rp = df_main_all.iloc[0]
        matched = int(rp.get("n_questions", 0))
        interval = int(rp.get("n_rows", 0))
        events = int(rp.get("n_events", 0))
    else:
        matched = interval = events = 0
    lines.append(
        rf"Pooled (Models A \& B) & {_fmt(n_unique_users)} & {_fmt(n_unique_questions)} & "
        rf"{matched:,} & {interval:,} & {events:,} \\"
    )

    # Per tenure bucket (from results_main.csv), matched/interval/events partition the pool.
    if df_main is not None and not df_main.empty:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{6}{@{}l}{\textit{Model A, by tenure bucket}} \\")
        dfb = df_main.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
        dfb = dfb.dropna(subset=["n_rows"])
        for _, r in dfb.iterrows():
            matched_b = int(r.get("n_questions", r.get("n_rows", 0)))
            interval_b = int(r.get("n_rows", 0))
            events_b = int(r.get("n_events", 0))
            lines.append(
                rf"\hspace{{1em}}{_latex_bucket(str(r['bucket']))} & — & — & "
                rf"{matched_b:,} & {interval_b:,} & {events_b:,} \\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines += _table_notes_block([
        r"Matched rows are treated $+$ control question-level rows in the Cox sample "
        r"($=2\times$ matched pairs); a control reused across pairs contributes one row per "
        r"pair it anchors. Interval rows are the person-interval rows expanded from the "
        r"matched rows and fit by the Cox partial likelihood; the point estimates use the "
        r"full interval set, while the matched-pair bootstrap (Table~\ref{tab:pair_bootstrap}) "
        r"caps each replicate at 8M interval rows.",
        r"Users and Questions are dataset-wide distinct counts and do not partition by tenure "
        r"bucket (a user's questions, and reused control questions, can fall in several "
        r"buckets), so the bucket rows leave those columns blank.",
        r"Events = helping events in the full analysis sample (after time rounding).",
    ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =====================================================================
# Figure 1: Reciprocity Strength Across Experience
# =====================================================================

def generate_reciprocity_figure(df: pd.DataFrame):
    """
    Connected-dots plot of the hazard ratio for isTreatedActive by tenure bucket,
    with 95% CI error bars.
    """
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    # ISS-24 re-headline: under HEADLINE_ESTIMAND=="arrival" plot the answer-arrival
    # increment (beta_4, is_treated_active) as the primary series so this headline figure
    # matches the re-headlined tables; under "summed" plot the summed DiD. Fall back to
    # whichever column set is present.
    have_did = "did_hr" in df.columns and df["did_hr"].notna().any()
    have_arrival = "treat_hr" in df.columns and df["treat_hr"].notna().any()
    if HEADLINE_ESTIMAND == "arrival" and have_arrival:
        hr_col, lo_col, hi_col, p_col = ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    elif have_did:
        hr_col, lo_col, hi_col, p_col = ("did_hr", "did_ci_lo", "did_ci_hi", "did_p")
    else:
        hr_col, lo_col, hi_col, p_col = ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    df = df.dropna(subset=[hr_col])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    x = np.arange(len(df))
    hrs = df[hr_col].values
    ci_lo = df[lo_col].values
    ci_hi = df[hi_col].values

    err_lo = hrs - ci_lo
    err_hi = ci_hi - hrs

    # Connected dots with 95% CI error bars
    ax.errorbar(
        x, hrs,
        yerr=[err_lo, err_hi],
        fmt="o-",
        color="#2563eb",
        linewidth=2,
        markersize=9,
        capsize=5,
        capthick=1.2,
        ecolor="#2d2d2d",
        elinewidth=1.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )

    # Null effect line and subtle band for reference. Tighten the y-limit to the data:
    # the CIs top out near 1.09, so leave only modest headroom above the tallest error
    # bar for the value labels (drawn at ci_hi+0.015) and the significance stars
    # (ci_hi+0.055), rather than the old fixed 1.35 ceiling that left ~half the panel empty.
    y_min = min(0.97, ci_lo.min() - 0.02)
    y_max = ci_hi.max() + 0.09
    ax.axhspan(0.98, 1.02, color="gray", alpha=0.12, zorder=0)
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.2, label="No effect (HR = 1)", zorder=1)
    ax.set_ylim(y_min, y_max)

    # Horizontal grid for easier reading
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_SHORT, fontsize=10)
    ax.set_xlabel("User tenure at time of question", fontsize=11)
    ax.set_ylabel("Hazard ratio (receiving answer → helping)", fontsize=11)
    ax.set_title("Strength of the generalized reciprocity effect across user experience", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    # Value labels above points
    for i, (hr, hi) in enumerate(zip(hrs, ci_hi)):
        ax.text(i, hi + 0.015, f"{hr:.2f}", ha="center", va="bottom", fontsize=8, fontweight="500", color="#333333")

    # Significance stars above value labels
    for i, (_, r) in enumerate(df.iterrows()):
        p = r[p_col]
        stars = _sig_stars(p).replace("\\textdagger", "†")
        if stars:
            ax.text(i, ci_hi[i] + 0.055, stars, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a1a")

    plt.tight_layout()

    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"strength_rec.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved strength_rec.[eps/png/pdf]")


# =====================================================================
# Figure 2: Speed Moderation Across Experience
# =====================================================================

def generate_speed_figure(df: pd.DataFrame):
    """
    Connected-dots plot of the speed interaction coefficient by tenure bucket,
    with 95% CI error bars (1.96 * SE). Matches strength_rec style for consistency.
    """
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["speed_coef"])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    x = np.arange(len(df))
    coefs = df["speed_coef"].values
    ses = df["speed_se"].values
    err = 1.96 * ses

    ax.errorbar(
        x, coefs,
        yerr=err,
        fmt="o-",
        color="#059669",
        linewidth=2,
        markersize=9,
        capsize=5,
        capthick=1.2,
        ecolor="#2d2d2d",
        elinewidth=1.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )

    ax.axhline(y=0, color="#555555", linestyle="--", linewidth=1.2, label="No moderation (coef = 0)", zorder=0)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_SHORT, fontsize=10)
    ax.set_xlabel("User tenure at time of question", fontsize=11)
    ax.set_ylabel("Coefficient: Treatment × log(Response Time)", fontsize=11)
    ax.set_title("Response time moderation of the reciprocity effect across user experience", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="best", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    # Y limits with headroom for labels
    y_margin = max(0.015, err.max() + 0.01)
    ax.set_ylim(coefs.min() - y_margin - err[np.argmin(coefs)], coefs.max() + y_margin + err[np.argmax(coefs)])

    # Significance stars above/below error bars
    for i, (_, r) in enumerate(df.iterrows()):
        p = r["speed_p"]
        stars = _sig_stars(p).replace("\\textdagger", "†")
        if stars:
            y_pos = coefs[i] + err[i] + 0.008 if coefs[i] >= 0 else coefs[i] - err[i] - 0.008
            va = "bottom" if coefs[i] >= 0 else "top"
            ax.text(i, y_pos, stars, ha="center", va=va, fontsize=10, fontweight="bold", color="#1a1a1a")

    plt.tight_layout()

    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"speed_moderation.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved speed_moderation.[eps/png/pdf]")


# =====================================================================
# Figure: Interaction effect — treatment effect by response time bin (or parametric fallback)
# =====================================================================

def generate_interaction_effect_figure(
    df_rt_bins: pd.DataFrame = None,
    df_speed_all: pd.DataFrame = None,
):
    """
    Plot treatment effect (hazard ratio) by response time.
    If df_rt_bins is provided and non-empty, plot a bar chart by bin (non-parametric).
    Otherwise plot the parametric curve from Model B (speed interaction).
    """
    if df_rt_bins is not None and not df_rt_bins.empty and "treat_hr" in df_rt_bins.columns:
        _plot_interaction_effect_bins(df_rt_bins)
        return
    # Fallback: parametric curve from Model B
    if df_speed_all is not None and not df_speed_all.empty and "speed_coef" in df_speed_all.columns:
        row = df_speed_all.iloc[0]
        treat = row["treat_coef"]
        speed = row["speed_coef"]
        rt_hours = np.linspace(0.5, 720, 400)
        log_rt = np.log1p(rt_hours)
        log_hr = treat + speed * log_rt
        hr = np.exp(log_hr)
        title_suffix = " (Model B: linear)"
    else:
        print("  ⚠ Skipping interaction_effect figure: no response-time bin results or Model B.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.plot(rt_hours, hr, color="#2563eb", linewidth=2.5, label="Predicted hazard ratio")
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.2, label="No effect (HR = 1)", zorder=1)
    ax.fill_between(rt_hours, 1.0, hr, where=(hr >= 1.0), alpha=0.15, color="#2563eb", zorder=0)
    ax.fill_between(rt_hours, hr, 1.0, where=(hr < 1.0), alpha=0.15, color="#c62828", zorder=0)
    ax.set_xlabel("Response time (hours from question to answer)", fontsize=11)
    ax.set_ylabel("Predicted hazard ratio\n(receiving answer → helping)", fontsize=11)
    ax.set_title("Treatment effect (hazard ratio) by response time, pooled" + title_suffix, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, rt_hours.max())
    ax.set_ylim(min(0.92, hr.min() - 0.02), max(1.25, hr.max() + 0.03))
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="best", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"interaction_effect.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved interaction_effect.[eps/png/pdf]")


def _plot_interaction_effect_bins(df: pd.DataFrame):
    """Connected dots and line for treatment effect (HR) by response time bin."""
    # ISS-24 re-headline: match the re-headlined response-time-bins TABLE. Under
    # HEADLINE_ESTIMAND=="arrival" plot the per-bin answer-arrival increment (beta_4) as the
    # primary series; under "summed" plot the summed DiD per bin. Both per-bin gradients are
    # in the CSV; the summed contrast is also reported in the pooled tables.
    have_did = "did_hr" in df.columns and df["did_hr"].notna().any()
    have_arrival = "treat_hr" in df.columns and df["treat_hr"].notna().any()
    if HEADLINE_ESTIMAND == "arrival" and have_arrival:
        hr_col, lo_col, hi_col, p_col = ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    elif have_did:
        hr_col, lo_col, hi_col, p_col = ("did_hr", "did_ci_lo", "did_ci_hi", "did_p")
    else:
        hr_col, lo_col, hi_col, p_col = ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    labels = df["bucket"].tolist()
    hrs = df[hr_col].values
    ci_lo = df[lo_col].values
    ci_hi = df[hi_col].values
    err_lo = hrs - ci_lo
    err_hi = ci_hi - hrs

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    x = np.arange(len(df))
    ax.errorbar(
        x, hrs,
        yerr=[err_lo, err_hi],
        fmt="o-",
        color="#2563eb",
        linewidth=2,
        markersize=9,
        capsize=5,
        capthick=1.2,
        ecolor="#2d2d2d",
        elinewidth=1.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    ax.axhspan(0.98, 1.02, color="gray", alpha=0.12, zorder=0)
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.2, label="No effect (HR = 1)", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
    ax.set_xlabel("Response time (question to answer)", fontsize=11)
    ax.set_ylabel("Hazard ratio (receiving answer → helping)", fontsize=11)
    ax.set_title("Treatment effect (hazard ratio) by response time bin, pooled across tenure", fontsize=12, fontweight="bold", pad=10)
    y_min = min(0.92, ci_lo.min() - 0.02)
    y_max = max(1.2, ci_hi.max() + 0.06)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    for i, (_, r) in enumerate(df.iterrows()):
        stars = _sig_stars(r[p_col]).replace("\\textdagger", "†")
        if stars:
            ax.text(i, ci_hi[i] + 0.015, stars, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a1a")
    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"interaction_effect.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved interaction_effect.[eps/png/pdf]")


# =====================================================================
# Main
# =====================================================================

def main():
    _ensure_dirs()

    # --- Load cached results (all result CSVs live in model_cache) ---
    main_path = os.path.join(CACHE_DIR, "results_main.csv")
    main_all_path = os.path.join(CACHE_DIR, "results_main_all.csv")
    speed_all_path = os.path.join(CACHE_DIR, "results_speed_all.csv")
    rt_bins_path = os.path.join(CACHE_DIR, "results_response_time_bins.csv")
    rt_bins_quality_path = os.path.join(CACHE_DIR, "results_response_time_bins_quality.csv")
    pair_bootstrap_path = os.path.join(CACHE_DIR, "results_pair_bootstrap.csv")
    speed_path = os.path.join(CACHE_DIR, "results_speed.csv")
    desc_path = os.path.join(CACHE_DIR, "descriptives.pkl")

    if not os.path.exists(main_path):
        print(f"ERROR: {main_path} not found. Run fit_cox_models.py first.")
        return

    print(f"Reading cache from: {os.path.abspath(CACHE_DIR)}")
    df_main = pd.read_csv(main_path)
    df_main_all = pd.read_csv(main_all_path) if os.path.exists(main_all_path) else pd.DataFrame()
    df_speed_all = pd.read_csv(speed_all_path) if os.path.exists(speed_all_path) else pd.DataFrame()
    df_rt_bins = pd.read_csv(rt_bins_path) if os.path.exists(rt_bins_path) else pd.DataFrame()
    df_rt_bins_quality = (
        pd.read_csv(rt_bins_quality_path) if os.path.exists(rt_bins_quality_path) else pd.DataFrame()
    )
    df_pair_bootstrap = pd.read_csv(pair_bootstrap_path) if os.path.exists(pair_bootstrap_path) else pd.DataFrame()
    df_speed = pd.read_csv(speed_path) if os.path.exists(speed_path) else pd.DataFrame()
    descriptives = {}
    if os.path.exists(desc_path):
        with open(desc_path, "rb") as f:
            descriptives = pickle.load(f)
        print(f"Loaded descriptives from: {os.path.abspath(desc_path)}")
    else:
        print(f"WARNING: {os.path.abspath(desc_path)} not found; desc_stats table will have missing values.")

    # Fix pooled Events when CSVs still store MAX_FIT_ROWS subsample counts.
    if _reconcile_pooled_events(df_main, df_main_all, df_speed_all):
        _persist_reconciled_pooled_events(df_main_all, df_speed_all)

    bootstrap_available = (
        not df_pair_bootstrap.empty and "bootstrap_hr_ci_lo" in df_pair_bootstrap.columns
    )

    # --- Generate Tables ---
    print("\n=== Generating LaTeX Tables ===")

    # Pooled regressions table (Main, Main+Speed)
    if not df_main_all.empty:
        tex = generate_regression_all_table(
            df_main_all,
            df_speed_all,
            bootstrap_available=bootstrap_available,
            df_pair_bootstrap=df_pair_bootstrap if bootstrap_available else None,
        )
        out = os.path.join(TABLE_DIR, "regression_all.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # Table 1: Descriptive Statistics (N by bucket from df_main so it matches main_results)
    tex = generate_desc_stats_table(descriptives, df_main=df_main)
    out = os.path.join(TABLE_DIR, "desc_stats.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ Wrote {os.path.abspath(out)}")

    # Table 3: Main effect (by tenure bucket)
    tex = generate_main_results_table(df_main, bootstrap_available=bootstrap_available)
    out = os.path.join(TABLE_DIR, "main_results.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ Wrote {os.path.abspath(out)}")

    # Table 4: Speed interaction (by tenure bucket)
    if not df_speed.empty:
        tex = generate_speed_table(df_speed, bootstrap_available=bootstrap_available)
        out = os.path.join(TABLE_DIR, "speed_results.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # Non-parametric response-time-bin treatment effects
    if not df_rt_bins.empty:
        tex = generate_response_time_bins_table(
            df_rt_bins, bootstrap_available=bootstrap_available
        )
        out = os.path.join(TABLE_DIR, "response_time_bins.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # Appendix: baseline vs quality-controlled RT bins (#27)
    if not df_rt_bins.empty and not df_rt_bins_quality.empty:
        tex = generate_response_time_bins_quality_table(
            df_rt_bins, df_rt_bins_quality, bootstrap_available=bootstrap_available
        )
        if tex:
            out = os.path.join(TABLE_DIR, "response_time_bins_quality.tex")
            with open(out, "w") as f:
                f.write(tex)
            print(f"✓ {out}")

    if not df_pair_bootstrap.empty:
        tex = generate_pair_bootstrap_table(df_pair_bootstrap)
        if tex:
            out = os.path.join(TABLE_DIR, "pair_bootstrap.tex")
            with open(out, "w") as f:
                f.write(tex)
            print(f"✓ {out}")

    # Revision robustness tables (from revision_robustness.py)
    revision_tables = [
        ("results_observable_controls.csv", "Observable Selection Controls", "tab:observable_controls", "observable_controls.tex", "spec"),
        ("results_answer_quality.csv", "Answer-Quality Robustness", "tab:answer_quality_robustness", "answer_quality_robustness.tex", "spec"),
        ("results_newcomer_robustness.csv", "Newcomer Bucket Robustness ($<$ 1 Week)", "tab:newcomer_robustness", "newcomer_robustness.tex", "model"),
        ("results_cohort_robustness.csv", "Cohort Heterogeneity Robustness", "tab:cohort_robustness", "cohort_robustness.tex", "model"),
    ]
    for csv_name, caption, label, out_name, spec_col in revision_tables:
        path = os.path.join(CACHE_DIR, csv_name)
        if os.path.exists(path) and os.path.getsize(path) > 1:
            df_rev = pd.read_csv(path)
            if df_rev.empty:
                continue
            tex = generate_revision_robustness_table(df_rev, caption, label, spec_col=spec_col)
            if tex:
                out = os.path.join(TABLE_DIR, out_name)
                with open(out, "w") as f:
                    f.write(tex)
                print(f"✓ {out}")

    # ISS-04 outcome decomposition (dedicated table: summed DiD + beta_2/beta_4 by help type)
    composite_path = os.path.join(CACHE_DIR, "results_composite_outcome.csv")
    if os.path.exists(composite_path) and os.path.getsize(composite_path) > 1:
        df_comp = pd.read_csv(composite_path)
        if not df_comp.empty:
            tex = generate_outcome_decomposition_table(df_comp)
            if tex:
                out = os.path.join(TABLE_DIR, "composite_outcome.tex")
                with open(out, "w") as f:
                    f.write(tex)
                print(f"✓ {out}")

    # ISS-06: answer-score coding sensitivity (dedicated appendix table; separate CSV so
    # answer_quality_robustness.tex stays byte-identical to its linear-only rendering).
    score_coding_path = os.path.join(CACHE_DIR, "results_score_coding_sensitivity.csv")
    if os.path.exists(score_coding_path) and os.path.getsize(score_coding_path) > 1:
        df_sc = pd.read_csv(score_coding_path)
        if not df_sc.empty:
            tex = generate_score_coding_table(df_sc)
            if tex:
                out = os.path.join(TABLE_DIR, "score_coding_sensitivity.tex")
                with open(out, "w") as f:
                    f.write(tex)
                print(f"✓ {out}")

    # ViewCount placebo is deprecated for the manuscript (cumulative dump ViewCount
    # is not a valid contemporaneous exposure placebo). Do not regenerate tex.
    placebo_tex = os.path.join(TABLE_DIR, "viewcount_placebo.tex")
    if os.path.exists(placebo_tex):
        os.remove(placebo_tex)
        print(f"✓ Removed deprecated {placebo_tex}")

    # Conservative absolute-effect and NNT proxies
    df_absolute = build_absolute_effects()
    if not df_absolute.empty:
        write_absolute_effects_csv(df_absolute)
        tex = generate_absolute_effects_latex_table(df_absolute)
        out = os.path.join(TABLE_DIR, "absolute_effects.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # --- Generate Figures ---
    print("\n=== Generating Figures ===")

    generate_reciprocity_figure(df_main)

    if not df_speed.empty:
        generate_speed_figure(df_speed)

    generate_interaction_effect_figure(df_rt_bins, df_speed_all)

    print("\n=== All outputs generated ===")
    print(f"Tables: {TABLE_DIR}/")
    print(f"Figures: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
