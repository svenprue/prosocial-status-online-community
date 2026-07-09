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
from cox_config import RT_BIN_LABELS
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
    return rf"@{{}}>{{\raggedright\arraybackslash}}p{{2.55cm}}*{{{n_buckets}}}{{c}}@{{}}"


def _tenure_table_preamble(caption: str, label: str) -> list[str]:
    return [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\resizebox{\linewidth}{!}{%",
    ]


def _tenure_table_postamble() -> list[str]:
    return [
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]


def _latex_bucket(label: str) -> str:
    """Escape < and > for LaTeX math mode in table labels."""
    return label.replace("<", r"$<$").replace(">", r"$>$")


def _standard_error_note(n_cols: int) -> str:
    return (
        rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\footnotesize Cox-table standard errors are "
        r"model-based; matched-pair bootstrap uncertainty is reported separately when generated.} \\"
    )


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
        rf"Help events per window & {_f(he_mean)} & {_f(he_std)} & {_f(he_med)} & \\",
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
            return rf"[{rr[klo]:.2f}, {rr[khi]:.2f}]"
        return "—"

    def _hr_row(label, klo, khi):
        cells = [_ci_cell(r, klo, khi)] + ([_ci_cell(s, klo, khi)] if has_speed else [])
        lines.append(rf"{label} & " + " & ".join(cells) + r" \\[4pt]")

    have_did = pd.notna(r.get("did_coef", np.nan))
    if have_did:
        # Headline: the DiD treatment effect is the SUM of the two nested treated
        # indicators (post-answer vs. pre-question baseline, treated vs. control),
        # not the is_treated_active coefficient alone.
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Treatment Effect (DiD): post-answer vs.\ pre-question}}}} \\")
        _coef_row(r"\hspace{1em} Received Answer (net post-answer effect)", "did_coef", "did_p", "did_se")
        _hr_row(r"\hspace{1em} Hazard Ratio [95\% CI]", "did_ci_lo", "did_ci_hi")
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{\quad Decomposition (nested time-varying terms)}}}} \\")
        _coef_row(r"\hspace{2em} Waiting period: Received Answer $\times$ Post-Question", "gap_coef", "gap_p")
        _coef_row(r"\hspace{2em} Answer arrival: $\times$ Post-Answer Received", "treat_coef", "treat_p", "treat_se")
    else:
        # Legacy fallback (CSVs without did_* columns): report is_treated_active alone.
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Treatment Effect (DID)}}}} \\")
        _coef_row(r"\hspace{1em} Received Answer $\times$ Post-Answer Received", "treat_coef", "treat_p", "treat_se")
        _hr_row(r"\hspace{1em} Hazard Ratio [95\% CI]", "treat_ci_lo", "treat_ci_hi")
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Waiting Period}}}} \\")
        _coef_row(r"\hspace{1em} Received Answer $\times$ Post-Question", "gap_coef", "gap_p")

    if has_speed:
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Response Time Interaction}}}} \\")
        # Net RT moderation of the DiD (sum of the two RT interactions), then component.
        if pd.notna(s.get("did_speed_coef", np.nan)):
            net_cell = _fmt_coef(s["did_speed_coef"], s.get("did_speed_p", np.nan))
            lines.append(rf"\hspace{{1em}} Net: Treatment $\times$ log(RT), summed & — & {net_cell} \\")
            if pd.notna(s.get("did_speed_se", np.nan)):
                lines.append(rf" & — & {_fmt_se(s['did_speed_se'])} \\[2pt]")
        row = r"\hspace{1em} Component: Post-Answer $\times$ log(RT) & —"
        row += rf" & {_fmt_coef(s['speed_coef'], s['speed_p'])}"
        lines.append(row + r" \\")
        row = r" & —"
        row += rf" & {_fmt_se(s['speed_se'])}"
        lines.append(row + r" \\[4pt]")

    n = int(r.get("n_questions", r["n_rows"]))
    n_events = int(r["n_events"])
    lines.append(r"\midrule")
    row = rf"N & {n:,}"
    for _ in range(n_cols - 1):
        row += rf" & {n:,}"
    lines.append(row + r" \\")
    row = rf"Events & {n_events:,}"
    for _ in range(n_cols - 1):
        row += rf" & {n_events:,}"
    lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        _standard_error_note(n_cols + 1),
        rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\footnotesize $^{{***}}p<0.001$; $^{{**}}p<0.01$; $^{{*}}p<0.05$; $^{{\dagger}}p<0.1$}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_revision_robustness_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    spec_col: str = "spec",
) -> str:
    """Generic HR table for revision_robustness.py CSV outputs."""
    if df.empty or "HR" not in df.columns:
        return ""
    spec_name = spec_col if spec_col in df.columns else ("model" if "model" in df.columns else None)
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
        label_txt = str(r[spec_name]) if spec_name else str(r.get("model", ""))
        if "tenure_bucket" in r and pd.notna(r["tenure_bucket"]):
            label_txt = f"{label_txt} ({r['tenure_bucket']})"
        if "outcome" in r and pd.notna(r["outcome"]):
            label_txt = str(r["outcome"])
        label_tex = label_txt.replace("_", r"\_")
        n_col = "N_questions" if "N_questions" in r else "N"
        n_val = int(r.get(n_col, 0))
        lines.append(
            rf"{label_tex} & {r['HR']:.3f} "
            rf"& [{r['CI_low']:.3f}, {r['CI_high']:.3f}] "
            rf"& {n_val:,} & {int(r.get('events', 0)):,} \\"
        )
    lines += [
        r"\bottomrule",
        _standard_error_note(5),
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_outcome_decomposition_table(df: pd.DataFrame) -> str:
    """ISS-04 decomposition: the summed DiD treatment effect by help type, split into
    its waiting-period (beta_2, anticipatory engagement) and answer-arrival (beta_4)
    components. Makes visible where a reciprocity signal is cleanly identified
    (answers) versus dominated by the pre-answer activity pre-trend (comments)."""
    if df.empty or "HR" not in df.columns:
        return ""
    label_map = {
        "answers_only": "Answers only",
        "comments_only": "Comments only",
        "accepts_only": r"Accepts only$^{a}$",
        "answers_comments": "Answers $+$ comments",
        "composite_all": r"All types$^{a}$",
    }
    order = {k: i for i, k in enumerate(
        ["answers_only", "comments_only", "accepts_only", "answers_comments", "composite_all"]
    )}
    df = df.copy()
    if "outcome" in df.columns:
        df["_o"] = df["outcome"].map(order).fillna(99)
        df = df.sort_values("_o")
    lines = [
        r"\begin{table}[H]",
        r"\caption{Reciprocity Effect Decomposed by Help Type (ISS-04)}",
        r"\label{tab:outcome_decomposition}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcccr@{}}",
        r"\toprule",
        r"\textbf{Outcome} & \textbf{Treatment HR [95\% CI]} & \textbf{Waiting $\beta_2$} & \textbf{Arrival $\beta_4$} & \textbf{Events} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        name = label_map.get(str(r.get("outcome", "")), str(r.get("outcome", "")).replace("_", r"\_"))
        hr, lo, hi = r.get("HR", np.nan), r.get("CI_low", np.nan), r.get("CI_high", np.nan)
        hr_txt = f"{hr:.2f}" if pd.notna(hr) else "—"
        ci_txt = f"[{lo:.2f}, {hi:.2f}]" if pd.notna(lo) and pd.notna(hi) else ""
        b2, b4 = r.get("waiting_coef", np.nan), r.get("arrival_coef", np.nan)
        b2_txt = f"{b2:+.2f}" if pd.notna(b2) else "—"
        b4_txt = f"{b4:+.2f}" if pd.notna(b4) else "—"
        events = int(r.get("events", 0)) if pd.notna(r.get("events", np.nan)) else 0
        lines.append(rf"{name} & {hr_txt}\,{ci_txt} & {b2_txt} & {b4_txt} & {events:,} \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{@{}l}{\footnotesize Treatment HR is the summed DiD, $\exp(\beta_2+\beta_4)$; $\beta_2$ is the waiting-period (anticipatory-engagement) term and $\beta_4$ the answer-arrival increment.} \\",
        r"\multicolumn{5}{@{}l}{\footnotesize $^{a}$ Accept events are treated-only by construction (a control never receives an answer to accept); rows including them are degenerate and shown for reference.} \\",
        _standard_error_note(5),
        r"\end{tabular}",
        r"\end{table}",
    ]
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

def generate_main_results_table(df: pd.DataFrame) -> str:
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
        rf"\begin{{tabular}}{{{_tenure_table_col_spec(n_buckets)}}}",
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

    have_did = "did_coef" in df.columns and df["did_coef"].notna().any()

    if have_did:
        # Headline: the DiD treatment effect is the SUM of the two nested treated terms
        # (post-answer vs. pre-question baseline, treated vs. control). The raw
        # is_treated_active coefficient alone is only the post-answer increment over the
        # (now large) waiting-period term, so it is not the treatment effect.
        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Treatment Effect (DiD): post-answer vs.\ pre-question}} \\")
        did_coef_cells, did_se_cells, did_hr_cells = [], [], []
        for _, r in df.iterrows():
            did_coef_cells.append(_fmt_coef(r["did_coef"], r.get("did_p", np.nan)) if pd.notna(r.get("did_coef")) else "—")
            did_se_cells.append(_fmt_se(r["did_se"]) if pd.notna(r.get("did_se")) else "")
            if pd.notna(r.get("did_ci_lo")) and pd.notna(r.get("did_ci_hi")):
                did_hr_cells.append(f"[{r['did_ci_lo']:.2f}, {r['did_ci_hi']:.2f}]")
            else:
                did_hr_cells.append("—")
        lines.append(rf"\hspace{{1em}}Received Answer (net post-answer) & " + " & ".join(did_coef_cells) + r" \\")
        lines.append(rf" & " + " & ".join(did_se_cells) + r" \\")
        lines.append(rf"\hspace{{1em}}\textit{{Hazard Ratio [95\% CI]}} & " + " & ".join(did_hr_cells) + r" \\[4pt]")

        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{\quad Decomposition (nested time-varying terms)}} \\")
        gap_cells = [
            _fmt_coef(r["gap_coef"], r.get("gap_p", np.nan)) if pd.notna(r.get("gap_coef")) else "—"
            for _, r in df.iterrows()
        ]
        lines.append(rf"\hspace{{2em}}Waiting period ($\times$ Post-Question) & " + " & ".join(gap_cells) + r" \\")
        inc_cells = [
            _fmt_coef(r["treat_coef"], r.get("treat_p", np.nan)) if pd.notna(r.get("treat_coef")) else "—"
            for _, r in df.iterrows()
        ]
        lines.append(rf"\hspace{{2em}}Answer arrival ($\times$ Post-Answer Received) & " + " & ".join(inc_cells) + r" \\[4pt]")
    else:
        # Legacy fallback (old CSVs without did_* columns): report is_treated_active alone.
        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Treatment Effect (DID)}} \\")
        cells_coef, cells_se, cells_hr = [], [], []
        for _, r in df.iterrows():
            cells_coef.append(_fmt_coef(r["treat_coef"], r["treat_p"]))
            cells_se.append(_fmt_se(r["treat_se"]))
            cells_hr.append(f"[{r['treat_ci_lo']:.2f}, {r['treat_ci_hi']:.2f}]")
        lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post-Answer Received & " + " & ".join(cells_coef) + r" \\")
        lines.append(rf" & " + " & ".join(cells_se) + r" \\")
        lines.append(rf"\hspace{{1em}}\textit{{Hazard Ratio [95\% CI]}} & " + " & ".join(cells_hr) + r" \\[4pt]")
        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Waiting Period}} \\")
        gap_cells = [
            _fmt_coef(r["gap_coef"], r.get("gap_p", np.nan)) if pd.notna(r.get("gap_coef")) else "—"
            for _, r in df.iterrows()
        ]
        lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post-Question & " + " & ".join(gap_cells) + r" \\[4pt]")

    # N = unique questions; Events = helping events
    lines.append(r"\midrule")
    n_cells = [f"{int(r.get('n_questions', r['n_rows'])):,}" for _, r in df.iterrows()]
    evt_cells = [f"{int(r['n_events']):,}" for _, r in df.iterrows()]
    lines.append(rf"N & " + " & ".join(n_cells) + r" \\")
    lines.append(rf"Events & " + " & ".join(evt_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\footnotesize N = unique questions (treated + control) in the Cox sample. Within each column, treated vs.\ control counts can differ because tenure is defined per question.} \\",
        _standard_error_note(n_buckets + 1),
        r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\footnotesize $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$; $^{\dagger}p<0.1$} \\",
    ]
    lines += _tenure_table_postamble()
    return "\n".join(lines)


# =====================================================================
# Table 4: Speed interaction (by tenure bucket)
# =====================================================================

def generate_speed_table(df: pd.DataFrame) -> str:
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
        rf"\begin{{tabular}}{{{_tenure_table_col_spec(n_buckets)}}}",
        r"\toprule",
        rf" & {header_labels} \\",
        r"\midrule",
    ]

    have_did = "did_coef" in df.columns and df["did_coef"].notna().any()
    have_did_speed = "did_speed_coef" in df.columns and df["did_speed_coef"].notna().any()

    # Base treatment effect (at mean response time; covariates are mean-centred).
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Base Treatment Effect (DiD, at mean response time)}} \\")
    cells, se_cells = [], []
    for _, r in df.iterrows():
        if have_did and pd.notna(r.get("did_coef")):
            cells.append(_fmt_coef(r["did_coef"], r.get("did_p", np.nan)))
            se_cells.append(_fmt_se(r["did_se"]) if pd.notna(r.get("did_se")) else "")
        else:
            cells.append(_fmt_coef(r["treat_coef"], r["treat_p"]))
            se_cells.append(_fmt_se(r["treat_se"]))
    lines.append(rf"\hspace{{1em}}Received Answer (net post-answer) & " + " & ".join(cells) + r" \\")
    lines.append(rf" & " + " & ".join(se_cells) + r" \\[4pt]")

    # Net response-time moderation of the DiD = sum of the two RT interactions.
    if have_did_speed:
        lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Net Response-Time Moderation of DiD (summed)}} \\")
        net_cells, net_se_cells = [], []
        for _, r in df.iterrows():
            net_cells.append(_fmt_coef(r["did_speed_coef"], r.get("did_speed_p", np.nan)) if pd.notna(r.get("did_speed_coef")) else "—")
            net_se_cells.append(_fmt_se(r["did_speed_se"]) if pd.notna(r.get("did_speed_se")) else "")
        lines.append(rf"\hspace{{1em}}Treatment $\times$ log(RT), summed & " + " & ".join(net_cells) + r" \\")
        lines.append(rf" & " + " & ".join(net_se_cells) + r" \\[4pt]")

    # Components of the RT interaction (each attaches to one nested base term).
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{\quad Components}} \\")
    speed_cells, speed_se_cells = [], []
    for _, r in df.iterrows():
        speed_cells.append(_fmt_coef(r["speed_coef"], r["speed_p"]))
        speed_se_cells.append(_fmt_se(r["speed_se"]))
    lines.append(rf"\hspace{{2em}}Answer arrival $\times$ log(RT) & " + " & ".join(speed_cells) + r" \\")
    lines.append(rf" & " + " & ".join(speed_se_cells) + r" \\[2pt]")
    gs_cells = []
    for _, r in df.iterrows():
        gs_cells.append(_fmt_coef(r["gap_speed_coef"], r["gap_speed_p"]))
    lines.append(rf"\hspace{{2em}}Waiting period $\times$ log(RT) & " + " & ".join(gs_cells) + r" \\[4pt]")

    # N = unique questions; Events = helping events
    lines.append(r"\midrule")
    n_cells = [f"{int(r.get('n_questions', r['n_rows'])):,}" for _, r in df.iterrows()]
    evt_cells = [f"{int(r['n_events']):,}" for _, r in df.iterrows()]
    lines.append(rf"N & " + " & ".join(n_cells) + r" \\")
    lines.append(rf"Events & " + " & ".join(evt_cells) + r" \\")

    lines += [
        r"\bottomrule",
        _standard_error_note(n_buckets + 1),
        r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\footnotesize $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$; $^{\dagger}p<0.1$} \\",
    ]
    lines += _tenure_table_postamble()
    return "\n".join(lines)


def generate_response_time_bins_table(df: pd.DataFrame) -> str:
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
    use_did = "did_hr" in df.columns and df["did_hr"].notna().any()
    hr_key, lo_key, hi_key, p_key = (
        ("did_hr", "did_ci_lo", "did_ci_hi", "did_p") if use_did
        else ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    )
    for _, r in df.iterrows():
        p = r.get(p_key, np.nan)
        stars = _sig_stars(p) if pd.notna(p) else ""
        n = int(r.get("n_questions", r.get("n_rows", 0)))
        events = int(r.get("n_events", 0))
        hr = r.get(hr_key, np.nan)
        lo, hi = r.get(lo_key, np.nan), r.get(hi_key, np.nan)
        hr_txt = f"{hr:.2f}{stars}" if pd.notna(hr) else "—"
        ci_txt = f"[{lo:.2f}, {hi:.2f}]" if pd.notna(lo) and pd.notna(hi) else "—"
        lines.append(
            rf"{_latex_bucket(str(r['bucket']))} & {hr_txt} "
            rf"& {ci_txt} "
            rf"& {n:,} & {events:,} \\"
        )
    detail_note = (
        r"\multicolumn{5}{@{}l}{\footnotesize HR is the summed DiD contrast (post-answer vs.\ pre-question, treated vs.\ control); comparable across bins.} \\"
        if use_did
        else r"\multicolumn{5}{@{}l}{\footnotesize HR is the post-answer increment only (is\_treated\_active); not comparable across bins when the waiting-period term is nonzero.} \\"
    )
    lines += [
        r"\bottomrule",
        _standard_error_note(5),
        r"\multicolumn{5}{@{}l}{\footnotesize Each row fits Model A to treated questions in that response-time bin plus the full no-answer control pool.} \\",
        detail_note,
        r"\multicolumn{5}{@{}l}{\footnotesize $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$; $^{\dagger}p<0.1$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_selection_bounds_table(df: pd.DataFrame) -> str:
    """Generate a compact table from selection_sensitivity.py outputs."""
    if df.empty or "adjusted_event_rate_rr_proxy" not in df.columns:
        return ""
    keep_fracs = [0.0, 0.10, 0.25, 0.50]
    df = df.loc[
        df["assumed_control_zero_post_event_fraction"].round(2).isin(keep_fracs)
    ].copy()
    if df.empty:
        return ""
    scope_order = {scope: i for i, scope in enumerate(["All"] + BUCKET_ORDER)}
    df["scope_order"] = df["scope"].map(scope_order).fillna(999)
    df = df.sort_values(["scope_order", "assumed_control_zero_post_event_fraction"])

    lines = [
        r"\begin{table}[H]",
        r"\caption{Selection Sensitivity Based on Zero Post-Question Helping}",
        r"\label{tab:selection_bounds}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Scope} & \textbf{Assumed frac.} & \textbf{Base HR} & \textbf{Control zero share} & \textbf{Adjusted RR proxy} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            rf"{_latex_bucket(str(r['scope']))} & {r['assumed_control_zero_post_event_fraction']:.2f} "
            rf"& {r['base_hr']:.2f} & {r['control_zero_post_help_share'] * 100:.1f}\% "
            rf"& {r['adjusted_event_rate_rr_proxy']:.2f} \\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{@{}l}{\footnotesize Assumed frac. is the fraction of zero-post-help control questions assigned one latent help event.} \\",
        r"\multicolumn{5}{@{}l}{\footnotesize Adjusted RR is an event-rate proxy, not a refitted Cox hazard ratio.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_pair_bootstrap_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for matched-pair bootstrap uncertainty."""
    if df.empty or "bootstrap_hr_ci_lo" not in df.columns:
        return ""
    lines = [
        r"\begin{table}[H]",
        r"\caption{Matched-Pair Bootstrap Uncertainty for the Summed DiD Treatment Effect (Model A)}",
        r"\label{tab:pair_bootstrap}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Scope} & \textbf{HR} & \textbf{Bootstrap 95\% CI} & \textbf{Replicates} & \textbf{N} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            rf"{_latex_bucket(str(r['scope']))} & {r['base_hr']:.2f} "
            rf"& [{r['bootstrap_hr_ci_lo']:.2f}, {r['bootstrap_hr_ci_hi']:.2f}] "
            rf"& {int(r['n_bootstrap_success'])}/{int(r['n_bootstrap_requested'])} "
            rf"& {int(r['n_questions']):,} \\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{@{}l}{\footnotesize Replicates resample whole matched pairs with replacement.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
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
    # Prefer the summed DiD hazard ratio (post-answer vs pre-question, treated vs
    # control); fall back to the legacy is_treated_active HR only if did_* is absent.
    use_did = "did_hr" in df.columns and df["did_hr"].notna().any()
    hr_col, lo_col, hi_col, p_col = (
        ("did_hr", "did_ci_lo", "did_ci_hi", "did_p") if use_did
        else ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    )
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

    # Null effect line and subtle band for reference
    y_min = min(0.92, ci_lo.min() - 0.02)
    y_max = max(1.35, ci_hi.max() + 0.08)
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
    # Prefer the summed DiD contrast so bins are comparable (the is_treated_active
    # increment alone is not, because the waiting-period term varies with response time).
    use_did = "did_hr" in df.columns and df["did_hr"].notna().any()
    hr_col, lo_col, hi_col, p_col = (
        ("did_hr", "did_ci_lo", "did_ci_hi", "did_p") if use_did
        else ("treat_hr", "treat_ci_lo", "treat_ci_hi", "treat_p")
    )
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
    selection_bounds_path = os.path.join(CACHE_DIR, "results_selection_bounds.csv")
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
    df_selection_bounds = pd.read_csv(selection_bounds_path) if os.path.exists(selection_bounds_path) else pd.DataFrame()
    df_pair_bootstrap = pd.read_csv(pair_bootstrap_path) if os.path.exists(pair_bootstrap_path) else pd.DataFrame()
    df_speed = pd.read_csv(speed_path) if os.path.exists(speed_path) else pd.DataFrame()
    descriptives = {}
    if os.path.exists(desc_path):
        with open(desc_path, "rb") as f:
            descriptives = pickle.load(f)
        print(f"Loaded descriptives from: {os.path.abspath(desc_path)}")
    else:
        print(f"WARNING: {os.path.abspath(desc_path)} not found; desc_stats table will have missing values.")

    # --- Generate Tables ---
    print("\n=== Generating LaTeX Tables ===")

    # Pooled regressions table (Main, Main+Speed)
    if not df_main_all.empty:
        tex = generate_regression_all_table(df_main_all, df_speed_all)
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
    tex = generate_main_results_table(df_main)
    out = os.path.join(TABLE_DIR, "main_results.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ Wrote {os.path.abspath(out)}")

    # Table 4: Speed interaction (by tenure bucket)
    if not df_speed.empty:
        tex = generate_speed_table(df_speed)
        out = os.path.join(TABLE_DIR, "speed_results.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # Non-parametric response-time-bin treatment effects
    if not df_rt_bins.empty:
        tex = generate_response_time_bins_table(df_rt_bins)
        out = os.path.join(TABLE_DIR, "response_time_bins.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    if not df_selection_bounds.empty:
        tex = generate_selection_bounds_table(df_selection_bounds)
        if tex:
            out = os.path.join(TABLE_DIR, "selection_bounds.tex")
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
        ("results_observable_controls.csv", "Observable Selection Controls (ISS-02)", "tab:observable_controls", "observable_controls.tex", "spec"),
        ("results_answer_quality.csv", "Answer-Quality Robustness (ISS-06)", "tab:answer_quality_robustness", "answer_quality_robustness.tex", "spec"),
        ("results_newcomer_robustness.csv", "Newcomer Bucket Robustness ($<$ 1 Week)", "tab:newcomer_robustness", "newcomer_robustness.tex", "model"),
        ("results_cohort_robustness.csv", "Cohort Heterogeneity Robustness (ISS-10)", "tab:cohort_robustness", "cohort_robustness.tex", "model"),
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

    placebo_path = os.path.join(CACHE_DIR, "results_viewcount_placebo.csv")
    if os.path.exists(placebo_path):
        df_placebo = pd.read_csv(placebo_path)
        tex = generate_viewcount_placebo_table(df_placebo)
        if tex:
            out = os.path.join(TABLE_DIR, "viewcount_placebo.tex")
            with open(out, "w") as f:
                f.write(tex)
            print(f"✓ {out}")

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
