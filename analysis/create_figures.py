"""
create_figures.py
==================
Loads cached model results and descriptive statistics produced by fit_cox_models.py
(from model_cache/*.csv and model_cache/descriptives.pkl), then generates LaTeX tables
and figures for the paper's Results section.

Outputs (written to output_tables/ and output_figures/):
  - desc_stats.tex              Descriptive statistics (Table 2)
  - regression_all.tex          Pooled Cox regressions (Main, Main+Speed, Main+Speed+Quad.)
  - main_results.tex            Main effect by tenure bucket (Table 3)
  - speed_results.tex           Speed interaction by tenure bucket (Table 4)
  - strength_rec.*              Reciprocity HR by tenure bucket (Figure 2)
  - speed_moderation.*           Speed interaction by tenure bucket (Figure 3)
  - interaction_effect.*        Predicted treatment effect by response time (pooled Model C/B)

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

# =====================================================================
# Configuration
# =====================================================================
CACHE_DIR = "model_cache"
TABLE_DIR = "output_tables"
FIGURE_DIR = "output_figures"

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


# =====================================================================
# Table 1: Descriptive Statistics
# =====================================================================

def generate_desc_stats_table(desc: dict) -> str:
    """Generate LaTeX for the descriptive statistics table."""

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

    # Tenure bucket breakdown
    bucket_counts = desc.get("tenure_bucket_counts", {})
    if bucket_counts:
        lines.append(r"\multicolumn{5}{@{}l}{\textit{Observations by Tenure Bucket}} \\")
        for b in BUCKET_ORDER:
            ct = bucket_counts.get(b, 0)
            lines.append(rf"\hspace{{1em}} {b} & & & & {_f(ct)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =====================================================================
# Table: Pooled regressions (regression_all.tex) — Main, Main+Speed, Main+Speed+Quadratic
# =====================================================================

def generate_regression_all_table(
    df_main_all: pd.DataFrame,
    df_speed_all: pd.DataFrame = None,
    df_model_c_all: pd.DataFrame = None,
) -> str:
    """
    Generate LaTeX table for pooled Cox regressions (all experience levels).
    Columns: Main; Main + Speed (linear RT interaction); Main + Speed + Quadratic (Model C).
    """
    if df_main_all.empty or "treat_coef" not in df_main_all.columns:
        return ""
    r = df_main_all.iloc[0]
    has_speed = (
        df_speed_all is not None
        and not df_speed_all.empty
        and "speed_coef" in df_speed_all.columns
    )
    has_model_c = (
        df_model_c_all is not None
        and not df_model_c_all.empty
        and "speed_coef" in df_model_c_all.columns
    )
    s = df_speed_all.iloc[0] if has_speed else None
    c = df_model_c_all.iloc[0] if has_model_c else None

    n_cols = 1 + int(has_speed) + int(has_model_c)
    col_spec = "@{}l" + "c" * n_cols + "@{}"
    header_cells = [r"\textbf{Main}"]
    if has_speed:
        header_cells.append(r"\textbf{Main + Speed}")
    if has_model_c:
        header_cells.append(r"\textbf{Main + Speed + Quad.}")
    header = " & ".join(header_cells) + r" \\"

    lines = [
        r"\begin{table}[H]",
        r"\caption{Pooled Cox Regressions (All Experience Levels)}",
        r"\label{tab:regression_all}",
        r"\centering",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & " + header,
        r"\midrule",
        rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Treatment Effect (DID)}}}} \\",
    ]
    # Treatment coef row
    row = rf"\hspace{{1em}} Received Answer $\times$ Post-Answer Received & {_fmt_coef(r['treat_coef'], r['treat_p'])}"
    if has_speed:
        row += rf" & {_fmt_coef(s['treat_coef'], s['treat_p'])}"
    if has_model_c:
        row += rf" & {_fmt_coef(c['treat_coef'], c['treat_p'])}"
    lines.append(row + r" \\")
    # SE row
    row = rf" & {_fmt_se(r['treat_se'])}"
    if has_speed:
        row += rf" & {_fmt_se(s['treat_se'])}"
    if has_model_c:
        row += rf" & {_fmt_se(c['treat_se'])}"
    lines.append(row + r" \\")
    # HR [95% CI] row
    row = rf"\hspace{{1em}} Hazard Ratio [95\% CI] & [{r['treat_ci_lo']:.2f}, {r['treat_ci_hi']:.2f}]"
    if has_speed:
        row += r" & —"
    if has_model_c:
        row += rf" & [{c['treat_ci_lo']:.2f}, {c['treat_ci_hi']:.2f}]"
    lines.append(row + r" \\[4pt]")

    lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Waiting Period}}}} \\")
    gap_coef = r.get("gap_coef", np.nan)
    gap_p = r.get("gap_p", np.nan)
    waiting_cell = _fmt_coef(gap_coef, gap_p) if pd.notna(gap_coef) else "—"
    row = rf"\hspace{{1em}} Received Answer $\times$ Post-Question & {waiting_cell}"
    if has_speed:
        row += r" & —"
    if has_model_c:
        row += r" & —"
    lines.append(row + r" \\[4pt]")

    if has_speed or has_model_c:
        lines.append(rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\textit{{Response Time Interaction}}}} \\")
        row = r"\hspace{1em} Treatment $\times$ log(Response Time) & —"
        if has_speed:
            row += rf" & {_fmt_coef(s['speed_coef'], s['speed_p'])}"
        else:
            row += r" & —"
        if has_model_c:
            row += rf" & {_fmt_coef(c['speed_coef'], c['speed_p'])}"
        lines.append(row + r" \\")
        row = r" & —"
        if has_speed:
            row += rf" & {_fmt_se(s['speed_se'])}"
        if has_model_c:
            row += rf" & {_fmt_se(c['speed_se'])}"
        lines.append(row + (r" \\[4pt]" if not (has_model_c and "speed_sq_coef" in c) else r" \\"))
        if has_model_c and "speed_sq_coef" in c:
            row = r"\hspace{1em} Treatment $\times$ [log(Response Time)]$^2$ & —"
            if has_speed:
                row += r" & —"
            row += rf" & {_fmt_coef(c['speed_sq_coef'], c['speed_sq_p'])}"
            lines.append(row + r" \\")
            row = r" & —"
            if has_speed:
                row += r" & —"
            row += rf" & {_fmt_se(c.get('speed_sq_se', np.nan))}"
            lines.append(row + r" \\[4pt]")

    n_rows = int(r["n_rows"])
    n_events = int(r["n_events"])
    lines.append(r"\midrule")
    row = rf"Intervals & {n_rows:,}"
    for _ in range(n_cols - 1):
        row += rf" & {n_rows:,}"
    lines.append(row + r" \\")
    row = rf"Events & {n_events:,}"
    for _ in range(n_cols - 1):
        row += rf" & {n_events:,}"
    lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        rf"\multicolumn{{{n_cols + 1}}}{{@{{}}l}}{{\footnotesize $^{{***}}p<0.001$; $^{{**}}p<0.01$; $^{{*}}p<0.05$; $^{{\dagger}}p<0.1$}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =====================================================================
# Table 3: Main effect (by tenure bucket)
# =====================================================================

def generate_main_results_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table for the main effect (Model A) by tenure bucket.
    Each column is one tenure bucket.
    """
    # Ensure correct order
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["treat_coef"])

    n_buckets = len(df)

    # Header
    col_spec = "@{}l" + "c" * n_buckets + "@{}"
    header_labels = " & ".join(df["bucket"].tolist())

    lines = [
        r"\begin{table}[H]",
        r"\caption{Cox Regression Results: Effect of Receiving an Answer on Helping Hazard}",
        r"\label{tab:main_results}",
        r"\centering",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
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

    # Rows for key coefficients
    # isTreatedActive (main DID treatment)
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Treatment Effect (DID)}} \\")

    # is_treated_active
    cells_coef = []
    cells_se = []
    cells_hr = []
    for _, r in df.iterrows():
        c = r["treat_coef"]
        se = r["treat_se"]
        p = r["treat_p"]
        hr = r["treat_hr"]
        cells_coef.append(_fmt_coef(c, p))
        cells_se.append(_fmt_se(se))
        cells_hr.append(f"[{r['treat_ci_lo']:.2f}, {r['treat_ci_hi']:.2f}]")

    lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post-Answer Received & " + " & ".join(cells_coef) + r" \\")
    lines.append(rf" & " + " & ".join(cells_se) + r" \\")
    lines.append(rf"\hspace{{1em}}\textit{{Hazard Ratio [95\% CI]}} & " + " & ".join(cells_hr) + r" \\[4pt]")

    # Waiting period
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Waiting Period}} \\")
    gap_cells = []
    gap_se_cells = []
    for _, r in df.iterrows():
        gc = r.get("gap_coef", np.nan)
        gp = r.get("gap_p", np.nan)
        if pd.notna(gc):
            gap_cells.append(_fmt_coef(gc, gp))
            # Approximate SE from coef and HR
            gap_se_cells.append("")
        else:
            gap_cells.append("—")
            gap_se_cells.append("")

    lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post-Question & " + " & ".join(gap_cells) + r" \\[4pt]")

    # Observations
    lines.append(r"\midrule")
    obs_cells = [f"{int(r['n_rows']):,}" for _, r in df.iterrows()]
    evt_cells = [f"{int(r['n_events']):,}" for _, r in df.iterrows()]
    lines.append(rf"Intervals & " + " & ".join(obs_cells) + r" \\")
    lines.append(rf"Events & " + " & ".join(evt_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\footnotesize $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$; $^{\dagger}p<0.1$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
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
    col_spec = "@{}l" + "c" * n_buckets + "@{}"
    header_labels = " & ".join(df["bucket"].tolist())

    lines = [
        r"\begin{table}[H]",
        r"\caption{Response Time Moderation of the Reciprocity Effect}",
        r"\label{tab:speed_results}",
        r"\centering",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {header_labels} \\",
        r"\midrule",
    ]

    # Base treatment effect
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Base Treatment Effect}} \\")
    cells = []
    se_cells = []
    for _, r in df.iterrows():
        cells.append(_fmt_coef(r["treat_coef"], r["treat_p"]))
        se_cells.append(_fmt_se(r["treat_se"]))
    lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post-Answer Received & " + " & ".join(cells) + r" \\")
    lines.append(rf" & " + " & ".join(se_cells) + r" \\[4pt]")

    # Speed interaction
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Response Time Interaction}} \\")
    speed_cells = []
    speed_se_cells = []
    for _, r in df.iterrows():
        speed_cells.append(_fmt_coef(r["speed_coef"], r["speed_p"]))
        speed_se_cells.append(_fmt_se(r["speed_se"]))
    lines.append(rf"\hspace{{1em}}Treatment $\times$ log(Response Time) & " + " & ".join(speed_cells) + r" \\")
    lines.append(rf" & " + " & ".join(speed_se_cells) + r" \\[4pt]")

    # Waiting period × speed
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Waiting Period $\times$ Response Time}} \\")
    gs_cells = []
    for _, r in df.iterrows():
        gs_cells.append(_fmt_coef(r["gap_speed_coef"], r["gap_speed_p"]))
    lines.append(rf"\hspace{{1em}}Post-Question $\times$ log(Response Time) & " + " & ".join(gs_cells) + r" \\[4pt]")

    # Observations
    lines.append(r"\midrule")
    obs_cells = [f"{int(r['n_rows']):,}" for _, r in df.iterrows()]
    evt_cells = [f"{int(r['n_events']):,}" for _, r in df.iterrows()]
    lines.append(rf"Intervals & " + " & ".join(obs_cells) + r" \\")
    lines.append(rf"Events & " + " & ".join(evt_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\footnotesize $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$; $^{\dagger}p<0.1$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =====================================================================
# Figure 1: Reciprocity Strength Across Experience
# =====================================================================

def generate_reciprocity_figure(df: pd.DataFrame):
    """
    Bar chart of the hazard ratio for isTreatedActive by tenure bucket,
    with 95% CI error bars.
    """
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["treat_hr"])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    x = np.arange(len(df))
    hrs = df["treat_hr"].values
    ci_lo = df["treat_ci_lo"].values
    ci_hi = df["treat_ci_hi"].values

    err_lo = hrs - ci_lo
    err_hi = ci_hi - hrs

    # Single-hue gradient: deep blue to teal for clarity and print-friendly contrast
    n_bars = len(df)
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, n_bars))  # distinct from default Blues

    bar_width = 0.62
    bars = ax.bar(
        x, hrs, width=bar_width, color=colors,
        yerr=[err_lo, err_hi], capsize=5, error_kw={"linewidth": 1.5, "color": "#2d2d2d", "capthick": 1.2},
        edgecolor="white", linewidth=1.0,
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
    ax.set_title("Strength of generalized reciprocity across user experience", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    # Value labels on top of bars
    for i, (hr, hi) in enumerate(zip(hrs, ci_hi)):
        ax.text(i, hi + 0.015, f"{hr:.2f}", ha="center", va="bottom", fontsize=8, fontweight="500", color="#333333")

    # Significance stars above value labels
    for i, (_, r) in enumerate(df.iterrows()):
        p = r["treat_p"]
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
    Bar chart of the speed interaction coefficient by tenure bucket,
    with SE-based error bars.
    """
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["speed_coef"])

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(df))
    coefs = df["speed_coef"].values
    ses = df["speed_se"].values

    # Color bars by sign
    colors = ["#d9534f" if c < 0 else "#5cb85c" for c in coefs]
    # Override: use a gradient for visual clarity
    colors_pos = plt.cm.Greens(np.linspace(0.4, 0.8, len(df)))
    colors_neg = plt.cm.Reds(np.linspace(0.4, 0.8, len(df)))
    bar_colors = [colors_pos[i] if c >= 0 else colors_neg[i] for i, c in enumerate(coefs)]

    bars = ax.bar(
        x, coefs, width=0.65, color=bar_colors,
        yerr=1.96 * ses, capsize=4,
        edgecolor="white", linewidth=0.5,
        error_kw={"linewidth": 1.2, "color": "#333333"},
    )

    ax.axhline(y=0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_SHORT, fontsize=9)
    ax.set_xlabel("User Tenure at Time of Question", fontsize=10)
    ax.set_ylabel("Coefficient: Treatment × log(Response Time)", fontsize=10)
    ax.set_title("Response Time Moderation of Reciprocity\nAcross User Experience", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate significance
    for i, (_, r) in enumerate(df.iterrows()):
        p = r["speed_p"]
        stars = _sig_stars(p).replace("\\textdagger", "†")
        if stars:
            y_pos = coefs[i] + 1.96 * ses[i] + 0.002 if coefs[i] >= 0 else coefs[i] - 1.96 * ses[i] - 0.002
            va = "bottom" if coefs[i] >= 0 else "top"
            ax.text(i, y_pos, stars, ha="center", va=va, fontsize=9)

    # Add interpretive annotation
    ax.annotate(
        "Positive = longer wait\nincreases reciprocity",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=7, fontstyle="italic",
        color="#555555",
    )

    plt.tight_layout()

    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"speed_moderation.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved speed_moderation.[eps/png/pdf]")


# =====================================================================
# Combined Figure: Reciprocity + Speed side by side
# =====================================================================

def generate_combined_figure(df_main: pd.DataFrame, df_speed: pd.DataFrame):
    """Side-by-side panels for the paper."""
    df_m = df_main.set_index("bucket").reindex(BUCKET_ORDER).reset_index().dropna(subset=["treat_hr"])
    df_s = df_speed.set_index("bucket").reindex(BUCKET_ORDER).reset_index().dropna(subset=["speed_coef"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: Reciprocity HR ---
    x = np.arange(len(df_m))
    hrs = df_m["treat_hr"].values
    ci_lo = df_m["treat_ci_lo"].values
    ci_hi = df_m["treat_ci_hi"].values

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(df_m)))
    ax1.bar(x, hrs, width=0.65, color=colors,
            yerr=[hrs - ci_lo, ci_hi - hrs], capsize=4,
            edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 1.2, "color": "#333333"})
    ax1.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(BUCKET_SHORT, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Hazard Ratio", fontsize=10)
    ax1.set_title("(A) Reciprocity Effect", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, (_, r) in enumerate(df_m.iterrows()):
        stars = _sig_stars(r["treat_p"]).replace("\\textdagger", "†")
        if stars:
            ax1.text(i, ci_hi[i] + 0.005, stars, ha="center", va="bottom", fontsize=8)

    # --- Panel B: Speed interaction ---
    x2 = np.arange(len(df_s))
    coefs = df_s["speed_coef"].values
    ses = df_s["speed_se"].values
    bar_colors = ["#5cb85c" if c >= 0 else "#d9534f" for c in coefs]

    ax2.bar(x2, coefs, width=0.65, color=bar_colors,
            yerr=1.96 * ses, capsize=4,
            edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 1.2, "color": "#333333"})
    ax2.axhline(y=0, color="#999999", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(BUCKET_SHORT, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Interaction Coefficient", fontsize=10)
    ax2.set_title("(B) Response Time Moderation", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i, (_, r) in enumerate(df_s.iterrows()):
        stars = _sig_stars(r["speed_p"]).replace("\\textdagger", "†")
        if stars:
            y = coefs[i] + 1.96 * ses[i] + 0.001 if coefs[i] >= 0 else coefs[i] - 1.96 * ses[i] - 0.001
            va = "bottom" if coefs[i] >= 0 else "top"
            ax2.text(i, y, stars, ha="center", va=va, fontsize=8)

    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"combined_results.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved combined_results.[eps/png/pdf]")


# =====================================================================
# Figure: Interaction effect — treatment effect by response time bin (or parametric fallback)
# =====================================================================

def generate_interaction_effect_figure(
    df_rt_bins: pd.DataFrame = None,
    df_model_c_all: pd.DataFrame = None,
    df_speed_all: pd.DataFrame = None,
):
    """
    Plot treatment effect (hazard ratio) by response time.
    If df_rt_bins is provided and non-empty, plot a bar chart by bin (non-parametric).
    Otherwise plot the parametric curve from Model C or Model B.
    """
    if df_rt_bins is not None and not df_rt_bins.empty and "treat_hr" in df_rt_bins.columns:
        _plot_interaction_effect_bins(df_rt_bins)
        return
    # Fallback: parametric curve
    if df_model_c_all is not None and not df_model_c_all.empty and "speed_sq_coef" in df_model_c_all.columns:
        row = df_model_c_all.iloc[0]
        treat = row["treat_coef"]
        speed = row["speed_coef"]
        speed_sq = row["speed_sq_coef"]
        use_quadratic = True
        title_suffix = " (Model C: linear + quadratic)"
    elif df_speed_all is not None and not df_speed_all.empty and "speed_coef" in df_speed_all.columns:
        row = df_speed_all.iloc[0]
        treat = row["treat_coef"]
        speed = row["speed_coef"]
        speed_sq = 0.0
        use_quadratic = False
        title_suffix = " (Model B: linear)"
    else:
        print("  ⚠ Skipping interaction_effect figure: no response-time bin results or Model C/B.")
        return

    rt_hours = np.linspace(0.5, 720, 400)
    log_rt = np.log1p(rt_hours)
    log_hr = treat + speed * log_rt + (speed_sq * (log_rt ** 2) if use_quadratic else 0.0)
    hr = np.exp(log_hr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.plot(rt_hours, hr, color="#2e7d32", linewidth=2.5, label="Predicted hazard ratio")
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.0, label="No effect (HR = 1)")
    ax.fill_between(rt_hours, 1.0, hr, where=(hr >= 1.0), alpha=0.15, color="#2e7d32")
    ax.fill_between(rt_hours, hr, 1.0, where=(hr < 1.0), alpha=0.15, color="#c62828")
    ax.set_xlabel("Response time (hours from question to answer)", fontsize=11)
    ax.set_ylabel("Predicted hazard ratio\n(receiving answer → helping)", fontsize=11)
    ax.set_title("Treatment effect by response time" + title_suffix, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, rt_hours.max())
    ax.set_ylim(min(0.92, hr.min() - 0.02), max(1.25, hr.max() + 0.03))
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="best", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"interaction_effect.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved interaction_effect.[eps/png/pdf]")


def _plot_interaction_effect_bins(df: pd.DataFrame):
    """Points and line of treatment effect (HR) by response time bin."""
    labels = df["bucket"].tolist()
    hrs = df["treat_hr"].values
    ci_lo = df["treat_ci_lo"].values
    ci_hi = df["treat_ci_hi"].values
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
        color="#2e7d32",
        linewidth=2,
        markersize=8,
        capsize=4,
        capthick=1.2,
        ecolor="#2d2d2d",
        elinewidth=1.2,
    )
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.2, label="No effect (HR = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
    ax.set_xlabel("Response time (question to answer)", fontsize=11)
    ax.set_ylabel("Hazard ratio (receiving answer → helping)", fontsize=11)
    ax.set_title("Treatment effect by response time bin", fontsize=12, fontweight="bold", pad=10)
    y_min = min(0.92, ci_lo.min() - 0.02)
    y_max = max(1.2, ci_hi.max() + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (_, r) in enumerate(df.iterrows()):
        stars = _sig_stars(r["treat_p"]).replace("\\textdagger", "†")
        if stars:
            ax.text(i, ci_hi[i] + 0.01, stars, ha="center", va="bottom", fontsize=9)
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
    model_c_all_path = os.path.join(CACHE_DIR, "results_model_c_all.csv")
    rt_bins_path = os.path.join(CACHE_DIR, "results_response_time_bins.csv")
    speed_path = os.path.join(CACHE_DIR, "results_speed.csv")
    desc_path = os.path.join(CACHE_DIR, "descriptives.pkl")

    if not os.path.exists(main_path):
        print(f"ERROR: {main_path} not found. Run fit_cox_models.py first.")
        return

    df_main = pd.read_csv(main_path)
    df_main_all = pd.read_csv(main_all_path) if os.path.exists(main_all_path) else pd.DataFrame()
    df_speed_all = pd.read_csv(speed_all_path) if os.path.exists(speed_all_path) else pd.DataFrame()
    df_model_c_all = pd.read_csv(model_c_all_path) if os.path.exists(model_c_all_path) else pd.DataFrame()
    df_rt_bins = pd.read_csv(rt_bins_path) if os.path.exists(rt_bins_path) else pd.DataFrame()
    df_speed = pd.read_csv(speed_path) if os.path.exists(speed_path) else pd.DataFrame()
    descriptives = {}
    if os.path.exists(desc_path):
        with open(desc_path, "rb") as f:
            descriptives = pickle.load(f)

    # --- Generate Tables ---
    print("\n=== Generating LaTeX Tables ===")

    # Pooled regressions table (Main, Main+Speed, Main+Speed+Quadratic)
    if not df_main_all.empty:
        tex = generate_regression_all_table(df_main_all, df_speed_all, df_model_c_all)
        out = os.path.join(TABLE_DIR, "regression_all.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # Table 1: Descriptive Statistics
    tex = generate_desc_stats_table(descriptives)
    out = os.path.join(TABLE_DIR, "desc_stats.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ {out}")

    # Table 3: Main effect (by tenure bucket)
    tex = generate_main_results_table(df_main)
    out = os.path.join(TABLE_DIR, "main_results.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ {out}")

    # Table 4: Speed interaction (by tenure bucket)
    if not df_speed.empty:
        tex = generate_speed_table(df_speed)
        out = os.path.join(TABLE_DIR, "speed_results.tex")
        with open(out, "w") as f:
            f.write(tex)
        print(f"✓ {out}")

    # --- Generate Figures ---
    print("\n=== Generating Figures ===")

    generate_reciprocity_figure(df_main)

    if not df_speed.empty:
        generate_speed_figure(df_speed)
        generate_combined_figure(df_main, df_speed)

    generate_interaction_effect_figure(df_rt_bins, df_model_c_all, df_speed_all)

    print("\n=== All outputs generated ===")
    print(f"Tables: {TABLE_DIR}/")
    print(f"Figures: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()