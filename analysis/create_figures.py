"""
generate_outputs.py
====================
Loads cached model results and descriptive statistics produced by fit_models.py,
then generates all LaTeX tables and figures for the paper's Results section.

Outputs (written to output_tables/ and output_figures/):
  - desc_stats.tex          Descriptive statistics (Table 2)
  - main_results.tex        Cox regression: main reciprocity effect (Table 3)
  - speed_results.tex       Cox regression: response-time moderation (Table 4)
  - strength_rec.eps/.png   Reciprocity HR across tenure buckets (Figure 2)
  - speed_moderation.eps/.png  Speed interaction across buckets (Figure 3)

Usage:
    python generate_outputs.py
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
# Table 3: Main Cox Regression Results
# =====================================================================

def generate_main_results_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table for the main Cox model (Model A) across tenure buckets.
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

    lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post & " + " & ".join(cells_coef) + r" \\")
    lines.append(rf" & " + " & ".join(cells_se) + r" \\")
    lines.append(rf"\hspace{{1em}}\textit{{Hazard Ratio [95\% CI]}} & " + " & ".join(cells_hr) + r" \\[4pt]")

    # Gap placebo
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Gap Period (Placebo)}} \\")
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
# Table 4: Response-Time Moderation
# =====================================================================

def generate_speed_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table for Model B (speed interaction) across tenure buckets.
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
    lines.append(rf"\hspace{{1em}}Received Answer $\times$ Post & " + " & ".join(cells) + r" \\")
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

    # Gap × speed (placebo for speed)
    lines.append(r"\multicolumn{" + str(n_buckets + 1) + r"}{@{}l}{\textit{Gap Period $\times$ Response Time}} \\")
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
    Bar chart of the hazard ratio for isTreatedActive across tenure buckets,
    with 95% CI error bars.
    """
    df = df.set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    df = df.dropna(subset=["treat_hr"])

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(df))
    hrs = df["treat_hr"].values
    ci_lo = df["treat_ci_lo"].values
    ci_hi = df["treat_ci_hi"].values

    err_lo = hrs - ci_lo
    err_hi = ci_hi - hrs

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(df)))

    bars = ax.bar(
        x, hrs, width=0.65, color=colors,
        yerr=[err_lo, err_hi], capsize=4,
        edgecolor="white", linewidth=0.5,
        error_kw={"linewidth": 1.2, "color": "#333333"},
    )

    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, label="No effect (HR=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_SHORT, fontsize=9)
    ax.set_xlabel("User Tenure at Time of Question", fontsize=10)
    ax.set_ylabel("Hazard Ratio (Receiving Answer → Helping)", fontsize=10)
    ax.set_title("Strength of Generalized Reciprocity\nAcross User Experience", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate significance
    for i, (_, r) in enumerate(df.iterrows()):
        p = r["treat_p"]
        stars = _sig_stars(p).replace("\\textdagger", "†")
        if stars:
            ax.text(i, ci_hi[i] + 0.01, stars, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(FIGURE_DIR, f"strength_rec.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved strength_rec.[eps/png/pdf]")


# =====================================================================
# Figure 2: Speed Moderation Across Experience
# =====================================================================

def generate_speed_figure(df: pd.DataFrame):
    """
    Bar chart of the speed interaction coefficient across tenure buckets,
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
# Main
# =====================================================================

def main():
    _ensure_dirs()

    # --- Load cached results ---
    main_path = os.path.join(CACHE_DIR, "results_main.csv")
    speed_path = os.path.join(CACHE_DIR, "results_speed.csv")
    desc_path = os.path.join(CACHE_DIR, "descriptives.pkl")

    if not os.path.exists(main_path):
        print(f"ERROR: {main_path} not found. Run fit_models.py first.")
        return

    df_main = pd.read_csv(main_path)
    df_speed = pd.read_csv(speed_path) if os.path.exists(speed_path) else pd.DataFrame()
    descriptives = {}
    if os.path.exists(desc_path):
        with open(desc_path, "rb") as f:
            descriptives = pickle.load(f)

    # --- Generate Tables ---
    print("\n=== Generating LaTeX Tables ===")

    # Table 1: Descriptive Statistics
    tex = generate_desc_stats_table(descriptives)
    out = os.path.join(TABLE_DIR, "desc_stats.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ {out}")

    # Table 3: Main Results
    tex = generate_main_results_table(df_main)
    out = os.path.join(TABLE_DIR, "main_results.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"✓ {out}")

    # Table 4: Speed Moderation
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

    print("\n=== All outputs generated ===")
    print(f"Tables: {TABLE_DIR}/")
    print(f"Figures: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()