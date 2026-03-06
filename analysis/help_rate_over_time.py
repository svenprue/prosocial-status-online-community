"""
help_rate_over_time.py
======================
Plots empirical help rate over the study window (question-relative time),
separately for treated and control, by tenure bucket.

For each tenure bucket (or key ones: newcomers vs experienced):
  - X-axis: time relative to question (day 0 = question posted).
  - Y-axis: help rate (answers per user per hour, optionally normalized to pre-question baseline).
  - Vertical line at TQ (question time = 0).
  - For treated: vertical line at median (or mean) answer arrival time (TAT_A).
  - Binned rates plotted as points connected by lines (no smoothing).
  - 95% CI error bars from Poisson SE: SE(rate) = sqrt(rate / exposure).

Outputs:
  - help_rate_2panel.eps/.png/.pdf   — 2×1 layout: newcomers (<1 week) vs experienced (pooled).
  - help_rate_all_buckets.eps/.png/.pdf — Small multiples for all tenure buckets (optional).

Usage:
    python help_rate_over_time.py [--input ../data/event_history] [--sample 200000]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# Configuration
# =====================================================================
FIGURE_DIR = "output_figures"
BUCKET_ORDER = [
    "< 1 Week", "1 Week - 1 Month", "1 - 6 Months",
    "6 - 12 Months", "1 - 3 Years", "3 - 6 Years", "> 6 Years",
]
BUCKET_SHORT = ["<1W", "1W–1M", "1–6M", "6–12M", "1–3Y", "3–6Y", ">6Y"]

# Newcomers = first bucket; Experienced = rest pooled
NEWCOMER_BUCKET = "< 1 Week"
EXPERIENCED_LABEL = "Experienced (1 Week – 6+ Years)"

# Default window: -2 to +2 days relative to question
WINDOW_DAYS = 2  # plot and bin from -WINDOW_DAYS to +WINDOW_DAYS


def create_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 7, 30, 180, 365, 1095, 2190, np.inf]
    df["tenure_bucket"] = pd.cut(
        df["user_tenure_days"], bins=bins, labels=BUCKET_ORDER, right=True
    )
    return df


def load_timelines_and_events(input_folder: str, sample_size: int = None):
    """Load study_timelines and study_events; optionally subsample."""
    timelines_path = os.path.join(input_folder, "study_timelines.parquet")
    events_path = os.path.join(input_folder, "study_events.parquet")

    if not os.path.exists(timelines_path) or not os.path.exists(events_path):
        raise FileNotFoundError(
            f"Event history not found. Expected:\n  {timelines_path}\n  {events_path}\n"
            "Run preprocessing/create_matched_event_histories.py first."
        )

    timelines = pd.read_parquet(timelines_path)
    events = pd.read_parquet(events_path)

    for col in ["t_start", "t_question", "t_answer", "t_end", "user_tenure_days"]:
        if col in timelines.columns:
            timelines[col] = pd.to_numeric(timelines[col], errors="coerce")
    if "t_event" in events.columns:
        events["t_event"] = pd.to_numeric(events["t_event"], errors="coerce")

    timelines = create_tenure_buckets(timelines)
    timelines["hasAnswer"] = timelines["hasAnswer"].astype(int)

    if sample_size and sample_size < timelines["match_id"].nunique():
        ids = np.random.RandomState(42).choice(
            timelines["match_id"].unique(), sample_size, replace=False
        )
        timelines = timelines[timelines["match_id"].isin(ids)].copy()
        events = events[events["match_id"].isin(ids)].copy()

    return timelines, events


def compute_binned_rates(
    timelines: pd.DataFrame,
    events: pd.DataFrame,
    bin_width_hours: float = 4.0,
    time_min_hours: float = None,
    time_max_hours: float = None,
) -> pd.DataFrame:
    """
    For each (tenure_bucket, hasAnswer), compute help rate per bin.
    Rate = (count of events in bin) / (n_observations * bin_width_hours) = answers per user per hour.

    Returns DataFrame with columns:
      tenure_bucket, hasAnswer, time_center_hours, time_center_days,
      n_obs, event_count, rate_raw.
    """
    ev = events.merge(
        timelines[["match_id", "question_id", "hasAnswer", "tenure_bucket"]],
        on=["match_id", "question_id"],
        how="inner",
    )

    if time_min_hours is None:
        time_min_hours = float(timelines["t_start"].min())
    if time_max_hours is None:
        time_max_hours = float(timelines["t_end"].max())

    # Only count events inside the window (avoid assigning out-of-range to edge bins)
    ev = ev[(ev["t_event"] >= time_min_hours) & (ev["t_event"] <= time_max_hours)]

    # Build bin edges so that 0 (question time) is always a boundary: no bin spans pre and post
    left_edges = np.arange(time_min_hours, 0, bin_width_hours)
    right_edges = np.arange(0, time_max_hours + bin_width_hours * 0.5, bin_width_hours)
    bin_edges = np.union1d(left_edges, right_edges)
    bin_edges = np.sort(bin_edges)
    time_centers_h = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths_h = np.diff(bin_edges)  # per-bin width (may differ at boundaries)
    time_centers_d = time_centers_h / 24.0

    ev["bin_idx"] = np.searchsorted(bin_edges, ev["t_event"].values, side="right") - 1
    ev["bin_idx"] = ev["bin_idx"].clip(0, len(bin_edges) - 2)

    obs_per_group = (
        timelines.groupby(["tenure_bucket", "hasAnswer"]).size().reset_index(name="n_obs")
    )
    counts = (
        ev.groupby(["tenure_bucket", "hasAnswer", "bin_idx"])
        .agg(event_count=("t_event", "count"))
        .reset_index()
    )

    # Full grid: every (tenure_bucket, hasAnswer, bin)
    full_rows = []
    for _, row in obs_per_group.iterrows():
        for bi in range(len(time_centers_h)):
            full_rows.append({
                "tenure_bucket": row["tenure_bucket"],
                "hasAnswer": row["hasAnswer"],
                "bin_idx": bi,
                "n_obs": row["n_obs"],
            })
    full_df = pd.DataFrame(full_rows)
    full_df["time_center_hours"] = full_df["bin_idx"].map(lambda i: time_centers_h[i])
    full_df["time_center_days"] = full_df["time_center_hours"] / 24.0
    full_df["bin_width_h"] = full_df["bin_idx"].map(lambda i: bin_widths_h[i] if i < len(bin_widths_h) else bin_width_hours)

    full_df = full_df.merge(
        counts,
        on=["tenure_bucket", "hasAnswer", "bin_idx"],
        how="left",
    )
    full_df["event_count"] = full_df["event_count"].fillna(0)
    exposure = full_df["n_obs"] * full_df["bin_width_h"]
    full_df["rate_raw"] = full_df["event_count"] / exposure
    # Poisson SE: Var(count)=count, so SE(rate) = sqrt(count)/exposure = sqrt(rate/exposure)
    full_df["rate_raw_se"] = np.sqrt(full_df["rate_raw"] / exposure)
    full_df.loc[exposure <= 0, "rate_raw_se"] = 0
    return full_df.drop(columns=["bin_idx", "bin_width_h"], errors="ignore")


def normalize_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add column rate_norm = rate_raw / pre_question_mean (per group), with SE propagation."""
    df = df.copy()
    pre = df[df["time_center_hours"] < 0].groupby(["tenure_bucket", "hasAnswer"])["rate_raw"].mean().reset_index()
    pre = pre.rename(columns={"rate_raw": "baseline"})
    df = df.merge(pre, on=["tenure_bucket", "hasAnswer"], how="left")
    df["baseline"] = df["baseline"].replace(0, np.nan)
    df["rate_norm"] = df["rate_raw"] / df["baseline"]
    df["rate_norm"] = df["rate_norm"].fillna(1.0)
    # SE(rate_norm) ≈ rate_norm * (SE(rate_raw)/rate_raw) when rate_raw > 0 (treat baseline as fixed)
    df["rate_norm_se"] = np.where(
        df["rate_raw"] > 0,
        df["rate_norm"] * (df["rate_raw_se"] / df["rate_raw"]),
        0.0,
    )
    return df


def median_answer_time_hours(timelines: pd.DataFrame, tenure_bucket: str) -> float:
    """Median t_answer (hours from question) for treated in this bucket."""
    sub = timelines[(timelines["tenure_bucket"] == tenure_bucket) & (timelines["hasAnswer"] == 1)]
    if sub.empty or sub["t_answer"].isna().all():
        return np.nan
    return float(sub["t_answer"].median())


def mean_answer_time_hours(timelines: pd.DataFrame, tenure_bucket: str) -> float:
    """Mean t_answer for treated in this bucket."""
    sub = timelines[(timelines["tenure_bucket"] == tenure_bucket) & (timelines["hasAnswer"] == 1)]
    if sub.empty or sub["t_answer"].isna().all():
        return np.nan
    return float(sub["t_answer"].mean())


def _aggregate_pooled_rates(rates_df: pd.DataFrame, buckets: list, rate_col: str) -> pd.DataFrame:
    """Pool rates across buckets: weighted average by n_obs per (hasAnswer, time), with pooled SE."""
    sub = rates_df[rates_df["tenure_bucket"].isin(buckets)].copy()
    if sub.empty:
        return sub
    se_col = "rate_raw_se" if rate_col == "rate_raw" else "rate_norm_se"
    # Weighted average rate
    sub["w"] = sub[rate_col] * sub["n_obs"]
    sub["se2_w"] = (sub["n_obs"] * sub[se_col]) ** 2
    agg = (
        sub.groupby(["hasAnswer", "time_center_days", "time_center_hours"], as_index=False)
        .agg({"w": "sum", "n_obs": "sum", "se2_w": "sum"})
    )
    agg[rate_col] = agg["w"] / agg["n_obs"]
    agg[se_col] = np.sqrt(agg["se2_w"]) / agg["n_obs"]
    return agg.drop(columns=["se2_w"], errors="ignore")


def plot_help_rate_one_panel(
    ax,
    rates_df: pd.DataFrame,
    timelines: pd.DataFrame,
    tenure_label: str,
    tenure_bucket_or_buckets,  # str or list (for pooled)
    use_normalized: bool = False,
    median_ta_hours: float = None,
    xlim_days: tuple = (-7, 7),
    show_ci: bool = True,
    min_yerr_frac: float = 0.0,
):
    """
    Plot treated and control help rate on ax.
    tenure_bucket_or_buckets: single bucket string or list of buckets (pooled "Experienced").
    """
    if isinstance(tenure_bucket_or_buckets, str):
        buckets = [tenure_bucket_or_buckets]
    else:
        buckets = list(tenure_bucket_or_buckets)

    rate_col = "rate_norm" if use_normalized else "rate_raw"
    se_col = "rate_norm_se" if use_normalized else "rate_raw_se"
    sub = rates_df[rates_df["tenure_bucket"].isin(buckets)].copy()
    if sub.empty:
        ax.set_title(tenure_label)
        return

    # If pooling multiple buckets, aggregate to one series per (hasAnswer, time)
    if len(buckets) > 1:
        sub = _aggregate_pooled_rates(rates_df, buckets, rate_col)
        if sub.empty:
            return

    # Optional: enforce minimum error bar length for visibility when SE is very small (large N)
    all_y = sub[rate_col].values
    y_range = float(np.ptp(all_y)) or 1.0

    for has_ans, label, color, ls in [
        (0, "Control", "#2166ac", "-"),
        (1, "Treated", "#b2182b", "-"),
    ]:
        g = sub[sub["hasAnswer"] == has_ans].sort_values("time_center_days")
        if g.empty:
            continue
        x = g["time_center_days"].values
        y = g[rate_col].values
        yerr = None
        if show_ci and se_col in g.columns:
            yerr = 1.96 * g[se_col].values
            if min_yerr_frac > 0:
                min_yerr = min_yerr_frac * y_range
                yerr = np.maximum(yerr, min_yerr)
        ax.errorbar(
            x, y, yerr=yerr,
            color=color, linestyle=ls, marker="o", markersize=4, label=label, linewidth=1.5,
            capsize=3, capthick=1,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.8, label="Question (TQ)")
    if median_ta_hours is not None and not np.isnan(median_ta_hours):
        ta_days = median_ta_hours / 24.0
        ax.axvline(ta_days, color="gray", linestyle=":", linewidth=1.2, alpha=0.9, label=f"Median answer (TAT_A)")

    ax.set_xlabel("Time relative to question (days)")
    ax.set_ylabel("Help rate (norm. to baseline)" if use_normalized else "Help rate (answers per user per hour)")
    if show_ci and min_yerr_frac > 0:
        ax.text(0.02, 0.98, "95% CI (min. length for visibility)", transform=ax.transAxes, fontsize=6, va="top", color="gray")
    ax.set_title(tenure_label)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(xlim_days[0], xlim_days[1])


def plot_2panel(
    rates_df: pd.DataFrame,
    timelines: pd.DataFrame,
    use_normalized: bool = True,
    output_dir: str = FIGURE_DIR,
    xlim_days: tuple = (-7, 7),
    show_ci: bool = True,
    min_yerr_frac: float = 0.02,
):
    """2×1 layout: newcomers (<1 week) on top, experienced (pooled) on bottom."""
    os.makedirs(output_dir, exist_ok=True)

    experienced_buckets = [b for b in BUCKET_ORDER if b != NEWCOMER_BUCKET]
    median_ta_newcomer = median_answer_time_hours(timelines, NEWCOMER_BUCKET)
    # Median TAT_A for experienced: pool treated from all experienced buckets
    treated_exp = timelines[(timelines["tenure_bucket"].isin(experienced_buckets)) & (timelines["hasAnswer"] == 1)]
    median_ta_experienced = float(treated_exp["t_answer"].median()) if len(treated_exp) else np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    plot_help_rate_one_panel(
        ax1, rates_df, timelines,
        tenure_label=f"Newcomers ({NEWCOMER_BUCKET})",
        tenure_bucket_or_buckets=NEWCOMER_BUCKET,
        use_normalized=use_normalized,
        median_ta_hours=median_ta_newcomer,
        xlim_days=xlim_days,
        show_ci=show_ci,
        min_yerr_frac=min_yerr_frac,
    )
    plot_help_rate_one_panel(
        ax2, rates_df, timelines,
        tenure_label=EXPERIENCED_LABEL,
        tenure_bucket_or_buckets=experienced_buckets,
        use_normalized=use_normalized,
        median_ta_hours=median_ta_experienced,
        xlim_days=xlim_days,
        show_ci=show_ci,
        min_yerr_frac=min_yerr_frac,
    )

    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"help_rate_2panel.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved help_rate_2panel.[eps/png/pdf]")


def plot_all_buckets(
    rates_df: pd.DataFrame,
    timelines: pd.DataFrame,
    use_normalized: bool = True,
    output_dir: str = FIGURE_DIR,
    xlim_days: tuple = (-7, 7),
    show_ci: bool = True,
    min_yerr_frac: float = 0.02,
):
    """Small multiples: one panel per tenure bucket."""
    os.makedirs(output_dir, exist_ok=True)

    n_buckets = len(BUCKET_ORDER)
    n_cols = 2
    n_rows = (n_buckets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    for i, bucket in enumerate(BUCKET_ORDER):
        ax = axes.flat[i]
        median_ta = median_answer_time_hours(timelines, bucket)
        plot_help_rate_one_panel(
            ax, rates_df, timelines,
            tenure_label=bucket,
            tenure_bucket_or_buckets=bucket,
            use_normalized=use_normalized,
            median_ta_hours=median_ta,
            xlim_days=xlim_days,
            show_ci=show_ci,
            min_yerr_frac=min_yerr_frac,
        )

    # Hide unused subplots
    for j in range(i + 1, len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"help_rate_all_buckets.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved help_rate_all_buckets.[eps/png/pdf]")


def main():
    parser = argparse.ArgumentParser(description="Plot empirical help rate over time by tenure")
    parser.add_argument("--input", default="../data/event_history", help="Folder with study_timelines.parquet, study_events.parquet")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs")
    parser.add_argument("--bin-hours", type=float, default=4.0, help="Bin width in hours (smaller = more bins)")
    parser.add_argument("--no-normalize", action="store_true", help="Plot raw rate instead of normalized to pre-question baseline")
    parser.add_argument("--all-buckets", action="store_true", help="Also create small-multiples figure for all tenure buckets")
    parser.add_argument("--no-ci", action="store_true", help="Do not plot 95%% CI error bars")
    parser.add_argument("--min-errorbar-pct", type=float, default=2.0, metavar="PCT", help="Minimum error bar length as %% of y-range (for visibility when SE is tiny; 0 = true scale)")
    parser.add_argument("--output-dir", default=FIGURE_DIR, help="Output directory for figures")
    parser.add_argument("--window-days", type=float, default=WINDOW_DAYS, help="Plot and bin from -N to +N days relative to question (default 7)")
    args = parser.parse_args()

    window_days = args.window_days
    time_min_hours = -window_days * 24
    time_max_hours = window_days * 24
    xlim_days = (-window_days, window_days)

    print("Loading data…")
    timelines, events = load_timelines_and_events(args.input, sample_size=args.sample)
    print(f"  Timelines: {len(timelines):,}, Events: {len(events):,}")
    print(f"  Plot window: {xlim_days[0]:.0f} to +{xlim_days[1]:.0f} days")

    print("Computing binned help rates…")
    rates_df = compute_binned_rates(
        timelines, events,
        bin_width_hours=args.bin_hours,
        time_min_hours=time_min_hours,
        time_max_hours=time_max_hours,
    )
    use_normalized = not args.no_normalize
    if use_normalized:
        rates_df = normalize_to_baseline(rates_df)
    else:
        rates_df["rate_norm"] = rates_df["rate_raw"].copy()
        rates_df["rate_norm_se"] = rates_df["rate_raw_se"].copy()

    show_ci = not args.no_ci
    min_yerr_frac = (args.min_errorbar_pct / 100.0) if args.min_errorbar_pct else 0.0
    print("Generating 2-panel figure (newcomers vs experienced)…")
    plot_2panel(rates_df, timelines, use_normalized=use_normalized, output_dir=args.output_dir, xlim_days=xlim_days, show_ci=show_ci, min_yerr_frac=min_yerr_frac)

    if args.all_buckets:
        print("Generating small-multiples (all buckets)…")
        plot_all_buckets(rates_df, timelines, use_normalized=use_normalized, output_dir=args.output_dir, xlim_days=xlim_days, show_ci=show_ci, min_yerr_frac=min_yerr_frac)

    print("Done.")


if __name__ == "__main__":
    main()
