"""
help_rate_over_time.py
======================
Plots empirical help rate over the study window (question-relative time),
separately for treated and control, by tenure bucket.

All plots use 15-minute bins (configurable via --bin-hours).
Stable groups: pooled figure (all tenure) + by-tenure small multiples for appendix.
Continuous treatment: one figure with 7 tenure panels, 0–12h, joint legend.
  - X-axis: time relative to question (day 0 = question posted).
  - Y-axis: help rate (answers per user per hour, optionally normalized to pre-question baseline).
  - Vertical line at TQ (question time = 0); for treated, median answer time (TAT_A).
  - 95% CI error bars from Poisson SE: SE(rate) = sqrt(rate / exposure).

Outputs (file names match manuscript includegraphics paths):
  - help_rate_pooled.*       — Help rate over ±2-day window, pooled (fig:help_rate_2panel).
  - help_rate_by_tenure.*    — Help rate by tenure bucket, small multiples (fig:help_rate_by_tenure, appendix).
  - help_rate_adoption_pooled.* — Help rate by answer-receipt status, pooled (fig:help_rate_adoption_pooled).
  - help_rate_adoption_by_tenure.* — Help rate by adoption status, by tenure (fig:help_rate_adoption_by_tenure, appendix).

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


def compute_binned_rates_adoption(
    timelines: pd.DataFrame,
    events: pd.DataFrame,
    bin_width_hours: float = 2.0,
    time_max_hours: float = 24.0,
    tenure_bucket: str = None,
) -> pd.DataFrame:
    """
    For each time bin [t_lo, t_hi], compare help rate of:
      - Treated: units who received an answer *before* the bucket (t_answer <= t_lo).
      - Control: units who received an answer *after* the end of the bucket (t_answer > t_hi or no answer).
    Units who receive their answer *during* the bucket (t_lo < t_answer <= t_hi) are excluded from
    both groups in that bucket, so the comparison is clean (before-bucket vs after-bucket only).

    Exposure = n_units * bin_width (full bin for each included unit). If tenure_bucket is set,
    restrict to that tenure bucket.

    Returns DataFrame with columns:
      time_center_hours, time_center_days, group (control/treated),
      n_obs (person-hours), event_count, rate_raw, rate_raw_se.
    """
    time_min_hours = 0.0
    if tenure_bucket is not None and "tenure_bucket" in timelines.columns:
        tl = timelines[timelines["tenure_bucket"] == tenure_bucket][["match_id", "question_id", "t_answer"]].copy()
    else:
        tl = timelines[["match_id", "question_id", "t_answer"]].copy()
    tl["t_answer"] = pd.to_numeric(tl["t_answer"], errors="coerce")

    ev = events.merge(tl, on=["match_id", "question_id"], how="inner")
    ev = ev[(ev["t_event"] >= time_min_hours) & (ev["t_event"] <= time_max_hours)]

    bin_edges = np.arange(0, time_max_hours + bin_width_hours * 0.5, bin_width_hours)
    time_centers_h = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths_h = np.diff(bin_edges)
    time_centers_d = time_centers_h / 24.0

    ev["bin_idx"] = np.searchsorted(bin_edges, ev["t_event"].values, side="right") - 1
    ev["bin_idx"] = ev["bin_idx"].clip(0, len(bin_edges) - 2)

    # For each bin [t_lo, t_hi]: treated = got answer before bucket (t_answer <= t_lo),
    # control = got answer after bucket (t_answer > t_hi or no answer). Exclude during-bucket.
    bi = ev["bin_idx"].values.astype(int)
    t_lo = np.asarray(bin_edges)[bi]
    t_hi = np.asarray(bin_edges)[bi + 1]
    ta = ev["t_answer"].values
    ev = ev.copy()
    ev["group"] = pd.NA
    ev.loc[np.isnan(ta) | (ta > t_hi), "group"] = "control"
    ev.loc[(~np.isnan(ta)) & (ta <= t_lo), "group"] = "treated"
    ev = ev[ev["group"].notna()]  # drop events from units who got answer during this bin

    counts = (
        ev.groupby(["group", "bin_idx"])
        .agg(event_count=("t_event", "count"))
        .reset_index()
    )

    # Exposure: only units with t_answer <= t_lo (treated) or t_answer > t_hi / no answer (control)
    t_answer = tl["t_answer"].values
    rows = []
    for bi in range(len(time_centers_h)):
        t_lo = float(bin_edges[bi])
        t_hi = float(bin_edges[bi + 1])
        width = float(bin_widths_h[bi])
        no_ans = np.isnan(t_answer)
        after_bin = t_answer > t_hi
        control_units = no_ans | after_bin
        treated_units = (~no_ans) & (t_answer <= t_lo)
        control_hours = np.sum(control_units) * width
        treated_hours = np.sum(treated_units) * width
        rows.append({"bin_idx": bi, "control_hours": control_hours, "treated_hours": treated_hours})
    exposure_df = pd.DataFrame(rows)

    # Build result: one row per (group, bin).
    # Control (no answer yet): include first bin (0 to bin_width). Treated (answer received): skip first bin (no one received before t=0).
    result = []
    for g in ["control", "treated"]:
        exp_col = "control_hours" if g == "control" else "treated_hours"
        bin_start = 0 if g == "control" else 1
        for bi in range(bin_start, len(time_centers_h)):
            exp = exposure_df.loc[exposure_df["bin_idx"] == bi, exp_col].iloc[0]
            cnt = counts[(counts["group"] == g) & (counts["bin_idx"] == bi)]["event_count"]
            cnt = cnt.iloc[0] if len(cnt) else 0
            rate = cnt / exp if exp > 0 else 0.0
            rate_se = np.sqrt(rate / exp) if exp > 0 else 0.0
            result.append({
                "time_center_hours": time_centers_h[bi],
                "time_center_days": time_centers_d[bi],
                "group": g,
                "n_obs": exp,
                "event_count": cnt,
                "rate_raw": rate,
                "rate_raw_se": rate_se,
            })
    return pd.DataFrame(result)


def compute_binned_rates_stable_control(
    timelines: pd.DataFrame,
    events: pd.DataFrame,
    bin_width_hours: float = 2.0,
    time_max_hours: float = 24.0,
    tenure_bucket: str = None,
) -> pd.DataFrame:
    """
    Binned help rate for the stable control group (hasAnswer == 0, never receive an answer).
    Same bin edges as adoption (0 to time_max_hours). For use as reference in adoption plot.

    Returns DataFrame with columns:
      time_center_hours, time_center_days, rate_raw, rate_raw_se, n_obs, event_count.
    """
    time_min_hours = 0.0
    tl = timelines[timelines["hasAnswer"] == 0].copy()
    if tenure_bucket is not None and "tenure_bucket" in tl.columns:
        tl = tl[tl["tenure_bucket"] == tenure_bucket]
    tl = tl[["match_id", "question_id"]]
    if tl.empty:
        bin_edges = np.arange(0, time_max_hours + bin_width_hours * 0.5, bin_width_hours)
        time_centers_h = (bin_edges[:-1] + bin_edges[1:]) / 2
        time_centers_d = time_centers_h / 24.0
        return pd.DataFrame({
            "time_center_hours": time_centers_h,
            "time_center_days": time_centers_d,
            "rate_raw": 0.0,
            "rate_raw_se": 0.0,
            "n_obs": 0.0,
            "event_count": 0,
        })

    ev = events.merge(tl, on=["match_id", "question_id"], how="inner")
    ev = ev[(ev["t_event"] >= time_min_hours) & (ev["t_event"] <= time_max_hours)]

    bin_edges = np.arange(0, time_max_hours + bin_width_hours * 0.5, bin_width_hours)
    time_centers_h = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths_h = np.diff(bin_edges)
    time_centers_d = time_centers_h / 24.0
    n_control = len(tl)

    ev["bin_idx"] = np.searchsorted(bin_edges, ev["t_event"].values, side="right") - 1
    ev["bin_idx"] = ev["bin_idx"].clip(0, len(bin_edges) - 2)
    counts = ev.groupby("bin_idx").agg(event_count=("t_event", "count")).reset_index()

    rows = []
    for bi in range(len(time_centers_h)):
        width = float(bin_widths_h[bi])
        exposure = n_control * width
        cnt = counts[counts["bin_idx"] == bi]["event_count"]
        cnt = int(cnt.iloc[0]) if len(cnt) else 0
        rate = cnt / exposure if exposure > 0 else 0.0
        rate_se = np.sqrt(rate / exposure) if exposure > 0 else 0.0
        rows.append({
            "time_center_hours": time_centers_h[bi],
            "time_center_days": time_centers_d[bi],
            "rate_raw": rate,
            "rate_raw_se": rate_se,
            "n_obs": exposure,
            "event_count": cnt,
        })
    return pd.DataFrame(rows)


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
            color=color, linestyle=ls, marker="o", markersize=2.5, label=label, linewidth=0.8,
            capsize=2, capthick=0.8,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.8, label="Question (TQ)")
    if median_ta_hours is not None and not np.isnan(median_ta_hours):
        ta_days = median_ta_hours / 24.0
        ax.axvline(ta_days, color="gray", linestyle=":", linewidth=0.7, alpha=0.9, label=f"Median answer (TAT_A)")

    ax.set_xlabel("Time relative to question (days)")
    ax.set_ylabel("Help rate (norm. to baseline)" if use_normalized else "Help rate (answers per user per hour)")
    if show_ci and min_yerr_frac > 0:
        ax.text(0.02, 0.98, "95% CI (min. length for visibility)", transform=ax.transAxes, fontsize=6, va="top", color="gray")
    ax.set_title(tenure_label)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(xlim_days[0], xlim_days[1])


def plot_help_rate_pooled(
    rates_df: pd.DataFrame,
    timelines: pd.DataFrame,
    use_normalized: bool = True,
    output_dir: str = FIGURE_DIR,
    xlim_days: tuple = (-7, 7),
    show_ci: bool = True,
    min_yerr_frac: float = 0.02,
):
    """Single panel: help rate pooled over all tenure buckets."""
    os.makedirs(output_dir, exist_ok=True)
    treated = timelines[timelines["hasAnswer"] == 1]
    median_ta = float(treated["t_answer"].median()) if len(treated) else np.nan
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_help_rate_one_panel(
        ax, rates_df, timelines,
        tenure_label="Help rate over the ±2-day observation window (pooled)",
        tenure_bucket_or_buckets=BUCKET_ORDER,
        use_normalized=use_normalized,
        median_ta_hours=median_ta,
        xlim_days=xlim_days,
        show_ci=show_ci,
        min_yerr_frac=min_yerr_frac,
    )
    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"help_rate_pooled.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved help_rate_pooled.[eps/png/pdf]")


def plot_help_rate_by_tenure(
    rates_df: pd.DataFrame,
    timelines: pd.DataFrame,
    use_normalized: bool = True,
    output_dir: str = FIGURE_DIR,
    xlim_days: tuple = (-7, 7),
    show_ci: bool = True,
    min_yerr_frac: float = 0.02,
):
    """Small multiples: one panel per tenure bucket (appendix)."""
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

    for j in range(n_buckets, len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"help_rate_by_tenure.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved help_rate_by_tenure.[eps/png/pdf]")


def _plot_adoption_one_panel(
    ax,
    rates_adoption: pd.DataFrame,
    timelines_bucket: pd.DataFrame,
    show_ci: bool,
    title: str,
    show_legend: bool = True,
    show_share_ylabel: bool = True,
    xlim_hours: tuple = (0, 24),
    rates_control_stable: pd.DataFrame = None,
):
    """Draw one adoption panel on ax: help rate (control/treated) + control group (no answer) + share with answer on twin axis."""
    colors = {"control": "#2166ac", "treated": "#b2182b"}
    labels = {"control": "No answer yet", "treated": "Answer received"}

    # Control group (never received answer): reference line
    if rates_control_stable is not None and not rates_control_stable.empty:
        sub = rates_control_stable.sort_values("time_center_hours")
        x = sub["time_center_hours"].values
        y = sub["rate_raw"].values
        yerr = (1.96 * sub["rate_raw_se"].values) if show_ci and "rate_raw_se" in sub.columns else None
        ax.plot(
            x, y,
            color="#4d4d4d", linestyle="--", linewidth=1.5, marker="s", markersize=3,
            label="Control (no answer)", zorder=2,
        )
        if yerr is not None:
            ax.fill_between(x, y - yerr, y + yerr, color="#4d4d4d", alpha=0.15, zorder=1)

    for g in ["control", "treated"]:
        sub = rates_adoption[rates_adoption["group"] == g].sort_values("time_center_hours")
        if sub.empty:
            continue
        x = sub["time_center_hours"].values
        y = sub["rate_raw"].values
        yerr = (1.96 * sub["rate_raw_se"].values) if show_ci and "rate_raw_se" in sub.columns else None
        ax.plot(
            x, y,
            color=colors[g], linestyle="-", linewidth=2, marker="o", markersize=4,
            label=labels[g], zorder=2,
        )
        if yerr is not None:
            ax.fill_between(x, y - yerr, y + yerr, color=colors[g], alpha=0.2, zorder=1)

    t_answer = pd.to_numeric(timelines_bucket["t_answer"], errors="coerce")
    time_points = np.sort(rates_adoption["time_center_hours"].unique())
    share = np.array([(t_answer <= t).mean() for t in time_points])

    ax2 = ax.twinx()
    ax2.plot(
        time_points, share,
        color="#2d7a3e", linestyle="--", linewidth=1.5, marker="s", markersize=3,
        label="Share with answer", zorder=2,
    )
    if show_share_ylabel:
        ax2.set_ylabel("Share with answer", fontsize=9, color="#2d7a3e")
    ax2.tick_params(axis="y", labelcolor="#2d7a3e", labelsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#2d7a3e")

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Hours since question", fontsize=9)
    ax.set_ylabel("Help rate (per user per hour)", fontsize=9)
    ax.set_title(title, fontsize=10)
    if show_legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax.set_xlim(xlim_hours[0], xlim_hours[1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def plot_adoption_pooled(
    timelines: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: str = FIGURE_DIR,
    show_ci: bool = True,
    bin_width_hours: float = 0.5,
    time_max_hours: float = 12.0,
):
    """
    Single panel: adoption help rate pooled over all tenure buckets.
    Includes No answer yet, Answer received, Control (no answer), Share with answer.
    Saves help_rate_adoption_pooled.[eps/png/pdf].
    """
    os.makedirs(output_dir, exist_ok=True)
    rates = compute_binned_rates_adoption(
        timelines, events,
        bin_width_hours=bin_width_hours,
        time_max_hours=time_max_hours,
        tenure_bucket=None,
    )
    rates_control = compute_binned_rates_stable_control(
        timelines, events,
        bin_width_hours=bin_width_hours,
        time_max_hours=time_max_hours,
        tenure_bucket=None,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_adoption_one_panel(
        ax, rates, timelines, show_ci,
        title="Help rate in the post-question window by answer-receipt status (pooled)",
        show_legend=True,
        show_share_ylabel=True,
        xlim_hours=(0, time_max_hours),
        rates_control_stable=rates_control,
    )
    plt.tight_layout()
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(
            os.path.join(output_dir, "help_rate_adoption_pooled.{}".format(ext)),
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    print("✓ Saved help_rate_adoption_pooled.[eps/png/pdf]")


def plot_adoption_by_tenure(
    timelines: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: str = FIGURE_DIR,
    show_ci: bool = True,
    bin_width_hours: float = 0.25,
    time_max_hours: float = 12.0,
):
    """
    One figure: 7 panels (one per tenure bucket), continuous treatment adoption.
    X-axis 0 to 12 hours. One joint legend for all panels.
    Saves help_rate_adoption_by_tenure.[eps/png/pdf].
    """
    os.makedirs(output_dir, exist_ok=True)
    n_buckets = len(BUCKET_ORDER)
    n_cols = 2
    n_rows = (n_buckets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    for i, bucket in enumerate(BUCKET_ORDER):
        ax = axes.flat[i]
        tl_b = timelines[timelines["tenure_bucket"] == bucket]
        if tl_b.empty:
            ax.set_title(bucket)
            ax.set_visible(True)
            continue
        rates_b = compute_binned_rates_adoption(
            timelines, events,
            bin_width_hours=bin_width_hours,
            time_max_hours=time_max_hours,
            tenure_bucket=bucket,
        )
        rates_control_b = compute_binned_rates_stable_control(
            timelines, events,
            bin_width_hours=bin_width_hours,
            time_max_hours=time_max_hours,
            tenure_bucket=bucket,
        )
        show_share_ylabel = (i % n_cols == 1)
        _plot_adoption_one_panel(
            ax, rates_b, tl_b, show_ci,
            title=bucket,
            show_legend=False,
            show_share_ylabel=show_share_ylabel,
            xlim_hours=(0, time_max_hours),
            rates_control_stable=rates_control_b,
        )
    for j in range(n_buckets, len(axes.flat)):
        axes.flat[j].set_visible(False)

    # Joint legend: proxy artists so all series are shown (panels have show_legend=False)
    from matplotlib.lines import Line2D
    proxy_no_answer_yet = Line2D([0], [0], color="#2166ac", linestyle="-", linewidth=2, marker="o", markersize=4, label="No answer yet")
    proxy_answer_received = Line2D([0], [0], color="#b2182b", linestyle="-", linewidth=2, marker="o", markersize=4, label="Answer received")
    proxy_control_stable = Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1.5, marker="s", markersize=3, label="Control (no answer)")
    proxy_share = Line2D([0], [0], color="#2d7a3e", linestyle="--", linewidth=1.5, label="Share with answer")
    fig.legend(
        [proxy_no_answer_yet, proxy_answer_received, proxy_control_stable, proxy_share],
        ["No answer yet", "Answer received", "Control (no answer)", "Share with answer"],
        loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave space for legend below
    for ext in ["eps", "png", "pdf"]:
        fig.savefig(
            os.path.join(output_dir, "help_rate_adoption_by_tenure.{}".format(ext)),
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    print("✓ Saved help_rate_adoption_by_tenure.[eps/png/pdf]")


def main():
    parser = argparse.ArgumentParser(description="Plot empirical help rate over time by tenure")
    parser.add_argument("--input", default="../data/event_history", help="Folder with study_timelines.parquet, study_events.parquet")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N matched pairs")
    parser.add_argument("--bin-hours", type=float, default=0.25, help="Bin width in hours (default 0.25 = 15 min)")
    parser.add_argument("--no-normalize", action="store_true", help="Plot raw rate instead of normalized to pre-question baseline")
    parser.add_argument("--no-ci", action="store_true", help="Do not plot 95%% CI error bars")
    parser.add_argument("--min-errorbar-pct", type=float, default=2.0, metavar="PCT", help="Minimum error bar length as %% of y-range (for visibility when SE is tiny; 0 = true scale)")
    parser.add_argument("--output-dir", default=FIGURE_DIR, help="Output directory for figures")
    parser.add_argument("--window-days", type=float, default=WINDOW_DAYS, help="Plot and bin from -N to +N days relative to question (default 7)")
    parser.add_argument("--no-adoption", action="store_true", help="Skip adoption-over-time figure (first 24h, time-varying control/treated)")
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
    print("Generating pooled help rate figure…")
    plot_help_rate_pooled(rates_df, timelines, use_normalized=use_normalized, output_dir=args.output_dir, xlim_days=xlim_days, show_ci=show_ci, min_yerr_frac=min_yerr_frac)
    print("Generating help rate by tenure (appendix)…")
    plot_help_rate_by_tenure(rates_df, timelines, use_normalized=use_normalized, output_dir=args.output_dir, xlim_days=xlim_days, show_ci=show_ci, min_yerr_frac=min_yerr_frac)

    if not args.no_adoption:
        print("Generating pooled adoption figure (0–12h, all groups)…")
        plot_adoption_pooled(
            timelines, events,
            output_dir=args.output_dir,
            show_ci=show_ci,
            bin_width_hours=0.5,
            time_max_hours=12.0,
        )
        print("Generating adoption-by-tenure figure (0–12h, joint legend)…")
        plot_adoption_by_tenure(
            timelines, events,
            output_dir=args.output_dir,
            show_ci=show_ci,
            bin_width_hours=0.5,
            time_max_hours=12.0,
        )

    print("Done.")


if __name__ == "__main__":
    main()
