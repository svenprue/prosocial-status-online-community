"""
Investigate where questions are lost in the data pipeline.

Counts questions at each stage:
1. Source: posts_questions.parquet (total, with owner)
2. Raw: question_centered_model_7d_all_questions.parquet (Phase_One_Start = one per question)
3. Processed: question_centered_model_7d_processed.parquet (after cutoff filter)
4. Matched: matched_questions.parquet (after PSM, in data/input)

Uses duckdb if available (same as preprocessing), else pyarrow, else pandas.
Install one of: pip install duckdb   OR   pip install pyarrow   OR   pip install pandas
"""

import csv
import math
from pathlib import Path

BUCKET_ORDER = [
    "< 1 Week", "1 Week - 1 Month", "1 - 6 Months",
    "6 - 12 Months", "1 - 3 Years", "3 - 6 Years", "> 6 Years",
]

_BACKEND = None  # "duckdb" | "pyarrow" | "pandas"
try:
    import duckdb
    _BACKEND = "duckdb"
except ImportError:
    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        _BACKEND = "pyarrow"
    except ImportError:
        try:
            import pandas as pd
            _BACKEND = "pandas"
        except ImportError:
            print("This script needs one of: duckdb, pyarrow, or pandas.")
            print("  pip install duckdb   # same as preprocessing pipeline")
            print("  pip install pyarrow")
            print("  pip install pandas")
            raise


def main():
    base = Path(__file__).resolve().parent.parent
    input_dir = base / "data" / "input"
    study_dir = base / "data" / "study_datasets"
    cutoff_date = "2025-04-01"

    print("=" * 70)
    print("QUESTION PIPELINE LOSS INVESTIGATION")
    print("=" * 70)

    try:
        if _BACKEND == "duckdb":
            _main_duckdb(input_dir, study_dir, cutoff_date)
        elif _BACKEND == "pandas":
            _main_pandas(input_dir, study_dir, cutoff_date)
        else:
            _main_pyarrow(input_dir, study_dir, cutoff_date)
    except Exception as e:
        print(f"[WARN] Pipeline-loss stage failed ({e}); continuing to discard characterization.")

    # ISS-19: characterize which questions the matching step discards (off-support).
    characterize_discards(base, input_dir, study_dir)


def _main_duckdb(input_dir, study_dir, cutoff_date):
    con = duckdb.connect(":memory:")
    n_source_total = n_source_with_owner = None
    q_path = input_dir / "posts_questions.parquet"
    if q_path.exists():
        qp = str(q_path.resolve())
        r = con.execute(f"SELECT COUNT(*) AS n, COUNT(OwnerUserId) AS with_owner FROM read_parquet('{qp}')").fetchone()
        n_source_total, n_source_with_owner = r[0], r[1]
        n_source_null_owner = n_source_total - n_source_with_owner
        print(f"\n1. SOURCE: {q_path.name}")
        print(f"   Total questions:        {n_source_total:,}")
        print(f"   With OwnerUserId:       {n_source_with_owner:,}")
        print(f"   Lost (null owner):      {n_source_null_owner:,} ({100*n_source_null_owner/max(1,n_source_total):.1f}%)")
    else:
        print(f"\n[SKIP] Source not found: {q_path}")

    n_raw_questions = None
    raw_path = input_dir / "question_centered_model_7d_all_questions.parquet"
    if raw_path.exists():
        rp = str(raw_path.resolve())
        r = con.execute(f"SELECT COUNT(*) FROM read_parquet('{rp}')").fetchone()
        n_raw_rows = r[0]
        r2 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{rp}') WHERE event = 'Phase_One_Start'").fetchone()
        n_raw_questions = r2[0]
        print(f"\n2. RAW: {raw_path.name}")
        print(f"   Total event rows:       {n_raw_rows:,}")
        print(f"   Phase_One_Start rows:   {n_raw_questions:,} (= questions in raw)")
        if n_source_with_owner is not None:
            print(f"   Lost vs source (owner): {n_source_with_owner - n_raw_questions:,} (expected 0)")
        try:
            r3 = con.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{rp}')
                WHERE event = 'Phase_One_Start' AND CAST(phase_two_end AS TIMESTAMP) > CAST('{cutoff_date}' AS TIMESTAMP)
            """).fetchone()
            print(f"   Questions with phase_two_end > {cutoff_date}: {r3[0]:,} (dropped in processing)")
        except Exception as e:
            print(f"   (Could not compute cutoff drop: {e})")
    else:
        print(f"\n[SKIP] Raw not found: {raw_path}")

    n_proc = None
    proc_path = study_dir / "question_centered_model_7d_processed.parquet"
    if proc_path.exists():
        pp = str(proc_path.resolve())
        n_proc = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pp}')").fetchone()[0]
        print(f"\n3. PROCESSED: {proc_path.name}")
        print(f"   Rows (one per question): {n_proc:,}")
        print(f"   Cutoff filter:           phase_two_end <= {cutoff_date}")
        if n_raw_questions is not None:
            lost = n_raw_questions - n_proc
            print(f"   Lost in processing:      {lost:,} ({100*lost/max(1,n_raw_questions):.1f}% of raw)")
    else:
        print(f"\n[SKIP] Processed not found: {proc_path}")

    unique_questions_matched = n_pairs = None
    matched_path = input_dir / "matched_questions.parquet"
    if matched_path.exists():
        mp = str(matched_path.resolve())
        r = con.execute(f"SELECT COUNT(*) AS n, COUNT(DISTINCT match_id) AS pairs, COUNT(DISTINCT questionId) AS uq FROM read_parquet('{mp}')").fetchone()
        n_matched, n_pairs, unique_questions_matched = r[0], r[1], r[2]
        print(f"\n4. MATCHED: {matched_path.name}")
        print(f"   Rows:                    {n_matched:,}")
        print(f"   Unique match_id (pairs): {n_pairs:,}")
        print(f"   Unique questions:        {unique_questions_matched:,}")
        if n_proc is not None:
            print(f"   Lost in matching:        {n_proc - unique_questions_matched:,} questions not in any pair")
    else:
        print(f"\n[SKIP] Matched not found: {matched_path}")

    con.close()
    _print_summary(n_source_total, n_source_with_owner, n_raw_questions, n_proc, unique_questions_matched, n_pairs, cutoff_date)
    _print_zero_post_question_help(input_dir)


def _main_pandas(input_dir, study_dir, cutoff_date):
    import pandas as pd
    n_source_total = n_source_with_owner = None
    q_path = input_dir / "posts_questions.parquet"
    if q_path.exists():
        df = pd.read_parquet(q_path, columns=["Id", "OwnerUserId"])
        n_source_total = len(df)
        n_source_with_owner = df["OwnerUserId"].notna().sum()
        n_source_null_owner = n_source_total - n_source_with_owner
        print(f"\n1. SOURCE: {q_path.name}")
        print(f"   Total questions:        {n_source_total:,}")
        print(f"   With OwnerUserId:       {n_source_with_owner:,}")
        print(f"   Lost (null owner):      {n_source_null_owner:,} ({100*n_source_null_owner/max(1,n_source_total):.1f}%)")
    else:
        print(f"\n[SKIP] Source not found: {q_path}")

    n_raw_questions = None
    raw_path = input_dir / "question_centered_model_7d_all_questions.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path, columns=["event"])
        n_raw_rows = len(df)
        n_raw_questions = (df["event"] == "Phase_One_Start").sum()
        print(f"\n2. RAW: {raw_path.name}")
        print(f"   Total event rows:       {n_raw_rows:,}")
        print(f"   Phase_One_Start rows:   {n_raw_questions:,} (= questions in raw)")
        if n_source_with_owner is not None:
            print(f"   Lost vs source (owner): {n_source_with_owner - n_raw_questions:,} (expected 0)")
        try:
            df2 = pd.read_parquet(raw_path, columns=["event", "phase_two_end"])
            phase_one = df2[df2["event"] == "Phase_One_Start"]
            beyond = (pd.to_datetime(phase_one["phase_two_end"], errors="coerce") > pd.Timestamp(cutoff_date)).sum()
            print(f"   Questions with phase_two_end > {cutoff_date}: {beyond:,} (dropped in processing)")
        except Exception as e:
            print(f"   (Could not compute cutoff drop: {e})")
    else:
        print(f"\n[SKIP] Raw not found: {raw_path}")

    n_proc = None
    proc_path = study_dir / "question_centered_model_7d_processed.parquet"
    if proc_path.exists():
        df = pd.read_parquet(proc_path, columns=["eventId"])
        n_proc = len(df)
        print(f"\n3. PROCESSED: {proc_path.name}")
        print(f"   Rows (one per question): {n_proc:,}")
        print(f"   Cutoff filter:           phase_two_end <= {cutoff_date}")
        if n_raw_questions is not None:
            lost = n_raw_questions - n_proc
            print(f"   Lost in processing:      {lost:,} ({100*lost/max(1,n_raw_questions):.1f}% of raw)")
    else:
        print(f"\n[SKIP] Processed not found: {proc_path}")

    unique_questions_matched = n_pairs = None
    matched_path = input_dir / "matched_questions.parquet"
    if matched_path.exists():
        df = pd.read_parquet(matched_path, columns=["match_id", "questionId"])
        n_matched = len(df)
        n_pairs = df["match_id"].nunique()
        unique_questions_matched = df["questionId"].nunique()
        print(f"\n4. MATCHED: {matched_path.name}")
        print(f"   Rows:                    {n_matched:,}")
        print(f"   Unique match_id (pairs): {n_pairs:,}")
        print(f"   Unique questions:        {unique_questions_matched:,}")
        if n_proc is not None:
            print(f"   Lost in matching:        {n_proc - unique_questions_matched:,} questions not in any pair")
    else:
        print(f"\n[SKIP] Matched not found: {matched_path}")

    _print_summary(n_source_total, n_source_with_owner, n_raw_questions, n_proc, unique_questions_matched, n_pairs, cutoff_date)
    _print_zero_post_question_help(input_dir)


def _main_pyarrow(input_dir, study_dir, cutoff_date):
    import datetime
    n_source_total = n_source_with_owner = None
    q_path = input_dir / "posts_questions.parquet"
    if q_path.exists():
        t = pq.read_table(q_path, columns=["Id", "OwnerUserId"])
        n_source_total = t.num_rows
        owner = t.column("OwnerUserId")
        n_source_with_owner = n_source_total - owner.null_count
        n_source_null_owner = n_source_total - n_source_with_owner
        print(f"\n1. SOURCE: {q_path.name}")
        print(f"   Total questions:        {n_source_total:,}")
        print(f"   With OwnerUserId:       {n_source_with_owner:,}")
        print(f"   Lost (null owner):      {n_source_null_owner:,} ({100*n_source_null_owner/max(1,n_source_total):.1f}%)")
    else:
        print(f"\n[SKIP] Source not found: {q_path}")

    n_raw_questions = None
    raw_path = input_dir / "question_centered_model_7d_all_questions.parquet"
    if raw_path.exists():
        t = pq.read_table(raw_path, columns=["event"])
        n_raw_rows = t.num_rows
        event = t.column("event")
        try:
            n_raw_questions = pc.sum(pc.equal(event, "Phase_One_Start")).as_py()
        except Exception:
            n_raw_questions = sum(1 for i in range(n_raw_rows) if event[i].as_py() == "Phase_One_Start")
        if n_raw_questions is None:
            n_raw_questions = sum(1 for i in range(n_raw_rows) if event[i].as_py() == "Phase_One_Start")
        print(f"\n2. RAW: {raw_path.name}")
        print(f"   Total event rows:       {n_raw_rows:,}")
        print(f"   Phase_One_Start rows:   {n_raw_questions:,} (= questions in raw)")
        if n_source_with_owner is not None:
            print(f"   Lost vs source (owner): {n_source_with_owner - n_raw_questions:,} (expected 0)")
        try:
            t2 = pq.read_table(raw_path, columns=["event", "phase_two_end"])
            event2, phase_end = t2.column("event"), t2.column("phase_two_end")
            cutoff_val = datetime.datetime(2025, 4, 1, tzinfo=datetime.timezone.utc)
            beyond = 0
            for i in range(t2.num_rows):
                if event2[i].as_py() != "Phase_One_Start":
                    continue
                val = phase_end[i]
                if val is None:
                    continue
                val = val.as_py() if hasattr(val, "as_py") else val
                if val is not None and val > cutoff_val:
                    beyond += 1
            print(f"   Questions with phase_two_end > {cutoff_date}: {beyond:,} (dropped in processing)")
        except Exception as e:
            print(f"   (Could not compute cutoff drop: {e})")
    else:
        print(f"\n[SKIP] Raw not found: {raw_path}")

    n_proc = None
    proc_path = study_dir / "question_centered_model_7d_processed.parquet"
    if proc_path.exists():
        t = pq.read_table(proc_path, columns=["eventId"])
        n_proc = t.num_rows
        print(f"\n3. PROCESSED: {proc_path.name}")
        print(f"   Rows (one per question): {n_proc:,}")
        print(f"   Cutoff filter:           phase_two_end <= {cutoff_date}")
        if n_raw_questions is not None:
            lost = n_raw_questions - n_proc
            print(f"   Lost in processing:      {lost:,} ({100*lost/max(1,n_raw_questions):.1f}% of raw)")
    else:
        print(f"\n[SKIP] Processed not found: {proc_path}")

    unique_questions_matched = n_pairs = None
    matched_path = input_dir / "matched_questions.parquet"
    if matched_path.exists():
        t = pq.read_table(matched_path, columns=["match_id", "questionId"])
        n_matched = t.num_rows
        match_ids = t.column("match_id")
        qids = t.column("questionId")
        n_pairs = len({match_ids[i].as_py() for i in range(n_matched)})
        unique_questions_matched = len({qids[i].as_py() for i in range(n_matched) if qids[i].as_py() is not None})
        print(f"\n4. MATCHED: {matched_path.name}")
        print(f"   Rows:                    {n_matched:,}")
        print(f"   Unique match_id (pairs): {n_pairs:,}")
        print(f"   Unique questions:        {unique_questions_matched:,}")
        if n_proc is not None:
            print(f"   Lost in matching:        {n_proc - unique_questions_matched:,} questions not in any pair")
    else:
        print(f"\n[SKIP] Matched not found: {matched_path}")

    _print_summary(n_source_total, n_source_with_owner, n_raw_questions, n_proc, unique_questions_matched, n_pairs, cutoff_date)
    _print_zero_post_question_help(input_dir)


def _print_summary(n_source_total, n_source_with_owner, n_raw_questions, n_proc, unique_questions_matched, n_pairs, cutoff_date):
    print("\n" + "=" * 70)
    print("SUMMARY: Where questions are lost")
    print("=" * 70)
    if n_source_total is not None:
        print(f"  Source (posts_questions):     {n_source_total:,} total  →  {n_source_with_owner:,} with owner")
    if n_raw_questions is not None:
        print(f"  Raw (all_questions):          {n_raw_questions:,} questions")
    if n_proc is not None:
        print(f"  Processed (after cutoff):      {n_proc:,} questions")
        if n_raw_questions is not None:
            print(f"  → Lost in processing:         {n_raw_questions - n_proc:,} (mainly cutoff {cutoff_date})")
    if unique_questions_matched is not None and n_proc is not None:
        print(f"  Matched (in pairs):            {unique_questions_matched:,} questions ({n_pairs:,} pairs)")
        print(f"  → Lost in PSM:                 {n_proc - unique_questions_matched:,}")
    print()


def _print_zero_post_question_help(input_dir):
    """Report retained questions with no observed post-question help events."""
    base = input_dir.parent
    timelines_path = base / "event_history" / "study_timelines.parquet"
    events_path = base / "event_history" / "study_events.parquet"
    if not timelines_path.exists() or not events_path.exists():
        print("[SKIP] Event-history files not found; cannot compute zero post-question help.")
        return

    print("=" * 70)
    print("ZERO POST-QUESTION HELP AMONG RETAINED QUESTIONS")
    print("=" * 70)
    print("Counts use study_events help events with t_event >= 0 within each question window.")

    if _BACKEND == "duckdb":
        _print_zero_post_question_help_duckdb(timelines_path, events_path)
    else:
        _print_zero_post_question_help_pandas(timelines_path, events_path)


def _print_zero_post_question_help_duckdb(timelines_path, events_path):
    con = duckdb.connect(":memory:")
    tp = str(timelines_path.resolve())
    ep = str(events_path.resolve())
    tenure_case = """
        CASE
            WHEN treated_tenure_days <= 7 THEN '< 1 Week'
            WHEN treated_tenure_days <= 30 THEN '1 Week - 1 Month'
            WHEN treated_tenure_days <= 180 THEN '1 - 6 Months'
            WHEN treated_tenure_days <= 365 THEN '6 - 12 Months'
            WHEN treated_tenure_days <= 1095 THEN '1 - 3 Years'
            WHEN treated_tenure_days <= 2190 THEN '3 - 6 Years'
            ELSE '> 6 Years'
        END
    """
    query = f"""
        WITH timelines AS (
            SELECT match_id, question_id, hasAnswer, user_tenure_days
            FROM read_parquet('{tp}')
        ),
        treated_tenure AS (
            SELECT match_id, MAX(CASE WHEN hasAnswer = 1 THEN user_tenure_days END) AS treated_tenure_days
            FROM timelines
            GROUP BY match_id
        ),
        post_events AS (
            SELECT match_id, question_id, COUNT(*) AS n_post_help
            FROM read_parquet('{ep}')
            WHERE t_event >= 0
            GROUP BY match_id, question_id
        ),
        base AS (
            SELECT
                t.match_id,
                t.question_id,
                t.hasAnswer,
                {tenure_case} AS tenure_bucket,
                COALESCE(p.n_post_help, 0) AS n_post_help
            FROM timelines t
            LEFT JOIN treated_tenure tt ON t.match_id = tt.match_id
            LEFT JOIN post_events p
              ON t.match_id = p.match_id AND t.question_id = p.question_id
        )
        SELECT
            tenure_bucket,
            hasAnswer,
            COUNT(*) AS n_questions,
            SUM(CASE WHEN n_post_help = 0 THEN 1 ELSE 0 END) AS n_zero_post_help,
            SUM(n_post_help) AS n_post_help_events
        FROM base
        GROUP BY GROUPING SETS ((tenure_bucket, hasAnswer), (hasAnswer))
        ORDER BY tenure_bucket NULLS FIRST, hasAnswer
    """
    rows = con.execute(query).fetchall()
    con.close()
    _print_zero_post_question_help_rows(rows)


def _print_zero_post_question_help_pandas(timelines_path, events_path):
    try:
        import pandas as pd
    except ImportError:
        print("  pandas unavailable; skipping zero post-question help summary.")
        return

    timelines = pd.read_parquet(
        timelines_path,
        columns=["match_id", "question_id", "hasAnswer", "user_tenure_days"],
    )
    events = pd.read_parquet(events_path, columns=["match_id", "question_id", "t_event"])
    treated_tenure = (
        timelines.loc[timelines["hasAnswer"] == 1, ["match_id", "user_tenure_days"]]
        .drop_duplicates("match_id")
        .rename(columns={"user_tenure_days": "treated_tenure_days"})
    )
    timelines = timelines.merge(treated_tenure, on="match_id", how="left")
    bins = [-float("inf"), 7, 30, 180, 365, 1095, 2190, float("inf")]
    timelines["tenure_bucket"] = pd.cut(
        timelines["treated_tenure_days"], bins=bins, labels=BUCKET_ORDER, right=True
    )
    post_counts = (
        events.loc[events["t_event"] >= 0]
        .groupby(["match_id", "question_id"])
        .size()
        .reset_index(name="n_post_help")
    )
    merged = timelines.merge(post_counts, on=["match_id", "question_id"], how="left")
    merged["n_post_help"] = merged["n_post_help"].fillna(0).astype(int)
    grouped = (
        merged.groupby(["tenure_bucket", "hasAnswer"], observed=True)
        .agg(
            n_questions=("question_id", "size"),
            n_zero_post_help=("n_post_help", lambda s: int((s == 0).sum())),
            n_post_help_events=("n_post_help", "sum"),
        )
        .reset_index()
    )
    overall = (
        merged.groupby("hasAnswer")
        .agg(
            n_questions=("question_id", "size"),
            n_zero_post_help=("n_post_help", lambda s: int((s == 0).sum())),
            n_post_help_events=("n_post_help", "sum"),
        )
        .reset_index()
    )
    rows = []
    for _, r in overall.iterrows():
        rows.append((None, int(r["hasAnswer"]), int(r["n_questions"]), int(r["n_zero_post_help"]), int(r["n_post_help_events"])))
    for _, r in grouped.iterrows():
        rows.append((str(r["tenure_bucket"]), int(r["hasAnswer"]), int(r["n_questions"]), int(r["n_zero_post_help"]), int(r["n_post_help_events"])))
    _print_zero_post_question_help_rows(rows)


def _print_zero_post_question_help_rows(rows):
    print("  Tenure bucket              Answered  Questions     Zero-post-help  Share    Help events")
    for tenure_bucket, has_answer, n_questions, n_zero, n_events in rows:
        bucket = tenure_bucket if tenure_bucket is not None else "All"
        share = n_zero / max(1, n_questions)
        print(
            f"  {bucket:<25} {int(has_answer):>8}  {int(n_questions):>10,}  "
            f"{int(n_zero):>14,}  {share:>6.1%}  {int(n_events):>11,}"
        )
    print()


# ---------------------------------------------------------------------------
# ISS-19: off-support / discard characterization
# ---------------------------------------------------------------------------
# The matching step (matching/create_matched_dataset.py) discards questions in
# TWO structurally different ways:
#   (A) STRATA filter: build_prematch_frame drops every year x mainTag cell with
#       < MIN_QUESTIONS_PER_STRATUM (=300) questions before matching runs. This
#       takes the pre-matching universe (processed, minus self-answered, with
#       responseTime>7d recoded to hasAnswer=0) down to the prematch_years pool.
#   (B) CALIPER loop: iterating over treated units, a treated question with no
#       control inside caliper 0.05 in its year x mainTag x numTags cell is
#       dropped outright; controls are matched WITH REPLACEMENT, so a control is
#       discarded only by never being selected as any treated unit's neighbour
#       (the off-support tail).
# We report, per arm (hasAnswer): the discard split, the strata/caliper
# decomposition, retained-vs-discarded standardized mean differences (SMDs, same
# pooled-SD convention as tab:balance), and a tenure-bucket breakdown.

# Covariates present in the processed file (universe-level comparison).
_DISCARD_COVS_UNIVERSE = [
    "timeSinceFirstActivityDays", "ownerReputation", "bodyLenChars", "titleLenChars",
    "numQuestionsAskedAT", "numHelpProvidedAT", "numQuestionsAsked30D",
    "numHelpProvided30D", "numTags",
]
# tag_accept_share_avg is derived during matching and only lives in the pool
# (prematch_years); it is the direct tag-level answerability proxy.
_DISCARD_COVS_POOL = ["tag_accept_share_avg"] + _DISCARD_COVS_UNIVERSE

_TENURE_CASE = """
    CASE
        WHEN timeSinceFirstActivityDays <= 7 THEN '< 1 Week'
        WHEN timeSinceFirstActivityDays <= 30 THEN '1 Week - 1 Month'
        WHEN timeSinceFirstActivityDays <= 180 THEN '1 - 6 Months'
        WHEN timeSinceFirstActivityDays <= 365 THEN '6 - 12 Months'
        WHEN timeSinceFirstActivityDays <= 1095 THEN '1 - 3 Years'
        WHEN timeSinceFirstActivityDays <= 2190 THEN '3 - 6 Years'
        ELSE '> 6 Years'
    END
"""

_CSV_FIELDS = [
    "kind", "arm", "label", "covariate",
    "n_a", "n_b", "mean_a", "mean_b", "sd_a", "sd_b", "smd",
    "n_questions", "n_users", "discard_share",
]


def _arm_name(arm):
    return "control_unanswered" if int(arm) == 0 else "treated_answered"


def _smd(mean_a, mean_b, var_a, var_b):
    """Pooled-SD SMD (Austin/Stuart/cobalt convention), matching tab:balance."""
    denom = math.sqrt((var_a + var_b) / 2) if (var_a is not None and var_b is not None
                                               and (var_a + var_b) > 0) else 0.0
    return (mean_a - mean_b) / denom if denom else 0.0


def characterize_discards(base, input_dir, study_dir):
    proc_path = study_dir / "question_centered_model_7d_processed.parquet"
    matched_path = input_dir / "matched_questions.parquet"
    pool_glob = input_dir / "prematch_years" / "*.parquet"
    if not (proc_path.exists() and matched_path.exists()):
        print("[SKIP] Discard characterization: processed and/or matched file not found.")
        return
    if _BACKEND != "duckdb":
        print("[SKIP] Discard characterization requires duckdb "
              "(21M-row anti-joins); install duckdb and re-run.")
        return

    print("=" * 70)
    print("OFF-SUPPORT / DISCARD CHARACTERIZATION (ISS-19)")
    print("=" * 70)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    proc = str(proc_path.resolve())
    matched = str(matched_path.resolve())
    pool = str(pool_glob.resolve())
    have_pool = any((input_dir / "prematch_years").glob("*.parquet"))

    # Pre-matching universe, arms defined exactly as build_prematch_frame does.
    con.execute(f"""
        CREATE VIEW universe AS
        SELECT questionId, userId,
               CASE WHEN responseTimeHours / 24.0 > 7 THEN 0 ELSE hasAnswer END AS arm,
               {_TENURE_CASE} AS tenure_bucket,
               {', '.join(_DISCARD_COVS_UNIVERSE)}
        FROM read_parquet('{proc}')
        WHERE hasSelfAnswer = 0
    """)
    con.execute(f"CREATE VIEW ret AS SELECT DISTINCT questionId FROM read_parquet('{matched}')")
    con.execute("""
        CREATE VIEW flagged AS
        SELECT u.*,
               CASE WHEN r.questionId IS NULL THEN 'discarded' ELSE 'retained' END AS status
        FROM universe u LEFT JOIN ret r ON u.questionId = r.questionId
    """)

    rows = []

    # --- Stage counts + discard split by arm (the headline deliverable) ---
    print("\n1. DISCARD SPLIT BY ARM (arm 0 = unanswered/control, 1 = answered/treated)")
    print(f"   {'arm':<20}{'status':>10}{'questions':>14}{'users':>14}{'share_of_arm':>14}")
    arm_totals = dict(con.execute(
        "SELECT arm, COUNT(*) FROM universe GROUP BY arm").fetchall())
    for arm, status, n_q, n_u in con.execute("""
        SELECT arm, status, COUNT(*) n, COUNT(DISTINCT userId) u
        FROM flagged GROUP BY arm, status ORDER BY arm, status
    """).fetchall():
        share = n_q / max(1, arm_totals.get(arm, 0))
        print(f"   {_arm_name(arm):<20}{status:>10}{n_q:>14,}{n_u:>14,}{share:>13.1%}")
        rows.append({"kind": "discard_split", "arm": _arm_name(arm), "label": status,
                     "n_questions": n_q, "n_users": n_u, "discard_share": round(share, 4)})

    # --- Strata (A) vs caliper (B) decomposition ---
    if have_pool:
        con.execute(f"CREATE VIEW poolids AS SELECT DISTINCT questionId FROM read_parquet('{pool}')")
        print("\n2. DISCARD MECHANISM: strata filter (<300/cell) vs caliper off-support tail")
        print(f"   {'arm':<20}{'strata_drop':>14}{'caliper_drop':>14}{'retained':>14}{'total':>14}")
        for arm, s, c, keep, tot in con.execute("""
            SELECT u.arm,
                   SUM(CASE WHEN pi.questionId IS NULL THEN 1 ELSE 0 END) strata_dropped,
                   SUM(CASE WHEN pi.questionId IS NOT NULL AND r.questionId IS NULL THEN 1 ELSE 0 END) caliper_dropped,
                   SUM(CASE WHEN r.questionId IS NOT NULL THEN 1 ELSE 0 END) retained,
                   COUNT(*) total
            FROM universe u
            LEFT JOIN poolids pi ON u.questionId = pi.questionId
            LEFT JOIN ret r ON u.questionId = r.questionId
            GROUP BY u.arm ORDER BY u.arm
        """).fetchall():
            print(f"   {_arm_name(arm):<20}{s:>14,}{c:>14,}{keep:>14,}{tot:>14,}")
            for lbl, val in [("strata_dropped", s), ("caliper_dropped", c),
                             ("retained", keep), ("total", tot)]:
                rows.append({"kind": "mechanism", "arm": _arm_name(arm),
                             "label": lbl, "n_questions": val})

    # --- Universe-level retained-vs-discarded SMDs (covariates in processed) ---
    print("\n3. RETAINED vs DISCARDED SMDs (universe; pooled-SD, same scale as tab:balance)")
    _emit_smd_block(con, rows, "smd_universe", "flagged", _DISCARD_COVS_UNIVERSE,
                    group_a="retained", group_b="discarded")

    # --- Pool-level retained-vs-neverchosen SMDs (P3-relevant, adds answerability) ---
    if have_pool:
        con.execute(f"""
            CREATE VIEW poolf AS
            SELECT p.questionId, p.hasAnswer AS arm,
                   {', '.join(_DISCARD_COVS_POOL)},
                   CASE WHEN r.questionId IS NULL THEN 'discarded' ELSE 'retained' END AS status
            FROM read_parquet('{pool}') p
            LEFT JOIN ret r ON p.questionId = r.questionId
        """)
        print("\n4. POOL: retained vs caliper NEVER-CHOSEN SMDs (adds tag_accept_share_avg)")
        _emit_smd_block(con, rows, "smd_pool_neverchosen", "poolf", _DISCARD_COVS_POOL,
                        group_a="retained", group_b="discarded")

    # --- Tenure-bucket discard breakdown ---
    print("\n5. DISCARD SHARE BY TENURE BUCKET (universe; < 1 Week carries the headline)")
    print(f"   {'arm':<20}{'tenure_bucket':<20}{'questions':>14}{'discarded':>14}{'share':>10}")
    for arm, bucket, n_q, n_disc in con.execute("""
        SELECT arm, tenure_bucket, COUNT(*) n,
               SUM(CASE WHEN status = 'discarded' THEN 1 ELSE 0 END) disc
        FROM flagged GROUP BY arm, tenure_bucket
        ORDER BY arm, n DESC
    """).fetchall():
        share = n_disc / max(1, n_q)
        star = "  <== newcomer" if bucket == "< 1 Week" else ""
        print(f"   {_arm_name(arm):<20}{bucket:<20}{n_q:>14,}{n_disc:>14,}{share:>9.1%}{star}")
        rows.append({"kind": "tenure", "arm": _arm_name(arm), "label": bucket,
                     "n_questions": n_q, "n_a": n_disc, "discard_share": round(share, 4)})

    con.close()

    out_dir = base / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "offsupport_characterization.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _CSV_FIELDS})
    print(f"\nWrote {len(rows)} rows to {out_path}")


def _emit_smd_block(con, rows, kind, view, covs, group_a, group_b):
    """Compute and print per-arm SMDs (group_a vs group_b) for each covariate."""
    sel = ", ".join([f"avg({c}) m_{c}, var_samp({c}) v_{c}" for c in covs])
    for arm in (0, 1):
        agg = {r[0]: r for r in con.execute(
            f"SELECT status, COUNT(*) n, {sel} FROM {view} WHERE arm = {arm} "
            f"GROUP BY status ORDER BY status").fetchall()}
        colnames = ["status", "n"] + sum([[f"m_{c}", f"v_{c}"] for c in covs], [])
        R = {k: dict(zip(colnames, v)) for k, v in agg.items()}
        if group_a not in R or group_b not in R:
            continue
        n_a, n_b = R[group_a]["n"], R[group_b]["n"]
        print(f"   arm={_arm_name(arm)}: {group_a} n={n_a:,} vs {group_b} n={n_b:,}")
        print(f"     {'covariate':<28}{'mean_'+group_a:>16}{'mean_'+group_b:>16}{'SMD':>9}")
        for c in covs:
            ma, mb = R[group_a][f"m_{c}"], R[group_b][f"m_{c}"]
            va, vb = R[group_a][f"v_{c}"], R[group_b][f"v_{c}"]
            smd = _smd(ma, mb, va, vb)
            flag = "  *" if abs(smd) >= 0.1 else ""
            print(f"     {c:<28}{ma:>16.3f}{mb:>16.3f}{smd:>9.3f}{flag}")
            rows.append({"kind": kind, "arm": _arm_name(arm), "covariate": c,
                         "n_a": n_a, "n_b": n_b,
                         "mean_a": round(ma, 4), "mean_b": round(mb, 4),
                         "sd_a": round(math.sqrt(va), 4) if va and va > 0 else 0.0,
                         "sd_b": round(math.sqrt(vb), 4) if vb and vb > 0 else 0.0,
                         "smd": round(smd, 4)})


if __name__ == "__main__":
    main()
