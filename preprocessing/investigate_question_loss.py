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

    if _BACKEND == "duckdb":
        _main_duckdb(input_dir, study_dir, cutoff_date)
    elif _BACKEND == "pandas":
        _main_pandas(input_dir, study_dir, cutoff_date)
    else:
        _main_pyarrow(input_dir, study_dir, cutoff_date)


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
        r = con.execute(f"SELECT COUNT(*) FROM read_parquet('{rp}', columns=['event'])").fetchone()
        n_raw_rows = r[0]
        r2 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{rp}', columns=['event']) WHERE event = 'Phase_One_Start'").fetchone()
        n_raw_questions = r2[0]
        print(f"\n2. RAW: {raw_path.name}")
        print(f"   Total event rows:       {n_raw_rows:,}")
        print(f"   Phase_One_Start rows:   {n_raw_questions:,} (= questions in raw)")
        if n_source_with_owner is not None:
            print(f"   Lost vs source (owner): {n_source_with_owner - n_raw_questions:,} (expected 0)")
        try:
            r3 = con.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{rp}', columns=['event','phase_two_end'])
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


if __name__ == "__main__":
    main()
