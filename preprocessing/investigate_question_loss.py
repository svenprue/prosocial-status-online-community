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


if __name__ == "__main__":
    main()
