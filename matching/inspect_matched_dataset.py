"""
Inspect the matched dataset (output of create_matched_dataset.py).

Prints:
  - Total rows (each row = one question in a matched pair)
  - Number of pairs (each pair = 1 treated + 1 control)
  - Unique questions (= rows, since each question appears once)
  - Treated vs control counts

Run from project root: python3 matching/inspect_matched_dataset.py
"""

from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "input" / "matched_questions.parquet"


def main():
    if not _PATH.exists():
        print(f"Matched dataset not found: {_PATH}")
        print("Run matching/create_matched_dataset.py first to create it.")
        return

    # Prefer duckdb (same as pipeline), then pandas, then pyarrow
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        p = str(_PATH.resolve())
        row = con.execute(f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT match_id) AS n_pairs,
                COUNT(DISTINCT "questionId") AS n_unique_questions,
                SUM(CASE WHEN "hasAnswer" = 1 THEN 1 ELSE 0 END) AS n_treated,
                SUM(CASE WHEN "hasAnswer" = 0 THEN 1 ELSE 0 END) AS n_control
            FROM read_parquet('{p}')
        """).fetchone()
        con.close()
        n_rows, n_pairs, n_unique, n_treated, n_control = row
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_parquet(_PATH)
            n_rows = len(df)
            n_pairs = df["match_id"].nunique()
            n_unique = df["questionId"].nunique()
            n_treated = (df["hasAnswer"] == 1).sum()
            n_control = (df["hasAnswer"] == 0).sum()
        except ImportError:
            import pyarrow.parquet as pq
            t = pq.read_table(_PATH)
            n_rows = t.num_rows
            mid = t.column("match_id")
            qid = t.column("questionId")
            ha = t.column("hasAnswer")
            n_pairs = len(set(mid[i].as_py() for i in range(n_rows)))
            n_unique = len(set(qid[i].as_py() for i in range(n_rows) if qid[i].as_py() is not None))
            n_treated = sum(1 for i in range(n_rows) if ha[i].as_py() == 1)
            n_control = n_rows - n_treated

    print("Matched dataset:", _PATH.name)
    print("  Total rows:              ", f"{n_rows:,}")
    print("  Matched pairs (match_id):", f"{n_pairs:,}")
    print("  Unique questions:        ", f"{n_unique:,}")
    print("  Treated (hasAnswer=1):   ", f"{n_treated:,}")
    print("  Control (hasAnswer=0):   ", f"{n_control:,}")
    print()
    print("Each pair has exactly one treated and one control question, so")
    print("  unique questions = rows = 2 × pairs.")


if __name__ == "__main__":
    main()
