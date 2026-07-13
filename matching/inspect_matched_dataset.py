"""
Inspect the matched dataset (output of create_matched_dataset.py).

Matching is nearest-neighbor WITH replacement, so:
  - Each treated question is in exactly one pair  -> distinct treated = pairs
  - A control question may be reused across pairs  -> distinct controls <= pairs
  - Every pair still contributes 1 treated row + 1 control row -> total rows = 2 x pairs
    (but total DISTINCT questions < 2 x pairs, because controls repeat)

Prints row/pair counts, distinct treated/control questions, and the control-reuse
distribution (max/where available).

Run from project root: python3 matching/inspect_matched_dataset.py
"""

from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "input" / "matched_questions.parquet"


def _stats_duckdb(p):
    import duckdb
    con = duckdb.connect(":memory:")
    row = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT match_id) AS n_pairs,
            COUNT(DISTINCT "questionId") AS n_unique_questions,
            COUNT(DISTINCT CASE WHEN "hasAnswer" = 1 THEN "questionId" END) AS n_treated_q,
            COUNT(DISTINCT CASE WHEN "hasAnswer" = 0 THEN "questionId" END) AS n_control_q,
            SUM(CASE WHEN "hasAnswer" = 1 THEN 1 ELSE 0 END) AS n_treated,
            SUM(CASE WHEN "hasAnswer" = 0 THEN 1 ELSE 0 END) AS n_control
        FROM read_parquet('{p}')
    """).fetchone()
    max_reuse = con.execute(f"""
        SELECT COALESCE(MAX(m), 0) FROM (
            SELECT COUNT(DISTINCT match_id) AS m
            FROM read_parquet('{p}') WHERE "hasAnswer" = 0 GROUP BY "questionId"
        )
    """).fetchone()[0]
    con.close()
    keys = ["n_rows", "n_pairs", "n_unique", "n_treated_q", "n_control_q", "n_treated", "n_control"]
    s = dict(zip(keys, row))
    s["max_reuse"] = max_reuse
    return s


def _stats_pandas(p):
    import pandas as pd
    df = pd.read_parquet(p)
    ctrl = df[df["hasAnswer"] == 0]
    mi = ctrl.groupby("questionId")["match_id"].nunique() if len(ctrl) else None
    return {
        "n_rows": len(df),
        "n_pairs": df["match_id"].nunique(),
        "n_unique": df["questionId"].nunique(),
        "n_treated_q": df.loc[df["hasAnswer"] == 1, "questionId"].nunique(),
        "n_control_q": df.loc[df["hasAnswer"] == 0, "questionId"].nunique(),
        "n_treated": int((df["hasAnswer"] == 1).sum()),
        "n_control": int((df["hasAnswer"] == 0).sum()),
        "max_reuse": int(mi.max()) if mi is not None and len(mi) else 0,
    }


def main():
    if not _PATH.exists():
        print(f"Matched dataset not found: {_PATH}")
        print("Run matching/create_matched_dataset.py first to create it.")
        return

    try:
        s = _stats_duckdb(str(_PATH.resolve()))
    except ImportError:
        s = _stats_pandas(_PATH)

    print("Matched dataset:", _PATH.name)
    print("  Total rows:                 ", f"{s['n_rows']:,}")
    print("  Matched pairs (match_id):   ", f"{s['n_pairs']:,}")
    print("  Distinct questions:         ", f"{s['n_unique']:,}")
    print("  Distinct treated questions: ", f"{s['n_treated_q']:,}", " (= pairs; matched 1:1)")
    print("  Distinct control questions: ", f"{s['n_control_q']:,}", " (<= pairs; reused with replacement)")
    print("  Treated rows (hasAnswer=1): ", f"{s['n_treated']:,}")
    print("  Control rows (hasAnswer=0): ", f"{s['n_control']:,}")
    print("  Max control reuse (M_i):    ", f"{s['max_reuse']:,}")
    print()
    print("Matching is WITH replacement: distinct treated = pairs, distinct controls <= pairs,")
    print("and total rows = 2 x pairs (a control reused in M_i pairs contributes M_i control rows).")


if __name__ == "__main__":
    main()
