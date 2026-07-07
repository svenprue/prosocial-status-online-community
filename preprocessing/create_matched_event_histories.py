import os
import duckdb
import pandas as pd
from pathlib import Path

def generate_event_history_dataset(
        matched_questions_path: str,
        input_folder: str,
        output_folder: str,
        days_before_question: int = 2,
        days_after_answer: int = 2
) -> None:
    """
    Creates an event-history dataset for survival analysis (Cox/Poisson).

    Unit of analysis: question_id. Matching and timelines are defined per matched
    question (one row per question; match_id pairs treatment/control questions).
    user_id is the asker of that question, used for tenure and for finding that
    asker's help events (answers to others) within the question's time window.
    Updated to include 'user_tenure_days' (days since asker's registration).
    """
    print(f"\n=== Generating Event History Dataset ===")
    print(f"Parameters: Pre-Window={days_before_question} days, Post-Window={days_after_answer} days")

    # Initialize DuckDB
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='20GB';") 
    con.execute("PRAGMA threads=8;")

    # 1. Load Matched Questions
    print(f"Loading matched questions from {matched_questions_path}...")
    matched_df = pd.read_parquet(matched_questions_path)
    con.register("matched_raw", matched_df)

    # Question-level covariates that newer matched files carry (answer quality
    # for the quality-confound check; ViewCount for the exposure placebo).
    # Absent columns are tolerated so older matched files still run.
    optional_question_covs = [
        c for c in ('hasAcceptedAnswer', 'firstAnswerScore', 'firstAnswerVoteCount', 'viewCount')
        if c in matched_df.columns
    ]
    if optional_question_covs:
        print(f"Carrying question covariates into timelines: {optional_question_covs}")
    else:
        print("No optional question covariates found in matched file (older matching run?)")
    opt_select_m = "".join(f",\n            m.{c}" for c in optional_question_covs)
    opt_select_mt = "".join(f",\n            mt.{c}" for c in optional_question_covs)
    opt_select_plain = "".join(f", {c}" for c in optional_question_covs)

    # 2. Create match timelines (one row per matched question; timelines keyed by question_id)
    print("Aligning timelines for matched pairs (per question_id)...")
    con.execute(f"""
        CREATE TEMPORARY TABLE match_timelines AS
        WITH treatment_times AS (
            SELECT match_id, responseTimeHours
            FROM matched_raw
            WHERE hasAnswer = 1
        )
        SELECT
            m.match_id,
            m.questionId AS question_id,
            m.userId AS user_id,
            m.hasAnswer,
            m.year AS question_year,
            COALESCE(t.responseTimeHours, m.responseTimeHours) AS match_response_hours{opt_select_m}
        FROM matched_raw m
        LEFT JOIN treatment_times t ON m.match_id = t.match_id
        WHERE COALESCE(t.responseTimeHours, m.responseTimeHours) IS NOT NULL
    """)

    # 3. Load Raw Data (Questions, Answers, AND USERS)
    questions_path = os.path.join(input_folder, 'posts_questions.parquet')
    answers_path = os.path.join(input_folder, 'posts_answers.parquet')
    users_path = os.path.join(input_folder, 'Users.parquet')  # <--- NEW
    
    print("Loading raw questions, answers, and users...")
    con.execute(f"CREATE VIEW raw_questions AS SELECT Id, OwnerUserId, CreationDate::TIMESTAMP as ts FROM '{questions_path}'")
    con.execute(f"CREATE VIEW raw_answers AS SELECT Id, OwnerUserId, ParentId, CreationDate::TIMESTAMP as ts FROM '{answers_path}'")
    
    # Check if users file exists, otherwise warn
    if os.path.exists(users_path):
        con.execute(f"CREATE VIEW raw_users AS SELECT Id, CreationDate::TIMESTAMP as registration_ts FROM '{users_path}'")
    else:
        print(f"WARNING: {users_path} not found. Tenure will be NULL.")
        con.execute("CREATE VIEW raw_users AS SELECT NULL::INTEGER as Id, NULL::TIMESTAMP as registration_ts WHERE 1=0")

    # 4. Define absolute time boundaries per question & calculate asker tenure
    print("Calculating study windows per question and asker tenure...")
    con.execute(f"""
        CREATE TEMPORARY TABLE study_windows AS
        SELECT
            mt.match_id,
            mt.question_id,
            mt.user_id,
            mt.hasAnswer,
            mt.question_year,
            mt.match_response_hours,{opt_select_mt}
            rq.ts AS question_ts,

            -- Asker tenure: days between asker's registration and this question
            GREATEST(0, date_diff('day', u.registration_ts, rq.ts)) AS user_tenure_days,

            rq.ts - INTERVAL '{days_before_question} DAYS' AS window_start_ts,
            rq.ts + INTERVAL '1 SECOND' * CAST(mt.match_response_hours * 3600 AS BIGINT) AS answer_ts,
            (rq.ts + INTERVAL '1 SECOND' * CAST(mt.match_response_hours * 3600 AS BIGINT))
                + INTERVAL '{days_after_answer} DAYS' AS window_end_ts

        FROM match_timelines mt
        JOIN raw_questions rq ON mt.question_id = rq.Id
        LEFT JOIN raw_users u ON mt.user_id = u.Id
    """)

    # 5. Find help events: for each question timeline, answers by that question's asker to others
    print("Finding help events (asker's answers to others) within each question's window...")
    con.execute("""
        CREATE TEMPORARY TABLE user_helps AS
        SELECT
            sw.match_id,
            sw.question_id,
            sw.user_id,
            sw.hasAnswer,
            sw.question_ts,
            sw.window_start_ts,
            sw.answer_ts,
            sw.window_end_ts,
            ra.ts AS help_ts,
            date_diff('second', sw.question_ts, ra.ts) / 3600.0 AS relative_help_time_hours,
            date_diff('second', sw.question_ts, sw.answer_ts) / 3600.0 AS relative_answer_time_hours
        FROM study_windows sw
        JOIN raw_answers ra ON sw.user_id = ra.OwnerUserId
        LEFT JOIN raw_questions q_parent ON ra.ParentId = q_parent.Id
        WHERE ra.ts >= sw.window_start_ts
          AND ra.ts <= sw.window_end_ts
          AND (q_parent.OwnerUserId IS NULL OR ra.OwnerUserId != q_parent.OwnerUserId)
    """)

    # 6. Export Results
    print("Exporting Event Log...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Export study definitions (one row per question; question_id is the timeline unit)
    con.execute(f"""
        COPY (
            SELECT
                match_id, question_id, user_id, hasAnswer,
                question_year,
                user_tenure_days{opt_select_plain},
                date_diff('second', question_ts, window_start_ts) / 3600.0 AS t_start,
                0.0 AS t_question,
                date_diff('second', question_ts, answer_ts) / 3600.0 AS t_answer,
                date_diff('second', question_ts, window_end_ts) / 3600.0 AS t_end
            FROM study_windows
        ) TO '{os.path.join(output_folder, "study_timelines.parquet")}' (FORMAT PARQUET);
    """)

    # Export help events (keyed by match_id, question_id)
    con.execute(f"""
        COPY (
            SELECT
                match_id, question_id, user_id, relative_help_time_hours AS t_event
            FROM user_helps
        ) TO '{os.path.join(output_folder, "study_events.parquet")}' (FORMAT PARQUET);
    """)

    # 7. Composite reciprocity events (R2 P1/P4): answers + comments on others'
    # posts + accept-actions on the asker's OTHER questions. Written to a
    # SEPARATE file with a help_type column so the primary answers-only
    # study_events.parquet (and every model reading it) is unchanged; the
    # composite robustness re-estimation opts in explicitly.
    # Upvotes are NOT included: the public dump anonymizes Votes.UserId except
    # VoteTypeId 5/8, so upvotes-given cannot be attributed to the asker.
    comments_path = os.path.join(input_folder, 'Comments.parquet')
    votes_path = os.path.join(input_folder, 'Votes.parquet')
    if os.path.exists(comments_path):
        print("Finding composite help events (comments, accepts) within each question's window...")
        con.execute(f"""
            CREATE VIEW raw_comments AS
            SELECT PostId, UserId, CreationDate::TIMESTAMP AS ts
            FROM '{comments_path}'
            WHERE UserId IS NOT NULL
        """)
        con.execute("""
            CREATE VIEW raw_post_owners AS
            SELECT Id, OwnerUserId FROM raw_questions
            UNION ALL
            SELECT Id, OwnerUserId FROM raw_answers
        """)
        # Comments by the asker on posts they do not own (full-precision timestamps)
        con.execute("""
            CREATE TEMPORARY TABLE user_comment_helps AS
            SELECT
                sw.match_id,
                sw.question_id,
                sw.user_id,
                date_diff('second', sw.question_ts, c.ts) / 3600.0 AS t_event
            FROM study_windows sw
            JOIN raw_comments c ON sw.user_id = c.UserId
            JOIN raw_post_owners po ON c.PostId = po.Id
            WHERE c.ts >= sw.window_start_ts
              AND c.ts <= sw.window_end_ts
              AND (po.OwnerUserId IS NULL OR po.OwnerUserId != sw.user_id)
        """)
        # Accept-actions: the asker accepts an answer on one of their OTHER
        # questions (the focal question's accept is mechanically tied to
        # treatment and excluded). NOTE: dump vote timestamps are date-only,
        # so these events have day granularity.
        if os.path.exists(votes_path):
            con.execute(f"""
                CREATE TEMPORARY TABLE user_accept_helps AS
                SELECT
                    sw.match_id,
                    sw.question_id,
                    sw.user_id,
                    date_diff('second', sw.question_ts, v.CreationDate::TIMESTAMP) / 3600.0 AS t_event
                FROM study_windows sw
                JOIN raw_questions q2 ON q2.OwnerUserId = sw.user_id AND q2.Id != sw.question_id
                JOIN raw_answers a2 ON a2.ParentId = q2.Id
                JOIN (SELECT PostId, CreationDate FROM '{votes_path}' WHERE VoteTypeId = 1) v
                  ON v.PostId = a2.Id
                WHERE v.CreationDate::TIMESTAMP >= sw.window_start_ts
                  AND v.CreationDate::TIMESTAMP <= sw.window_end_ts
            """)
            accept_union = """
            UNION ALL
            SELECT match_id, question_id, user_id, t_event, 'accept' AS help_type
            FROM user_accept_helps"""
        else:
            print(f"WARNING: {votes_path} not found; composite events will lack accept-actions.")
            accept_union = ""
        con.execute(f"""
            COPY (
                SELECT match_id, question_id, user_id,
                       relative_help_time_hours AS t_event, 'answer' AS help_type
                FROM user_helps
                UNION ALL
                SELECT match_id, question_id, user_id, t_event, 'comment' AS help_type
                FROM user_comment_helps{accept_union}
            ) TO '{os.path.join(output_folder, "study_events_composite.parquet")}' (FORMAT PARQUET);
        """)
        n_comment_events = con.execute("SELECT COUNT(*) FROM user_comment_helps").fetchone()[0]
        print(f" - Composite: comment help events found: {n_comment_events:,}")
        if accept_union:
            n_accept_events = con.execute("SELECT COUNT(*) FROM user_accept_helps").fetchone()[0]
            print(f" - Composite: accept help events found: {n_accept_events:,} (day-granular timing)")
    else:
        print(f"NOTE: {comments_path} not found; skipping composite reciprocity events "
              f"(run preprocessing/processing_data_dump.py with Comments.xml to enable).")

    print(f"Success! Output saved to {output_folder}")

    # Stats (unit = question_id / matched question)
    n_questions = con.execute("SELECT COUNT(*) FROM study_windows").fetchone()[0]
    n_events = con.execute("SELECT COUNT(*) FROM user_helps").fetchone()[0]
    avg_tenure = con.execute("SELECT AVG(user_tenure_days) FROM study_windows").fetchone()[0]

    print(f"\nStats:")
    print(f" - Questions (timelines) analyzed: {n_questions:,}")
    print(f" - Help events found: {n_events:,}")
    print(f" - Average asker tenure: {avg_tenure:.1f} days" if avg_tenure else " - Average asker tenure: N/A")

    con.close()

if __name__ == "__main__":
    # Paths relative to project root (parent of preprocessing/) so script works from any cwd
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent
    calculate_for_path = _project_root / "data" / "input" / "matched_questions.parquet"
    input_folder = _project_root / "data" / "input"
    output_folder = _project_root / "data" / "event_history"

    generate_event_history_dataset(
        matched_questions_path=str(calculate_for_path),
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        days_before_question=2, 
        days_after_answer=2
    )
