import os
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

from input_paths import resolve_input_file, input_file_exists


def generate_event_history_dataset(
        matched_questions_path: str,
        input_folder: str,
        output_folder: str,
        days_before_question: int = 2,
        days_after_answer: int = 2,
        include_composite_help: bool = True,
) -> None:
    """
    Creates an event-history dataset for survival analysis (Cox/Poisson).

    Unit of analysis: question_id. Matching and timelines are defined per matched
    question (one row per question; match_id pairs treatment/control questions).
    user_id is the asker of that question, used for tenure and for finding that
    asker's help events (answers/comments/accepts to others) within the window.
    """
    print(f"\n=== Generating Event History Dataset ===")
    print(f"Parameters: Pre-Window={days_before_question} days, Post-Window={days_after_answer} days")

    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='20GB';")
    con.execute("PRAGMA threads=8;")

    print(f"Loading matched questions from {matched_questions_path}...")
    matched_df = pd.read_parquet(matched_questions_path)
    for col in [
        "hasAcceptedAnswer", "firstAnswerScore", "firstAnswerBodyLenChars", "viewCount",
        "postHour", "postDayOfWeek", "numTags", "bodyLenChars", "titleLenChars", "ownerReputation",
    ]:
        if col not in matched_df.columns:
            matched_df[col] = np.nan
    con.register("matched_raw", matched_df)

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
            m.hasAcceptedAnswer,
            m.firstAnswerScore,
            m.firstAnswerBodyLenChars,
            m.viewCount,
            m.postHour,
            m.postDayOfWeek,
            m.numTags,
            m.bodyLenChars,
            m.titleLenChars,
            m.ownerReputation,
            COALESCE(t.responseTimeHours, m.responseTimeHours) AS match_response_hours
        FROM matched_raw m
        LEFT JOIN treatment_times t ON m.match_id = t.match_id
        WHERE COALESCE(t.responseTimeHours, m.responseTimeHours) IS NOT NULL
    """)

    questions_path = resolve_input_file("posts_questions")
    answers_path = resolve_input_file("posts_answers")
    users_path = resolve_input_file("users")

    print("Loading raw questions, answers, and users...")
    con.execute(f"""
        CREATE VIEW raw_questions AS
        SELECT Id, OwnerUserId, CreationDate::TIMESTAMP AS ts
        FROM '{questions_path}'
    """)
    con.execute(f"""
        CREATE VIEW raw_answers AS
        SELECT Id, OwnerUserId, ParentId, CreationDate::TIMESTAMP AS ts
        FROM '{answers_path}'
    """)

    if os.path.exists(users_path):
        con.execute(f"""
            CREATE VIEW raw_users AS
            SELECT Id, CreationDate::TIMESTAMP AS registration_ts
            FROM '{users_path}'
        """)
    else:
        print(f"WARNING: {users_path} not found. Tenure will be NULL.")
        con.execute("CREATE VIEW raw_users AS SELECT NULL::INTEGER AS Id, NULL::TIMESTAMP AS registration_ts WHERE 1=0")

    print("Calculating study windows per question and asker tenure...")
    con.execute(f"""
        CREATE TEMPORARY TABLE study_windows AS
        SELECT
            mt.match_id,
            mt.question_id,
            mt.user_id,
            mt.hasAnswer,
            mt.question_year,
            mt.hasAcceptedAnswer,
            mt.firstAnswerScore,
            mt.firstAnswerBodyLenChars,
            mt.viewCount,
            mt.postHour,
            mt.postDayOfWeek,
            mt.numTags,
            mt.bodyLenChars,
            mt.titleLenChars,
            mt.ownerReputation,
            mt.match_response_hours,
            rq.ts AS question_ts,
            GREATEST(0, date_diff('day', u.registration_ts, rq.ts)) AS user_tenure_days,
            rq.ts - INTERVAL '{days_before_question} DAYS' AS window_start_ts,
            rq.ts + INTERVAL '1 SECOND' * CAST(mt.match_response_hours * 3600 AS BIGINT) AS answer_ts,
            (rq.ts + INTERVAL '1 SECOND' * CAST(mt.match_response_hours * 3600 AS BIGINT))
                + INTERVAL '{days_after_answer} DAYS' AS window_end_ts
        FROM match_timelines mt
        JOIN raw_questions rq ON mt.question_id = rq.Id
        LEFT JOIN raw_users u ON mt.user_id = u.Id
    """)

    print("Finding answer help events within each question's window...")
    con.execute("""
        CREATE TEMPORARY TABLE answer_help_events AS
        SELECT
            sw.match_id,
            sw.question_id,
            sw.user_id,
            ra.ts AS help_ts,
            date_diff('second', sw.question_ts, ra.ts) / 3600.0 AS relative_help_time_hours,
            'answer' AS help_type
        FROM study_windows sw
        JOIN raw_answers ra ON sw.user_id = ra.OwnerUserId
        LEFT JOIN raw_questions q_parent ON ra.ParentId = q_parent.Id
        WHERE ra.ts >= sw.window_start_ts
          AND ra.ts <= sw.window_end_ts
          AND (q_parent.OwnerUserId IS NULL OR ra.OwnerUserId != q_parent.OwnerUserId)
    """)

    if include_composite_help and input_file_exists("comments"):
        comments_path = resolve_input_file("comments")
        print("Finding comment help events (composite outcome)...")
        con.execute(f"""
            CREATE VIEW raw_comments AS
            SELECT
                Id,
                PostId,
                UserId,
                CreationDate::TIMESTAMP AS ts
            FROM '{comments_path}'
            WHERE UserId IS NOT NULL
        """)
        con.execute("""
            CREATE TEMPORARY TABLE comment_help_events AS
            SELECT
                sw.match_id,
                sw.question_id,
                sw.user_id,
                c.ts AS help_ts,
                date_diff('second', sw.question_ts, c.ts) / 3600.0 AS relative_help_time_hours,
                'comment' AS help_type
            FROM study_windows sw
            JOIN raw_comments c ON sw.user_id = c.UserId
            JOIN raw_questions q ON c.PostId = q.Id
            WHERE c.ts >= sw.window_start_ts
              AND c.ts <= sw.window_end_ts
              AND q.OwnerUserId IS NOT NULL
              AND c.UserId != q.OwnerUserId
        """)
    else:
        con.execute("""
            CREATE TEMPORARY TABLE comment_help_events AS
            SELECT NULL::VARCHAR AS match_id, NULL::BIGINT AS question_id, NULL::BIGINT AS user_id,
                   NULL::TIMESTAMP AS help_ts, NULL::DOUBLE AS relative_help_time_hours,
                   NULL::VARCHAR AS help_type
            WHERE 1=0
        """)

    print("Finding accept help events (asker accepted an answer on own question)...")
    con.execute(f"""
        CREATE VIEW raw_questions_accept AS
        SELECT Id, OwnerUserId, AcceptedAnswerId, CreationDate::TIMESTAMP AS ts
        FROM '{questions_path}'
        WHERE AcceptedAnswerId IS NOT NULL
    """)
    con.execute(f"""
        CREATE VIEW raw_answers_accept AS
        SELECT Id, ParentId, CreationDate::TIMESTAMP AS ts
        FROM '{answers_path}'
    """)
    con.execute("""
        CREATE TEMPORARY TABLE accept_help_events AS
        SELECT
            sw.match_id,
            sw.question_id,
            sw.user_id,
            a.ts AS help_ts,
            date_diff('second', sw.question_ts, a.ts) / 3600.0 AS relative_help_time_hours,
            'accept' AS help_type
        FROM study_windows sw
        JOIN raw_questions_accept q ON sw.question_id = q.Id AND sw.user_id = q.OwnerUserId
        JOIN raw_answers_accept a ON q.AcceptedAnswerId = a.Id
        WHERE a.ts >= sw.window_start_ts
          AND a.ts <= sw.window_end_ts
    """)

    con.execute("""
        CREATE TEMPORARY TABLE all_help_events AS
        SELECT match_id, question_id, user_id, relative_help_time_hours AS t_event, help_type
        FROM answer_help_events
        UNION ALL
        SELECT match_id, question_id, user_id, relative_help_time_hours, help_type
        FROM comment_help_events
        UNION ALL
        SELECT match_id, question_id, user_id, relative_help_time_hours, help_type
        FROM accept_help_events
    """)

    print("Exporting Event Log...")
    os.makedirs(output_folder, exist_ok=True)

    timelines_path = os.path.join(output_folder, "study_timelines.parquet")
    events_path = os.path.join(output_folder, "study_events.parquet")

    con.execute(f"""
        COPY (
            SELECT
                match_id,
                question_id,
                user_id,
                hasAnswer,
                question_year,
                hasAcceptedAnswer,
                firstAnswerScore,
                firstAnswerBodyLenChars,
                viewCount,
                postHour,
                postDayOfWeek,
                numTags,
                bodyLenChars,
                titleLenChars,
                ownerReputation,
                user_tenure_days,
                date_diff('second', question_ts, window_start_ts) / 3600.0 AS t_start,
                0.0 AS t_question,
                date_diff('second', question_ts, answer_ts) / 3600.0 AS t_answer,
                date_diff('second', question_ts, window_end_ts) / 3600.0 AS t_end
            FROM study_windows
        ) TO '{timelines_path}' (FORMAT PARQUET);
    """)

    con.execute(f"""
        COPY (
            SELECT match_id, question_id, user_id, t_event, help_type
            FROM all_help_events
        ) TO '{events_path}' (FORMAT PARQUET);
    """)

    print(f"Success! Output saved to {output_folder}")

    n_questions = con.execute("SELECT COUNT(*) FROM study_windows").fetchone()[0]
    n_events = con.execute("SELECT COUNT(*) FROM all_help_events").fetchone()[0]
    n_answers = con.execute("SELECT COUNT(*) FROM answer_help_events").fetchone()[0]
    n_comments = con.execute("SELECT COUNT(*) FROM comment_help_events").fetchone()[0]
    n_accepts = con.execute("SELECT COUNT(*) FROM accept_help_events").fetchone()[0]
    avg_tenure = con.execute("SELECT AVG(user_tenure_days) FROM study_windows").fetchone()[0]

    print(f"\nStats:")
    print(f" - Questions (timelines) analyzed: {n_questions:,}")
    print(f" - Help events (all types): {n_events:,}")
    print(f"   - answers: {n_answers:,}, comments: {n_comments:,}, accepts: {n_accepts:,}")
    print(f" - Average asker tenure: {avg_tenure:.1f} days" if avg_tenure else " - Average asker tenure: N/A")

    con.close()


if __name__ == "__main__":
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
        days_after_answer=2,
    )
