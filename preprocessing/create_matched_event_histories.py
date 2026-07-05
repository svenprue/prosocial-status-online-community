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

    # 2. Create match timelines (one row per matched question; timelines keyed by question_id)
    print("Aligning timelines for matched pairs (per question_id)...")
    con.execute("""
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
            COALESCE(t.responseTimeHours, m.responseTimeHours) AS match_response_hours
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
            mt.match_response_hours,
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
                user_tenure_days,
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
