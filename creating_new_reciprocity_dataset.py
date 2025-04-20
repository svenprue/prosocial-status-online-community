import os
import duckdb
import pandas as pd


def process_question_data(
        input_folder: str,
        output_folder: str,
        window_length: int,
        include_all_questions: bool = False
) -> None:
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA max_temp_directory_size='200GiB'")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar;")

    questions_path = os.path.join(input_folder, 'posts_questions.parquet')
    answers_path = os.path.join(input_folder, 'posts_answers.parquet')
    votes_path = os.path.join(input_folder, 'Votes.parquet')

    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            AcceptedAnswerId AS accepted_answer_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{questions_path}'
        WHERE OwnerUserId IS NOT NULL;
    """)

    # Modified to include Score for answers
    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            OwnerUserId AS owner_user_id,
            ParentId AS parent_question_id,
            CAST(Score AS INTEGER) AS score,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{answers_path}'
        WHERE OwnerUserId IS NOT NULL;
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW votes AS
        SELECT
            PostId AS post_id,
            CAST(CreationDate AS TIMESTAMP) AS vote_date
        FROM '{votes_path}'
        WHERE VoteTypeId = 1;  -- accepted answer vote
    """)

    # Define eligible questions with all the metrics we need
    if include_all_questions:
        con.execute("""
            CREATE OR REPLACE TEMPORARY VIEW eligible_questions AS
            WITH first_answers AS (
                SELECT
                    parent_question_id AS question_id,
                    MIN(creation_date) AS first_answer_timestamp
                FROM answers
                WHERE score >= 0  -- Only consider non-negative score answers
                GROUP BY parent_question_id
            ),
            accepted_answers AS (
                SELECT
                    q.question_id,
                    a.creation_date AS accepted_answer_timestamp,
                    v.vote_date AS accepted_answer_vote_timestamp
                FROM questions q
                LEFT JOIN answers a ON q.accepted_answer_id = a.answer_id
                LEFT JOIN votes v ON a.answer_id = v.post_id
            ),
            answer_counts AS (
                SELECT
                    parent_question_id AS question_id,
                    COUNT(*) AS answer_count
                FROM answers
                WHERE score >= 0  -- Only count answers with non-negative scores
                GROUP BY parent_question_id
            )
            SELECT
                q.question_id,
                q.owner_user_id,
                q.creation_date AS question_timestamp,
                COALESCE(ac.answer_count > 0, FALSE) AS has_answer,
                q.accepted_answer_id IS NOT NULL AS has_accepted_answer,
                fa.first_answer_timestamp,
                aa.accepted_answer_timestamp,
                aa.accepted_answer_vote_timestamp
            FROM questions q
            LEFT JOIN first_answers fa ON q.question_id = fa.question_id
            LEFT JOIN accepted_answers aa ON q.question_id = aa.question_id
            LEFT JOIN answer_counts ac ON q.question_id = ac.question_id
            WHERE q.owner_user_id IN (
                SELECT DISTINCT owner_user_id 
                FROM questions
            );
        """)
    else:
        con.execute("""
            CREATE OR REPLACE TEMPORARY VIEW eligible_questions AS
            WITH first_answers AS (
                SELECT
                    parent_question_id AS question_id,
                    MIN(creation_date) AS first_answer_timestamp
                FROM answers
                WHERE score >= 0  -- Only consider non-negative score answers
                GROUP BY parent_question_id
            ),
            accepted_answers AS (
                SELECT
                    q.question_id,
                    a.creation_date AS accepted_answer_timestamp,
                    v.vote_date AS accepted_answer_vote_timestamp
                FROM questions q
                LEFT JOIN answers a ON q.accepted_answer_id = a.answer_id
                LEFT JOIN votes v ON a.answer_id = v.post_id
            ),
            answer_counts AS (
                SELECT
                    parent_question_id AS question_id,
                    COUNT(*) AS answer_count
                FROM answers
                WHERE score >= 0  -- Only count answers with non-negative scores
                GROUP BY parent_question_id
            ),
            raw AS (
                SELECT
                    q.question_id,
                    q.owner_user_id,
                    q.creation_date AS question_timestamp,
                    COALESCE(ac.answer_count > 0, FALSE) AS has_answer,
                    q.accepted_answer_id IS NOT NULL AS has_accepted_answer,
                    fa.first_answer_timestamp,
                    aa.accepted_answer_timestamp,
                    aa.accepted_answer_vote_timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY q.owner_user_id
                        ORDER BY RANDOM()
                    ) AS rn
                FROM questions q
                LEFT JOIN first_answers fa ON q.question_id = fa.question_id
                LEFT JOIN accepted_answers aa ON q.question_id = aa.question_id
                LEFT JOIN answer_counts ac ON q.question_id = ac.question_id
                WHERE q.owner_user_id IN (
                    SELECT DISTINCT owner_user_id 
                    FROM questions
                )
            )
            SELECT 
                question_id,
                owner_user_id,
                question_timestamp,
                has_answer,
                has_accepted_answer,
                first_answer_timestamp,
                accepted_answer_timestamp,
                accepted_answer_vote_timestamp
            FROM raw
            WHERE rn = 1;
        """)

    # Add the summary print statements here
    question_count = con.execute("SELECT COUNT(*) FROM eligible_questions").fetchone()[0]
    user_count = con.execute("SELECT COUNT(DISTINCT owner_user_id) FROM eligible_questions").fetchone()[0]

    print(f"Summary for window_length={window_length}d:")
    print(f"  - Including {question_count} questions from {user_count} unique users")
    print(f"  - {'All questions' if include_all_questions else 'One question'} per user mode")

    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            question_id,
            owner_user_id,
            question_timestamp,
            has_answer,
            has_accepted_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            DENSE_RANK() OVER (ORDER BY owner_user_id, question_id) AS event_id,
            question_timestamp AS phase_two_start
        FROM eligible_questions;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            *,
            (phase_two_start - INTERVAL '{window_length} DAYS') AS phase_one_start,
            (phase_two_start + INTERVAL '{window_length} DAYS') AS phase_two_end
        FROM phase_definitions;
    """)

    # Historical events: Question Asked
    questions_asked_df = con.execute("""
        SELECT
            NULL AS event_id,
            q.owner_user_id AS user_id,
            q.creation_date AS timestamp,
            'Question' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'Question' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp
        FROM questions q
        WHERE q.owner_user_id IN (
            SELECT DISTINCT owner_user_id FROM eligible_questions
        );
    """).fetchdf()

    # Historical events: Answers provided
    answers_provided_df = con.execute("""
        SELECT
            NULL AS event_id,
            a.owner_user_id AS user_id,
            a.creation_date AS timestamp,
            'Answer' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'Answer' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp
        FROM answers a
        JOIN questions q ON a.parent_question_id = q.question_id
        WHERE a.owner_user_id IN (
            SELECT DISTINCT owner_user_id FROM eligible_questions
        )
        AND a.owner_user_id != q.owner_user_id  -- Exclude self-answers
    """).fetchdf()

    # Historical events: AcceptedAnswers received
    accepted_answers_df = con.execute("""
        SELECT
            NULL AS event_id,
            q.owner_user_id AS user_id,
            a.creation_date AS timestamp,
            'AcceptedAnswer' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'AcceptedAnswer' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp
        FROM questions q
        JOIN answers a 
          ON q.accepted_answer_id = a.answer_id
        WHERE q.owner_user_id IN (
            SELECT DISTINCT owner_user_id FROM eligible_questions
        );
    """).fetchdf()

    # Historical events: Accepted Answer Vote received
    answer_votes_df = con.execute("""
        SELECT
            NULL AS event_id,
            a.owner_user_id AS user_id,
            v.vote_date AS timestamp,
            'AcceptedAnswerVote' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'AcceptedAnswerVote' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp
        FROM answers a
        JOIN votes v ON a.answer_id = v.post_id
        WHERE a.owner_user_id IN (
            SELECT DISTINCT owner_user_id FROM eligible_questions
        )
    """).fetchdf()

    historical_events_df = pd.concat([
        questions_asked_df,
        answers_provided_df,
        accepted_answers_df,
        answer_votes_df
    ], ignore_index=True)

    con.register("historical_events_df", historical_events_df)
    con.execute("CREATE TEMPORARY TABLE historical_events AS SELECT * FROM historical_events_df;")

    # Current events
    con.execute("""
        CREATE TEMPORARY TABLE phase_one_events AS
        SELECT
            event_id,
            owner_user_id AS user_id,
            phase_one_start AS timestamp,
            'Phase_One_Start' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_One_Start' AS event_history,
            0 AS is_history,
            has_answer,
            has_accepted_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp
        FROM phase_definitions;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE phase_two_start_events AS
        SELECT
            event_id,
            owner_user_id AS user_id,
            phase_two_start AS timestamp,
            'Phase_Two_Start' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_Start' AS event_history,
            0 AS is_history,
            has_answer,
            has_accepted_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp
        FROM phase_definitions;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE phase_two_end_events AS
        SELECT
            event_id,
            owner_user_id AS user_id,
            phase_two_end AS timestamp,
            'Phase_Two_End' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_End' AS event_history,
            0 AS is_history,
            has_answer,
            has_accepted_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp
        FROM phase_definitions;
    """)

    # Also filter window answers to only include non-negative score answers
    con.execute("""
        CREATE TEMPORARY TABLE window_answers AS
        SELECT
            p.event_id,
            a.owner_user_id AS user_id,
            a.creation_date AS timestamp,
            'Window_Answer' AS event,
            p.question_id,
            p.phase_one_start,
            p.phase_two_end,
            'Window_Answer' AS event_history,
            0 AS is_history,
            p.has_answer,
            p.has_accepted_answer,
            p.first_answer_timestamp,
            p.accepted_answer_timestamp,
            p.accepted_answer_vote_timestamp,
            p.question_timestamp
        FROM phase_definitions p
        JOIN answers a
          ON a.owner_user_id = p.owner_user_id
         AND a.creation_date BETWEEN p.phase_one_start AND p.phase_two_end
        JOIN questions q
          ON a.parent_question_id = q.question_id
        WHERE q.owner_user_id <> p.owner_user_id;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE current_events AS
        SELECT * FROM phase_one_events
        UNION ALL
        SELECT * FROM phase_two_start_events
        UNION ALL
        SELECT * FROM phase_two_end_events
        UNION ALL
        SELECT * FROM window_answers;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE all_events AS
        SELECT * FROM historical_events
        UNION ALL
        SELECT * FROM current_events;
    """)

    output_path = os.path.join(
        output_folder,
        f"question_centered_model_{window_length}d_{'all_questions' if include_all_questions else 'one_question'}.parquet"
    )

    con.execute(f"""
        COPY (
            SELECT
                event_id,
                user_id,
                timestamp,
                event,
                question_id,
                phase_one_start,
                phase_two_end,
                event_history,
                is_history,
                has_answer,
                has_accepted_answer,
                first_answer_timestamp,
                accepted_answer_timestamp,
                accepted_answer_vote_timestamp,
                question_timestamp
            FROM all_events
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.close()
    print(f"Done! Saved dataset to {output_path}")


if __name__ == "__main__":
    input_data_folder = r".\01_input_data\processed_data_dump"
    output_data_folder = r".\02_raw_datasets"
    os.makedirs(output_data_folder, exist_ok=True)

    for days in [7]:
        process_question_data(
            input_folder=input_data_folder,
            output_folder=output_data_folder,
            window_length=days,
            include_all_questions=True
        )