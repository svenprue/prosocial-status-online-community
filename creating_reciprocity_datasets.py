import os
import duckdb
import pandas as pd

def process_data(
    input_folder: str,
    output_folder: str,
    window_length: int,
    include_all_accepted_answers: bool = False
    ) -> None:
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA max_temp_directory_size='30GiB'")
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
        FROM '{questions_path}';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            OwnerUserId AS owner_user_id,
            ParentId AS parent_question_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{answers_path}';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW votes AS
        SELECT
            PostId AS answer_id,
            CAST(CreationDate AS TIMESTAMP) AS vote_date
        FROM '{votes_path}'
        WHERE VoteTypeId = 1;
    """)

    if include_all_accepted_answers:
        con.execute("""
            CREATE OR REPLACE TEMPORARY VIEW eligible_answers AS
            SELECT
                q.question_id,
                q.owner_user_id AS asker_id,
                q.accepted_answer_id,
                a.answer_id,
                a.owner_user_id AS responder_id,
                a.creation_date AS answer_date,
                q.creation_date AS question_date,
                v.vote_date,
                EXTRACT(EPOCH FROM (a.creation_date - q.creation_date)) / 60.0 AS response_time
            FROM questions q
            JOIN answers a
              ON q.accepted_answer_id = a.answer_id
            JOIN votes v
              ON a.answer_id = v.answer_id
            WHERE q.accepted_answer_id IS NOT NULL
              AND q.owner_user_id <> a.owner_user_id;
        """)
    else:
        con.execute("""
            CREATE OR REPLACE TEMPORARY VIEW eligible_answers AS
            WITH raw AS (
                SELECT
                    q.question_id,
                    q.owner_user_id AS asker_id,
                    q.accepted_answer_id,
                    a.answer_id,
                    a.owner_user_id AS responder_id,
                    a.creation_date AS answer_date,
                    q.creation_date AS question_date,
                    v.vote_date,
                    EXTRACT(EPOCH FROM (a.creation_date - q.creation_date)) / 60.0 AS response_time,
                    ROW_NUMBER() OVER (
                        PARTITION BY q.owner_user_id
                        ORDER BY RANDOM()
                    ) AS rn
                FROM questions q
                JOIN answers a
                  ON q.accepted_answer_id = a.answer_id
                JOIN votes v
                  ON a.answer_id = v.answer_id
                WHERE q.accepted_answer_id IS NOT NULL
                  AND q.owner_user_id <> a.owner_user_id
            )
            SELECT *
            FROM raw
            WHERE rn = 1;
        """)

    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            question_id,
            asker_id,
            responder_id,
            answer_id,
            answer_date,
            question_date,
            vote_date,
            DENSE_RANK() OVER (ORDER BY asker_id, question_id) AS event_id,
            CASE
                WHEN CAST(answer_date AS DATE) = CAST(vote_date AS DATE)
                     OR vote_date IS NULL
                THEN answer_date
                ELSE vote_date
            END AS phase_two_start,
            response_time
        FROM eligible_answers;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            *,
            (phase_two_start - INTERVAL '{window_length} DAYS') AS phase_one_start,
            (phase_two_start + INTERVAL '{window_length} DAYS') AS phase_two_end
        FROM phase_definitions;
    """)

    # Historical events: 10 columns in the order:
    # event_id, user_id, timestamp, event, question_id, phase_one_start, phase_two_end, event_history, is_history, response_time
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
            NULL AS response_time
        FROM questions q
        WHERE q.owner_user_id IN (
            SELECT DISTINCT asker_id FROM eligible_answers
        );
    """).fetchdf()

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
            NULL AS response_time
        FROM questions q
        JOIN answers a 
          ON q.accepted_answer_id = a.answer_id
        WHERE q.owner_user_id IN (
            SELECT DISTINCT asker_id FROM eligible_answers
        );
    """).fetchdf()

    provided_answers_df = con.execute("""
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
            NULL AS response_time
        FROM answers a
        WHERE a.owner_user_id IN (
            SELECT DISTINCT asker_id FROM eligible_answers
        );
    """).fetchdf()

    historical_events_df = pd.concat([
        questions_asked_df,
        accepted_answers_df,
        provided_answers_df
    ], ignore_index=True)

    con.register("historical_events_df", historical_events_df)
    con.execute("CREATE TEMPORARY TABLE historical_events AS SELECT * FROM historical_events_df;")

    # Current events: 10 columns with matching order
    con.execute("""
        CREATE TEMPORARY TABLE phase_one_events AS
        SELECT
            event_id,
            asker_id AS user_id,
            phase_one_start AS timestamp,
            'Phase_One_Start' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_One_Start' AS event_history,
            0 AS is_history,
            response_time
        FROM phase_definitions;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE phase_two_start_events AS
        SELECT
            event_id,
            asker_id AS user_id,
            phase_two_start AS timestamp,
            'Phase_Two_Start' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_Start' AS event_history,
            0 AS is_history,
            response_time
        FROM phase_definitions;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE phase_two_end_events AS
        SELECT
            event_id,
            asker_id AS user_id,
            phase_two_end AS timestamp,
            'Phase_Two_End' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_End' AS event_history,
            0 AS is_history,
            response_time
        FROM phase_definitions;
    """)

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
            NULL AS response_time
        FROM phase_definitions p
        JOIN answers a
          ON a.owner_user_id = p.asker_id
         AND a.creation_date BETWEEN p.phase_two_start AND p.phase_two_end
        JOIN questions q
          ON a.parent_question_id = q.question_id
        WHERE q.owner_user_id <> p.asker_id;
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
        f"reciprocity_model_{window_length}d_{'all_answers' if include_all_accepted_answers else 'one_answer'}.parquet"
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
                response_time
            FROM all_events
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.close()
    print(f"Done! Saved dataset to {output_path}")

if __name__ == "__main__":
    input_data_folder = r".\01_input_data\processed_SO_data_dump"
    output_data_folder = r".\02_raw_datasets"
    os.makedirs(output_data_folder, exist_ok=True)

    for days in [14, 7, 3]:
        process_data(
            input_folder=input_data_folder,
            output_folder=output_data_folder,
            window_length=days,
            include_all_accepted_answers=True
        )
        process_data(
            input_folder=input_data_folder,
            output_folder=output_data_folder,
            window_length=days,
            include_all_accepted_answers=False
        )