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

    # Load data with explicit schema and pushdown predicates
    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            AcceptedAnswerId AS accepted_answer_id,
            CreationDate AS creation_date
        FROM '{questions_path}'
        WHERE AcceptedAnswerId IS NOT NULL;
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            OwnerUserId AS owner_user_id,
            ParentId AS parent_question_id,
            CreationDate AS creation_date
        FROM '{answers_path}';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW votes AS
        SELECT
            PostId AS answer_id,
            CreationDate AS vote_date
        FROM '{votes_path}'
        WHERE VoteTypeId = 1;
    """)

    # Efficient eligible answers using temporary tables
    con.execute("""
        CREATE TEMPORARY TABLE eligible_base AS
        SELECT
            q.question_id,
            q.owner_user_id AS asker_id,
            a.answer_id,
            a.owner_user_id AS responder_id,
            a.creation_date AS answer_date,
            q.creation_date AS question_date,
            v.vote_date,
            EXTRACT(EPOCH FROM (a.creation_date - q.creation_date)) / 60.0 AS response_time
        FROM questions q
        JOIN answers a ON q.accepted_answer_id = a.answer_id
        JOIN votes v ON a.answer_id = v.answer_id
        WHERE q.owner_user_id <> a.owner_user_id;
    """)

    if include_all_accepted_answers:
        con.execute("""
            CREATE TEMPORARY VIEW eligible_answers AS
            SELECT * FROM eligible_base;
        """)
    else:
        # Optimized sampling using hash instead of random()
        con.execute("""
            CREATE TEMPORARY VIEW eligible_answers AS
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY asker_id
                        ORDER BY HASH(answer_id)
                    ) AS rn
                FROM eligible_base
            ) sub
            WHERE rn = 1;
        """)

    # Replace both phase_definitions creations with this single block:
    con.execute(f"""
        CREATE TEMPORARY TABLE phase_definitions AS
        SELECT
            *,
            DENSE_RANK() OVER (ORDER BY asker_id, question_id) AS event_id,
            COALESCE(
                CASE WHEN CAST(answer_date AS DATE) = CAST(vote_date AS DATE) THEN answer_date END,
                answer_date
            ) AS phase_two_start,
            phase_two_start - INTERVAL '{window_length} DAYS' AS phase_one_start,
            phase_two_start + INTERVAL '{window_length} DAYS' AS phase_two_end
        FROM eligible_answers;
    """)

    # Historical events entirely in SQL
    con.execute("""
        CREATE TEMPORARY TABLE historical_events AS
        SELECT
            NULL::INTEGER AS event_id,
            owner_user_id AS user_id,
            creation_date AS timestamp,
            'Question' AS event,
            NULL::INTEGER AS question_id,
            NULL::TIMESTAMP AS phase_one_start,
            NULL::TIMESTAMP AS phase_two_end,
            'Question' AS event_history,
            1 AS is_history,
            NULL::DOUBLE AS response_time
        FROM questions
        WHERE owner_user_id IN (SELECT asker_id FROM eligible_answers)
        UNION ALL
        SELECT
            NULL::INTEGER,
            q.owner_user_id,
            a.creation_date,
            'AcceptedAnswer',
            NULL::INTEGER,
            NULL::TIMESTAMP,
            NULL::TIMESTAMP,
            'AcceptedAnswer',
            1,
            NULL::DOUBLE
        FROM questions q
        JOIN answers a ON q.accepted_answer_id = a.answer_id
        WHERE q.owner_user_id IN (SELECT asker_id FROM eligible_answers)
        UNION ALL
        SELECT
            NULL::INTEGER,
            owner_user_id,
            creation_date,
            'Answer',
            NULL::INTEGER,
            NULL::TIMESTAMP,
            NULL::TIMESTAMP,
            'Answer',
            1,
            NULL::DOUBLE
        FROM answers
        WHERE owner_user_id IN (SELECT asker_id FROM eligible_answers);
    """)

    # Current events using optimized UNIONs
    con.execute(f"""
        CREATE TEMPORARY TABLE current_events AS
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
        FROM phase_definitions
        UNION ALL
        SELECT
            event_id,
            asker_id,
            phase_two_start,
            'Phase_Two_Start',
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_Start',
            0,
            response_time
        FROM phase_definitions
        UNION ALL
        SELECT
            event_id,
            asker_id,
            phase_two_end,
            'Phase_Two_End',
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_End',
            0,
            response_time
        FROM phase_definitions
        UNION ALL
        SELECT
            p.event_id,
            a.owner_user_id,
            a.creation_date,
            'Window_Answer',
            p.question_id,
            p.phase_one_start,
            p.phase_two_end,
            'Window_Answer',
            0,
            NULL
        FROM phase_definitions p
        JOIN answers a
          ON a.owner_user_id = p.asker_id
         AND a.creation_date BETWEEN p.phase_two_start AND p.phase_two_end
        JOIN questions q
          ON a.parent_question_id = q.question_id
        WHERE q.owner_user_id <> p.asker_id;
    """)

    # Final output with direct Parquet write
    output_path = os.path.join(
        output_folder,
        f"reciprocity_model_{window_length}d_{'all_answers' if include_all_accepted_answers else 'one_answer'}.parquet"
    )

    con.execute(f"""
        COPY (
            SELECT * FROM historical_events
            UNION ALL
            SELECT * FROM current_events
        )
        TO '{output_path}' (FORMAT PARQUET);
    """)

    con.close()
    print(f"Done! Saved dataset to {output_path}")

if __name__ == "__main__":
    input_data_folder = r".\01_input_data\processed_SO_data_dump"
    output_data_folder = r".\02_raw_datasets"
    os.makedirs(output_data_folder, exist_ok=True)

    for days in [3, 7, 14, 30]:
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