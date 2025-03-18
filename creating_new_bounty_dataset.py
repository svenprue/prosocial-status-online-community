import os
import duckdb
import pandas as pd
from pathlib import Path


def create_user_answers_dataset(
        input_folder: str,
        output_folder: str,
        memory_limit: str = '10GB',
        temp_dir_size: str = '30GiB',
        threads: int = 4
) -> None:
    """
    Create a dataset of user answers with bounties from Stack Overflow data.

    Args:
        input_folder: Folder containing input Parquet files
        output_folder: Folder where output dataset will be saved
        memory_limit: DuckDB memory limit
        temp_dir_size: DuckDB temporary directory size
        threads: Number of threads for DuckDB to use
    """
    try:
        con = duckdb.connect(database=':memory:')
        con.execute(f"PRAGMA memory_limit='{memory_limit}';")
        con.execute(f"PRAGMA max_temp_directory_size='{temp_dir_size}'")
        con.execute(f"PRAGMA threads={threads};")
        con.execute("PRAGMA enable_progress_bar;")

        answers_path = os.path.join(input_folder, 'posts_answers.parquet')
        votes_path = os.path.join(input_folder, 'Votes.parquet')
        questions_path = os.path.join(input_folder, 'posts_questions.parquet')

        # Verify input files exist
        for file_path in [answers_path, votes_path, questions_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")

        # Load ALL answers
        con.execute(f"""
            CREATE TEMPORARY VIEW answers AS
            SELECT
                Id AS answer_id,
                OwnerUserId AS owner_user_id,
                ParentId AS parent_question_id,
                CAST(CreationDate AS TIMESTAMP) AS creation_date
            FROM '{answers_path}'
            WHERE OwnerUserId IS NOT NULL;
        """)

        # Load questions for historical events
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

        # Load bounty votes
        con.execute(f"""
            CREATE TEMPORARY VIEW bounty_votes AS
            SELECT
                PostId AS post_id,
                VoteTypeId AS vote_type_id,
                CAST(CreationDate AS TIMESTAMP) AS vote_date,
                CAST(BountyAmount AS INTEGER) AS bounty_amount
            FROM '{votes_path}'
            WHERE VoteTypeId = 8;
        """)

        # Create the main dataset of ALL user answers with bounty information
        # Excluding self-answers
        con.execute("""
            CREATE TEMPORARY TABLE user_answers AS
            SELECT
                a.owner_user_id AS user_id,
                a.answer_id,
                a.parent_question_id AS question_id,
                a.creation_date AS timestamp,
                CASE WHEN b.post_id IS NOT NULL THEN 1 ELSE 0 END AS is_bounty,
                COALESCE(b.bounty_amount, 0) AS bounty_amount,
                DENSE_RANK() OVER (PARTITION BY a.owner_user_id ORDER BY a.creation_date) AS answer_sequence,
                q.owner_user_id AS question_owner_id
            FROM answers a
            JOIN questions q ON a.parent_question_id = q.question_id
            LEFT JOIN bounty_votes b ON a.answer_id = b.post_id
            WHERE a.owner_user_id <> q.owner_user_id;  -- Exclude self-answers
        """)

        # Create event IDs for each unique user
        con.execute("""
            CREATE TEMPORARY TABLE user_events AS
            SELECT
                user_id,
                DENSE_RANK() OVER (ORDER BY user_id) AS event_id
            FROM user_answers
            GROUP BY user_id;
        """)

        # Current answer events
        con.execute("""
            CREATE TEMPORARY TABLE user_answer_events AS
            SELECT
                e.event_id,
                a.user_id,
                a.timestamp,
                'Answer' AS event,
                a.answer_id,
                a.question_id,
                a.is_bounty,
                a.bounty_amount,
                a.answer_sequence,
                0 AS is_history
            FROM user_answers a
            JOIN user_events e ON a.user_id = e.user_id;
        """)

        # Create historical events - Questions asked by users
        con.execute("""
            CREATE TEMPORARY TABLE questions_asked_history AS
            SELECT
                e.event_id,
                q.owner_user_id AS user_id,
                q.creation_date AS timestamp,
                'Question' AS event,
                NULL AS answer_id,
                q.question_id,
                0 AS is_bounty, 
                0 AS bounty_amount,
                NULL AS answer_sequence,
                1 AS is_history
            FROM questions q
            JOIN user_events e ON q.owner_user_id = e.user_id;
        """)

        # Create historical events - Answers provided by users (explicitly excluding self-answers)
        con.execute("""
            CREATE TEMPORARY TABLE answers_provided_history AS
            SELECT
                e.event_id,
                a.user_id,
                a.timestamp,
                'History_Answer' AS event,
                a.answer_id,
                a.question_id,
                a.is_bounty,
                a.bounty_amount,
                a.answer_sequence,
                1 AS is_history
            FROM user_answers a
            JOIN user_events e ON a.user_id = e.user_id
            WHERE a.user_id <> a.question_owner_id;  -- Double-check to ensure no self-answers
        """)

        # Create historical events - Accepted answers received by users (excluding self-accepted answers)
        con.execute("""
            CREATE TEMPORARY VIEW accepted_answers AS
            SELECT
                a.answer_id,
                a.owner_user_id,
                a.parent_question_id AS question_id,
                a.creation_date,
                q.owner_user_id AS question_owner_id
            FROM answers a
            JOIN questions q ON a.answer_id = q.accepted_answer_id
            WHERE a.owner_user_id <> q.owner_user_id;  -- Exclude self-accepted answers
        """)

        con.execute("""
            CREATE TEMPORARY TABLE accepted_answers_history AS
            SELECT
                e.event_id,
                aa.owner_user_id AS user_id,
                aa.creation_date AS timestamp,
                'AcceptedAnswer' AS event,
                aa.answer_id,
                aa.question_id,
                0 AS is_bounty,
                0 AS bounty_amount,
                NULL AS answer_sequence,
                1 AS is_history
            FROM accepted_answers aa
            JOIN user_events e ON aa.owner_user_id = e.user_id;
        """)

        # Combine all events
        con.execute("""
            CREATE TEMPORARY TABLE all_events AS
            SELECT
                event_id,
                user_id,
                timestamp,
                event,
                answer_id,
                question_id,
                is_bounty,
                bounty_amount,
                answer_sequence,
                is_history
            FROM user_answer_events

            UNION ALL

            SELECT
                event_id,
                user_id,
                timestamp,
                event,
                answer_id,
                question_id,
                is_bounty,
                bounty_amount,
                answer_sequence,
                is_history
            FROM questions_asked_history

            UNION ALL

            SELECT
                event_id,
                user_id,
                timestamp,
                event,
                answer_id,
                question_id,
                is_bounty,
                bounty_amount,
                answer_sequence,
                is_history
            FROM answers_provided_history

            UNION ALL

            SELECT
                event_id,
                user_id,
                timestamp,
                event,
                answer_id,
                question_id,
                is_bounty,
                bounty_amount,
                answer_sequence,
                is_history
            FROM accepted_answers_history;
        """)

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "user_answers_bounty_dataset.parquet")

        con.execute(f"""
            COPY (
                SELECT
                    event_id,
                    user_id,
                    timestamp,
                    event,
                    answer_id,
                    question_id,
                    is_bounty,
                    bounty_amount,
                    answer_sequence,
                    is_history
                FROM all_events
                ORDER BY user_id, is_history, timestamp
            )
            TO '{output_path}'
            (FORMAT PARQUET, COMPRESSION 'GZIP');
        """)

        con.close()
        print(f"Done! Saved dataset to {output_path}")

    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise


if __name__ == "__main__":
    # Use Path for more portable paths
    base_dir = Path(".")
    input_data_folder = base_dir / "01_input_data" / "processed_data_dump"
    output_data_folder = base_dir / "02_raw_datasets"

    create_user_answers_dataset(
        input_folder=str(input_data_folder),
        output_folder=str(output_data_folder)
    )