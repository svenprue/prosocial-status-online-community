import os
import duckdb
import pandas as pd
from tqdm import tqdm


def process_bounty_data(
        input_folder: str,
        output_folder: str
) -> None:
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA max_temp_directory_size='30GiB'")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar;")

    # Define paths
    bounty_timeline_path = os.path.join(input_folder, 'scraped_datasets', 'bounty_timeline_results_cleaned.parquet')
    questions_path = os.path.join(input_folder, 'processed_data_dump', 'posts_questions.parquet')
    answers_path = os.path.join(input_folder, 'processed_data_dump', 'posts_answers.parquet')
    votes_path = os.path.join(input_folder, 'processed_data_dump', 'Votes.parquet')

    # Create view for bounty timeline with row number to handle duplicates
    con.execute(f"""
        CREATE TEMPORARY VIEW bounty_timeline AS
        WITH ranked_bounties AS (
            SELECT
                question_id,
                CAST(bounty_start AS TIMESTAMP) AS bounty_start,
                CAST(bounty_end AS TIMESTAMP) AS bounty_end,
                ROW_NUMBER() OVER (
                    PARTITION BY question_id 
                    ORDER BY CAST(bounty_start AS TIMESTAMP)
                ) AS rn
            FROM '{bounty_timeline_path}'
        )
        SELECT
            question_id,
            bounty_start,
            bounty_end
        FROM ranked_bounties
        WHERE rn = 1;
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            AcceptedAnswerId,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{questions_path}';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            ParentId AS question_id,
            OwnerUserId AS owner_user_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{answers_path}';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW bounty_votes AS
        SELECT
            PostId AS question_id,
            BountyAmount
        FROM '{votes_path}'
        WHERE VoteTypeId = 8;
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW accept_votes AS
        SELECT
            PostId AS post_id,
            UserId AS voter_id,
            CAST(CreationDate AS TIMESTAMP) AS vote_date
        FROM '{votes_path}'
        WHERE VoteTypeId = 1;
    """)

    # Create main bounty dataset - mark these as non-historical (is_history = 0)
    con.execute("""
        CREATE TEMPORARY TABLE bounty_answers AS
        WITH bounty_data AS (
            SELECT
                bt.question_id,
                q.creation_date AS question_posted,
                bt.bounty_start,
                bt.bounty_end,
                bv.BountyAmount
            FROM bounty_timeline bt
            JOIN questions q ON bt.question_id = q.question_id
            LEFT JOIN bounty_votes bv ON bt.question_id = bv.question_id
        )
        SELECT
            b.question_id,
            b.question_posted,
            b.bounty_start,
            b.bounty_end,
            b.BountyAmount,
            a.owner_user_id AS user_id,
            a.creation_date AS answer_timestamp,
            CASE
                WHEN a.creation_date BETWEEN b.question_posted AND b.bounty_start THEN 0
                WHEN a.creation_date BETWEEN b.bounty_start AND b.bounty_end THEN 1
            END AS after_bounty,
            'Bounty_Answer' AS event_history,
            0 AS is_history
        FROM bounty_data b
        JOIN answers a ON b.question_id = a.question_id
        WHERE a.creation_date BETWEEN b.question_posted AND b.bounty_end;
    """)

    # Create historical events - all three types consistent with the first code file
    # 1. Historical questions asked
    con.execute("""
        CREATE TEMPORARY VIEW historical_questions AS
        SELECT
            owner_user_id AS user_id,
            creation_date AS timestamp,
            'Question' AS event_history,
            1 AS is_history
        FROM questions
        WHERE owner_user_id IN (
            SELECT DISTINCT user_id FROM bounty_answers
        );
    """)

    # 2. Historical answers provided
    con.execute("""
        CREATE TEMPORARY VIEW historical_answers AS
        SELECT
            a.owner_user_id AS user_id,
            a.creation_date AS timestamp,
            'Answer' AS event_history,
            1 AS is_history
        FROM answers a
        WHERE a.owner_user_id IN (
            SELECT DISTINCT user_id FROM bounty_answers
        );
    """)

    # 3. Historical accepted answers received
    con.execute("""
        CREATE TEMPORARY VIEW historical_accepted_answers AS
        SELECT
            q.owner_user_id AS user_id,
            a.creation_date AS timestamp,
            'AcceptedAnswer' AS event_history,
            1 AS is_history
        FROM questions q
        JOIN answers a ON q.AcceptedAnswerId = a.answer_id
        WHERE q.owner_user_id IN (
            SELECT DISTINCT user_id FROM bounty_answers
        )
        AND q.AcceptedAnswerId IS NOT NULL;
    """)

    # 4. Historical accepted answers provided
    con.execute("""
        CREATE TEMPORARY VIEW historical_answers_accepted AS
        SELECT
            a.owner_user_id AS user_id,
            a.creation_date AS answer_timestamp,
            v.vote_date AS acceptance_date,
            a.answer_id,
            q.question_id,
            'AcceptedAnswerProvided' AS event_history,
            1 AS is_history
        FROM answers a
        JOIN questions q ON a.answer_id = q.AcceptedAnswerId
        JOIN accept_votes v ON a.answer_id = v.post_id
        WHERE a.owner_user_id IN (
            SELECT DISTINCT user_id FROM bounty_answers
        )
        AND q.AcceptedAnswerId IS NOT NULL;
    """)

    # Combine all events
    output_path = os.path.join(output_folder, 'bounty_raw_dataset.parquet')

    con.execute(f"""
        COPY (
            SELECT
                question_id,
                question_posted,
                bounty_start,
                bounty_end,
                BountyAmount,
                user_id,
                answer_timestamp,
                NULL AS acceptance_date,
                after_bounty,
                event_history,
                is_history
            FROM bounty_answers

            UNION ALL

            SELECT
                NULL AS question_id,
                NULL AS question_posted,
                NULL AS bounty_start,
                NULL AS bounty_end,
                NULL AS BountyAmount,
                user_id,
                timestamp AS answer_timestamp,
                NULL AS acceptance_date,
                NULL AS after_bounty,
                event_history,
                is_history
            FROM historical_questions

            UNION ALL

            SELECT
                NULL AS question_id,
                NULL AS question_posted,
                NULL AS bounty_start,
                NULL AS bounty_end,
                NULL AS BountyAmount,
                user_id,
                timestamp AS answer_timestamp,
                NULL AS acceptance_date,
                NULL AS after_bounty,
                event_history,
                is_history
            FROM historical_answers

            UNION ALL

            SELECT
                NULL AS question_id,
                NULL AS question_posted,
                NULL AS bounty_start,
                NULL AS bounty_end,
                NULL AS BountyAmount,
                user_id,
                timestamp AS answer_timestamp,
                NULL AS acceptance_date,
                NULL AS after_bounty,
                event_history,
                is_history
            FROM historical_accepted_answers

            UNION ALL

            SELECT
                question_id,
                NULL AS question_posted,
                NULL AS bounty_start,
                NULL AS bounty_end,
                NULL AS BountyAmount,
                user_id,
                answer_timestamp,
                acceptance_date,
                NULL AS after_bounty,
                event_history,
                is_history
            FROM historical_answers_accepted

            ORDER BY user_id, answer_timestamp
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.close()
    print(f"Done! Saved dataset to {output_path}")


if __name__ == "__main__":
    input_data_folder = r".\01_input_data"
    output_data_folder = r".\02_raw_datasets"
    os.makedirs(output_data_folder, exist_ok=True)

    process_bounty_data(
        input_folder=input_data_folder,
        output_folder=output_data_folder
    )