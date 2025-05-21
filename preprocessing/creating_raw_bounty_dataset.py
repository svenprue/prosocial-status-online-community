import os
import duckdb
import pandas as pd
from pathlib import Path

import tqdm
from tqdm import tqdm


def create_user_answers_dataset(
        input_folder: str,
        output_folder: str,
        bounty_timeline_path: str,
        memory_limit: str = '10GB',
        temp_dir_size: str = '300GiB',
        threads: int = 4,
        batch_size: int = 500000
) -> None:
    """
    Create a dataset of user answers with bounties from Stack Overflow data.

    Args:
        input_folder: Folder containing input Parquet files
        output_folder: Folder where output dataset will be saved
        bounty_timeline_path: Path to the cleaned bounty timeline data
        memory_limit: DuckDB memory limit
        temp_dir_size: DuckDB temporary directory size
        threads: Number of threads for DuckDB to use
        batch_size: Number of users to process in each batch
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
        for file_path in [answers_path, votes_path, questions_path, bounty_timeline_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")

        # Load bounty timeline data
        # TODO: 2) Remove deleted Qs as of scraper from sample + those without start & end
        # TODO: 3) Check if we correctly create dataset for multiple bounty timeframes
        con.execute(f"""
           CREATE TEMPORARY VIEW bounty_timeline AS
           SELECT 
               question_id,
               bounty_start,
               bounty_end
           FROM '{bounty_timeline_path}' 
           WHERE 
               bounty_start IS NOT NULL 
               AND bounty_end IS NOT NULL;
       """)

        # Load ALL answers
        con.execute(f"""
           CREATE TEMPORARY VIEW answers AS
           SELECT
               Id AS answer_id,
               OwnerUserId AS owner_user_id,
               ParentId AS parent_question_id,
               CAST(Score AS INTEGER) AS score,
               CAST(CreationDate AS TIMESTAMP) AS creation_date
           FROM '{answers_path}'
       """)

        # Load all questions
        con.execute(f"""
           CREATE TEMPORARY VIEW questions AS
           SELECT
               Id AS question_id,
               OwnerUserId AS owner_user_id,
               AcceptedAnswerId AS accepted_answer_id,
               CAST(CreationDate AS TIMESTAMP) AS creation_date
           FROM '{questions_path}';
       """)

        # Load all votes
        con.execute(f"""
           CREATE TEMPORARY VIEW votes AS
           SELECT
               PostId AS post_id,
               VoteTypeId AS vote_type_id,
               UserId AS voter_user_id,
               CAST(CreationDate AS TIMESTAMP) AS vote_date,
               CAST(BountyAmount AS INTEGER) AS bounty_amount
           FROM '{votes_path}';
       """)

        # Count total answers before filtering
        total_answers_result = con.execute("SELECT COUNT(*) as total FROM answers").fetchone()
        total_answers = total_answers_result[0]
        print(f"Total answers in dataset: {total_answers:,}")

        # Create view with answers joined to questions using LEFT JOIN
        con.execute("""
           CREATE TEMPORARY VIEW answers_with_questions AS
           SELECT
               a.owner_user_id AS user_id,
               a.answer_id,
               a.parent_question_id AS question_id,
               a.creation_date AS timestamp,
               DENSE_RANK() OVER (PARTITION BY a.owner_user_id ORDER BY a.creation_date) AS answer_sequence,
               q.owner_user_id AS question_owner_id
           FROM answers a
           LEFT JOIN questions q ON a.parent_question_id = q.question_id
           WHERE a.owner_user_id IS NOT NULL
           AND (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)
       """)

        # Count answers after filtering
        filtered_answers_result = con.execute("SELECT COUNT(*) as filtered FROM answers_with_questions").fetchone()
        filtered_answers = filtered_answers_result[0]
        print(f"Answers after excluding self-answers: {filtered_answers:,}")
        print(f"Self-answers excluded: {total_answers - filtered_answers:,}")

        # Now check if any answer falls within any bounty period
        con.execute("""
                    CREATE
                    TEMPORARY TABLE answers_with_bounty_info AS
                    SELECT a.user_id,
                           a.answer_id,
                           a.question_id,
                           a.timestamp,
                           a.answer_sequence,
                           a.question_owner_id,
                           CASE
                               WHEN EXISTS (SELECT 1
                                            FROM bounty_timeline bt
                                            WHERE a.question_id = bt.question_id
                                              AND a.timestamp BETWEEN bt.bounty_start AND bt.bounty_end)
                                   THEN 1
                               ELSE 0
                               END AS is_bounty
                    FROM answers_with_questions a;
                    """)

        # Count bounty answers
        bounty_answers_result = con.execute(
            "SELECT COUNT(*) as bounty_count FROM answers_with_bounty_info WHERE is_bounty = 1").fetchone()
        bounty_answers = bounty_answers_result[0]
        print(f"Answers during bounty periods: {bounty_answers:,}")

        # Add bounty amount information
        con.execute("""
                    CREATE
                    TEMPORARY TABLE user_answers AS
                    SELECT a.user_id,
                           a.answer_id,
                           a.question_id,
                           a.timestamp,
                           a.is_bounty,
                           COALESCE(b.bounty_amount, 0) AS bounty_amount,
                           a.answer_sequence,
                           a.question_owner_id
                    FROM answers_with_bounty_info a
                             LEFT JOIN (SELECT post_id, bounty_amount
                                        FROM votes
                                        WHERE vote_type_id = 8) b ON a.answer_id = b.post_id;
                    """)

        # Get all unique users to process in batches
        unique_users_result = con.execute("SELECT DISTINCT user_id FROM user_answers ORDER BY user_id").fetchall()
        all_user_ids = [row[0] for row in unique_users_result]
        total_users = len(all_user_ids)
        print(f"Total unique users to process: {total_users:,}")

        # Calculate number of batches
        num_batches = (total_users + batch_size - 1) // batch_size
        print(f"Processing in {num_batches} batches of {batch_size:,} users each")

        # Prepare output path
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "user_answers_bounty_dataset.parquet")

        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Track if this is the first write
        first_write = True

        # Process users in batches
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_users)
            batch_user_ids = all_user_ids[start_idx:end_idx]

            print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (users {start_idx:,} to {end_idx - 1:,})...")

            # Create a temporary table with current batch user IDs
            batch_users_df = pd.DataFrame({'user_id': batch_user_ids})
            con.register("batch_users", batch_users_df)

            # Helper function to append DataFrame to output file
            def append_to_output(df, event_type_name):
                nonlocal first_write
                if df.empty:
                    print(f"    {event_type_name}: 0 events")
                    return

                if first_write:
                    # First write: create the file
                    df.to_parquet(output_path, compression='gzip', index=False)
                    first_write = False
                else:
                    # Subsequent writes: append to existing file
                    existing_df = pd.read_parquet(output_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_parquet(output_path, compression='gzip', index=False)

                print(f"    {event_type_name}: {len(df):,} events")

                # Clear the DataFrame from memory
                del df
                import gc
                gc.collect()

            # Current answer events for this batch
            user_answer_events_df = con.execute("""
                                                SELECT a.user_id,
                                                       a.timestamp,
                                                       'Answer' AS event,
                                                       a.answer_id,
                                                       a.question_id,
                                                       a.is_bounty,
                                                       a.bounty_amount,
                                                       a.answer_sequence,
                                                       0        AS is_history
                                                FROM user_answers a
                                                WHERE a.user_id IN (SELECT user_id FROM batch_users)
                                                """).fetchdf()
            append_to_output(user_answer_events_df, "Current Answer events")

            # Historical events: Questions asked by users
            questions_asked_df = con.execute("""
                                             SELECT q.owner_user_id AS user_id,
                                                    q.creation_date AS timestamp,
                   'Question' AS event,
                   NULL AS answer_id,
                   q.question_id,
                   0 AS is_bounty, 
                   0 AS bounty_amount,
                   NULL AS answer_sequence,
                   1 AS is_history
                                             FROM questions q
                                             WHERE q.owner_user_id IN (SELECT user_id FROM batch_users)
                                             """).fetchdf()
            append_to_output(questions_asked_df, "Historical Question events")

            # Historical events: Answers provided by users (excluding self-answers)
            answers_provided_df = con.execute("""
                                              SELECT a.user_id,
                                                     a.timestamp,
                                                     'AnswerProvided' AS event,
                                                     a.answer_id,
                                                     a.question_id,
                                                     a.is_bounty,
                                                     a.bounty_amount,
                                                     a.answer_sequence,
                                                     1                AS is_history
                                              FROM user_answers a
                                              WHERE a.user_id IN (SELECT user_id FROM batch_users)
                                              """).fetchdf()
            append_to_output(answers_provided_df, "Historical AnswerProvided events")

            # Historical events: Answers received by users to their questions
            answers_received_df = con.execute("""
                                              SELECT q.owner_user_id AS user_id,
                                                     a.creation_date AS timestamp,
                   'AnswerReceived' AS event,
                   a.answer_id,
                   q.question_id,
                   0 AS is_bounty,
                   0 AS bounty_amount,
                   NULL AS answer_sequence,
                   1 AS is_history
                                              FROM questions q
                                                  JOIN answers a
                                              ON q.question_id = a.parent_question_id
                                              WHERE q.owner_user_id IN (SELECT user_id FROM batch_users)
                                                AND (a.owner_user_id <> q.owner_user_id OR a.owner_user_id IS NULL)
                                                AND a.score >= 0 -- only non-negative answers
                                              """).fetchdf()
            append_to_output(answers_received_df, "Historical AnswerReceived events")

            # Historical events: Accepted answers received by users
            accepted_answers_received_df = con.execute("""
                                                       SELECT q.owner_user_id AS user_id,
                                                              a.creation_date AS timestamp,
                   'AcceptedAnswerReceived' AS event,
                   a.answer_id,
                   q.question_id,
                   0 AS is_bounty,
                   0 AS bounty_amount,
                   NULL AS answer_sequence,
                   1 AS is_history
                                                       FROM questions q
                                                           JOIN answers a
                                                       ON q.accepted_answer_id = a.answer_id
                                                       WHERE q.owner_user_id IN (SELECT user_id FROM batch_users)
                                                         AND (a.owner_user_id <> q.owner_user_id OR a.owner_user_id IS NULL)
                                                       """).fetchdf()
            append_to_output(accepted_answers_received_df, "Historical AcceptedAnswerReceived events")

            # Accept votes received (vote type 1)
            accepted_vote_received_df = con.execute("""
                                                    SELECT a.owner_user_id AS user_id,
                                                           v.vote_date AS timestamp,
                   'AcceptedAnswerVote' AS event,
                   a.answer_id,
                   a.parent_question_id AS question_id,
                   0 AS is_bounty,
                   0 AS bounty_amount,
                   NULL AS answer_sequence,
                   1 AS is_history
                                                    FROM answers a
                                                        JOIN votes v
                                                    ON a.answer_id = v.post_id
                                                    WHERE a.owner_user_id IN (SELECT user_id FROM batch_users)
                                                      AND v.vote_type_id = 1 -- Accept vote
                                                    """).fetchdf()
            append_to_output(accepted_vote_received_df, "Historical AcceptedAnswerVote events")

            # Historical events: AcceptedAnswerPosted
            accepted_answers_posted_df = con.execute("""
                                                     SELECT a.owner_user_id AS user_id,
                                                            a.creation_date AS timestamp,
                   'AcceptedAnswerPosted' AS event,
                   a.answer_id,
                   q.question_id,
                   0 AS is_bounty,
                   0 AS bounty_amount,
                   NULL AS answer_sequence,
                   1 AS is_history
                                                     FROM answers a
                                                         JOIN questions q
                                                     ON q.accepted_answer_id = a.answer_id
                                                     WHERE a.owner_user_id IN (SELECT user_id FROM batch_users)
                                                       AND (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)
                                                     """).fetchdf()
            append_to_output(accepted_answers_posted_df, "Historical AcceptedAnswerPosted events")

        # Final summary
        final_df = pd.read_parquet(output_path)
        total_final_events = len(final_df)
        print(f"\nFinal dataset summary:")
        print(f"Total events in final dataset: {total_final_events:,}")

        # Count events by type in final dataset
        print("\nFinal event type breakdown:")
        event_breakdown = final_df.groupby(['event', 'is_history']).size().reset_index(name='count')
        for _, row in event_breakdown.iterrows():
            history_label = "Historical" if row['is_history'] else "Current"
            print(f"  {history_label} {row['event']}: {row['count']:,}")

        con.close()
        print(f"\nDone! Saved dataset to {output_path}")

    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise


if __name__ == "__main__":
    base_dir = Path("..")
    input_data_folder = base_dir / "data" / "input"
    output_data_folder = base_dir / "data" / "input"
    bounty_timeline_path = base_dir / "data" / "input" / "bounty_timeline.parquet"

    create_user_answers_dataset(
        input_folder=str(input_data_folder),
        output_folder=str(output_data_folder),
        bounty_timeline_path=str(bounty_timeline_path)
    )