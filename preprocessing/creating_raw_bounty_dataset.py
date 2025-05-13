import os
import duckdb
import pandas as pd
from pathlib import Path


def create_user_answers_dataset(
        input_folder: str,
        output_folder: str,
        bounty_timeline_path: str,
        memory_limit: str = '10GB',
        temp_dir_size: str = '65GiB',
        threads: int = 4
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

        # Load bounty timeline data - using the cleaned version that already has proper timestamps
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
               CAST(CreationDate AS TIMESTAMP) AS creation_date
           FROM '{answers_path}'
           WHERE OwnerUserId IS NOT NULL;
       """)

        # Load ALL questions - without filtering NULL owners
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
           WHERE a.owner_user_id IS NULL 
              OR q.owner_user_id IS NULL 
              OR a.owner_user_id <> q.owner_user_id;  -- Include: self-answer exclusion, answers to deleted/orphaned questions, answers to questions with NULL owner
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

        # Create event IDs for each unique user
        con.execute("""
                    CREATE
                    TEMPORARY TABLE user_events AS
                    SELECT user_id,
                           DENSE_RANK() OVER (ORDER BY user_id) AS event_id
                    FROM user_answers
                    GROUP BY user_id;
                    """)

        # Count unique users
        unique_users_result = con.execute("SELECT COUNT(*) as users FROM user_events").fetchone()
        unique_users = unique_users_result[0]
        print(f"Unique users with answers: {unique_users:,}")

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
           JOIN user_events e ON q.owner_user_id = e.user_id
           WHERE q.owner_user_id IS NOT NULL;  -- Only include questions with a valid owner for history
       """)

        # Create historical events - Answers provided by users (explicitly excluding self-answers)
        con.execute("""
           CREATE TEMPORARY TABLE answers_provided_history AS
           SELECT
               e.event_id,
               a.user_id,
               a.timestamp,
               'AnswerProvided' AS event,
               a.answer_id,
               a.question_id,
               a.is_bounty,
               a.bounty_amount,
               a.answer_sequence,
               1 AS is_history
           FROM user_answers a
           JOIN user_events e ON a.user_id = e.user_id
           WHERE a.user_id <> a.question_owner_id  -- Exclude self-answers
           OR a.question_owner_id IS NULL;  -- Include answers to questions without owners
       """)

        # Create historical events - Answers received by users to their questions
        con.execute("""
           CREATE TEMPORARY TABLE answers_received_history AS
           SELECT
               e.event_id,
               q.owner_user_id AS user_id,
               a.creation_date AS timestamp,
               'AnswerReceived' AS event,
               a.answer_id,
               q.question_id,
               0 AS is_bounty,
               0 AS bounty_amount,
               NULL AS answer_sequence,
               1 AS is_history
           FROM questions q
           JOIN answers a ON q.question_id = a.parent_question_id
           JOIN user_events e ON q.owner_user_id = e.user_id
           WHERE a.owner_user_id <> q.owner_user_id  -- Exclude self-answers
           AND q.owner_user_id IS NOT NULL;  -- Only include questions with a valid owner
       """)

        # Create historical events - Accepted answers received by users
        con.execute("""
           CREATE TEMPORARY TABLE accepted_answers_received_history AS
           SELECT
               e.event_id,
               q.owner_user_id AS user_id,
               a.creation_date AS timestamp,
               'AcceptedAnswerReceived' AS event,
               a.answer_id,
               q.question_id,
               0 AS is_bounty,
               0 AS bounty_amount,
               NULL AS answer_sequence,
               1 AS is_history
           FROM questions q
           JOIN answers a ON q.accepted_answer_id = a.answer_id
           JOIN user_events e ON q.owner_user_id = e.user_id
           WHERE a.owner_user_id <> q.owner_user_id  -- Exclude self-accepted answers
           AND q.owner_user_id IS NOT NULL;  -- Only include questions with a valid owner
       """)

        # Accept votes received (vote type 1)
        con.execute("""
           CREATE TEMPORARY TABLE accepted_vote_received_history AS
           SELECT
               e.event_id,
               a.owner_user_id AS user_id,
               v.vote_date AS timestamp,
               'AcceptedAnswerVote' AS event,
               a.answer_id,
               a.parent_question_id AS question_id,
               0 AS is_bounty,
               0 AS bounty_amount,
               NULL AS answer_sequence,
               1 AS is_history
           FROM answers a
           JOIN votes v ON a.answer_id = v.post_id
           JOIN user_events e ON a.owner_user_id = e.user_id
           WHERE v.vote_type_id = 1;  -- Accept vote
       """)

        # Create historical events - AcceptedAnswerPosted
        con.execute("""
           CREATE TEMPORARY TABLE accepted_answers_posted_history AS
           SELECT
               e.event_id,
               a.owner_user_id AS user_id,
               a.creation_date AS timestamp,
               'AcceptedAnswerPosted' AS event,
               a.answer_id,
               q.question_id,
               0 AS is_bounty,
               0 AS bounty_amount,
               NULL AS answer_sequence,
               1 AS is_history
           FROM answers a
           JOIN questions q ON q.accepted_answer_id = a.answer_id
           JOIN user_events e ON a.owner_user_id = e.user_id
           WHERE a.owner_user_id <> q.owner_user_id  -- Exclude self-accepted answers
           AND q.owner_user_id IS NOT NULL;  -- Only include questions with a valid owner
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
           FROM answers_received_history

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
           FROM accepted_answers_received_history

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
           FROM accepted_vote_received_history

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
           FROM accepted_answers_posted_history;
       """)

        # Count total events
        total_events_result = con.execute("SELECT COUNT(*) as total_events FROM all_events").fetchone()
        total_events = total_events_result[0]
        print(f"Total events in final dataset: {total_events:,}")

        # Count events by type
        print("\nEvent type breakdown:")
        event_breakdown = con.execute("""
                                      SELECT event, is_history, COUNT(*) as count
                                      FROM all_events
                                      GROUP BY event, is_history
                                      ORDER BY is_history, event
                                      """).fetchall()

        for event, is_history, count in event_breakdown:
            history_label = "Historical" if is_history else "Current"
            print(f"  {history_label} {event}: {count:,}")

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