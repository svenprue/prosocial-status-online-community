import os
import duckdb
import pandas as pd


def count_eligible_questions(input_folder: str) -> None:
    """Count eligible questions for both all_questions and one_question modes."""

    con = duckdb.connect(database=':memory:')

    # Load questions data
    questions_path = os.path.join(input_folder, 'posts_questions.parquet')

    print(f"Loading questions from: {questions_path}")

    # Create questions view with basic filtering
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

    # Count total questions after basic filtering
    total_questions = con.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    total_users = con.execute("SELECT COUNT(DISTINCT owner_user_id) FROM questions").fetchone()[0]

    print(f"\nTotal questions (after OwnerUserId IS NOT NULL filter): {total_questions:,}")
    print(f"Total unique users: {total_users:,}")

    # Count for "all questions" mode (all questions from users who have asked questions)
    con.execute("""
        CREATE TEMPORARY VIEW eligible_questions_all AS
        SELECT
            q.question_id,
            q.owner_user_id,
            q.creation_date
        FROM questions q
        WHERE q.owner_user_id IN (
            SELECT DISTINCT owner_user_id 
            FROM questions
        );
    """)

    all_questions_count = con.execute("SELECT COUNT(*) FROM eligible_questions_all").fetchone()[0]
    all_questions_users = con.execute("SELECT COUNT(DISTINCT owner_user_id) FROM eligible_questions_all").fetchone()[0]

    print(f"\n--- ALL QUESTIONS MODE ---")
    print(f"Eligible questions: {all_questions_count:,}")
    print(f"Unique users: {all_questions_users:,}")

    # Count for "one question per user" mode
    con.execute("""
        CREATE TEMPORARY VIEW eligible_questions_one AS
        WITH ranked AS (
            SELECT
                q.question_id,
                q.owner_user_id,
                q.creation_date,
                ROW_NUMBER() OVER (
                    PARTITION BY q.owner_user_id
                    ORDER BY RANDOM()
                ) AS rn
            FROM questions q
            WHERE q.owner_user_id IN (
                SELECT DISTINCT owner_user_id 
                FROM questions
            )
        )
        SELECT 
            question_id,
            owner_user_id,
            creation_date
        FROM ranked
        WHERE rn = 1;
    """)

    one_question_count = con.execute("SELECT COUNT(*) FROM eligible_questions_one").fetchone()[0]
    one_question_users = con.execute("SELECT COUNT(DISTINCT owner_user_id) FROM eligible_questions_one").fetchone()[0]

    print(f"\n--- ONE QUESTION PER USER MODE ---")
    print(f"Eligible questions: {one_question_count:,}")
    print(f"Unique users: {one_question_users:,}")

    # Additional stats
    print(f"\n--- SUMMARY ---")
    print(f"Average questions per user: {total_questions / total_users:.2f}")
    print(f"Reduction from all to one per user: {(1 - one_question_count / all_questions_count) * 100:.1f}%")

    con.close()


if __name__ == "__main__":
    # Update this path to match your data location
    input_data_folder = r"..\data\input"

    if not os.path.exists(input_data_folder):
        print(f"Error: Input folder '{input_data_folder}' does not exist.")
        print("Please update the input_data_folder path in the script.")
    else:
        count_eligible_questions(input_data_folder)