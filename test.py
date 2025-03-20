import os
import duckdb


def count_answer_users_with_questions_join(input_folder: str) -> None:
    """
    Count unique users who have posted answers, joining with questions table
    to distinguish between self-answers and answers to others' questions.

    Args:
        input_folder: Directory containing the parquet files
    """
    # Path to parquet files
    answers_path = os.path.join(input_folder, "posts_answers.parquet")
    questions_path = os.path.join(input_folder, "posts_questions.parquet")

    # Check if files exist
    for path in [answers_path, questions_path]:
        if not os.path.exists(path):
            print(f"Input file not found: {path}")
            return

    print(f"Querying answers and questions data...")

    # Connect to DuckDB
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA enable_progress_bar;")

    # All answers (simple count without joins)
    query_all = f"""
    SELECT 
        COUNT(DISTINCT OwnerUserId) AS unique_users,
        COUNT(*) AS total_answers
    FROM '{answers_path}'
    WHERE OwnerUserId IS NOT NULL
    """

    # Joined query (similar to dataset creation)
    query_joined = f"""
    WITH answers AS (
        SELECT
            Id AS answer_id,
            OwnerUserId AS owner_user_id,
            ParentId AS parent_question_id,
            CreationDate AS creation_date
        FROM '{answers_path}'
        WHERE OwnerUserId IS NOT NULL
    ),
    questions AS (
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            CreationDate AS creation_date
        FROM '{questions_path}'
        WHERE OwnerUserId IS NOT NULL
    ),
    joined_answers AS (
        SELECT
            a.owner_user_id AS user_id,
            a.answer_id,
            a.parent_question_id AS question_id,
            CASE WHEN a.owner_user_id = q.owner_user_id THEN 1 ELSE 0 END AS is_self_answer
        FROM answers a
        JOIN questions q ON a.parent_question_id = q.question_id
    )
    SELECT
        COUNT(DISTINCT user_id) AS total_users_after_join,
        COUNT(DISTINCT CASE WHEN is_self_answer = 0 THEN user_id END) AS users_with_non_self_answers,
        COUNT(DISTINCT CASE WHEN is_self_answer = 1 THEN user_id END) AS users_with_only_self_answers,
        COUNT(*) AS total_answers_after_join,
        SUM(CASE WHEN is_self_answer = 0 THEN 1 ELSE 0 END) AS non_self_answers,
        SUM(CASE WHEN is_self_answer = 1 THEN 1 ELSE 0 END) AS self_answers
    FROM joined_answers
    """

    # Execute queries
    print("Counting all answers...")
    all_result = con.execute(query_all).fetchone()

    print("Counting with questions join...")
    joined_result = con.execute(query_joined).fetchone()

    # Print results
    print("\n=== ANSWER COUNTS ===")
    print(f"Total unique users who posted answers: {all_result[0]:,}")
    print(f"Total answers: {all_result[1]:,}")
    print(f"Average answers per user: {all_result[1] / all_result[0]:.2f}")

    print("\n=== AFTER JOINING WITH QUESTIONS ===")
    print(f"Users with answers after join: {joined_result[0]:,}")
    print(f"Users with non-self answers: {joined_result[1]:,}")
    print(f"Users with self-answers: {joined_result[2]:,}")
    print(f"Total answers after join: {joined_result[3]:,}")
    print(f"Non-self answers: {joined_result[4]:,}")
    print(f"Self-answers: {joined_result[5]:,}")

    # Calculate how many users/answers were excluded by joining
    excluded_users = all_result[0] - joined_result[0]
    excluded_answers = all_result[1] - joined_result[3]
    print(f"\nExcluded by join: {excluded_users:,} users, {excluded_answers:,} answers")

    con.close()


if __name__ == "__main__":
    input_data_folder = "./01_input_data/processed_data_dump"
    count_answer_users_with_questions_join(input_data_folder)