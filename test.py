import pandas as pd
import duckdb


def test_multiple_answers_per_user_question():
    # Load the dataset
    file_path = r".\02_raw_datasets\bounty_dataset.parquet"
    df = pd.read_parquet(file_path)

    # Filter to keep only bounty data (non-historical rows)
    bounty_data = df[df['is_history'] == 0].copy()

    # Create connection and register dataframe
    con = duckdb.connect(database=':memory:')
    con.register('bounty_data', bounty_data)

    # Count total unique questions first
    total_questions = con.execute("""
        SELECT COUNT(DISTINCT question_id) AS total_unique_questions
        FROM bounty_data
        WHERE question_id IS NOT NULL
    """).fetchone()[0]

    # Count instances where a user provided multiple answers to the same question
    result = con.execute("""
        WITH user_question_counts AS (
            SELECT 
                user_id,
                question_id,
                COUNT(*) AS answer_count
            FROM bounty_data
            WHERE question_id IS NOT NULL
            GROUP BY user_id, question_id
            HAVING COUNT(*) > 1
        )
        SELECT 
            COUNT(*) AS total_multiple_answer_cases,
            COUNT(DISTINCT user_id) AS unique_users_with_multiple_answers,
            COUNT(DISTINCT question_id) AS unique_questions_with_multiple_answers,
            MAX(answer_count) AS max_answers_per_question,
            AVG(answer_count) AS avg_answers_when_multiple
        FROM user_question_counts
    """).fetchdf()

    # Get first 20 question IDs with multiple answers from same user
    first_20_questions = con.execute("""
        WITH user_question_counts AS (
            SELECT 
                user_id,
                question_id,
                COUNT(*) AS answer_count
            FROM bounty_data
            WHERE question_id IS NOT NULL
            GROUP BY user_id, question_id
            HAVING COUNT(*) > 1
        )
        SELECT DISTINCT question_id
        FROM user_question_counts
        ORDER BY question_id
        LIMIT 20
    """).fetchall()

    # Display results
    print(f"Total number of unique questions in the dataset: {total_questions}")
    print("\nStatistics on users providing multiple answers to the same question:")
    print(f"Total cases of multiple answers: {result['total_multiple_answer_cases'][0]}")
    print(f"Number of unique users who provided multiple answers: {result['unique_users_with_multiple_answers'][0]}")
    print(
        f"Number of unique questions receiving multiple answers from same user: {result['unique_questions_with_multiple_answers'][0]}")
    print(f"Maximum number of answers from same user to one question: {result['max_answers_per_question'][0]}")
    print(f"Average number of answers when a user answers multiple times: {result['avg_answers_when_multiple'][0]:.2f}")

    # Calculate percentage
    if total_questions > 0:
        percentage = (result['unique_questions_with_multiple_answers'][0] / total_questions) * 100
        print(f"\nPercentage of questions with multiple answers from same user: {percentage:.2f}%")

    # Print first 20 question IDs with multiple answers
    print("\nFirst 20 question IDs that received multiple answers from the same user:")
    for idx, (question_id,) in enumerate(first_20_questions, 1):
        print(f"{idx}. {question_id}")

    con.close()

    return result


if __name__ == "__main__":
    test_multiple_answers_per_user_question()