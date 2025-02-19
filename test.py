import duckdb
import os


def check_user_answers_count(user_id, timestamp):
    """
    Check the correct count of answers provided by a user before a specific timestamp,
    reading directly from original parquet files.

    Args:
        user_id: The user ID to check
        timestamp: Cut-off timestamp (answers before this time will be counted)

    Returns:
        Dict containing counts and detail information
    """
    # Create connection
    con = duckdb.connect(database=':memory:')

    # Define paths to original data files
    data_path = os.path.join('.', '01_input_data', 'processed_data_dump')
    answers_path = os.path.join(data_path, 'posts_answers.parquet')
    questions_path = os.path.join(data_path, 'posts_questions.parquet')

    # Ensure files exist
    if not os.path.exists(answers_path) or not os.path.exists(questions_path):
        return {"error": f"Data files not found at {data_path}"}

    # Get answers count (excluding self-answers)
    answer_count_query = f"""
    WITH 
    answers_data AS (
      SELECT * FROM '{answers_path}'
    ),
    questions_data AS (
      SELECT * FROM '{questions_path}'
    )

    SELECT
      COUNT(*) AS correct_answer_count
    FROM answers_data a
    JOIN questions_data q ON a.ParentId = q.Id
    WHERE 
      a.OwnerUserId = {user_id}
      AND a.CreationDate < TIMESTAMP '{timestamp}'
      -- Exclude self-answers
      AND a.OwnerUserId != q.OwnerUserId
    """

    answer_count = con.execute(answer_count_query).fetchone()[0]

    # Get details of the answers for debugging
    answer_details_query = f"""
    WITH 
    answers_data AS (
      SELECT * FROM '{answers_path}'
    ),
    questions_data AS (
      SELECT * FROM '{questions_path}'
    )

    SELECT
      a.Id AS answer_id,
      a.ParentId AS question_id,
      a.CreationDate AS timestamp,
      q.OwnerUserId AS question_owner_id
    FROM answers_data a
    JOIN questions_data q ON a.ParentId = q.Id
    WHERE 
      a.OwnerUserId = {user_id}
      AND a.CreationDate < TIMESTAMP '{timestamp}'
      AND a.OwnerUserId != q.OwnerUserId
    ORDER BY a.CreationDate
    """

    answer_details = con.execute(answer_details_query).fetchall()

    # Check for potential duplicates
    duplicate_check_query = f"""
    WITH 
    answers_data AS (
      SELECT * FROM '{answers_path}'
    )

    SELECT
      a.ParentId AS question_id,
      a.CreationDate,
      COUNT(*) AS answer_count
    FROM answers_data a
    WHERE 
      a.OwnerUserId = {user_id}
      AND a.CreationDate < TIMESTAMP '{timestamp}'
    GROUP BY a.ParentId, a.CreationDate
    HAVING COUNT(*) > 1
    """

    duplicates = con.execute(duplicate_check_query).fetchall()

    # Close connection
    con.close()

    # Return results
    return {
        "correct_answer_count": answer_count,
        "answer_details": answer_details,
        "potential_duplicates": duplicates,
        "user_id": user_id,
        "timestamp": timestamp
    }


# Check the specific case with discrepancy
result = check_user_answers_count(356, '2008-11-20 14:36:42.953')

print(
    f"Correct answer count for user {result['user_id']} before {result['timestamp']}: {result['correct_answer_count']}")

if result['potential_duplicates']:
    print("\nPotential duplicates found:")
    for dup in result['potential_duplicates']:
        print(f"  Question ID: {dup[0]}, Timestamp: {dup[1]}, Count: {dup[2]}")

print(f"\nFound {len(result['answer_details'])} answers:")
for i, answer in enumerate(result['answer_details'], 1):
    print(f"{i}. Answer ID: {answer[0]}, Question ID: {answer[1]}, Posted: {answer[2]}, Question Owner: {answer[3]}")

# You can save this to a file for further analysis
if result['answer_details']:
    import pandas as pd

    pd.DataFrame(result['answer_details'],
                 columns=['answer_id', 'question_id', 'timestamp', 'question_owner_id']
                 ).to_csv(f"user_{result['user_id']}_answers.csv", index=False)
    print(f"\nDetailed answer data saved to user_{result['user_id']}_answers.csv")