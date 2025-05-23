import duckdb
from pathlib import Path
from datetime import datetime

# Connect to in-memory DuckDB
con = duckdb.connect(database=':memory:')

# Paths to the parquet files
base_dir = Path("..")
input_data_folder = base_dir / "data" / "input"
answers_path = str(input_data_folder / 'posts_answers.parquet')
questions_path = str(input_data_folder / 'posts_questions.parquet')

# Parse the timestamp
target_date = datetime(2008, 8, 6, 14, 37, 0)

# Direct query joining answers with questions
result = con.execute(f"""
    SELECT 
        a.Id AS answer_id,
        a.OwnerUserId AS answerer_user_id,
        a.ParentId AS question_id,
        a.CreationDate AS answer_date,
        q.OwnerUserId AS question_owner_user_id
    FROM '{answers_path}' AS a
    JOIN '{questions_path}' AS q ON a.ParentId = q.Id
    WHERE a.OwnerUserId = 35
      AND CAST(a.CreationDate AS TIMESTAMP) < TIMESTAMP '{target_date}'
    ORDER BY CAST(a.CreationDate AS TIMESTAMP) ASC;
""").fetchall()

# Print the result
print(f"Answers by User ID 35 before {target_date} with Question Owner info:")
if result:
    column_names = ["answer_id", "answerer_user_id", "question_id", "answer_date", "question_owner_user_id"]
    print(f"Found {len(result)} answers")

    for i, row in enumerate(result):
        print(f"\nAnswer {i+1}:")
        for j, value in enumerate(row):
            print(f"{column_names[j]}: {value}")
else:
    print("No answers found for user ID 35 before the specified timestamp")

# Close the connection
con.close()


def print_specific_row(processed_file_path, question_id=3448, user_id=35):
    """
    Prints all columns for rows where questionId = question_id and userId = user_id
    """
    import duckdb

    # Create DuckDB connection
    print(f"Connecting to data at {processed_file_path}...")
    conn = duckdb.connect(database=':memory:')

    # Set memory limit and optimize for lower memory usage
    conn.execute("SET memory_limit='4GB'")
    conn.execute("PRAGMA temp_directory='/tmp'")

    # First, get the column names from the Parquet file
    column_info = conn.execute(f"DESCRIBE SELECT * FROM '{processed_file_path}'").fetchall()
    column_names = [col[0] for col in column_info]

    # Query for rows matching the criteria
    query = f"""
        SELECT *
        FROM '{processed_file_path}'
        WHERE questionId = {question_id}
          AND userId = {user_id}
    """

    results = conn.execute(query).fetchall()

    # Check if we found any matching rows
    if not results:
        print(f"No rows found with questionId={question_id} and userId={user_id}")
        return

    # Print the number of matching rows
    print(f"Found {len(results)} row(s) matching questionId={question_id} and userId={user_id}")

    # Print each row with all columns
    for row_index, row in enumerate(results):
        print(f"\nRow {row_index + 1}:")
        row_dict = dict(zip(column_names, row))

        # Print all columns and their values
        for col_name, value in row_dict.items():
            print(f"  {col_name}: {value}")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    processed_file_path = "../data/study_datasets/user_answers_bounty_processed.parquet"
    print_specific_row(processed_file_path)