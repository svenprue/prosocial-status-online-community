import duckdb
import os
from pathlib import Path


def check_invalid_parent_references(
        posts_answers_path,
        posts_questions_path
):
    """
    Checks for answers in posts_answers where:
    1. ParentId is NULL
    2. ParentId doesn't exist in posts_questions

    Args:
        posts_answers_path: Path to the posts_answers data
        posts_questions_path: Path to the posts_questions data
    """
    print(f"\n=== Checking for Invalid Parent References in Answers ===")

    # Create DuckDB connection
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='6GB'")
    conn.execute("PRAGMA temp_directory='/tmp'")

    # Verify files exist
    for file_path in [posts_answers_path, posts_questions_path]:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

    # Load posts_answers data
    print("Loading posts_answers data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW posts_answers AS
        SELECT 
            Id AS answer_id,
            ParentId AS parent_id,
            OwnerUserId AS owner_user_id
        FROM '{posts_answers_path}'
    """)

    # Load posts_questions data
    print("Loading posts_questions data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW posts_questions AS
        SELECT 
            Id AS question_id
        FROM '{posts_questions_path}'
    """)

    # Count total answers
    total_answers = conn.execute("SELECT COUNT(*) FROM posts_answers").fetchone()[0]
    print(f"Total answers in posts_answers: {total_answers:,}")

    # ===== CHECK 1: Answers with NULL ParentId =====
    print("\n===== CHECK 1: Answers with NULL ParentId =====")

    conn.execute("""
        CREATE TEMPORARY VIEW null_parent_answers AS
        SELECT 
            answer_id,
            parent_id,
            owner_user_id
        FROM posts_answers
        WHERE parent_id IS NULL
    """)

    # Count answers with NULL ParentId
    null_parent_count = conn.execute("SELECT COUNT(*) FROM null_parent_answers").fetchone()[0]
    print(f"Found {null_parent_count:,} answers with NULL ParentId")

    # Show examples
    if null_parent_count > 0:
        print("\nExample answers with NULL ParentId (10 max):")
        null_parent_examples = conn.execute("""
            SELECT *
            FROM null_parent_answers
            ORDER BY answer_id
            LIMIT 10
        """).fetchdf()
        print(null_parent_examples.to_string(index=False))

    # ===== CHECK 2: Answers with invalid ParentId (not in posts_questions) =====
    print("\n===== CHECK 2: Answers with invalid ParentId (not in posts_questions) =====")

    conn.execute("""
        CREATE TEMPORARY VIEW invalid_parent_answers AS
        SELECT 
            pa.answer_id,
            pa.parent_id,
            pa.owner_user_id
        FROM posts_answers pa
        LEFT JOIN posts_questions pq ON pa.parent_id = pq.question_id
        WHERE pa.parent_id IS NOT NULL
          AND pq.question_id IS NULL
    """)

    # Count answers with invalid ParentId
    invalid_parent_count = conn.execute("SELECT COUNT(*) FROM invalid_parent_answers").fetchone()[0]
    print(f"Found {invalid_parent_count:,} answers with invalid ParentId (not in posts_questions)")

    # Show examples
    if invalid_parent_count > 0:
        print("\nExample answers with invalid ParentId (10 max):")
        invalid_parent_examples = conn.execute("""
            SELECT *
            FROM invalid_parent_answers
            ORDER BY answer_id LIMIT 10
        """).fetchdf()
        print(invalid_parent_examples.to_string(index=False))

    # ===== Additional: Summary and analysis =====
    print("\n===== Summary of Invalid Parent References =====")
    total_invalid = null_parent_count + invalid_parent_count
    invalid_ratio = (total_invalid / total_answers) * 100 if total_answers > 0 else 0

    print(f"Total answers with invalid parent references: {total_invalid:,} ({invalid_ratio:.2f}% of all answers)")
    print(f"- Null ParentId: {null_parent_count:,} ({(null_parent_count / total_answers) * 100:.2f}% of all answers)")
    print(f"- Invalid ParentId: {invalid_parent_count:,} ({(invalid_parent_count / total_answers) * 100:.2f}% of all answers)")

    # Skip creating files if no invalid records found
    if total_invalid == 0:
        print("\nNo invalid parent references found. No output files created.")
        conn.close()
        return

    # Save invalid parent IDs to text files
    output_dir = Path(posts_answers_path).parent

    # Save answers with NULL ParentId
    if null_parent_count > 0:
        null_parent_file = output_dir / "answers_with_null_parent.txt"
        conn.execute(f"""
            COPY (
                SELECT answer_id, parent_id, owner_user_id
                FROM null_parent_answers
                ORDER BY answer_id
            ) TO '{null_parent_file}' (HEADER, DELIMITER '|')
        """)
        print(f"Saved {null_parent_count} answers with NULL ParentId to {null_parent_file}")

    # Save answers with invalid ParentId
    if invalid_parent_count > 0:
        invalid_parent_file = output_dir / "answers_with_invalid_parent.txt"
        conn.execute(f"""
            COPY (
                SELECT answer_id, parent_id, owner_user_id
                FROM invalid_parent_answers
                ORDER BY answer_id
                LIMIT 10000
            ) TO '{invalid_parent_file}' (HEADER, DELIMITER '|')
        """)
        print(f"Saved up to 10,000 answers with invalid ParentId to {invalid_parent_file}")

    conn.close()


if __name__ == "__main__":
    posts_answers_path = "../data/input/posts_answers.parquet"
    posts_questions_path = "../data/input/posts_questions.parquet"

    check_invalid_parent_references(
        posts_answers_path=posts_answers_path,
        posts_questions_path=posts_questions_path
    )