import duckdb
import os
from pathlib import Path


def check_missing_answers(
        processed_file_path,
        votes_path,
        posts_answers_path,
        posts_questions_path
):
    """
    Performs two checks, excluding self-answers:
    1. Find answers to questions with votetype 8 that exist in posts_answers but not in processed file
    2. Find any answers in posts_answers with non-null owner_user_id that don't exist in processed file

    Args:
        processed_file_path: Path to the processed dataset
        votes_path: Path to the votes data
        posts_answers_path: Path to the posts_answers data
        posts_questions_path: Path to the posts_questions data
    """
    print(f"=== Checking for Missing Answers (Excluding Self-Answers) ===")

    # Create DuckDB connection
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='6GB'")
    conn.execute("PRAGMA temp_directory='/tmp'")

    # Verify all files exist
    for file_path in [processed_file_path, votes_path, posts_answers_path, posts_questions_path]:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

    # Load the processed file
    print("Loading processed file data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW processed_data AS
        SELECT DISTINCT answerId, questionId, userId
        FROM '{processed_file_path}'
        WHERE answerId IS NOT NULL
    """)

    # Count distinct answerIds in processed data
    processed_count = conn.execute("SELECT COUNT(DISTINCT answerId) FROM processed_data").fetchone()[0]
    print(f"Found {processed_count:,} distinct answerIds in processed file")

    # Load posts_answers data
    print("Loading posts_answers data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW posts_answers AS
        SELECT 
            Id AS answer_id,
            ParentId AS question_id,
            OwnerUserId AS owner_user_id
        FROM '{posts_answers_path}'
        WHERE OwnerUserId IS NOT NULL
    """)

    # Load posts_questions data
    print("Loading posts_questions data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW posts_questions AS
        SELECT 
            Id AS question_id,
            OwnerUserId AS question_owner_id
        FROM '{posts_questions_path}'
    """)

    # Create a view with answers joined to questions
    print("Creating joined view of answers and questions...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW answers_with_questions AS
        SELECT 
            pa.answer_id,
            pa.question_id,
            pa.owner_user_id AS answer_owner_id,
            pq.question_owner_id
        FROM posts_answers pa
        LEFT JOIN posts_questions pq ON pa.question_id = pq.question_id
        WHERE pa.owner_user_id IS NOT NULL 
          AND (pa.owner_user_id <> pq.question_owner_id OR pq.question_owner_id IS NULL)  -- Exclude self-answers
    """)

    # Count answers after excluding self-answers
    filtered_count = conn.execute("SELECT COUNT(*) FROM answers_with_questions").fetchone()[0]
    total_count = conn.execute("SELECT COUNT(*) FROM posts_answers").fetchone()[0]
    self_answers_count = conn.execute("""
                                      SELECT COUNT(*)
                                      FROM posts_answers pa
                                               JOIN posts_questions pq ON pa.question_id = pq.question_id
                                      WHERE pa.owner_user_id = pq.question_owner_id
                                        AND pa.owner_user_id IS NOT NULL
                                        AND pq.question_owner_id IS NOT NULL
                                      """).fetchone()[0]

    print(f"Found {total_count:,} total answers with non-null owner_user_id in posts_answers")
    print(f"Excluded {self_answers_count:,} self-answers")
    print(f"Remaining {filtered_count:,} answers after excluding self-answers")

    # Load votes data with votetype 8 (bounty)
    print("Loading bounty votes data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW bounty_votes AS
        SELECT DISTINCT
            PostId AS question_id
        FROM '{votes_path}'
        WHERE VoteTypeId = 8
    """)

    # Count questions with bounty votes
    bounty_count = conn.execute("SELECT COUNT(*) FROM bounty_votes").fetchone()[0]
    print(f"Found {bounty_count:,} questions with bounty votes")

    # ===== CHECK 1: Answers to bountied questions missing from processed file =====
    print("\n===== CHECK 1: Answers to bountied questions missing from processed file (excluding self-answers) =====")

    conn.execute("""
        CREATE TEMPORARY VIEW missing_bounty_answers AS
        SELECT 
            aq.answer_id,
            aq.question_id,
            aq.answer_owner_id,
            aq.question_owner_id
        FROM answers_with_questions aq
        JOIN bounty_votes bv ON aq.question_id = bv.question_id
        LEFT JOIN processed_data pd ON aq.answer_id = pd.answerId
        WHERE pd.answerId IS NULL
    """)

    # Count missing bounty answers
    missing_bounty_count = conn.execute("SELECT COUNT(*) FROM missing_bounty_answers").fetchone()[0]
    print(f"Found {missing_bounty_count:,} answers to bountied questions missing from processed file")

    # Show examples
    if missing_bounty_count > 0:
        print("\nExample missing bounty answers (10 max):")
        missing_examples = conn.execute("""
                                        SELECT *
                                        FROM missing_bounty_answers
                                        ORDER BY question_id LIMIT 10
                                        """).fetchdf()
        print(missing_examples.to_string(index=False))

    # ===== CHECK 2: Any answers missing from processed file =====
    print("\n===== CHECK 2: Any answers missing from processed file (excluding self-answers) =====")

    conn.execute("""
        CREATE TEMPORARY VIEW all_missing_answers AS
        SELECT 
            aq.answer_id,
            aq.question_id,
            aq.answer_owner_id,
            aq.question_owner_id
        FROM answers_with_questions aq
        LEFT JOIN processed_data pd ON aq.answer_id = pd.answerId
        WHERE pd.answerId IS NULL
    """)

    # Count all missing answers
    all_missing_count = conn.execute("SELECT COUNT(*) FROM all_missing_answers").fetchone()[0]
    print(f"Found {all_missing_count:,} total answers missing from processed file")

    # Show examples
    if all_missing_count > 0:
        print("\nExample missing answers (10 max):")
        all_missing_examples = conn.execute("""
                                            SELECT *
                                            FROM all_missing_answers
                                            ORDER BY answer_id LIMIT 10
                                            """).fetchdf()
        print(all_missing_examples.to_string(index=False))

    # ===== Additional: Missing ratio analysis =====
    print("\n===== Missing Data Analysis =====")
    missing_ratio = (all_missing_count / filtered_count) * 100 if filtered_count > 0 else 0
    print(f"Overall missing ratio: {missing_ratio:.2f}% of non-self-answers are missing from processed file")

    bounty_missing_ratio = (missing_bounty_count / all_missing_count) * 100 if all_missing_count > 0 else 0
    print(f"Bounty missing ratio: {bounty_missing_ratio:.2f}% of missing answers are from bountied questions")

    # Save missing answer IDs to text files
    output_dir = Path(processed_file_path).parent

    # Save bounty missing answers
    if missing_bounty_count > 0:
        bounty_missing_file = output_dir / "missing_bounty_answers_no_self_answers.txt"
        conn.execute(f"""
            COPY (
                SELECT answer_id, question_id, answer_owner_id, question_owner_id
                FROM missing_bounty_answers
                ORDER BY question_id, answer_id
            ) TO '{bounty_missing_file}' (HEADER, DELIMITER '|')
        """)
        print(f"Saved {missing_bounty_count} missing bounty answers to {bounty_missing_file}")

    # Save all missing answers (sample)
    if all_missing_count > 0:
        all_missing_file = output_dir / "all_missing_answers_no_self_answers_sample.txt"
        conn.execute(f"""
            COPY (
                SELECT answer_id, question_id, answer_owner_id, question_owner_id
                FROM all_missing_answers
                ORDER BY answer_id
                LIMIT 1000
            ) TO '{all_missing_file}' (HEADER, DELIMITER '|')
        """)
        print(f"Saved sample of missing answers to {all_missing_file}")

    conn.close()


if __name__ == "__main__":
    processed_file_path = "../data/study_datasets/user_answers_bounty_processed.parquet"
    votes_path = "../data/input/Votes.parquet"
    posts_answers_path = "../data/input/posts_answers.parquet"
    posts_questions_path = "../data/input/posts_questions.parquet"

    check_missing_answers(
        processed_file_path=processed_file_path,
        votes_path=votes_path,
        posts_answers_path=posts_answers_path,
        posts_questions_path=posts_questions_path
    )