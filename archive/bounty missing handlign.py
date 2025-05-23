def handling_deleted_bountied_questions(processed_file_path, votes_path, bounty_timeline_path, posts_questions_path):
    """
    Identifies questions with votetype 8 (bounty) but not in the bounty timeline.
    Only includes questions that exist in posts_questions.
    Adds a missing_bounty column to the processed file.
    Saves all missing question IDs to a text file.

    Args:
        processed_file_path: Path to the processed dataset
        votes_path: Path to the votes data
        bounty_timeline_path: Path to the bounty timeline data
        posts_questions_path: Path to the posts_questions data
    """
    import duckdb
    import os
    from pathlib import Path

    print(f"\n=== Handling deleted bountied questions (only existing questions) ===")

    # Create DuckDB connection with memory settings
    print(f"Connecting to data...")
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='12GB'")
    conn.execute("PRAGMA temp_directory='/tmp'")

    # Load the bounty timeline data
    print("Loading bounty timeline data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW bounty_timeline AS
        SELECT 
            question_id
        FROM '{bounty_timeline_path}'
    """)

    # Load questions with votetype 8 (bounty votes)
    print("Loading questions with bounty votes...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW bounty_votes AS
        SELECT DISTINCT
            PostId AS question_id
        FROM '{votes_path}'
        WHERE VoteTypeId = 8
    """)

    # Load posts_questions data
    print("Loading posts_questions data...")
    conn.execute(f"""
        CREATE TEMPORARY VIEW posts_questions AS
        SELECT 
            Id AS question_id
        FROM '{posts_questions_path}'
    """)

    # Find questions with bounty votes but not in timeline AND that exist in posts_questions
    print("Identifying missing bounty questions that exist in posts_questions...")
    conn.execute("""
        CREATE TEMPORARY VIEW missing_bounty_questions AS
        SELECT
            bv.question_id
        FROM bounty_votes bv
        JOIN posts_questions pq ON bv.question_id = pq.question_id
        LEFT JOIN bounty_timeline bt ON bv.question_id = bt.question_id
        WHERE bt.question_id IS NULL
    """)

    # Count missing bounty questions
    missing_count = conn.execute("SELECT COUNT(*) FROM missing_bounty_questions").fetchone()[0]
    print(f"Found {missing_count:,} questions with bounty votes but not in timeline (that exist in posts_questions)")

    # Save missing question IDs to a text file
    output_dir = Path(processed_file_path).parent
    missing_ids_file = output_dir / "missing_bounty_question_ids_existing.txt"

    print(f"Saving missing question IDs to {missing_ids_file}")
    missing_ids = conn.execute("SELECT question_id FROM missing_bounty_questions ORDER BY question_id").fetchall()

    with open(missing_ids_file, 'w') as f:
        f.write("# Question IDs with bounty votes but missing from bounty timeline (that exist in posts_questions)\n")
        f.write(f"# Total: {missing_count} question IDs\n")
        for idx, (qid,) in enumerate(missing_ids):
            f.write(f"{qid}\n")

    print(f"Saved {missing_count} question IDs to {missing_ids_file}")

    # Load the processed file into a temporary view
    print(f"Processing file: {processed_file_path}")
    conn.execute(f"""
        CREATE TEMPORARY VIEW processed_data AS
        SELECT * FROM '{processed_file_path}'
    """)

    # Get column count to verify we're not losing data
    col_count = conn.execute("SELECT COUNT(*) FROM pragma_table_info('processed_data')").fetchone()[0]
    print(f"File has {col_count} columns before adding missing_bounty")

    # Add the missing_bounty column and save directly to the original file
    conn.execute(f"""
        COPY (
            SELECT 
                p.*,
                CASE 
                    WHEN m.question_id IS NOT NULL THEN TRUE
                    ELSE FALSE
                END AS missing_bounty
            FROM processed_data p
            LEFT JOIN missing_bounty_questions m ON p.questionId = m.question_id
        ) TO '{processed_file_path}' (FORMAT PARQUET)
    """)

    # Count the number of rows with missing_bounty = TRUE
    missing_row_count = conn.execute(f"""
        SELECT COUNT(*) 
        FROM '{processed_file_path}'
        WHERE missing_bounty = TRUE
    """).fetchone()[0]

    total_row_count = conn.execute(f"""
        SELECT COUNT(*) 
        FROM '{processed_file_path}'
    """).fetchone()[0]

    print(
        f"Added missing_bounty column: TRUE for {missing_row_count:,} rows, FALSE for {total_row_count - missing_row_count:,} rows")
    print(f"Updated file saved to: {processed_file_path}")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    input_file = "../data/input/user_answers_bounty_dataset.parquet"
    output_file = "../data/study_datasets/user_answers_bounty_processed.parquet"
    votes_path = "../data/input/Votes.parquet"
    bounty_timeline_path = "../data/input/bounty_timeline.parquet"
    posts_questions_path = "../data/input/posts_questions.parquet"

    # Add handling of deleted bountied questions
    handling_deleted_bountied_questions(
        processed_file_path=output_file,
        votes_path=votes_path,
        bounty_timeline_path=bounty_timeline_path,
        posts_questions_path=posts_questions_path
    )