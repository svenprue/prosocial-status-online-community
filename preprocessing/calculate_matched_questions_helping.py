import os
import duckdb
import pandas as pd


def calculate_helping_for_matched_questions(
        matched_questions_path: str,
        input_folder: str,
        output_folder: str
) -> None:
    """
    For ALL rows in matched_questions.parquet (both hasAnswer=0 and hasAnswer=1),
    calculate how many answers the user posted to OTHER users' questions between
    posting their question and (question_timestamp + responseTimeHours).

    For hasAnswer=0 rows (no answer received), use the responseTimeHours from the
    paired hasAnswer=1 row with the same match_id to define the time window.

    Note: Matched questions are paired by match_id (not global_index). Each match_id
    typically has two rows with different global_index values: one with hasAnswer=1
    and one with hasAnswer=0.

    Args:
        matched_questions_path: Path to matched_questions.parquet
        input_folder: Path to folder containing raw data (questions, answers)
        output_folder: Path to save output file
    """
    print(f"\n=== Calculating Helping Metrics for Matched Questions ===")

    # Initialize DuckDB connection
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA max_temp_directory_size='200GiB'")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar;")

    # Load matched questions
    print(f"Loading matched questions from {matched_questions_path}...")
    matched_df = pd.read_parquet(matched_questions_path)
    print(f"Loaded {len(matched_df):,} total matched question records")

    # Check distribution
    print(f"  - hasAnswer=0: {(matched_df['hasAnswer'] == 0).sum():,}")
    print(f"  - hasAnswer=1: {(matched_df['hasAnswer'] == 1).sum():,}")

    # Get unique global_index values to see how many pairs we have
    unique_global_indices = matched_df['global_index'].nunique()
    rows_per_gi = matched_df.groupby('global_index').size()
    print(f"\nMatching structure:")
    print(f"  - Unique global_index values: {unique_global_indices:,}")
    print(f"  - Average rows per global_index: {len(matched_df) / unique_global_indices:.2f}")
    print(f"  - global_index with 1 row: {(rows_per_gi == 1).sum():,}")
    print(f"  - global_index with 2 rows: {(rows_per_gi == 2).sum():,}")
    print(f"  - global_index with >2 rows: {(rows_per_gi > 2).sum():,}")

    # Register matched questions in DuckDB
    con.register("matched_questions", matched_df)

    # Load raw data
    questions_path = os.path.join(input_folder, 'posts_questions.parquet')
    answers_path = os.path.join(input_folder, 'posts_answers.parquet')

    print("\nLoading raw Stack Overflow data...")

    # Create questions view
    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{questions_path}'
    """)

    # Create answers view
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

    print("Raw data loaded successfully")

    # Step 1: Create a lookup table of responseTimeHours for each match_id
    # Use the responseTimeHours from the hasAnswer=1 row in each match_id
    print("\nCreating responseTimeHours lookup from hasAnswer=1 rows...")
    con.execute("""
        CREATE TEMPORARY TABLE response_time_lookup AS
        SELECT
            match_id,
            responseTimeHours
        FROM matched_questions
        WHERE hasAnswer = 1
          AND responseTimeHours IS NOT NULL;
    """)

    lookup_count = con.execute("SELECT COUNT(*) FROM response_time_lookup").fetchone()[0]
    print(f"  - Created lookup for {lookup_count:,} match_id values")

    # Step 2: Join ALL matched questions with their response times and question timestamps
    print("\nCalculating time windows for ALL rows (using paired responseTimeHours from match_id)...")
    con.execute("""
        CREATE TEMPORARY TABLE question_windows AS
        SELECT
            mq.match_id,
            mq.global_index,
            mq.questionId AS question_id,
            mq.userId AS user_id,
            mq.hasAnswer,
            mq.phase,
            mq.numHelpProvidedAT,
            mq.numHelpProvided30D,
            mq.numHelpProvided7D,
            q.creation_date AS question_timestamp,
            -- Use responseTimeHours from the lookup (from hasAnswer=1 row in same match_id)
            rtl.responseTimeHours AS reference_response_time,
            -- Calculate end of window as question_timestamp + responseTimeHours (convert to seconds)
            (q.creation_date + INTERVAL '1 SECOND' * CAST(rtl.responseTimeHours * 3600 AS BIGINT)) AS window_end_timestamp
        FROM matched_questions mq
        LEFT JOIN questions q ON mq.questionId = q.question_id
        LEFT JOIN response_time_lookup rtl ON mq.match_id = rtl.match_id
        WHERE q.creation_date IS NOT NULL  -- Ensure we have valid question data
          AND rtl.responseTimeHours IS NOT NULL;  -- Ensure we have reference response time
    """)

    # Check how many rows we have with valid data
    valid_count = con.execute("SELECT COUNT(*) FROM question_windows").fetchone()[0]
    has_answer_0 = con.execute("SELECT COUNT(*) FROM question_windows WHERE hasAnswer = 0").fetchone()[0]
    has_answer_1 = con.execute("SELECT COUNT(*) FROM question_windows WHERE hasAnswer = 1").fetchone()[0]
    print(f"  - {valid_count:,} rows with valid timestamps and response times")
    print(f"    - hasAnswer=0: {has_answer_0:,}")
    print(f"    - hasAnswer=1: {has_answer_1:,}")

    # Step 3: Calculate helps_given in the window for ALL rows
    # Count how many answers the user posted to OTHER users' questions
    # between question_timestamp and window_end_timestamp
    print("\nCalculating helping behavior for ALL rows (answers posted to others in the time window)...")
    con.execute("""
        CREATE TEMPORARY TABLE helps_given AS
        SELECT
            qw.match_id,
            qw.global_index,
            qw.question_id,
            qw.user_id,
            qw.phase,
            qw.hasAnswer,
            qw.reference_response_time AS responseTimeHours,
            qw.question_timestamp,
            qw.window_end_timestamp,
            qw.numHelpProvidedAT,
            qw.numHelpProvided30D,
            qw.numHelpProvided7D,
            COUNT(a.answer_id) AS helps_given_in_window
        FROM question_windows qw
        LEFT JOIN answers a
          ON a.owner_user_id = qw.user_id
         AND a.creation_date >= qw.question_timestamp
         AND a.creation_date < qw.window_end_timestamp
        LEFT JOIN questions q
          ON a.parent_question_id = q.question_id
        WHERE (a.owner_user_id IS NULL
               OR a.owner_user_id <> q.owner_user_id
               OR q.owner_user_id IS NULL)  -- Exclude self-answers
        GROUP BY qw.match_id, qw.global_index, qw.question_id, qw.user_id, qw.phase,
                 qw.hasAnswer, qw.reference_response_time,
                 qw.question_timestamp, qw.window_end_timestamp,
                 qw.numHelpProvidedAT, qw.numHelpProvided30D, qw.numHelpProvided7D;
    """)

    # Get summary statistics
    stats = con.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT global_index) as unique_global_indices,
            SUM(helps_given_in_window) as total_helps_given,
            AVG(helps_given_in_window) as avg_helps_given,
            MAX(helps_given_in_window) as max_helps_given,
            COUNT(CASE WHEN helps_given_in_window > 0 THEN 1 END) as rows_with_helping,
            AVG(responseTimeHours) as avg_response_time_hours
        FROM helps_given
    """).fetchdf()

    print("\nSummary Statistics:")
    print(f"  - Total rows analyzed: {stats['total_rows'].iloc[0]:,}")
    print(f"  - Unique global_index values: {stats['unique_global_indices'].iloc[0]:,}")
    print(f"  - Total helps given: {stats['total_helps_given'].iloc[0]:,}")
    print(f"  - Average helps per row: {stats['avg_helps_given'].iloc[0]:.2f}")
    print(f"  - Maximum helps in any window: {stats['max_helps_given'].iloc[0]:,}")
    print(f"  - Rows where user helped others (>0): {stats['rows_with_helping'].iloc[0]:,} ({100*stats['rows_with_helping'].iloc[0]/stats['total_rows'].iloc[0]:.1f}%)")
    print(f"  - Average response time: {stats['avg_response_time_hours'].iloc[0]:.2f} hours")


    # Step 4: Export results
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "matched_questions_helping_metrics.parquet")

    print(f"\nExporting results to {output_path}...")
    con.execute(f"""
        COPY (
            SELECT
                match_id,
                global_index,
                question_id,
                user_id,
                phase,
                hasAnswer,
                question_timestamp,
                window_end_timestamp,
                responseTimeHours,
                helps_given_in_window,
                numHelpProvidedAT,
                numHelpProvided30D,
                numHelpProvided7D
            FROM helps_given
            ORDER BY match_id, hasAnswer
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.close()
    print(f"\nSuccessfully saved results to {output_path}")

if __name__ == "__main__":
    # Define paths
    matched_questions_path = r"..\data\study_datasets\matched_questions.parquet"
    input_data_folder = r"..\data\input"
    output_data_folder = r"..\data\study_datasets"

    # Run the calculation
    calculate_helping_for_matched_questions(
        matched_questions_path=matched_questions_path,
        input_folder=input_data_folder,
        output_folder=output_data_folder
    )
