import os
import pandas as pd
import numpy as np
import duckdb
from tqdm import tqdm
import gc


def handling_start_of_SO(input_file: str, output_file: str):
    """
    Process the dataset to remove events that occurred within 7 days of the first
    Stack Overflow question and recalculate fixed effects for affected users only,
    using DuckDB for memory efficiency.

    Args:
        input_file: Path to the processed question-centered dataset
        output_file: Path to save the filtered dataset
    """
    print(f"\n=== Handling Start of Stack Overflow Period for {input_file} ===")

    # Connect to DuckDB
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='8GB';")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar;")

    # Get the first ever question on Stack Overflow
    first_question_timestamp = get_first_SO_question_timestamp()
    cutoff_timestamp = first_question_timestamp + pd.Timedelta(days=7)
    print(f"First SO question: {first_question_timestamp}")
    print(f"Cutoff timestamp (first question + 7 days): {cutoff_timestamp}")

    # Register the cutoff timestamp with DuckDB
    con.execute(f"CREATE OR REPLACE TEMPORARY VIEW cutoff_ts AS SELECT '{cutoff_timestamp}'::TIMESTAMP AS cutoff")

    # Create a view of the input data
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY VIEW input_data AS
        SELECT * FROM '{input_file}'
    """)

    # Get the total row count
    total_rows = con.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
    print(f"Loaded {total_rows:,} total rows")

    # Identify event_ids where phaseOneStart is before the cutoff
    con.execute("""
                CREATE
                TEMPORARY TABLE affected_event_ids AS
                SELECT DISTINCT "eventId"
                FROM input_data
                WHERE "phaseOneStart" < (SELECT cutoff FROM cutoff_ts)
                """)

    affected_count = con.execute("SELECT COUNT(*) FROM affected_event_ids").fetchone()[0]
    print(f"Found {affected_count:,} affected event_ids to remove")

    # Store affected user_ids before removal
    con.execute("""
        CREATE TEMPORARY TABLE affected_user_ids AS
        SELECT DISTINCT a."userId"
        FROM input_data a
        JOIN affected_event_ids b
          ON a."eventId" = b."eventId"
    """)

    affected_users = con.execute("SELECT COUNT(*) FROM affected_user_ids").fetchone()[0]
    print(f"Found {affected_users:,} affected users")

    # Create a filtered view excluding affected event_ids
    con.execute("""
        CREATE TEMPORARY VIEW filtered_data AS
        SELECT *
        FROM input_data
        WHERE "eventId" NOT IN (SELECT "eventId" FROM affected_event_ids)
    """)

    filtered_count = con.execute("SELECT COUNT(*) FROM filtered_data").fetchone()[0]
    removed_count = total_rows - filtered_count
    print(f"Removed {removed_count:,} rows for affected event_ids")

    # Calculate user means for fixed effects recalculation only for affected users
    print("Calculating user means for affected users...")
    con.execute("""
                CREATE
                TEMPORARY TABLE user_means AS
                SELECT fd."userId",
                       AVG(fd."numHelped")   AS avg_numHelped,
                       AVG(fd."hasHelped")   AS avg_hasHelped,
                       AVG(fd."lnNumHelped") AS avg_lnNumHelped
                FROM filtered_data fd
                WHERE fd."userId" IN (SELECT "userId" FROM affected_user_ids)
                GROUP BY fd."userId"
                """)

    # Create final table with recalculated fixed effects for affected users only
    print("Recalculating fixed effects for affected users...")
    con.execute("""
                CREATE
                TEMPORARY TABLE output_data AS
                SELECT fd.*,
                       CASE
                           WHEN fd."userId" IN (SELECT "userId" FROM affected_user_ids) THEN
                               fd."numHelped" - COALESCE(um.avg_numHelped, 0)
                           ELSE
                               fd."userFeNumHelped"
                           END AS new_userFeNumHelped,

                       CASE
                           WHEN fd."userId" IN (SELECT "userId" FROM affected_user_ids) THEN
                               fd."hasHelped" - COALESCE(um.avg_hasHelped, 0)
                           ELSE
                               fd."userFeHasHelped"
                           END AS new_userFeHasHelped,

                       CASE
                           WHEN fd."userId" IN (SELECT "userId" FROM affected_user_ids) THEN
                               fd."lnNumHelped" - COALESCE(um.avg_lnNumHelped, 0)
                           ELSE
                               fd."userFeLnNumHelped"
                           END AS new_userFeLnNumHelped
                FROM filtered_data fd
                         LEFT JOIN user_means um
                                   ON fd."userId" = um."userId"
                """)

    # Prepare final output by renaming columns
    con.execute("""
                CREATE
                TEMPORARY TABLE final_output AS
                SELECT o.*                        EXCLUDE (new_userFeNumHelped, new_userFeHasHelped, new_userFeLnNumHelped), o.new_userFeNumHelped AS "userFeNumHelped",
                       o.new_userFeHasHelped   AS "userFeHasHelped",
                       o.new_userFeLnNumHelped AS "userFeLnNumHelped"
                FROM output_data o
                """)

    # Save the results to parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    con.execute(f"""
        COPY (SELECT * FROM final_output)
        TO '{output_file}'
        (FORMAT PARQUET, COMPRESSION 'GZIP')
    """)

    final_count = con.execute("SELECT COUNT(*) FROM final_output").fetchone()[0]
    print(f"Saved filtered dataset with {final_count:,} rows to {output_file}")
    print(f"Finished handling Stack Overflow start period")

    con.close()


def get_first_SO_question_timestamp():
    """
    Fetch the timestamp of the first question ever posted on Stack Overflow.

    Returns:
        pd.Timestamp: The timestamp of the first SO question
    """
    # Connect to DuckDB
    con = duckdb.connect(database=':memory:')

    # Find the path to the questions file
    questions_path = "../data/input/posts_questions.parquet"

    # Query for the earliest question timestamp
    result = con.execute(f"""
        SELECT MIN(CAST(CreationDate AS TIMESTAMP)) AS first_timestamp
        FROM '{questions_path}'
    """).fetchone()

    first_timestamp = pd.Timestamp(result[0])
    con.close()

    return first_timestamp


if __name__ == "__main__":
    input_folder = "../data/study_datasets"
    output_folder = "../data/study_datasets"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process the question-centered dataset
    for days in [7]:
        input_file = f"{input_folder}/question_centered_model_{days}d_processed.parquet"
        output_file = f"{output_folder}/question_centered_model_{days}d_processed.parquet"

        handling_start_of_SO(
            input_file=input_file,
            output_file=output_file
        )