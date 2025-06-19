import duckdb
import os


def count_event_ids(file_path):
    """Count all event_ids in the parquet file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    conn = duckdb.connect()
    try:
        # Count total event_ids
        total_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()[0]
        print(f"Total rows: {total_count}")

        # Count distinct event_ids
        distinct_count = conn.execute(f"SELECT COUNT(DISTINCT event_id) FROM read_parquet('{file_path}')").fetchone()[0]
        print(f"Distinct event_ids: {distinct_count}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    file_path = ("../data"
                 "/input/question_centered_model_7d_all_questions.parquet")
    count_event_ids(file_path)