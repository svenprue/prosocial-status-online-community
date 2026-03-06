import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from numba import njit
import duckdb
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


@njit
def calculate_metrics_jit(
        timestamps, questionAsked, answerPosted, acceptedAnswerReceived,
        acceptedVoteReceived, acceptedAnswerPosted, target_time
):
    questions_asked_at = 0
    help_provided_at = 0
    accepted_answers_received_at = 0
    accepted_votes_received_at = 0
    accepted_answers_posted_at = 0

    questions_asked_30d = 0
    help_provided_30d = 0
    accepted_answers_received_30d = 0
    accepted_votes_received_30d = 0
    accepted_answers_posted_30d = 0

    questions_asked_14d = 0
    help_provided_14d = 0
    accepted_answers_received_14d = 0
    accepted_votes_received_14d = 0
    accepted_answers_posted_14d = 0

    questions_asked_7d = 0
    help_provided_7d = 0
    accepted_answers_received_7d = 0
    accepted_votes_received_7d = 0
    accepted_answers_posted_7d = 0

    questions_asked_3d = 0
    help_provided_3d = 0
    accepted_answers_received_3d = 0
    accepted_votes_received_3d = 0
    accepted_answers_posted_3d = 0

    cutoff_30d = target_time - 30 * 86400 * 1_000_000
    cutoff_14d = target_time - 14 * 86400 * 1_000_000
    cutoff_7d = target_time - 7 * 86400 * 1_000_000
    cutoff_3d = target_time - 3 * 86400 * 1_000_000

    # Find the timestamp of first activity (question asked or answer provided)
    first_activity_timestamp = np.int64(0)  # Initialize to 0 (no activity)
    for i in range(len(timestamps)):
        if (questionAsked[i] == 1 or answerPosted[i] == 1) and timestamps[i] < target_time:
            if first_activity_timestamp == 0 or timestamps[i] < first_activity_timestamp:
                first_activity_timestamp = timestamps[i]

    # Calculate time since first activity in days
    time_since_first_activity_days = 0.0
    if first_activity_timestamp > 0:  # If there was activity
        # Convert microseconds to days (86400 seconds per day, 1_000_000 microseconds per second)
        time_since_first_activity_days = (target_time - first_activity_timestamp) / (86400 * 1_000_000)

    for i in range(len(timestamps)):
        if timestamps[i] >= target_time:
            continue

        questions_asked_at += questionAsked[i]
        help_provided_at += answerPosted[i]
        accepted_answers_received_at += acceptedAnswerReceived[i]
        accepted_votes_received_at += acceptedVoteReceived[i]
        accepted_answers_posted_at += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_30d:
            questions_asked_30d += questionAsked[i]
            help_provided_30d += answerPosted[i]
            accepted_answers_received_30d += acceptedAnswerReceived[i]
            accepted_votes_received_30d += acceptedVoteReceived[i]
            accepted_answers_posted_30d += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_14d:
            questions_asked_14d += questionAsked[i]
            help_provided_14d += answerPosted[i]
            accepted_answers_received_14d += acceptedAnswerReceived[i]
            accepted_votes_received_14d += acceptedVoteReceived[i]
            accepted_answers_posted_14d += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_7d:
            questions_asked_7d += questionAsked[i]
            help_provided_7d += answerPosted[i]
            accepted_answers_received_7d += acceptedAnswerReceived[i]
            accepted_votes_received_7d += acceptedVoteReceived[i]
            accepted_answers_posted_7d += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_3d:
            questions_asked_3d += questionAsked[i]
            help_provided_3d += answerPosted[i]
            accepted_answers_received_3d += acceptedAnswerReceived[i]
            accepted_votes_received_3d += acceptedVoteReceived[i]
            accepted_answers_posted_3d += acceptedAnswerPosted[i]

    help_provided_ever = 1 if help_provided_at > 0 else 0

    if questions_asked_at == 0:
        initialExperienceReceiving_code = 0
    elif questions_asked_at > 0 and accepted_answers_received_at == 0:
        initialExperienceReceiving_code = 1
    elif questions_asked_at > 0 and accepted_answers_received_at > 0:
        initialExperienceReceiving_code = 2
    else:
        initialExperienceReceiving_code = 3

    if help_provided_at == 0:
        initialExperienceGiving_code = 0
    elif help_provided_at > 0 and accepted_answers_posted_at == 0:
        initialExperienceGiving_code = 1
    elif accepted_answers_posted_at > 0:
        initialExperienceGiving_code = 2
    else:
        initialExperienceGiving_code = 3

    return (
        questions_asked_at, help_provided_at, help_provided_ever,
        accepted_answers_received_at, accepted_votes_received_at,
        questions_asked_30d, help_provided_30d,
        accepted_answers_received_30d, accepted_votes_received_30d,
        accepted_answers_posted_30d,
        questions_asked_14d, help_provided_14d,
        accepted_answers_received_14d, accepted_votes_received_14d,
        accepted_answers_posted_14d,
        questions_asked_7d, help_provided_7d,
        accepted_answers_received_7d, accepted_votes_received_7d,
        accepted_answers_posted_7d,
        questions_asked_3d, help_provided_3d,
        accepted_answers_received_3d, accepted_votes_received_3d,
        accepted_answers_posted_3d,
        accepted_answers_posted_at, initialExperienceReceiving_code, initialExperienceGiving_code,
        time_since_first_activity_days
    )


def convert_experience_code_to_string(experience_receiving_code, experience_giving_code):
    if experience_receiving_code == 0:
        initialExperienceReceiving = "no help seeked"
    elif experience_receiving_code == 1:
        initialExperienceReceiving = "help seeked"
    elif experience_receiving_code == 2:
        initialExperienceReceiving = "help received"
    else:
        initialExperienceReceiving = "unknown"

    if experience_giving_code == 0:
        initialExperienceGiving = "no help attempted"
    elif experience_giving_code == 1:
        initialExperienceGiving = "help attempted"
    elif experience_giving_code == 2:
        initialExperienceGiving = "helped"
    else:
        initialExperienceGiving = "unknown"

    return initialExperienceReceiving, initialExperienceGiving


def prepare_user_history_for_jit(user_history_df):
    user_history_df = user_history_df.sort_values("timestamp")
    timestamps = user_history_df["timestamp"].astype(np.int64).values
    questionAsked = user_history_df["questionAsked"].values
    answerPosted = user_history_df["answerPosted"].values
    acceptedAnswerReceived = user_history_df["acceptedAnswerReceived"].values
    acceptedVoteReceived = user_history_df["acceptedVoteReceived"].values
    acceptedAnswerPosted = user_history_df["acceptedAnswerPosted"].values

    return (timestamps, questionAsked, answerPosted, acceptedAnswerReceived,
            acceptedVoteReceived, acceptedAnswerPosted)


def calculate_reciprocity_activation(user_histories):
    reciprocity_status = {}
    print("Calculating reciprocity activation...")

    for user_id, user_history in user_histories.items():
        first_answer = user_history[user_history['answerPosted'] == 1].sort_values('timestamp').head(1)

        if first_answer.empty:
            reciprocity_status[user_id] = (0, pd.NaT)
            continue

        first_answer_time = first_answer['timestamp'].iloc[0]

        accepted_answers = user_history[
            (user_history['acceptedAnswerReceived'] == 1) &
            (user_history['timestamp'] < first_answer_time) &
            (user_history['timestamp'] > first_answer_time - pd.Timedelta(days=7))
            ]

        if not accepted_answers.empty:
            reciprocity_status[user_id] = (1, first_answer_time)
        else:
            reciprocity_status[user_id] = (0, pd.NaT)

    return reciprocity_status


def process_single_bounty_chunk(chunk_user_ids, chunk_idx, input_file, temp_dir, columns_to_read):
    """
    Worker function to process a single chunk of users for the bounty dataset.
    """
    try:
        # Original print statement context
        print(f"\nProcessing chunk {chunk_idx} ({len(chunk_user_ids):,} users)")

        # Using pandas read_parquet with filters as in original (better for large ID lists than SQL strings)
        chunk_filter = [("user_id", "in", list(chunk_user_ids))]
        user_data_chunk = pd.read_parquet(input_file, filters=chunk_filter, columns=columns_to_read)

        print(f"Loaded {len(user_data_chunk):,} total rows for this chunk of users")

        user_data_chunk["timestamp"] = pd.to_datetime(user_data_chunk["timestamp"], errors="coerce")

        history_chunk = user_data_chunk[user_data_chunk["is_history"] == 1].copy()
        non_history_chunk = user_data_chunk[user_data_chunk["is_history"] == 0].copy()

        print(f"History rows: {len(history_chunk):,}")
        print(f"Non-history rows: {len(non_history_chunk):,}")

        if len(non_history_chunk) == 0:
            print("No non-history data in this chunk, skipping...")
            return 0

        # Map events as in original code
        history_chunk["questionAsked"] = history_chunk["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedVoteReceived"] = history_chunk["event"].map({"AcceptedAnswerVote": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedAnswerReceived"] = history_chunk["event"].map({"AcceptedAnswerReceived": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedAnswerPosted"] = history_chunk["event"].map({"AcceptedAnswerPosted": 1}).fillna(0).astype(np.int8)
        history_chunk["answerPosted"] = history_chunk["event"].map({"AnswerProvided": 1}).fillna(0).astype(np.int8)

        print("Building JIT-compatible user history cache for this chunk...")
        user_histories_jit = {}
        # Groupby is generally fast enough here
        for user_id, user_data in history_chunk.groupby("user_id"):
            user_histories_jit[user_id] = prepare_user_history_for_jit(user_data)

        # Original code also built a pandas cache, kept for strict logic adherence (though unused in metrics loop)
        user_histories_pandas = {}
        for user_id, user_data in history_chunk.groupby("user_id"):
            user_histories_pandas[user_id] = user_data

        # reciprocity_status = calculate_reciprocity_activation(user_histories_pandas)

        del history_chunk
        del user_data_chunk
        del user_histories_pandas

        non_history_chunk = non_history_chunk.sort_values(["user_id", "timestamp"])

        results = []

        print(f"Processing {len(non_history_chunk):,} rows with JIT...")
        
        # Iterate rows
        for _, row in non_history_chunk.iterrows():
            user_id = row["user_id"]
            target_time = row["timestamp"].to_numpy().astype(np.int64)

            if user_id in user_histories_jit:
                user_history_arrays = user_histories_jit[user_id]

                (
                    questions_asked_at, help_provided_at, help_provided_ever,
                    accepted_answers_received_at, accepted_votes_received_at,
                    questions_asked_30d, help_provided_30d,
                    accepted_answers_received_30d, accepted_votes_received_30d,
                    accepted_answers_posted_30d,
                    questions_asked_14d, help_provided_14d,
                    accepted_answers_received_14d, accepted_votes_received_14d,
                    accepted_answers_posted_14d,
                    questions_asked_7d, help_provided_7d,
                    accepted_answers_received_7d, accepted_votes_received_7d,
                    accepted_answers_posted_7d,
                    questions_asked_3d, help_provided_3d,
                    accepted_answers_received_3d, accepted_votes_received_3d,
                    accepted_answers_posted_3d,
                    accepted_answers_posted_at, initialExperienceReceiving_code, initialExperienceGiving_code,
                    time_since_first_activity_days
                ) = calculate_metrics_jit(*user_history_arrays, target_time)

                initialExperienceReceiving, initialExperienceGiving = convert_experience_code_to_string(
                    initialExperienceReceiving_code, initialExperienceGiving_code
                )
            else:
                questions_asked_at = help_provided_at = help_provided_ever = 0
                accepted_answers_received_at = accepted_votes_received_at = 0
                questions_asked_30d = help_provided_30d = 0
                accepted_answers_received_30d = accepted_votes_received_30d = accepted_answers_posted_30d = 0
                questions_asked_14d = help_provided_14d = 0
                accepted_answers_received_14d = accepted_votes_received_14d = accepted_answers_posted_14d = 0
                questions_asked_7d = help_provided_7d = 0
                accepted_answers_received_7d = accepted_votes_received_7d = accepted_answers_posted_7d = 0
                questions_asked_3d = help_provided_3d = 0
                accepted_answers_received_3d = accepted_votes_received_3d = accepted_answers_posted_3d = 0
                accepted_answers_posted_at = 0
                initialExperienceReceiving = "no help seeked"
                initialExperienceGiving = "no help attempted"
                time_since_first_activity_days = 0.0

            result = {
                "userId": user_id,
                "timestamp": row["timestamp"],
                "event": row["event"],
                "answerId": row.get("answer_id", None),
                "questionId": row.get("question_id", None),
                "isBounty": row.get("is_bounty", 0),
                "answeredBeforeBounty": row.get("answered_before_bounty", 0),
                "answeredAfterBountyEnded": row.get("answered_after_bounty_ended", 0),
                "questionEverHadBounty": row.get("question_ever_had_bounty", 0),
                "bountyAmount": row.get("bounty_amount", 0),
                "answerSequence": row.get("answer_sequence", None),

                "numQuestionsAskedAT": questions_asked_at,
                "numHelpProvidedAT": help_provided_at,
                "helpProvidedEver": help_provided_ever,
                "numAcceptedVotesReceivedAT": accepted_votes_received_at,
                "numAcceptedAnswersReceivedAT": accepted_answers_received_at,
                "numAcceptedAnswersPostedAT": accepted_answers_posted_at,

                "numQuestionsAsked30D": questions_asked_30d,
                "numHelpProvided30D": help_provided_30d,
                "numAcceptedVotesReceived30D": accepted_votes_received_30d,
                "numAcceptedAnswersReceived30D": accepted_answers_received_30d,
                "numAcceptedAnswersPosted30D": accepted_answers_posted_30d,

                "numQuestionsAsked14D": questions_asked_14d,
                "numHelpProvided14D": help_provided_14d,
                "numAcceptedVotesReceived14D": accepted_votes_received_14d,
                "numAcceptedAnswersReceived14D": accepted_answers_received_14d,
                "numAcceptedAnswersPosted14D": accepted_answers_posted_14d,

                "numQuestionsAsked7D": questions_asked_7d,
                "numHelpProvided7D": help_provided_7d,
                "numAcceptedVotesReceived7D": accepted_votes_received_7d,
                "numAcceptedAnswersReceived7D": accepted_answers_received_7d,
                "numAcceptedAnswersPosted7D": accepted_answers_posted_7d,

                "numQuestionsAsked3D": questions_asked_3d,
                "numHelpProvided3D": help_provided_3d,
                "numAcceptedVotesReceived3D": accepted_votes_received_3d,
                "numAcceptedAnswersReceived3D": accepted_answers_received_3d,
                "numAcceptedAnswersPosted3D": accepted_answers_posted_3d,

                "initialExperienceReceiving": initialExperienceReceiving,
                "initialExperienceGiving": initialExperienceGiving,
                "timeSinceFirstActivityDays": time_since_first_activity_days
            }

            results.append(result)

        chunk_output_df = pd.DataFrame(results)

        # Derived columns
        chunk_output_df["receivedAcceptedAnswerEver"] = (chunk_output_df["numAcceptedAnswersReceivedAT"] > 0).astype(int)
        chunk_output_df["receivedAcceptedVoteEver"] = (chunk_output_df["numAcceptedVotesReceivedAT"] > 0).astype(int)
        chunk_output_df["month"] = chunk_output_df["timestamp"].dt.month
        chunk_output_df["year"] = chunk_output_df["timestamp"].dt.year

        # Save Temp File
        rows_count = len(chunk_output_df)
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.parquet")
        chunk_output_df.to_parquet(temp_file, index=False)
        
        print(f"Processed {rows_count:,} rows in this chunk (Saved to temp)")

        del user_histories_jit
        del non_history_chunk
        del chunk_output_df
        # del reciprocity_status
        gc.collect()

        return rows_count

    except Exception as e:
        print(f"!!! Error in chunk {chunk_idx}: {e}")
        return 0

def process_bounty_dataset(input_file: str, output_file: str, chunk_size: int = 500000, num_workers: int = 4) -> None:
    print(f"\n=== Processing {input_file} (Parallel with {num_workers} workers) ===")

    print("Reading unique user IDs...")
    all_user_ids = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()
    print(f"Found {len(all_user_ids):,} unique users")

    # Note: Shuffling helps load balance complex vs simple users across workers
    user_ids = np.copy(all_user_ids)
    np.random.shuffle(user_ids)

    total_rows = pd.read_parquet(input_file, columns=["is_history"])
    total_non_history = len(total_rows[total_rows["is_history"] == 0])
    print(f"Total non-history rows: {total_non_history:,}")
    del total_rows

    columns_to_read = [
        "user_id", "timestamp", "event", "answer_id",
        "question_id", "is_bounty", "answered_before_bounty", "answered_after_bounty_ended",
        "question_ever_had_bounty", "bounty_amount", "answer_sequence", "is_history"
    ]

    # Setup Temp Directory
    temp_dir = os.path.join(os.path.dirname(output_file), "temp_bounty_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Cleanup old temps
    for f in glob.glob(os.path.join(temp_dir, "*.parquet")):
        os.remove(f)

    # Clean existing output
    if os.path.exists(output_file):
        os.remove(output_file)

    # Create Chunk Tasks
    # Note: If chunk_size is 500,000, this might create very few large chunks. 
    # For parallelization, smaller chunks (e.g. 5,000) are usually better, but keeping default.
    num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
    tasks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(user_ids))
        chunk_user_ids = user_ids[start_idx:end_idx]
        tasks.append((chunk_user_ids, i))

    processed_rows = 0

    print(f"Starting parallel processing of {len(tasks)} chunks...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_bounty_chunk, c_ids, c_idx, input_file, temp_dir, columns_to_read): c_idx 
            for c_ids, c_idx in tasks
        }
        
        # Progress bar
        with tqdm(total=len(user_ids), desc="Processing users", unit="users") as pbar:
            for future in as_completed(futures):
                try:
                    count = future.result()
                    processed_rows += count
                    # Approximation for progress bar update (using chunks is uneven, but works)
                    # Ideally we track exactly how many users were in that chunk
                    # Just update by chunk_size or track properly. 
                    # Simpler: update pbar by chunk size is usually close enough for visuals
                    pbar.update(chunk_size) 
                except Exception as e:
                    print(f"Chunk failed: {e}")

    # Consolidate results
    print(f"\nConsolidating temp files into {output_file}...")
    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")), 
                        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    if not temp_files:
        print("No processed data found!")
        return

    first_chunk = True
    for t_file in tqdm(temp_files, desc="Merging files"):
        df = pd.read_parquet(t_file)
        if first_chunk:
            df.to_parquet(output_file, index=False)
            first_chunk = False
        else:
            df.to_parquet(output_file, index=False, append=True, engine="fastparquet")
        os.remove(t_file)

    try:
        os.rmdir(temp_dir)
    except:
        pass

    print(f"Total processed: {processed_rows:,} / {total_non_history:,}")
    print(f"\nCompleted processing {processed_rows:,} total rows")
    print(f"Final output saved to {output_file}")


def handling_deleted_bountied_questions(processed_file_path, votes_path, bounty_timeline_path):
    """
    Identifies questions with votetype 8 (bounty) but not in the bounty timeline.
    Adds a missing_bounty column to the processed file.
    """
    print(f"\n=== Handling deleted bountied questions ===")

    # 1. SETUP: Use a disk-based database file instead of ':memory:' to handle larger data
    #    and increase the memory limit.
    db_file = "temp_processing.duckdb"
    print(f"Connecting to disk-backed database ({db_file})...")
    
    # Clean up previous temp db if it exists
    if os.path.exists(db_file):
        os.remove(db_file)

    conn = duckdb.connect(database=db_file)
    
    # --- FIX: INCREASE MEMORY LIMIT ---
    # Change '4GB' to '16GB' (or '32GB' depending on your server's capacity).
    # Or remove this line entirely to let DuckDB use all available RAM.
    try:
        conn.execute("SET memory_limit='16GB'") 
        print("Memory limit set to 16GB")
    except:
        print("Could not set memory limit, using defaults")

    # Ensure temp directory has space for spilling data to disk
    conn.execute("PRAGMA temp_directory='/tmp'")
    
    try:
        # Load the bounty timeline data
        print("Loading bounty timeline data...")
        conn.execute(f"""
            CREATE OR REPLACE VIEW bounty_timeline AS
            SELECT question_id FROM '{bounty_timeline_path}'
        """)

        # Load questions with votetype 8 (bounty votes)
        print("Loading questions with bounty votes...")
        conn.execute(f"""
            CREATE OR REPLACE VIEW bounty_votes AS
            SELECT DISTINCT PostId AS question_id
            FROM '{votes_path}'
            WHERE VoteTypeId = 8
        """)

        # Find questions with bounty votes but not in timeline
        print("Identifying missing bounty questions...")
        conn.execute("""
            CREATE OR REPLACE VIEW missing_bounty_questions AS
            SELECT bv.question_id
            FROM bounty_votes bv
            LEFT JOIN bounty_timeline bt ON bv.question_id = bt.question_id
            WHERE bt.question_id IS NULL
        """)

        # Count missing bounty questions
        missing_count = conn.execute("SELECT COUNT(*) FROM missing_bounty_questions").fetchone()[0]
        print(f"Found {missing_count:,} questions with bounty votes but not in timeline")

        # Save missing question IDs to a text file
        output_dir = Path(processed_file_path).parent
        missing_ids_file = output_dir / "missing_bounty_question_ids.txt"

        print(f"Saving missing question IDs to {missing_ids_file}")
        # Use COPY to export strictly the IDs instead of fetching to Python memory
        conn.execute(f"COPY (SELECT question_id FROM missing_bounty_questions ORDER BY question_id) TO '{missing_ids_file}' (HEADER 0, DELIMITER ',')")
        print(f"Saved missing IDs.")

        # Load the processed file
        print(f"Processing file: {processed_file_path}")
        conn.execute(f"""
            CREATE OR REPLACE VIEW processed_data AS
            SELECT * FROM '{processed_file_path}'
        """)

        # Perform the join and rewrite the file
        # Using a left join on the view
        print("Rewriting dataset with 'missing_bounty' column...")
        
        # We write to a temp file first to ensure atomic success
        temp_output = processed_file_path + ".temp"
        
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
            ) TO '{temp_output}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
        """)

        # Close connection before swapping files
        conn.close()
        
        # Swap files
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)
        os.rename(temp_output, processed_file_path)

        print(f"Successfully updated: {processed_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up temp file if it exists
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.remove(temp_output)
        if conn:
            conn.close()
        raise e
    finally:
        # Clean up the temp database file
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except:
                pass

if __name__ == "__main__":
    input_file = "../data/input/user_answers_bounty_dataset.parquet"
    output_file = "../data/study_datasets/user_answers_bounty_processed.parquet"
    votes_path = "../data/input/Votes.parquet"
    bounty_timeline_path = "../data/input/bounty_timeline.parquet"

    process_bounty_dataset(
        input_file=input_file,
        output_file=output_file,
        chunk_size=50000, 
        num_workers=16
    )

    handling_deleted_bountied_questions(
        processed_file_path=output_file,
        votes_path=votes_path,
        bounty_timeline_path=bounty_timeline_path
    )