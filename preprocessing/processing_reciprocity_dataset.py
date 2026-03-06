import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
from numba import njit
import duckdb
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set display options for better debugging output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


@njit
def calculate_metrics_jit(
        timestamps, questionAsked, answer, target_time
):
    """
    Calculate only the pre-treatment metrics needed for matching:
    - All-time and rolling-window counts of questions asked / help provided
    - Time since first activity (in days)
    """
    # Initialize variables for all time windows
    questions_asked_at = 0
    help_provided_at = 0

    questions_asked_30d = 0
    help_provided_30d = 0

    questions_asked_7d = 0
    help_provided_7d = 0

    # Set cutoff times for different windows (in microseconds)
    cutoff_30d = target_time - 30 * 86400 * 1_000_000
    cutoff_7d = target_time - 7 * 86400 * 1_000_000

    # Find the timestamp of first activity (question asked or answer provided)
    first_activity_timestamp = np.int64(0)  # Initialize to 0 (no activity)
    for i in range(len(timestamps)):
        if (questionAsked[i] == 1 or answer[i] == 1) and timestamps[i] < target_time:
            if first_activity_timestamp == 0 or timestamps[i] < first_activity_timestamp:
                first_activity_timestamp = timestamps[i]

    # Calculate time since first activity in days
    time_since_first_activity_days = 0.0
    if first_activity_timestamp > 0:  # If there was activity
        # Convert microseconds to days (86400 seconds per day, 1_000_000 microseconds per second)
        time_since_first_activity_days = (target_time - first_activity_timestamp) / (86400 * 1_000_000)

    # Process each event in user history
    for i in range(len(timestamps)):
        if timestamps[i] >= target_time:
            continue

        # Add to all-time metrics
        questions_asked_at += questionAsked[i]
        help_provided_at += answer[i]

        # Add to 30-day window metrics
        if timestamps[i] >= cutoff_30d:
            questions_asked_30d += questionAsked[i]
            help_provided_30d += answer[i]

        # Add to 7-day window metrics
        if timestamps[i] >= cutoff_7d:
            questions_asked_7d += questionAsked[i]
            help_provided_7d += answer[i]

    return (
        questions_asked_at, help_provided_at,
        questions_asked_30d, help_provided_30d,
        questions_asked_7d, help_provided_7d,
        time_since_first_activity_days
    )


def convert_experience_code_to_string(experience_receiving_code, experience_giving_code):
    """
    Convert numeric experience codes to human-readable string values.
    """
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
    """
    Convert pandas DataFrame to JIT-compatible arrays.
    """
    user_history_df = user_history_df.sort_values("timestamp")
    timestamps = user_history_df["timestamp"].astype(np.int64).values
    questionAsked = user_history_df["questionAsked"].values
    answer = user_history_df["answer"].values

    # JIT now only needs timestamps, questionAsked, and answer
    return (timestamps, questionAsked, answer)


def calculate_reciprocity_activation(user_histories):
    """
    Calculate reciprocity activation for each user, considering event timing.
    Tracks whether a user's first answer followed receiving a non-self accepted answer within 7 days.
    Note: All accepted answers in the dataset are already non-self answers (self-answers excluded in raw data).
    """
    reciprocity_status = {}
    print("Calculating reciprocity activation...")

    for user_id, user_history in tqdm(user_histories.items(), desc="Processing reciprocity", position=1, leave=False):
        # Get the first provided answer
        first_answer = user_history[user_history['answer'] == 1].sort_values('timestamp').head(1)

        if first_answer.empty:
            # User never provided an answer
            reciprocity_status[user_id] = (0, pd.NaT)
            continue

        first_answer_time = first_answer['timestamp'].iloc[0]

        # Check for accepted answers within 7 days before the first provided answer
        accepted_answers = user_history[
            (user_history['acceptedAnswerReceived'] == 1) &
            (user_history['timestamp'] < first_answer_time) &
            (user_history['timestamp'] > first_answer_time - pd.Timedelta(days=7))
            ]

        # Only mark as activated if there were accepted answers in the window
        if not accepted_answers.empty:
            reciprocity_status[user_id] = (1, first_answer_time)
        else:
            reciprocity_status[user_id] = (0, pd.NaT)

    return reciprocity_status


def process_single_chunk(chunk_user_ids, chunk_idx, chunk_size, input_file, temp_dir, cutoff_date):
    """
    Worker function: Processes a single chunk of users and saves a temporary file.
    """
    try:
        # Re-establish context inside the worker
        cutoff = pd.to_datetime(cutoff_date)
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + len(chunk_user_ids)
        
        print(f"\nProcessing users {start_idx:,} to {end_idx:,} ({len(chunk_user_ids):,} users)")

        # Use DuckDB for efficient filtering by user_id
        user_ids_str = ", ".join(str(id) for id in chunk_user_ids)
        
        # Create a fresh DuckDB connection for this process
        con = duckdb.connect()
        df = con.query(f"""
           SELECT * FROM read_parquet('{input_file}')
           WHERE user_id IN ({user_ids_str})
        """).to_df()
        con.close()

        print(f"Loaded {len(df):,} total rows for this chunk of users")

        # Convert timestamps to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["phase_one_start"] = pd.to_datetime(df["phase_one_start"], errors="coerce")
        df["phase_two_end"] = pd.to_datetime(df["phase_two_end"], errors="coerce")

        # Create binary event flags using minimal memory (int8)
        # Note: All Answer events in raw data exclude self-answers
        # For matching we only need Question/Answer history (no accepted/vote breakdown).
        df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        df["answer"] = df["event"].map({"Answer": 1, "Window_Answer": 1}).fillna(0).astype(np.int8)

        # Calculate response time in hours (time to first non-self, non-negative answer)
        df["responseTimeHours"] = None
        mask = ~df["first_answer_timestamp"].isna() & ~df["question_timestamp"].isna()
        if any(mask):
            df.loc[mask, "responseTimeHours"] = (
                    (pd.to_datetime(df.loc[mask, "first_answer_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "question_timestamp"])).dt.total_seconds() / 3600
            )

        # Calculate time-to-accepted-answer in hours (time to accepted non-self answer, if any)
        df["timeToAcceptedAnswerHours"] = None
        mask = ~df["accepted_answer_timestamp"].isna() & ~df["question_timestamp"].isna()
        if any(mask):
            df.loc[mask, "timeToAcceptedAnswerHours"] = (
                    (pd.to_datetime(df.loc[mask, "accepted_answer_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "question_timestamp"])).dt.total_seconds() / 3600
            )

        # Calculate time-to-accept-vote in hours
        df["timeToAcceptVoteHours"] = None
        mask = ~df["accepted_answer_vote_timestamp"].isna() & ~df["accepted_answer_timestamp"].isna()
        if any(mask):
            df.loc[mask, "timeToAcceptVoteHours"] = (
                    (pd.to_datetime(df.loc[mask, "accepted_answer_vote_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "accepted_answer_timestamp"])).dt.total_seconds() / 3600
            )

        # Separate history and non-history data
        history_chunk = df[df["is_history"] == 1].copy()
        non_history_chunk = df[df["is_history"] == 0].copy()

        print(f"History rows: {len(history_chunk):,}")
        print(f"Non-history rows: {len(non_history_chunk):,}")

        del df
        gc.collect()

        if len(non_history_chunk) == 0:
            print("No non-history data in this chunk, skipping...")
            return 0

        # Build JIT-compatible user history cache
        print("Building JIT-compatible user history cache for this chunk...")
        user_histories_jit = {}
        # user_histories_pandas = {} # Commented out as per original logic flow usually dropping it

        # Note: Using tqdm inside parallel workers can look messy, but keeping as requested
        for user_id, user_data in history_chunk.groupby("user_id"):
            user_histories_jit[user_id] = prepare_user_history_for_jit(user_data)
            # user_histories_pandas[user_id] = user_data

        # Calculate reciprocity activation
        # reciprocity_status = calculate_reciprocity_activation(user_histories_pandas)

        del history_chunk
        # del user_histories_pandas

        # Find all Phase_One_Start events
        phase_one_starts = non_history_chunk[non_history_chunk["event"] == "Phase_One_Start"]
        all_event_ids = non_history_chunk["event_id"].unique()

        # Calculate metrics for each Phase_One_Start event using JIT
        print("Calculating metrics for each Phase_One_Start event...")
        event_metrics = {}

        for _, row in phase_one_starts.iterrows():
            user_id = row["user_id"]
            event_id = row["event_id"]
            target_time = row["timestamp"].to_numpy().astype(np.int64)  # Convert to int64 for JIT

            # Calculate metrics using JIT
            if user_id not in user_histories_jit:
                # Default metrics if no history
                metrics_values = (
                    0, 0,      # AT: questions asked, help provided
                    0, 0,      # 30D: questions asked, help provided
                    0, 0,      # 7D: questions asked, help provided
                    0.0        # time_since_first_activity_days
                )
            else:
                # Calculate metrics using JIT
                metrics_values = calculate_metrics_jit(*user_histories_jit[user_id], target_time)

            event_metrics[event_id] = tuple(metrics_values)

        # Set default metrics for missing event_ids
        missing_event_ids = set(all_event_ids) - set(event_metrics.keys())
        if missing_event_ids:
            default_metrics = (
                0, 0,      # AT
                0, 0,      # 30D
                0, 0,      # 7D
                0.0        # time_since_first_activity_days
            )
            for eid in missing_event_ids:
                event_metrics[eid] = default_metrics

        # Define metric column names
        metric_columns = [
            "numQuestionsAskedAT", "numHelpProvidedAT",
            "numQuestionsAsked30D", "numHelpProvided30D",
            "numQuestionsAsked7D", "numHelpProvided7D",
            "timeSinceFirstActivityDays"
        ]

        # Convert metrics to DataFrame for efficient merging
        print("Converting metrics to DataFrame for merging...")
        metrics_data = []
        for event_id, metrics in event_metrics.items():
            metrics_data.append([event_id] + list(metrics))

        metrics_df = pd.DataFrame(metrics_data, columns=['event_id'] + metric_columns)

        # Merge metrics with non_history_chunk using event_id
        print("Merging metrics with non-history data...")
        non_history_chunk = pd.merge(non_history_chunk, metrics_df, on='event_id', how='left')

        # Add reciprocity activation data
        # (Commented out logic kept as is)

        # Mark Window_Answer events for numHelped calculation (count per question over full window)
        non_history_chunk["numHelped"] = np.where(non_history_chunk["event"] == "Window_Answer", 1, 0)

        non_history_chunk["month"] = non_history_chunk["timestamp"].dt.month
        non_history_chunk["year"] = non_history_chunk["timestamp"].dt.year

        # Aggregate to one row per question (event_id); all matching covariates at question level
        print("Aggregating data by event_id (question level)...")
        group_cols = ["event_id"]
        agg_dict = {
            "user_id": "first",
            "timestamp": "first",
            "event": "first",
            "question_id": "first",
            "phase_one_start": "first",
            "phase_two_end": "first",
            "has_answer": "first",
            "has_unhelpful_answer": "first",
            "has_accepted_answer": "first",
            "has_self_answer": "first",
            "responseTimeHours": "first",
            "timeToAcceptedAnswerHours": "first",
            "timeToAcceptVoteHours": "first",
            "numHelped": "sum",
            "year": "first",
            "month": "first",
            "numQuestionsAskedAT": "first",
            "numHelpProvidedAT": "first",
            "numQuestionsAsked30D": "first",
            "numHelpProvided30D": "first",
            "numQuestionsAsked7D": "first",
            "numHelpProvided7D": "first",
            "timeSinceFirstActivityDays": "first",
            "registration_date": "first",
            "days_since_registration_at_phase_one_start": "first",
            "helps_given_between_question_and_answer": "first",
            "tag_ids": "first",
        }

        # Apply aggregation (one row per question)
        non_history_chunk = non_history_chunk.sort_values(["event_id", "timestamp"])
        agg_df = non_history_chunk.groupby(group_cols, as_index=False).agg(agg_dict)

        # Filter out rows beyond cutoff date (main place questions are lost in processing:
        # questions whose 7-day window ends after cutoff_date are dropped)
        initial_count = len(agg_df)
        agg_df = agg_df[agg_df["phase_two_end"] <= cutoff]
        removed = initial_count - len(agg_df)
        if removed > 0:
            print(f"Removed {removed:,} rows beyond cutoff date")

        # Optimize data types
        print("Optimizing data types...")
        agg_df = agg_df.convert_dtypes()  # Convert to best possible dtypes
        for col in agg_df.select_dtypes(include=["Int32", "Int64"]).columns:
            # Only convert if there are no NULLs
            if not agg_df[col].isna().any():
                agg_df[col] = agg_df[col].astype("int64")
        for col in agg_df.select_dtypes(include=["Float32"]).columns:
            agg_df[col] = agg_df[col].astype("float64")


        if "tag_ids" in agg_df.columns:
            def _get_main_tag_id(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return pd.NA
                try:
                    v = list(val) if not isinstance(val, (list, tuple)) else val
                    if len(v) > 0:
                        return int(v[0])
                except (TypeError, ValueError, IndexError):
                    pass
                return pd.NA

            agg_df["mainTagId"] = agg_df["tag_ids"].apply(_get_main_tag_id)

        # Create additional metrics needed for matching
        print("Calculating additional metrics...")
        agg_df['hasHelped'] = (agg_df['numHelped'] > 0).astype(int)
        # Single row per question; keep phase=1 for downstream scripts that expect it
        agg_df['phase'] = 1

        # Standardize column naming
        print("Standardizing column names...")
        column_rename_map = {
            'user_id': 'userId',
            'event_id': 'eventId',
            'question_id': 'questionId',
            'phase_one_start': 'phaseOneStart',
            'phase_two_end': 'phaseTwoEnd',
            'has_answer': 'hasAnswer',
            'has_unhelpful_answer': 'hasUnhelpfulAnswer',
            'has_accepted_answer': 'hasAcceptedAnswer',
            'has_self_answer': 'hasSelfAnswer',
        }
        agg_df = agg_df.rename(columns=column_rename_map)

        # PARALLELIZATION CHANGE: Save to temporary chunk file
        chunk_count = len(agg_df)
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.parquet")
        agg_df.to_parquet(temp_file, index=False)
        print(f"Saved temporary chunk {chunk_idx} with {chunk_count:,} rows")

        # Free memory
        del non_history_chunk
        del user_histories_jit
        del agg_df
        del event_metrics
        # del reciprocity_status
        gc.collect()

        return chunk_count

    except Exception as e:
        print(f"!!! Error processing chunk {chunk_idx}: {e}")
        return 0

def process_question_centered_dataset(input_file: str, output_file: str, chunk_size: int = 1000,
                                      cutoff_date: str = "2025-04-01", num_workers: int = 16) -> None:
    """
    Process the question-centered dataset to calculate metrics for each user.
    Dataset is centered on when users post their question.
    Self-answers are excluded from all metrics.
    Uses JIT compilation for performance optimization.
    PARALLELIZED VERSION
    """
    print(f"\n=== Processing {input_file} (Parallel with {num_workers} workers) ===")
    
    # Get unique user IDs and shuffle for better parallelization
    print("Reading unique user IDs...")
    all_user_ids = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()
    print(f"Found {len(all_user_ids):,} unique users")

    user_ids = np.copy(all_user_ids)
    np.random.shuffle(user_ids)

    # Get total count of non-history rows
    total_rows = pd.read_parquet(input_file, columns=["is_history"])
    total_non_history = len(total_rows[total_rows["is_history"] == 0])
    print(f"Total non-history rows: {total_non_history:,}")
    del total_rows

    # Create output directory and temp directory for chunks
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    temp_dir = os.path.join(os.path.dirname(output_file), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clean up old chunks if any
    for f in glob.glob(os.path.join(temp_dir, "*.parquet")):
        os.remove(f)

    if os.path.exists(output_file):
        os.remove(output_file)

    # Prepare chunks
    num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
    all_result_count = 0
    
    chunk_tasks = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(user_ids))
        chunk_user_ids = user_ids[start_idx:end_idx]
        chunk_tasks.append((chunk_user_ids, chunk_idx))

    print(f"Starting parallel processing of {num_chunks} chunks...")

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_chunk, c_ids, c_idx, chunk_size, input_file, temp_dir, cutoff_date)
            for c_ids, c_idx in chunk_tasks
        ]
        
        # Use tqdm to track completed chunks
        with tqdm(total=len(user_ids), desc="Overall progress", unit="users", position=0) as pbar:
            for future in as_completed(futures):
                result_count = future.result()
                all_result_count += result_count
                # We update by chunk_size (approx) or we can pass actual count back, 
                # strictly we should update by len(chunk_user_ids) but simplified here:
                pbar.update(chunk_size)

    # Consolidate results
    print("\nConsolidating temporary chunk files into final output...")
    
    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")), 
                        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not temp_files:
        print("No data processed!")
        return

    # Merge chunks: concat all then write once (avoids fastparquet append schema
    # issues with list columns like tag_ids, which cause KeyError: 'element')
    chunk_dfs = []
    for temp_f in tqdm(temp_files, desc="Reading chunks"):
        chunk_dfs.append(pd.read_parquet(temp_f))
        os.remove(temp_f)
    merged_df = pd.concat(chunk_dfs, ignore_index=True)
    del chunk_dfs
    gc.collect()
    merged_df.to_parquet(output_file, index=False, engine="pyarrow")
    print(f"Written output file {output_file} with {len(merged_df):,} rows")
    del merged_df
    gc.collect()

    # Cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except:
        pass

    print(f"\nCompleted processing {all_result_count:,} total rows")
    print(f"Final output saved to {output_file}")
    

if __name__ == "__main__":
    # Paths relative to project root so script works from any cwd
    base_dir = Path(__file__).resolve().parent.parent
    input_folder = base_dir / "data" / "input"
    output_folder = base_dir / "data" / "study_datasets"
    cutoff_date = "2025-04-01"

    # Set this to True to process test mode files, False for full dataset
    test_mode = False
    test_user_limit = 100000  # Should match the limit used in creating script

    # Process each question-centered model dataset
    for days in [7]:
        # Build filename based on test mode
        if test_mode:
            input_file = input_folder / f"question_centered_model_{days}d_all_questions_TEST{test_user_limit}.parquet"
            output_file = output_folder / f"question_centered_model_{days}d_processed_TEST{test_user_limit}.parquet"
        else:
            input_file = input_folder / f"question_centered_model_{days}d_all_questions.parquet"
            output_file = output_folder / f"question_centered_model_{days}d_processed.parquet"

        process_question_centered_dataset(
            input_file=input_file,
            output_file=output_file,
            chunk_size=100000,
            cutoff_date=cutoff_date
        )