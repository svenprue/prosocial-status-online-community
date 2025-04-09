import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from numba import njit
import duckdb
# Set display options to show all rows and columns with full width
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of each cell
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import duckdb

# Set display options to show all rows and columns with full width
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of each cell


def calculate_metrics_pandas(user_history, target_time):
    """
    Calculate metrics using pandas filtering and aggregation
    """
    # Make sure history is sorted by timestamp
    user_history = user_history.sort_values("timestamp")

    # Get all history up to the target time
    user_history_before = user_history[user_history["timestamp"] <= target_time]

    # Calculate all-time metrics
    questions_asked_at = user_history_before["questionAsked"].sum()
    help_received_at = user_history_before["acceptedAnswer"].sum()
    help_provided_at = user_history_before["answer"].sum()
    answers_received_at = user_history_before["answerReceived"].sum()
    accepted_answers_received_at = user_history_before["acceptedAnswerReceived"].sum()
    accepted_votes_received_at = user_history_before["acceptedVoteReceived"].sum()
    help_provided_ever = 1 if help_provided_at > 0 else 0

    # Calculate 30-day window
    cutoff_30d = target_time - pd.Timedelta(days=30)
    window_30d = user_history_before[user_history_before["timestamp"] >= cutoff_30d]
    questions_asked_30d = window_30d["questionAsked"].sum()
    help_received_30d = window_30d["acceptedAnswer"].sum()
    help_provided_30d = window_30d["answer"].sum()
    answers_received_30d = window_30d["answerReceived"].sum()
    accepted_answers_received_30d = window_30d["acceptedAnswerReceived"].sum()
    accepted_votes_received_30d = window_30d["acceptedVoteReceived"].sum()

    # Calculate 14-day window
    cutoff_14d = target_time - pd.Timedelta(days=14)
    window_14d = user_history_before[user_history_before["timestamp"] >= cutoff_14d]
    questions_asked_14d = window_14d["questionAsked"].sum()
    help_received_14d = window_14d["acceptedAnswer"].sum()
    help_provided_14d = window_14d["answer"].sum()
    answers_received_14d = window_14d["answerReceived"].sum()
    accepted_answers_received_14d = window_14d["acceptedAnswerReceived"].sum()
    accepted_votes_received_14d = window_14d["acceptedVoteReceived"].sum()

    # Calculate 7-day window
    cutoff_7d = target_time - pd.Timedelta(days=7)
    window_7d = user_history_before[user_history_before["timestamp"] >= cutoff_7d]
    questions_asked_7d = window_7d["questionAsked"].sum()
    help_received_7d = window_7d["acceptedAnswer"].sum()
    help_provided_7d = window_7d["answer"].sum()
    answers_received_7d = window_7d["answerReceived"].sum()
    accepted_answers_received_7d = window_7d["acceptedAnswerReceived"].sum()
    accepted_votes_received_7d = window_7d["acceptedVoteReceived"].sum()

    # Calculate 3-day window
    cutoff_3d = target_time - pd.Timedelta(days=3)
    window_3d = user_history_before[user_history_before["timestamp"] >= cutoff_3d]
    questions_asked_3d = window_3d["questionAsked"].sum()
    help_received_3d = window_3d["acceptedAnswer"].sum()
    help_provided_3d = window_3d["answer"].sum()
    answers_received_3d = window_3d["answerReceived"].sum()
    accepted_answers_received_3d = window_3d["acceptedAnswerReceived"].sum()
    accepted_votes_received_3d = window_3d["acceptedVoteReceived"].sum()

    return (
        questions_asked_at, help_received_at, help_provided_at, help_provided_ever,
        answers_received_at, accepted_answers_received_at, accepted_votes_received_at,
        questions_asked_30d, help_received_30d, help_provided_30d,
        answers_received_30d, accepted_answers_received_30d, accepted_votes_received_30d,
        questions_asked_14d, help_received_14d, help_provided_14d,
        answers_received_14d, accepted_answers_received_14d, accepted_votes_received_14d,
        questions_asked_7d, help_received_7d, help_provided_7d,
        answers_received_7d, accepted_answers_received_7d, accepted_votes_received_7d,
        questions_asked_3d, help_received_3d, help_provided_3d,
        answers_received_3d, accepted_answers_received_3d, accepted_votes_received_3d
    )


def process_question_dataset(input_file: str, output_file: str, chunk_size: int = 1000,
                             cutoff_date: str = "2025-04-01") -> None:
    """
    Process the question-centered dataset to calculate metrics for each user.
    Uses a pandas-based approach for metric calculation.

    Args:
        input_file: Path to the input parquet file
        output_file: Path to save the processed data
        chunk_size: Number of users to process in each chunk
        cutoff_date: Cutoff date for phase_two_end
    """
    print(f"\n=== Processing {input_file} ===")
    cutoff = pd.to_datetime(cutoff_date)

    # Get unique user IDs
    user_ids_original = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()
    print(f"Found {len(user_ids_original):,} unique users")
    print("First 10 user IDs before shuffling:")
    print(user_ids_original[:10])

    # Create a copy and shuffle it
    user_ids = np.copy(user_ids_original)
    np.random.shuffle(user_ids)
    print("First 10 user IDs after shuffling:")
    print(user_ids[:10])

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if output file exists and remove if it does
    if os.path.exists(output_file):
        os.remove(output_file)

    # Calculate number of chunks
    num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
    all_result_count = 0

    # Process users in chunks
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks", total=num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(user_ids))
        chunk_user_ids = user_ids[start_idx:end_idx]

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} with {len(chunk_user_ids):,} users")

        # Read only data for current chunk of users
        user_ids_str = ", ".join(str(id) for id in chunk_user_ids)

        # Use DuckDB to efficiently filter the parquet file
        df = duckdb.query(f"""
           SELECT * FROM read_parquet('{input_file}')
           WHERE user_id IN ({user_ids_str})
       """).to_df()
        print(f"Loaded {len(df):,} rows for this chunk.")

        # Ensure the timestamp is in datetime format
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["phase_one_start"] = pd.to_datetime(df["phase_one_start"], errors="coerce")
        df["phase_two_end"] = pd.to_datetime(df["phase_two_end"], errors="coerce")

        # Create numeric flags from event type with minimal memory usage
        df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["answer"] = df["event"].map({"Answer": 1, "Window_Answer": 1}).fillna(0).astype(np.int8)
        df["answerReceived"] = df["event"].map({"AnswerReceived": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswerReceived"] = df["event"].map({"AcceptedAnswerReceived": 1}).fillna(0).astype(np.int8)
        df["acceptedVoteReceived"] = df["event"].map({"AcceptedVoteReceived": 1}).fillna(0).astype(np.int8)

        # Calculate time-to-first-answer in hours if both timestamps exist
        df["time_to_first_answer_hours"] = None
        mask = ~df["first_answer_timestamp"].isna() & ~df["question_timestamp"].isna()
        if any(mask):
            df.loc[mask, "time_to_first_answer_hours"] = (
                    (pd.to_datetime(df.loc[mask, "first_answer_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "question_timestamp"])).dt.total_seconds() / 3600
            )

        # Calculate time-to-accepted-answer in hours if both timestamps exist
        df["time_to_accepted_answer_hours"] = None
        mask = ~df["accepted_answer_timestamp"].isna() & ~df["question_timestamp"].isna()
        if any(mask):
            df.loc[mask, "time_to_accepted_answer_hours"] = (
                    (pd.to_datetime(df.loc[mask, "accepted_answer_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "question_timestamp"])).dt.total_seconds() / 3600
            )

        # Calculate time-to-accept-vote in hours if both timestamps exist
        df["time_to_accept_vote_hours"] = None
        mask = ~df["accepted_answer_vote_timestamp"].isna() & ~df["accepted_answer_timestamp"].isna()
        if any(mask):
            df.loc[mask, "time_to_accept_vote_hours"] = (
                    (pd.to_datetime(df.loc[mask, "accepted_answer_vote_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "accepted_answer_timestamp"])).dt.total_seconds() / 3600
            )

        # Separate history and non-history data
        history_df = df[df["is_history"] == 1].copy()
        non_history_df = df[df["is_history"] == 0].copy()

        # Free memory
        del df
        gc.collect()

        print(f"History rows in chunk: {len(history_df):,}")
        print(f"Non-history rows in chunk: {len(non_history_df):,}")

        if len(non_history_df) == 0:
            print("No non-history rows in this chunk, skipping...")
            continue

        # Build user history data
        print("Building user history cache...")
        user_histories = {}

        for user_id, group in tqdm(history_df.groupby("user_id"), desc="Preprocessing users"):
            user_histories[user_id] = group

        # Free memory
        del history_df
        gc.collect()

        # Process non-history rows
        print("Processing non-history rows...")

        # Create placeholder for Phase_One_Start events
        phase_one_start_metrics = {}

        # Sort non-history rows by user_id and timestamp for better efficiency
        non_history_df = non_history_df.sort_values(["user_id", "timestamp"])

        # Create result columns for all phases
        results = []

        # Choose a debug user to verify metrics
        debug_user_id = chunk_user_ids[0] if len(chunk_user_ids) > 0 else None
        debug_count = 0

        # Process each event row
        for _, row in tqdm(non_history_df.iterrows(), total=len(non_history_df), desc="Processing rows"):
            user_id = row["user_id"]
            event_id = row["event_id"]
            target_time = row["timestamp"]

            # If user not in history, use zeros for all metrics
            if user_id not in user_histories:
                metrics = (0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0)
            else:
                # Use pandas-based calculation
                metrics = calculate_metrics_pandas(user_histories[user_id], target_time)

                # Debug output for the first user
                if user_id == debug_user_id and debug_count < 3:
                    debug_count += 1
                    print(f"\nDEBUG: User {user_id}, Event {event_id}, Time {target_time}")
                    print(f"All-time metrics: {metrics[0:7]}")
                    print(f"30-day metrics: {metrics[7:13]}")
                    print(f"Recent events for this user:")
                    recent = user_histories[user_id][user_histories[user_id]["timestamp"] <= target_time].tail(5)
                    print(recent[["timestamp", "event", "questionAsked", "acceptedAnswer", "answer"]].to_string())

            # Store metrics for Phase_One_Start events for later lookup by event_id
            if row["event"] == "Phase_One_Start":
                phase_one_start_metrics[event_id] = metrics

            # Unpack metrics
            (q_at, a_at, ans_at, help_provided_ever, ans_rec_at, acc_ans_rec_at, acc_vote_rec_at,
             q_30d, a_30d, ans_30d, ans_rec_30d, acc_ans_rec_30d, acc_vote_rec_30d,
             q_14d, a_14d, ans_14d, ans_rec_14d, acc_ans_rec_14d, acc_vote_rec_14d,
             q_7d, a_7d, ans_7d, ans_rec_7d, acc_ans_rec_7d, acc_vote_rec_7d,
             q_3d, a_3d, ans_3d, ans_rec_3d, acc_ans_rec_3d, acc_vote_rec_3d) = metrics

            # Create result dictionary
            result = {
                "event_id": event_id,
                "user_id": user_id,
                "timestamp": target_time,
                "event": row["event"],
                "question_id": row.get("question_id", None),
                "phase_one_start": row.get("phase_one_start", None),
                "phase_two_end": row.get("phase_two_end", None),
                "event_history": row.get("event_history", None),
                "is_history": 0,  # Non-history rows only

                # Question-specific properties
                "has_answer": row.get("has_answer", 0),
                "has_accepted_answer": row.get("has_accepted_answer", 0),
                "time_to_first_answer_hours": row.get("time_to_first_answer_hours", None),
                "time_to_accepted_answer_hours": row.get("time_to_accepted_answer_hours", None),
                "time_to_accept_vote_hours": row.get("time_to_accept_vote_hours", None),

                # All-time metrics
                "numQuestionsAskedAT": q_at,
                "numHelpReceivedAT": a_at,
                "numHelpProvidedAT": ans_at,
                "numAnswersReceivedAT": ans_rec_at,
                "numAcceptedAnswersReceivedAT": acc_ans_rec_at,
                "numAcceptedVotesReceivedAT": acc_vote_rec_at,
                "numHelpProvidedEver": help_provided_ever,

                # 30-day window metrics
                "numQuestionsAsked30D": q_30d,
                "numHelpReceived30D": a_30d,
                "numHelpProvided30D": ans_30d,
                "numAnswersReceived30D": ans_rec_30d,
                "numAcceptedAnswersReceived30D": acc_ans_rec_30d,
                "numAcceptedVotesReceived30D": acc_vote_rec_30d,

                # 14-day window metrics
                "numQuestionsAsked14D": q_14d,
                "numHelpReceived14D": a_14d,
                "numHelpProvided14D": ans_14d,
                "numAnswersReceived14D": ans_rec_14d,
                "numAcceptedAnswersReceived14D": acc_ans_rec_14d,
                "numAcceptedVotesReceived14D": acc_vote_rec_14d,

                # 7-day window metrics
                "numQuestionsAsked7D": q_7d,
                "numHelpReceived7D": a_7d,
                "numHelpProvided7D": ans_7d,
                "numAnswersReceived7D": ans_rec_7d,
                "numAcceptedAnswersReceived7D": acc_ans_rec_7d,
                "numAcceptedVotesReceived7D": acc_vote_rec_7d,

                # 3-day window metrics
                "numQuestionsAsked3D": q_3d,
                "numHelpReceived3D": a_3d,
                "numHelpProvided3D": ans_3d,
                "numAnswersReceived3D": ans_rec_3d,
                "numAcceptedAnswersReceived3D": acc_ans_rec_3d,
                "numAcceptedVotesReceived3D": acc_vote_rec_3d,
            }

            results.append(result)

        # Create DataFrame from results
        chunk_output_df = pd.DataFrame(results)

        # Free memory for user histories
        del user_histories
        gc.collect()

        # Determine phase (1 or 2) based on timestamps
        chunk_output_df["TimestampInt"] = pd.to_numeric(chunk_output_df["timestamp"], errors="coerce",
                                                        downcast="integer")
        start_times = (
            chunk_output_df.loc[chunk_output_df["event"] == "Phase_One_Start"]
            .groupby("event_id")["TimestampInt"]
            .max()
        )
        end_times = (
            chunk_output_df.loc[chunk_output_df["event"] == "Phase_Two_End"]
            .groupby("event_id")["TimestampInt"]
            .max()
        )
        chunk_output_df["start_time"] = chunk_output_df["event_id"].map(start_times)
        chunk_output_df["end_time"] = chunk_output_df["event_id"].map(end_times)
        chunk_output_df["end_time"] = chunk_output_df["end_time"].fillna(chunk_output_df["start_time"])

        chunk_output_df["isPhase"] = np.where(
            (chunk_output_df["TimestampInt"] >= chunk_output_df["start_time"]) &
            (chunk_output_df["TimestampInt"] < chunk_output_df["end_time"]),
            0,
            1,
        )
        chunk_output_df["phase"] = chunk_output_df["isPhase"].replace({0: 1, 1: 2})

        # Add additional derived metrics
        chunk_output_df["receivedHelpEver"] = (chunk_output_df["numHelpReceivedAT"] > 0).astype(int)
        chunk_output_df["receivedAnswerEver"] = (chunk_output_df["numAnswersReceivedAT"] > 0).astype(int)
        chunk_output_df["receivedAcceptedAnswerEver"] = (chunk_output_df["numAcceptedAnswersReceivedAT"] > 0).astype(
            int)
        chunk_output_df["receivedAcceptedVoteEver"] = (chunk_output_df["numAcceptedVotesReceivedAT"] > 0).astype(int)
        chunk_output_df["month"] = chunk_output_df["timestamp"].dt.month
        chunk_output_df["year"] = chunk_output_df["timestamp"].dt.year

        # Set the numHelped column based on answer events
        chunk_output_df["numHelped"] = np.where(chunk_output_df["event"] == "Window_Answer", 1, 0)
        chunk_output_df["hasAnswer"] = (chunk_output_df["has_answer"] > 0).astype(int)

        # Group and aggregate data by event_id and phase
        group_cols = ["event_id", "phase"]
        agg_dict = {
            "user_id": "first",
            "timestamp": "first",
            "event": "first",
            "question_id": "first",
            "phase_one_start": "first",
            "phase_two_end": "first",
            "event_history": "first",
            "is_history": "first",
            "has_answer": "first",
            "has_accepted_answer": "first",
            "time_to_first_answer_hours": "first",
            "time_to_accepted_answer_hours": "first",
            "time_to_accept_vote_hours": "first",
            "numHelped": "sum",
            "hasAnswer": "max",
            "numHelpProvidedEver": "first",
            "receivedHelpEver": "first",
            "receivedAnswerEver": "first",
            "receivedAcceptedAnswerEver": "first",
            "receivedAcceptedVoteEver": "first",
            "year": "first",
            "month": "first",
            "numQuestionsAskedAT": "first",
            "numHelpReceivedAT": "first",
            "numHelpProvidedAT": "first",
            "numAnswersReceivedAT": "first",
            "numAcceptedAnswersReceivedAT": "first",
            "numAcceptedVotesReceivedAT": "first",
            "numQuestionsAsked30D": "first",
            "numHelpReceived30D": "first",
            "numHelpProvided30D": "first",
            "numAnswersReceived30D": "first",
            "numAcceptedAnswersReceived30D": "first",
            "numAcceptedVotesReceived30D": "first",
            "numQuestionsAsked14D": "first",
            "numHelpReceived14D": "first",
            "numHelpProvided14D": "first",
            "numAnswersReceived14D": "first",
            "numAcceptedAnswersReceived14D": "first",
            "numAcceptedVotesReceived14D": "first",
            "numQuestionsAsked7D": "first",
            "numHelpReceived7D": "first",
            "numHelpProvided7D": "first",
            "numAnswersReceived7D": "first",
            "numAcceptedAnswersReceived7D": "first",
            "numAcceptedVotesReceived7D": "first",
            "numQuestionsAsked3D": "first",
            "numHelpReceived3D": "first",
            "numHelpProvided3D": "first",
            "numAnswersReceived3D": "first",
            "numAcceptedAnswersReceived3D": "first",
            "numAcceptedVotesReceived3D": "first",
        }

        # Apply aggregation
        agg_df = chunk_output_df.groupby(group_cols).agg(agg_dict).reset_index()

        # Print the first few rows of the aggregated dataframe for verification
        if chunk_idx == 0:
            print("\nSample of aggregated data:")
            print(agg_df.head().to_string())

        # Remove any rows with phase_two_end after the cutoff date
        initial_count = len(agg_df)
        agg_df = agg_df[agg_df["phase_two_end"] <= cutoff]
        removed = initial_count - len(agg_df)
        print(f"Removed {removed} aggregated rows with phase_two_end after {cutoff_date}.")

        # Demean numHelped per user
        agg_df["meaned_numHelped"] = agg_df.groupby("user_id")["numHelped"].transform(lambda x: x - x.mean())

        # Count the number of rows in this chunk
        chunk_count = len(agg_df)
        all_result_count += chunk_count

        # Save the chunk to file
        print(f"Saving chunk {chunk_idx + 1}/{num_chunks} with {chunk_count:,} rows...")
        if chunk_idx == 0:
            # First chunk, create the file
            agg_df.to_parquet(output_file, index=False)
            print(f"Created new output file: {output_file}")
        else:
            # Subsequent chunks, append to existing file
            agg_df.to_parquet(output_file, index=False, append=True)
            print(f"Appended to output file (running total: {all_result_count:,} rows)")

        # Free memory
        del chunk_output_df
        del non_history_df
        del results
        del agg_df
        gc.collect()
        print(f"Memory cleared for next chunk")

    print(f"Processed a total of {all_result_count:,} rows")
    print(f"Saved output to {output_file}")


if __name__ == "__main__":
    input_folder = "./02_raw_datasets"
    output_folder = "./03_processed_datasets"
    cutoff_date = "2025-04-01"

    # Process each question-centered model dataset
    for days in [7]:
        input_file = f"{input_folder}/question_centered_model_{days}d_all_questions.parquet"
        output_file = f"{output_folder}/question_centered_model_{days}d_processed.parquet"

        process_question_dataset(
            input_file=input_file,
            output_file=output_file,
            chunk_size=250000,
            cutoff_date=cutoff_date
        )