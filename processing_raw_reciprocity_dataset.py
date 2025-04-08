import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from numba import njit


@njit
def calculate_metrics_numba(timestamps, cum_question, cum_accepted, cum_answer, cum_answers_received,
                            cum_accepted_answers_received, cum_accepted_votes_received,
                            target_time, days_30, days_14, days_7, days_3):
    """
    Numba-optimized function to calculate metrics at different time windows
    Returns metrics for all time windows in a single pass
    """
    # Find index for the target time (right before it)
    idx_at = np.searchsorted(timestamps, target_time, side='left') - 1

    # Find indices for different time windows
    idx_30d = np.searchsorted(timestamps, days_30, side='left') - 1
    idx_14d = np.searchsorted(timestamps, days_14, side='left') - 1
    idx_7d = np.searchsorted(timestamps, days_7, side='left') - 1
    idx_3d = np.searchsorted(timestamps, days_3, side='left') - 1

    # All-time metrics
    q_at = cum_question[idx_at] if idx_at >= 0 else 0
    a_at = cum_accepted[idx_at] if idx_at >= 0 else 0
    ans_at = cum_answer[idx_at] if idx_at >= 0 else 0
    ans_rec_at = cum_answers_received[idx_at] if idx_at >= 0 else 0
    acc_ans_rec_at = cum_accepted_answers_received[idx_at] if idx_at >= 0 else 0
    acc_vote_rec_at = cum_accepted_votes_received[idx_at] if idx_at >= 0 else 0
    help_provided_ever = 1 if ans_at > 0 else 0

    # 30-day metrics (calculate as difference between all-time and before the window)
    q_30d_before = cum_question[idx_30d] if idx_30d >= 0 else 0
    a_30d_before = cum_accepted[idx_30d] if idx_30d >= 0 else 0
    ans_30d_before = cum_answer[idx_30d] if idx_30d >= 0 else 0
    ans_rec_30d_before = cum_answers_received[idx_30d] if idx_30d >= 0 else 0
    acc_ans_rec_30d_before = cum_accepted_answers_received[idx_30d] if idx_30d >= 0 else 0
    acc_vote_rec_30d_before = cum_accepted_votes_received[idx_30d] if idx_30d >= 0 else 0

    q_30d = q_at - q_30d_before
    a_30d = a_at - a_30d_before
    ans_30d = ans_at - ans_30d_before
    ans_rec_30d = ans_rec_at - ans_rec_30d_before
    acc_ans_rec_30d = acc_ans_rec_at - acc_ans_rec_30d_before
    acc_vote_rec_30d = acc_vote_rec_at - acc_vote_rec_30d_before

    # 14-day metrics
    q_14d_before = cum_question[idx_14d] if idx_14d >= 0 else 0
    a_14d_before = cum_accepted[idx_14d] if idx_14d >= 0 else 0
    ans_14d_before = cum_answer[idx_14d] if idx_14d >= 0 else 0
    ans_rec_14d_before = cum_answers_received[idx_14d] if idx_14d >= 0 else 0
    acc_ans_rec_14d_before = cum_accepted_answers_received[idx_14d] if idx_14d >= 0 else 0
    acc_vote_rec_14d_before = cum_accepted_votes_received[idx_14d] if idx_14d >= 0 else 0

    q_14d = q_at - q_14d_before
    a_14d = a_at - a_14d_before
    ans_14d = ans_at - ans_14d_before
    ans_rec_14d = ans_rec_at - ans_rec_14d_before
    acc_ans_rec_14d = acc_ans_rec_at - acc_ans_rec_14d_before
    acc_vote_rec_14d = acc_vote_rec_at - acc_vote_rec_14d_before

    # 7-day metrics
    q_7d_before = cum_question[idx_7d] if idx_7d >= 0 else 0
    a_7d_before = cum_accepted[idx_7d] if idx_7d >= 0 else 0
    ans_7d_before = cum_answer[idx_7d] if idx_7d >= 0 else 0
    ans_rec_7d_before = cum_answers_received[idx_7d] if idx_7d >= 0 else 0
    acc_ans_rec_7d_before = cum_accepted_answers_received[idx_7d] if idx_7d >= 0 else 0
    acc_vote_rec_7d_before = cum_accepted_votes_received[idx_7d] if idx_7d >= 0 else 0

    q_7d = q_at - q_7d_before
    a_7d = a_at - a_7d_before
    ans_7d = ans_at - ans_7d_before
    ans_rec_7d = ans_rec_at - ans_rec_7d_before
    acc_ans_rec_7d = acc_ans_rec_at - acc_ans_rec_7d_before
    acc_vote_rec_7d = acc_vote_rec_at - acc_vote_rec_7d_before

    # 3-day metrics
    q_3d_before = cum_question[idx_3d] if idx_3d >= 0 else 0
    a_3d_before = cum_accepted[idx_3d] if idx_3d >= 0 else 0
    ans_3d_before = cum_answer[idx_3d] if idx_3d >= 0 else 0
    ans_rec_3d_before = cum_answers_received[idx_3d] if idx_3d >= 0 else 0
    acc_ans_rec_3d_before = cum_accepted_answers_received[idx_3d] if idx_3d >= 0 else 0
    acc_vote_rec_3d_before = cum_accepted_votes_received[idx_3d] if idx_3d >= 0 else 0

    q_3d = q_at - q_3d_before
    a_3d = a_at - a_3d_before
    ans_3d = ans_at - ans_3d_before
    ans_rec_3d = ans_rec_at - ans_rec_3d_before
    acc_ans_rec_3d = acc_ans_rec_at - acc_ans_rec_3d_before
    acc_vote_rec_3d = acc_vote_rec_at - acc_vote_rec_3d_before

    return (q_at, a_at, ans_at, help_provided_ever, ans_rec_at, acc_ans_rec_at, acc_vote_rec_at,
            q_30d, a_30d, ans_30d, ans_rec_30d, acc_ans_rec_30d, acc_vote_rec_30d,
            q_14d, a_14d, ans_14d, ans_rec_14d, acc_ans_rec_14d, acc_vote_rec_14d,
            q_7d, a_7d, ans_7d, ans_rec_7d, acc_ans_rec_7d, acc_vote_rec_7d,
            q_3d, a_3d, ans_3d, ans_rec_3d, acc_ans_rec_3d, acc_vote_rec_3d)


def process_bounty_dataset(input_file: str, output_file: str, chunk_size: int = 100000) -> None:
    """
    Process the user_answers_bounty_dataset.parquet file to calculate metrics for each user.
    Uses a chunking approach with Numba acceleration for fast processing.

    Args:
        input_file: Path to the input parquet file
        output_file: Path to save the processed data
        chunk_size: Number of users to process in each chunk
    """
    print(f"\n=== Processing {input_file} ===")

    # First, get unique user IDs from the file
    print("Reading unique user IDs...")
    user_ids = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()
    print(f"Found {len(user_ids):,} unique users")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if output file exists and remove if it does
    if os.path.exists(output_file):
        os.remove(output_file)

    # Calculate number of chunks
    num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
    all_result_count = 0

    # Process users in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(user_ids))
        chunk_user_ids = user_ids[start_idx:end_idx]

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} with {len(chunk_user_ids):,} users")

        # Read only data for current chunk of users
        df = pd.read_parquet(input_file, filters=[('user_id', 'in', list(chunk_user_ids))])
        print(f"Loaded {len(df):,} rows for this chunk.")

        # Ensure the timestamp is in datetime format
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Create numeric flags from event type with minimal memory usage
        df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["answer"] = df["event"].map({"Answer": 1, "History_Answer": 1}).fillna(0).astype(np.int8)
        df["answerReceived"] = df["event"].map({"AnswerReceived": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswerReceived"] = df["event"].map({"AcceptedAnswerReceived": 1}).fillna(0).astype(np.int8)
        df["acceptedVoteReceived"] = df["event"].map({"AcceptedVoteReceived": 1}).fillna(0).astype(np.int8)

        # Create vote timestamp for AcceptedVoteReceived (adds 1 day)
        df["vote_timestamp"] = df["timestamp"].copy()
        mask = df["event"] == "AcceptedVoteReceived"
        df.loc[mask, "vote_timestamp"] = df.loc[mask, "timestamp"] + pd.Timedelta(days=1)

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

        # Build user history data optimized for fast lookup
        print("Building user history cache...")
        user_histories = {}

        for user_id, group in tqdm(history_df.groupby("user_id"), desc="Preprocessing users"):
            # Sort by timestamp for cumulative calculations
            group = group.sort_values("timestamp")

            # Convert timestamps to epoch seconds for Numba compatibility
            timestamps = group["timestamp"].astype(np.int64).values // 10 ** 9  # seconds since epoch

            # For vote timestamps, handle specially for AcceptedVoteReceived events
            vote_timestamps = group["vote_timestamp"].astype(np.int64).values // 10 ** 9

            # Calculate cumulative sums for all metrics (vectorized operation)
            user_histories[user_id] = {
                "timestamps": timestamps,
                "cum_question": np.cumsum(group["questionAsked"].values),
                "cum_accepted": np.cumsum(group["acceptedAnswer"].values),
                "cum_answer": np.cumsum(group["answer"].values),
                "cum_answers_received": np.cumsum(group["answerReceived"].values),
                "cum_accepted_answers_received": np.cumsum(group["acceptedAnswerReceived"].values),
                "cum_accepted_votes_received": np.cumsum(group["acceptedVoteReceived"].values),
                "vote_timestamps": vote_timestamps
            }

        # Free memory
        del history_df
        gc.collect()

        # Process non-history rows
        print("Processing non-history rows...")
        results = []

        # Process rows by user_id for better cache efficiency
        non_history_df = non_history_df.sort_values(["user_id", "timestamp"])

        # Process each row
        for _, row in tqdm(non_history_df.iterrows(), total=len(non_history_df), desc="Processing rows"):
            user_id = row["user_id"]
            target_time = row["timestamp"]

            # Convert target_time to epoch seconds for Numba
            target_time_sec = target_time.timestamp()

            # Calculate cutoff times for all time windows at once
            cutoff_30d_sec = target_time_sec - (30 * 86400)  # 30 days in seconds
            cutoff_14d_sec = target_time_sec - (14 * 86400)  # 14 days in seconds
            cutoff_7d_sec = target_time_sec - (7 * 86400)  # 7 days in seconds
            cutoff_3d_sec = target_time_sec - (3 * 86400)  # 3 days in seconds

            # If user not in history, use zeros for all metrics
            if user_id not in user_histories:
                metrics = (0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0)
            else:
                # Use Numba function to calculate all metrics at once
                user_data = user_histories[user_id]
                metrics = calculate_metrics_numba(
                    user_data["timestamps"],
                    user_data["cum_question"],
                    user_data["cum_accepted"],
                    user_data["cum_answer"],
                    user_data["cum_answers_received"],
                    user_data["cum_accepted_answers_received"],
                    user_data["cum_accepted_votes_received"],
                    target_time_sec, cutoff_30d_sec, cutoff_14d_sec, cutoff_7d_sec, cutoff_3d_sec
                )

            # Unpack metrics (much faster than creating a dictionary each time)
            (q_at, a_at, ans_at, help_provided_ever, ans_rec_at, acc_ans_rec_at, acc_vote_rec_at,
             q_30d, a_30d, ans_30d, ans_rec_30d, acc_ans_rec_30d, acc_vote_rec_30d,
             q_14d, a_14d, ans_14d, ans_rec_14d, acc_ans_rec_14d, acc_vote_rec_14d,
             q_7d, a_7d, ans_7d, ans_rec_7d, acc_ans_rec_7d, acc_vote_rec_7d,
             q_3d, a_3d, ans_3d, ans_rec_3d, acc_ans_rec_3d, acc_vote_rec_3d) = metrics

            # Create result dictionary
            result = {
                "event_id": row["event_id"],
                "user_id": user_id,
                "timestamp": target_time,
                "event": row["event"],
                "answer_id": row.get("answer_id", None),
                "question_id": row.get("question_id", None),
                "is_bounty": row.get("is_bounty", 0),
                "bounty_amount": row.get("bounty_amount", 0),
                "answer_sequence": row.get("answer_sequence", None),

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

        # Free memory
        del user_histories
        gc.collect()

        # Create DataFrame from results
        chunk_output_df = pd.DataFrame(results)

        # Add additional derived metrics (vectorized operations)
        chunk_output_df["receivedHelpEver"] = (chunk_output_df["numHelpReceivedAT"] > 0).astype(int)
        chunk_output_df["receivedAnswerEver"] = (chunk_output_df["numAnswersReceivedAT"] > 0).astype(int)
        chunk_output_df["receivedAcceptedAnswerEver"] = (chunk_output_df["numAcceptedAnswersReceivedAT"] > 0).astype(
            int)
        chunk_output_df["receivedAcceptedVoteEver"] = (chunk_output_df["numAcceptedVotesReceivedAT"] > 0).astype(int)
        chunk_output_df["month"] = chunk_output_df["timestamp"].dt.month
        chunk_output_df["year"] = chunk_output_df["timestamp"].dt.year

        all_result_count += len(chunk_output_df)

        # Append to output file
        if chunk_idx == 0:
            # First chunk, create the file
            chunk_output_df.to_parquet(output_file, index=False)
        else:
            # Subsequent chunks, append to existing file
            chunk_output_df.to_parquet(
                output_file,
                index=False,
                append=True
            )

        # Free memory
        del chunk_output_df
        del non_history_df
        del results
        gc.collect()

    print(f"Processed a total of {all_result_count:,} rows")
    print(f"Saved output to {output_file}")


if __name__ == "__main__":
    input_file = "02_raw_datasets/user_answers_bounty_dataset.parquet"
    output_file = "03_processed_datasets/user_answers_bounty_processed.parquet"

    process_bounty_dataset(
        input_file=input_file,
        output_file=output_file
    )