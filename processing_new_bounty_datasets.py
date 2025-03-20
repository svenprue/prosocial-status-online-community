import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_bounty_dataset(input_file: str, output_file: str) -> None:
    """
    Process the user_answers_bounty_dataset.parquet file to calculate metrics for each user.
    High-performance implementation that uses binary search and pre-computed arrays.
    """
    print(f"\n=== Processing {input_file} ===")

    # Load the DataFrame
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows.")

    # Ensure the timestamp is in datetime format and convert to epoch seconds
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp_seconds"] = df["timestamp"].view(np.int64) // 10 ** 9  # Convert to seconds since epoch

    # Create numeric flags from event type
    df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
    df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
    df["answer"] = df["event"].map({"Answer": 1, "History_Answer": 1}).fillna(0).astype(np.int8)

    # Separate history and non-history data
    history_df = df[df["is_history"] == 1].copy()
    non_history_df = df[df["is_history"] == 0].copy()

    print(f"History rows: {len(history_df):,}")
    print(f"Non-history rows: {len(non_history_df):,}")

    # Pre-calculate window sizes in seconds
    seconds_30d = 30 * 24 * 60 * 60
    seconds_14d = 14 * 24 * 60 * 60
    seconds_7d = 7 * 24 * 60 * 60
    seconds_3d = 3 * 24 * 60 * 60

    # Build user history arrays with cumulative metrics
    print("Building optimized user history arrays...")
    user_data = {}

    for user_id, user_history in tqdm(history_df.groupby("user_id"), desc="Preprocessing users"):
        # Sort user history by timestamp
        user_history = user_history.sort_values("timestamp_seconds")

        # Get timestamp array
        timestamps = user_history["timestamp_seconds"].values

        # Compute cumulative metrics
        cum_questions = np.cumsum(user_history["questionAsked"].values)
        cum_accepted = np.cumsum(user_history["acceptedAnswer"].values)
        cum_answers = np.cumsum(user_history["answer"].values)

        # Store as numpy arrays for fast access
        user_data[user_id] = {
            "timestamps": timestamps,
            "cum_questions": cum_questions,
            "cum_accepted": cum_accepted,
            "cum_answers": cum_answers
        }

    # Process non-history rows by user for better cache efficiency
    print("Processing non-history rows...")
    results = []

    # Group non-history data by user_id and sort by timestamp
    user_groups = non_history_df.groupby("user_id")

    for user_id, group in tqdm(user_groups, desc="Processing users"):
        # Sort by timestamp
        group = group.sort_values("timestamp_seconds")

        # If user has no history, add zero-value rows
        if user_id not in user_data:
            for _, row in group.iterrows():
                result = {
                    "event_id": row["event_id"],
                    "user_id": user_id,
                    "timestamp": row["timestamp"],
                    "event": row["event"],
                    "answer_id": row.get("answer_id", None),
                    "question_id": row.get("question_id", None),
                    "is_bounty": row.get("is_bounty", 0),
                    "bounty_amount": row.get("bounty_amount", 0),
                    "answer_sequence": row.get("answer_sequence", None),
                    "numQuestionsAskedAT": 0,
                    "numHelpReceivedAT": 0,
                    "numHelpProvidedAT": 0,
                    "numHelpProvidedEver": 0,
                    "numQuestionsAsked30D": 0,
                    "numHelpReceived30D": 0,
                    "numHelpProvided30D": 0,
                    "numQuestionsAsked14D": 0,
                    "numHelpReceived14D": 0,
                    "numHelpProvided14D": 0,
                    "numQuestionsAsked7D": 0,
                    "numHelpReceived7D": 0,
                    "numHelpProvided7D": 0,
                    "numQuestionsAsked3D": 0,
                    "numHelpReceived3D": 0,
                    "numHelpProvided3D": 0
                }
                results.append(result)
            continue

        # Get user's history data
        user_history = user_data[user_id]
        hist_timestamps = user_history["timestamps"]
        cum_questions = user_history["cum_questions"]
        cum_accepted = user_history["cum_accepted"]
        cum_answers = user_history["cum_answers"]

        # Process each row for this user
        for _, row in group.iterrows():
            target_time = row["timestamp_seconds"]

            # Find the index of the last event STRICTLY BEFORE this timestamp
            idx_at = np.searchsorted(hist_timestamps, target_time, side='left') - 1

            # Skip if no history data before this time
            if idx_at < 0:
                result = {
                    "event_id": row["event_id"],
                    "user_id": user_id,
                    "timestamp": row["timestamp"],
                    "event": row["event"],
                    "answer_id": row.get("answer_id", None),
                    "question_id": row.get("question_id", None),
                    "is_bounty": row.get("is_bounty", 0),
                    "bounty_amount": row.get("bounty_amount", 0),
                    "answer_sequence": row.get("answer_sequence", None),
                    "numQuestionsAskedAT": 0,
                    "numHelpReceivedAT": 0,
                    "numHelpProvidedAT": 0,
                    "numHelpProvidedEver": 0,
                    "numQuestionsAsked30D": 0,
                    "numHelpReceived30D": 0,
                    "numHelpProvided30D": 0,
                    "numQuestionsAsked14D": 0,
                    "numHelpReceived14D": 0,
                    "numHelpProvided14D": 0,
                    "numQuestionsAsked7D": 0,
                    "numHelpReceived7D": 0,
                    "numHelpProvided7D": 0,
                    "numQuestionsAsked3D": 0,
                    "numHelpReceived3D": 0,
                    "numHelpProvided3D": 0
                }
                results.append(result)
                continue

            # Calculate time window cutoffs
            cutoff_30d = target_time - seconds_30d
            cutoff_14d = target_time - seconds_14d
            cutoff_7d = target_time - seconds_7d
            cutoff_3d = target_time - seconds_3d

            # Find indices for time window boundaries using binary search
            idx_30d = np.searchsorted(hist_timestamps, cutoff_30d, side='left') - 1
            idx_14d = np.searchsorted(hist_timestamps, cutoff_14d, side='left') - 1
            idx_7d = np.searchsorted(hist_timestamps, cutoff_7d, side='left') - 1
            idx_3d = np.searchsorted(hist_timestamps, cutoff_3d, side='left') - 1

            # Get all-time metrics
            q_at = cum_questions[idx_at]
            a_at = cum_accepted[idx_at]
            ans_at = cum_answers[idx_at]

            # Calculate metrics for each time window
            # If window index is below 0, use 0 as the starting count
            q_30d = q_at - (cum_questions[idx_30d] if idx_30d >= 0 else 0)
            a_30d = a_at - (cum_accepted[idx_30d] if idx_30d >= 0 else 0)
            ans_30d = ans_at - (cum_answers[idx_30d] if idx_30d >= 0 else 0)

            q_14d = q_at - (cum_questions[idx_14d] if idx_14d >= 0 else 0)
            a_14d = a_at - (cum_accepted[idx_14d] if idx_14d >= 0 else 0)
            ans_14d = ans_at - (cum_answers[idx_14d] if idx_14d >= 0 else 0)

            q_7d = q_at - (cum_questions[idx_7d] if idx_7d >= 0 else 0)
            a_7d = a_at - (cum_accepted[idx_7d] if idx_7d >= 0 else 0)
            ans_7d = ans_at - (cum_answers[idx_7d] if idx_7d >= 0 else 0)

            q_3d = q_at - (cum_questions[idx_3d] if idx_3d >= 0 else 0)
            a_3d = a_at - (cum_accepted[idx_3d] if idx_3d >= 0 else 0)
            ans_3d = ans_at - (cum_answers[idx_3d] if idx_3d >= 0 else 0)

            # Create result
            result = {
                "event_id": row["event_id"],
                "user_id": user_id,
                "timestamp": row["timestamp"],
                "event": row["event"],
                "answer_id": row.get("answer_id", None),
                "question_id": row.get("question_id", None),
                "is_bounty": row.get("is_bounty", 0),
                "bounty_amount": row.get("bounty_amount", 0),
                "answer_sequence": row.get("answer_sequence", None),

                "numQuestionsAskedAT": q_at,
                "numHelpReceivedAT": a_at,
                "numHelpProvidedAT": ans_at,
                "numHelpProvidedEver": 1 if ans_at > 0 else 0,

                "numQuestionsAsked30D": q_30d,
                "numHelpReceived30D": a_30d,
                "numHelpProvided30D": ans_30d,

                "numQuestionsAsked14D": q_14d,
                "numHelpReceived14D": a_14d,
                "numHelpProvided14D": ans_14d,

                "numQuestionsAsked7D": q_7d,
                "numHelpReceived7D": a_7d,
                "numHelpProvided7D": ans_7d,

                "numQuestionsAsked3D": q_3d,
                "numHelpReceived3D": a_3d,
                "numHelpProvided3D": ans_3d
            }

            results.append(result)

    # Create DataFrame from results
    output_df = pd.DataFrame(results)

    # Add additional derived metrics
    output_df["receivedHelpEver"] = (output_df["numHelpReceivedAT"] > 0).astype(int)
    output_df["month"] = output_df["timestamp"].dt.month
    output_df["year"] = output_df["timestamp"].dt.year

    # Save to output file
    output_df.to_parquet(output_file, index=False)

    print(f"Processed {len(output_df)} rows")
    print(f"Saved output to {output_file}")


if __name__ == "__main__":
    input_file = "02_raw_datasets/user_answers_bounty_dataset_sampled_500K.parquet"
    output_file = "03_processed_datasets/user_answers_bounty_processed_sampled_500K.parquet"

    process_bounty_dataset(
        input_file=input_file,
        output_file=output_file
    )