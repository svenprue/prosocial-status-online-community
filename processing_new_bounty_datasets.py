import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_bounty_dataset(input_file: str, output_file: str) -> None:
    """
    Process the user_answers_bounty_dataset.parquet file to calculate metrics for each user.
    Uses a straightforward approach with optimizations for speed.
    """
    print(f"\n=== Processing {input_file} ===")

    # Load the DataFrame
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows.")

    # Ensure the timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Create numeric flags from event type
    df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
    df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
    df["answer"] = df["event"].map({"Answer": 1, "History_Answer": 1}).fillna(0).astype(np.int8)

    # Separate history and non-history data
    history_df = df[df["is_history"] == 1].copy()
    non_history_df = df[df["is_history"] == 0].copy()

    print(f"History rows: {len(history_df):,}")
    print(f"Non-history rows: {len(non_history_df):,}")

    # Create a dictionary of user histories for faster lookup
    print("Building user history cache...")
    user_histories = {}
    for user_id, user_data in tqdm(history_df.groupby("user_id"), desc="Preprocessing users"):
        # Sort by timestamp
        user_data = user_data.sort_values("timestamp")
        user_histories[user_id] = user_data

    # Process non-history rows
    print("Processing non-history rows...")
    results = []

    # Process rows by user_id for better cache efficiency
    non_history_df = non_history_df.sort_values(["user_id", "timestamp"])

    current_user_id = None
    user_history_df = None

    for _, row in tqdm(non_history_df.iterrows(), total=len(non_history_df), desc="Processing rows"):
        user_id = row["user_id"]
        target_time = row["timestamp"]

        # Only fetch user history once for each user (caching optimization)
        if user_id != current_user_id:
            current_user_id = user_id
            if user_id in user_histories:
                user_history_df = user_histories[user_id]
            else:
                user_history_df = pd.DataFrame(columns=history_df.columns)

        # Get history up to target time (strict < comparison)
        user_history_before = user_history_df[user_history_df["timestamp"] < target_time]

        # Calculate all-time metrics
        questions_asked_at = user_history_before["questionAsked"].sum()
        help_received_at = user_history_before["acceptedAnswer"].sum()
        help_provided_at = user_history_before["answer"].sum()
        help_provided_ever = 1 if help_provided_at > 0 else 0

        # Calculate time window metrics
        cutoff_30d = target_time - pd.Timedelta(days=30)
        window_30d = user_history_before[user_history_before["timestamp"] >= cutoff_30d]
        questions_asked_30d = window_30d["questionAsked"].sum()
        help_received_30d = window_30d["acceptedAnswer"].sum()
        help_provided_30d = window_30d["answer"].sum()

        cutoff_14d = target_time - pd.Timedelta(days=14)
        window_14d = user_history_before[user_history_before["timestamp"] >= cutoff_14d]
        questions_asked_14d = window_14d["questionAsked"].sum()
        help_received_14d = window_14d["acceptedAnswer"].sum()
        help_provided_14d = window_14d["answer"].sum()

        cutoff_7d = target_time - pd.Timedelta(days=7)
        window_7d = user_history_before[user_history_before["timestamp"] >= cutoff_7d]
        questions_asked_7d = window_7d["questionAsked"].sum()
        help_received_7d = window_7d["acceptedAnswer"].sum()
        help_provided_7d = window_7d["answer"].sum()

        cutoff_3d = target_time - pd.Timedelta(days=3)
        window_3d = user_history_before[user_history_before["timestamp"] >= cutoff_3d]
        questions_asked_3d = window_3d["questionAsked"].sum()
        help_received_3d = window_3d["acceptedAnswer"].sum()
        help_provided_3d = window_3d["answer"].sum()

        # Create result
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

            "numQuestionsAskedAT": questions_asked_at,
            "numHelpReceivedAT": help_received_at,
            "numHelpProvidedAT": help_provided_at,
            "numHelpProvidedEver": help_provided_ever,

            "numQuestionsAsked30D": questions_asked_30d,
            "numHelpReceived30D": help_received_30d,
            "numHelpProvided30D": help_provided_30d,

            "numQuestionsAsked14D": questions_asked_14d,
            "numHelpReceived14D": help_received_14d,
            "numHelpProvided14D": help_provided_14d,

            "numQuestionsAsked7D": questions_asked_7d,
            "numHelpReceived7D": help_received_7d,
            "numHelpProvided7D": help_provided_7d,

            "numQuestionsAsked3D": questions_asked_3d,
            "numHelpReceived3D": help_received_3d,
            "numHelpProvided3D": help_provided_3d,
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
    input_file = "02_raw_datasets/user_answers_bounty_dataset_sampled_10k.parquet"
    output_file = "03_processed_datasets/user_answers_bounty_processed_sampled_10k.parquet"

    process_bounty_dataset(
        input_file=input_file,
        output_file=output_file
    )