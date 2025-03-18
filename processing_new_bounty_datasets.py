import os
import glob
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from pathlib import Path


def process_bounty_dataset(input_folder: str, output_folder: str) -> None:
    """
    Process the user_answers_bounty_dataset.parquet file to calculate metrics for each user.
    First builds user metrics from all data, then processes rows with is_history=0.

    Args:
        input_folder: Directory containing the input parquet file
        output_folder: Directory where the processed file will be saved
    """
    # Make output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Path to the bounty dataset
    input_file = os.path.join(input_folder, "user_answers_bounty_dataset.parquet")
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    print(f"\n=== Processing {input_file} ===")

    # ---------------------------------------------------------------------
    # 1. Load the DataFrame and ensure proper datetime conversion
    # ---------------------------------------------------------------------
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows.")

    # Ensure the timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ---------------------------------------------------------------------
    # 2. Create numeric flags from event type
    # ---------------------------------------------------------------------
    df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
    df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
    df["answer"] = df["event"].map({"Answer": 1, "History_Answer": 1}).fillna(0).astype(np.int8)

    # Optimize data types
    df["user_id"] = df["user_id"].astype(np.int32, errors="ignore")
    df["is_history"] = df["is_history"].astype(np.int8, errors="ignore")

    # ---------------------------------------------------------------------
    # 3. Sort by user_id and timestamp
    # ---------------------------------------------------------------------
    df.sort_values(["user_id", "timestamp"], inplace=True)

    # ---------------------------------------------------------------------
    # 4. Define the Numba function for calculating metrics
    # ---------------------------------------------------------------------
    @njit
    def calculate_metrics_numba(timestamps, cum_question, cum_accepted, cum_answer,
                                target_time, days_30, days_14, days_7, days_3):
        # Index for current time (AT) metrics
        idx_at = np.searchsorted(timestamps, target_time, side="left") - 1

        # Indices for different time windows
        idx_30d = np.searchsorted(timestamps, days_30, side="left") - 1
        idx_14d = np.searchsorted(timestamps, days_14, side="left") - 1
        idx_7d = np.searchsorted(timestamps, days_7, side="left") - 1
        idx_3d = np.searchsorted(timestamps, days_3, side="left") - 1

        # Initialize all metrics to 0
        metrics = {
            "AT": [0, 0, 0, 0],  # [q, a, ans, ever]
            "30D": [0, 0, 0],  # [q, a, ans]
            "14D": [0, 0, 0],  # [q, a, ans]
            "7D": [0, 0, 0],  # [q, a, ans]
            "3D": [0, 0, 0]  # [q, a, ans]
        }

        # Calculate AT metrics
        if idx_at >= 0:
            metrics["AT"][0] = cum_question[idx_at]
            metrics["AT"][1] = cum_accepted[idx_at]
            metrics["AT"][2] = cum_answer[idx_at]
            metrics["AT"][3] = 1 if cum_answer[idx_at] > 0 else 0

        # Calculate 30D metrics (counts within the last 30 days)
        if idx_at >= 0:
            q_30d = cum_question[idx_30d] if idx_30d >= 0 else 0
            a_30d = cum_accepted[idx_30d] if idx_30d >= 0 else 0
            ans_30d = cum_answer[idx_30d] if idx_30d >= 0 else 0

            metrics["30D"][0] = metrics["AT"][0] - q_30d
            metrics["30D"][1] = metrics["AT"][1] - a_30d
            metrics["30D"][2] = metrics["AT"][2] - ans_30d

        # Calculate 14D metrics
        if idx_at >= 0:
            q_14d = cum_question[idx_14d] if idx_14d >= 0 else 0
            a_14d = cum_accepted[idx_14d] if idx_14d >= 0 else 0
            ans_14d = cum_answer[idx_14d] if idx_14d >= 0 else 0

            metrics["14D"][0] = metrics["AT"][0] - q_14d
            metrics["14D"][1] = metrics["AT"][1] - a_14d
            metrics["14D"][2] = metrics["AT"][2] - ans_14d

        # Calculate 7D metrics
        if idx_at >= 0:
            q_7d = cum_question[idx_7d] if idx_7d >= 0 else 0
            a_7d = cum_accepted[idx_7d] if idx_7d >= 0 else 0
            ans_7d = cum_answer[idx_7d] if idx_7d >= 0 else 0

            metrics["7D"][0] = metrics["AT"][0] - q_7d
            metrics["7D"][1] = metrics["AT"][1] - a_7d
            metrics["7D"][2] = metrics["AT"][2] - ans_7d

        # Calculate 3D metrics
        if idx_at >= 0:
            q_3d = cum_question[idx_3d] if idx_3d >= 0 else 0
            a_3d = cum_accepted[idx_3d] if idx_3d >= 0 else 0
            ans_3d = cum_answer[idx_3d] if idx_3d >= 0 else 0

            metrics["3D"][0] = metrics["AT"][0] - q_3d
            metrics["3D"][1] = metrics["AT"][1] - a_3d
            metrics["3D"][2] = metrics["AT"][2] - ans_3d

        return (metrics["AT"][0], metrics["AT"][1], metrics["AT"][2], metrics["AT"][3],
                metrics["30D"][0], metrics["30D"][1], metrics["30D"][2],
                metrics["14D"][0], metrics["14D"][1], metrics["14D"][2],
                metrics["7D"][0], metrics["7D"][1], metrics["7D"][2],
                metrics["3D"][0], metrics["3D"][1], metrics["3D"][2])

    # Function to compute metrics for a given user and timestamp
    def compute_user_metrics(user_id: int, target_time: pd.Timestamp, user_data: dict) -> dict:
        """
        For a given user and target_time, compute metrics for different time windows:
        - All time before target_time (AT)
        - Last 30 days before target_time (30D)
        - Last 14 days before target_time (14D)
        - Last 7 days before target_time (7D)
        - Last 3 days before target_time (3D)
        """
        if user_id not in user_data:
            return {
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

        data = user_data[user_id]
        target_sec = target_time.timestamp()

        # Calculate cutoff timestamps for different windows
        days_30_sec = target_sec - (30 * 86400)  # 30 days in seconds
        days_14_sec = target_sec - (14 * 86400)  # 14 days in seconds
        days_7_sec = target_sec - (7 * 86400)  # 7 days in seconds
        days_3_sec = target_sec - (3 * 86400)  # 3 days in seconds

        # Get all metrics using the numba function
        (q_at, a_at, ans_at, ever,
         q_30d, a_30d, ans_30d,
         q_14d, a_14d, ans_14d,
         q_7d, a_7d, ans_7d,
         q_3d, a_3d, ans_3d) = calculate_metrics_numba(
            data["timestamps"],
            data["cum_question"],
            data["cum_accepted"],
            data["cum_answer"],
            target_sec, days_30_sec, days_14_sec, days_7_sec, days_3_sec
        )

        return {
            "numQuestionsAskedAT": q_at,
            "numHelpReceivedAT": a_at,
            "numHelpProvidedAT": ans_at,
            "numHelpProvidedEver": ever,
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

    # ---------------------------------------------------------------------
    # 5. Filter for non-history rows (is_history=0)
    # ---------------------------------------------------------------------
    print("Filtering for non-history events...")
    non_history_df = df[df["is_history"] == 0].copy()
    print(f"Kept {len(non_history_df):,} non-history rows.")

    # ---------------------------------------------------------------------
    # 6. Build user cumulative data from ALL rows (including history)
    # ---------------------------------------------------------------------
    print("Building cumulative user data...")
    user_groups = df.groupby("user_id", sort=False)
    total_users = df["user_id"].nunique()
    user_data = {}

    for user_id, group in tqdm(user_groups, total=total_users, desc="Building cumulative data"):
        ts = group["timestamp"].astype(np.int64).values / 1e9  # seconds since epoch
        q_flag = group["questionAsked"].values
        a_flag = group["acceptedAnswer"].values
        ans_flag = group["answer"].values

        user_data[user_id] = {
            "timestamps": ts,
            "cum_question": np.cumsum(q_flag),
            "cum_accepted": np.cumsum(a_flag),
            "cum_answer": np.cumsum(ans_flag),
        }

    # ---------------------------------------------------------------------
    # 7. Process each non-history row and calculate metrics
    # ---------------------------------------------------------------------
    print("Processing non-history rows...")

    # Initialize empty lists for results
    row_metrics = []

    # Group non-history rows by user_id
    user_non_history = non_history_df.groupby("user_id")

    for user_id, user_rows in tqdm(user_non_history, total=len(user_non_history), desc="Processing users"):
        # Process each row for this user
        for _, row in user_rows.iterrows():
            timestamp = row["timestamp"]
            event_id = row["event_id"]

            # Calculate metrics for this user at this timestamp
            metrics = compute_user_metrics(user_id, timestamp, user_data)

            # Create a result record with original row data and calculated metrics
            result = {
                "event_id": event_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "event": row["event"],
                "answer_id": row.get("answer_id", None),
                "question_id": row.get("question_id", None),
                "is_bounty": row.get("is_bounty", 0),
                "bounty_amount": row.get("bounty_amount", 0),
                "answer_sequence": row.get("answer_sequence", None),
                **metrics
            }

            row_metrics.append(result)

    # Create DataFrame from all processed rows
    metrics_df = pd.DataFrame(row_metrics)

    # ---------------------------------------------------------------------
    # 8. Add additional derived metrics
    # ---------------------------------------------------------------------
    metrics_df["receivedHelpEver"] = (metrics_df["numHelpReceivedAT"] > 0).astype(int)
    metrics_df["month"] = metrics_df["timestamp"].dt.month
    metrics_df["year"] = metrics_df["timestamp"].dt.year

    # ---------------------------------------------------------------------
    # 9. Export processed dataset
    # ---------------------------------------------------------------------
    output_file = os.path.join(output_folder, "user_answers_bounty_processed.parquet")
    metrics_df.to_parquet(output_file, index=False)

    print(f"Processed {len(metrics_df)} rows")
    print(f"Saved output to {output_file}")

if __name__ == "__main__":
    # Use Path for more portable paths
    base_dir = Path(".")
    input_data_folder = base_dir / "02_raw_datasets"
    output_data_folder = base_dir / "03_processed_datasets"

    process_bounty_dataset(
        input_folder=str(input_data_folder),
        output_folder=str(output_data_folder)
    )