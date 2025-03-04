import os
import glob
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit


def process_parquet_files(input_folder: str, output_folder: str, cutoff_date: str = "2025-01-03") -> None:
    """
    Processes each parquet file in the input_folder by:
      1. Calculating per-user cumulative metrics at different time windows before Phase_One_Start:
         - All time (AT)
         - Last 30 days (30D)
         - Last 14 days (14D)
         - Last 7 days (7D)
         - Last 3 days (3D)
      2. Propagating these metrics (for the same event_id) to all rows.
      3. Dropping all rows where is_history == 1.
      4. Determining the phase (1 or 2) for each row based on the event timestamps.
      5. Adding additional time-based columns (month, year) and help indicators.
      6. Grouping and aggregating data by event_id and phase.
      7. Demeaning the numHelped and hasHelped columns by user.
      8. Finally, removing any aggregated rows whose phase_two_end is after the cutoff date.
      9. Saving the final aggregated data as a parquet file with suffix "_processed.parquet".

    Expected columns in each file (in order):
      event_id, user_id, timestamp, event, question_id,
      phase_one_start, phase_two_end, event_history, is_history, response_time
    """
    # Convert cutoff_date string to Timestamp.
    cutoff = pd.to_datetime(cutoff_date)

    parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in '{input_folder}'.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Numba-optimized function to calculate cumulative metrics at different time points.
    # -----------------------------------------------------------------------------
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

    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        print(f"\n=== Processing {file_name} ===")

        # ---------------------------------------------------------------------
        # 1. Load the DataFrame and ensure proper datetime conversion.
        # ---------------------------------------------------------------------
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df):,} rows from {file_name}.")

        # Ensure the timestamp, phase_two_end, and phase_one_start columns are in datetime format.
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["phase_two_end"] = pd.to_datetime(df["phase_two_end"], errors="coerce")
        df["phase_one_start"] = pd.to_datetime(df["phase_one_start"], errors="coerce")

        # ---------------------------------------------------------------------
        # 2. Keep only the relevant columns.
        # ---------------------------------------------------------------------
        keep_cols = [
            "event_id",
            "user_id",
            "timestamp",
            "event",
            "question_id",
            "phase_one_start",
            "phase_two_end",
            "event_history",
            "is_history",
            "response_time",
        ]
        df = df[[col for col in keep_cols if col in df.columns]]

        # ---------------------------------------------------------------------
        # 3. Create numeric flags from event_history.
        # ---------------------------------------------------------------------
        df["questionAsked"] = df["event_history"].map({"Question": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswer"] = df["event_history"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["answer"] = df["event_history"].map({"Answer": 1}).fillna(0).astype(np.int8)

        # Optimize data types.
        df["user_id"] = df["user_id"].astype(np.int32, errors="ignore")
        # The 'event' column remains a string.
        df["is_history"] = df["is_history"].astype(np.int8, errors="ignore")

        # ---------------------------------------------------------------------
        # 4. Sort by user_id and timestamp.
        # ---------------------------------------------------------------------
        df.sort_values(["user_id", "timestamp"], inplace=True)

        # ---------------------------------------------------------------------
        # 5. Precompute per-user cumulative data.
        # ---------------------------------------------------------------------
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
        # 6. Compute metrics at the Phase_One_Start event.
        # ---------------------------------------------------------------------
        mask_phase1 = (df["event"] == "Phase_One_Start") & df["event_id"].notna()
        df_phase1 = df.loc[mask_phase1, ["user_id", "event_id", "timestamp"]].copy()

        def process_row(row):
            metrics = compute_user_metrics(row.user_id, row.timestamp, user_data)
            return {
                "user_id": row.user_id,
                "event_id": row.event_id,
                **metrics
            }

        results = [process_row(r) for r in tqdm(df_phase1.itertuples(index=False),
                                                total=len(df_phase1),
                                                desc="Calculating metrics")]
        df_metrics = pd.DataFrame(results)
        if not df_metrics.empty:
            df = df.merge(df_metrics, on=["user_id", "event_id"], how="left")

        del user_groups, user_data, df_phase1, results
        gc.collect()

        # ---------------------------------------------------------------------
        # 7. Propagate metrics from Phase_One_Start to all rows (per event_id).
        # ---------------------------------------------------------------------
        cols_to_propagate = [
            "numHelpProvidedEver",
            "numHelpProvidedAT",
            "numHelpReceivedAT",
            "numQuestionsAskedAT",
            "numQuestionsAsked30D",
            "numHelpReceived30D",
            "numHelpProvided30D",
            "numQuestionsAsked14D",
            "numHelpReceived14D",
            "numHelpProvided14D",
            "numQuestionsAsked7D",
            "numHelpReceived7D",
            "numHelpProvided7D",
            "numQuestionsAsked3D",
            "numHelpReceived3D",
            "numHelpProvided3D"
        ]
        existing_cols = [c for c in cols_to_propagate if c in df.columns]
        if existing_cols:
            df_phase1_group = (
                df[df["event"] == "Phase_One_Start"]
                .groupby("event_id")
                .first()
                .reset_index()
            )
            df_phase1_group = df_phase1_group[["event_id"] + existing_cols]
            df = df.merge(df_phase1_group, on="event_id", suffixes=("", "_from_phase1"), how="left")
            for col in existing_cols:
                df[col] = df[col + "_from_phase1"]
                df.drop(columns=[col + "_from_phase1"], inplace=True)

        # ***** Drop all rows where is_history == 1 *****
        df = df[df["is_history"] == 0]

        # ---------------------------------------------------------------------
        # 8. Determine phase (1 or 2) for each row using event timestamps.
        #     For each event_id, define:
        #       - start_time: maximum timestamp (as int) where event == "Phase_One_Start"
        #       - end_time: maximum timestamp (as int) where event == "Phase_Two_End"
        #     Then assign phase = 1 if the row's timestamp is in [start_time, end_time), otherwise phase = 2.
        # ---------------------------------------------------------------------
        df["TimestampInt"] = pd.to_numeric(df["timestamp"], errors="coerce", downcast="integer")
        start_times = (
            df.loc[df["event"] == "Phase_One_Start"]
            .groupby("event_id")["TimestampInt"]
            .max()
        )
        end_times = (
            df.loc[df["event"] == "Phase_Two_End"]
            .groupby("event_id")["TimestampInt"]
            .max()
        )
        df["start_time"] = df["event_id"].map(start_times)
        df["end_time"] = df["event_id"].map(end_times)
        df["end_time"] = df["end_time"].fillna(df["start_time"])

        df["isPhase"] = np.where(
            (df["TimestampInt"] >= df["start_time"]) & (df["TimestampInt"] < df["end_time"]),
            0,
            1,
        )
        df["phase"] = df["isPhase"].replace({0: 1, 1: 2})

        # ---------------------------------------------------------------------
        # 9. Final processing: add month/year and help indicators.
        # ---------------------------------------------------------------------
        def process_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
            df_in["month"] = df_in["timestamp"].dt.month.astype("Int16")
            df_in["year"] = df_in["timestamp"].dt.year.astype("Int16")
            df_in["receivedHelpEver"] = (df_in["numHelpReceivedAT"] > 0).astype(int)
            if "answer" in df_in.columns:
                df_in["numHelped"] = df_in["answer"].astype(int)
            else:
                df_in["numHelped"] = 0
            if "acceptedAnswer" in df_in.columns:
                df_in["hasAcceptedAnswer"] = (df_in["acceptedAnswer"] > 0).astype(int)
            else:
                df_in["hasAcceptedAnswer"] = 0
            df_in["hasAnswer"] = (df_in["numHelped"] > 0).astype(int)
            return df_in

        df = process_dataframe(df)

        # ---------------------------------------------------------------------
        # 10. Group and aggregate data by event_id and phase.
        # ---------------------------------------------------------------------
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
            "response_time": "first",
            "numHelped": "sum",
            "hasAcceptedAnswer": "first",
            "hasAnswer": "first",
            "numHelpProvidedEver": "first",
            "receivedHelpEver": "first",
            "year": "first",
            "month": "first",
            "numHelpProvidedAT": "first",
            "numHelpReceivedAT": "first",
            "numQuestionsAskedAT": "first",
            "numQuestionsAsked30D": "first",
            "numHelpReceived30D": "first",
            "numHelpProvided30D": "first",
            "numQuestionsAsked14D": "first",
            "numHelpReceived14D": "first",
            "numHelpProvided14D": "first",
            "numQuestionsAsked7D": "first",
            "numHelpReceived7D": "first",
            "numHelpProvided7D": "first",
            "numQuestionsAsked3D": "first",
            "numHelpReceived3D": "first",
            "numHelpProvided3D": "first"
        }

        # Filter to include only existing columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()

        # ---------------------------------------------------------------------
        # 11. Remove aggregated rows with phase_two_end after the cutoff date.
        # ---------------------------------------------------------------------
        initial_agg_count = len(agg_df)
        agg_df = agg_df[agg_df["phase_two_end"] <= cutoff]
        removed_agg = initial_agg_count - len(agg_df)
        print(f"Removed {removed_agg} aggregated rows with phase_two_end after {cutoff_date}.")

        # ---------------------------------------------------------------------
        # 12. Demean numHelped and hasHelped per user (if these columns exist).
        # ---------------------------------------------------------------------
        if "numHelped" in agg_df.columns:
            agg_df["meaned_numHelped"] = agg_df.groupby("user_id")["numHelped"].transform(lambda x: x - x.mean())
        if "hasHelped" in agg_df.columns:
            agg_df["meaned_hasHelped"] = agg_df.groupby("user_id")["hasHelped"].transform(lambda x: x - x.mean())

        # ---------------------------------------------------------------------
        # 13. Export the aggregated data.
        # ---------------------------------------------------------------------
        out_file = os.path.splitext(file_name)[0] + "processed.parquet"
        out_path = os.path.join(output_folder, out_file)
        agg_df.to_parquet(out_path, index=False)
        print(f"Finished processing {file_name}. Results saved to: {out_path}")

        del df, agg_df
        gc.collect()

    print("\nAll parquet files processed successfully.")


def main():
    input_folder = "./02_raw_datasets"  # Adjust input folder path as needed.
    output_folder = "./03_processed_datasets"  # Adjust output folder path as needed.
    cutoff_date = "2025-01-03"  # Phase two end cutoff.
    process_parquet_files(input_folder, output_folder, cutoff_date)


if __name__ == "__main__":
    main()