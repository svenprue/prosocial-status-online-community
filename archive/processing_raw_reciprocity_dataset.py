import os
import glob
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
import sys
import dask.dataframe as dd


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
        # Function code remains the same...
        # Index for current time (AT) metrics
        idx_at = np.searchsorted(timestamps, target_time, side="left") - 1

        # Indices for different time windows
        idx_30d = np.searchsorted(timestamps, days_30, side="left") - 1
        idx_14d = np.searchsorted(timestamps, days_14, side="left") - 1
        idx_7d = np.searchsorted(timestamps, days_7, side="left") - 1
        idx_3d = np.searchsorted(timestamps, days_3, side="left") - 1

        # Initialize all metrics
        q_at, a_at, ans_at = 0, 0, 0
        q_30d, a_30d, ans_30d = 0, 0, 0
        q_14d, a_14d, ans_14d = 0, 0, 0
        q_7d, a_7d, ans_7d = 0, 0, 0
        q_3d, a_3d, ans_3d = 0, 0, 0
        ever = 0

        # Calculate AT metrics
        if idx_at >= 0:
            q_at = cum_question[idx_at]
            a_at = cum_accepted[idx_at]
            ans_at = cum_answer[idx_at]
            ever = 1 if ans_at > 0 else 0

        # Calculate window metrics (counts within specific time windows)
        # 30D window
        if idx_at >= 0:
            base_q = cum_question[idx_30d] if idx_30d >= 0 else 0
            base_a = cum_accepted[idx_30d] if idx_30d >= 0 else 0
            base_ans = cum_answer[idx_30d] if idx_30d >= 0 else 0
            q_30d = q_at - base_q
            a_30d = a_at - base_a
            ans_30d = ans_at - base_ans

        # 14D window
        if idx_at >= 0:
            base_q = cum_question[idx_14d] if idx_14d >= 0 else 0
            base_a = cum_accepted[idx_14d] if idx_14d >= 0 else 0
            base_ans = cum_answer[idx_14d] if idx_14d >= 0 else 0
            q_14d = q_at - base_q
            a_14d = a_at - base_a
            ans_14d = ans_at - base_ans

        # 7D window
        if idx_at >= 0:
            base_q = cum_question[idx_7d] if idx_7d >= 0 else 0
            base_a = cum_accepted[idx_7d] if idx_7d >= 0 else 0
            base_ans = cum_answer[idx_7d] if idx_7d >= 0 else 0
            q_7d = q_at - base_q
            a_7d = a_at - base_a
            ans_7d = ans_at - base_ans

        # 3D window
        if idx_at >= 0:
            base_q = cum_question[idx_3d] if idx_3d >= 0 else 0
            base_a = cum_accepted[idx_3d] if idx_3d >= 0 else 0
            base_ans = cum_answer[idx_3d] if idx_3d >= 0 else 0
            q_3d = q_at - base_q
            a_3d = a_at - base_a
            ans_3d = ans_at - base_ans

        return (q_at, a_at, ans_at, ever,
                q_30d, a_30d, ans_30d,
                q_14d, a_14d, ans_14d,
                q_7d, a_7d, ans_7d,
                q_3d, a_3d, ans_3d)

    def compute_user_metrics(user_id: int, target_time: pd.Timestamp, user_data: dict) -> dict:
        """
        For a given user and target_time, compute metrics for different time windows:
        - All time before target_time (AT)
        - Last 30 days before target_time (30D)
        - Last 14 days before target_time (14D)
        - Last 7 days before target_time (7D)
        - Last 3 days before target_time (3D)
        """
        # Function code remains the same...
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

        print("Calculating metrics...")
        results = [process_row(r) for r in tqdm(df_phase1.itertuples(index=False),
                                                total=len(df_phase1),
                                                desc="Calculating metrics")]

        df_metrics = pd.DataFrame(results)

        # Convert to memory-efficient dtypes
        for col in df_metrics.columns:
            if col != "user_id" and col != "event_id":
                df_metrics[col] = df_metrics[col].astype(np.int32)

        # Drop all rows where is_history == 1 early to save memory
        df = df[df["is_history"] == 0].copy()
        print(f"After dropping is_history rows: {len(df):,} rows remaining.")

        # ---------------------------------------------------------------------
        # 7. Propagate metrics from Phase_One_Start to all rows (per event_id).
        # ---------------------------------------------------------------------
        print("Propagating metrics using dask (memory-efficient)...")
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
            "numHelpProvided3D",
        ]

        # Convert to dask dataframes for memory-efficient processing
        dask_df = dd.from_pandas(df, npartitions=10)  # Adjust npartitions based on your system
        dask_metrics = dd.from_pandas(
            df_metrics[["event_id"] + [col for col in cols_to_propagate if col in df_metrics.columns]],
            npartitions=10)

        # Perform merge using dask
        merged_dask = dask_df.merge(dask_metrics, on="event_id", how="left")

        # Compute the result, bringing it back to pandas but with better memory management
        print("Computing merge (this may take some time)...")
        df = merged_dask.compute()

        # Clean up to free memory
        del dask_df, dask_metrics, merged_dask, df_metrics
        gc.collect()

        # Fill NaN values and convert to efficient types
        for col in cols_to_propagate:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(np.int32)

        print("Metrics propagated successfully using dask.")

        # ---------------------------------------------------------------------
        # 8. Determine phase (1 or 2) for each row using event timestamps.
        # ---------------------------------------------------------------------
        print("Determining phases...")
        df["TimestampInt"] = pd.to_numeric(df["timestamp"], errors="coerce", downcast="integer")

        # Get the timestamps for each phase boundary
        phase_one_start_times = (
            df.loc[df["event"] == "Phase_One_Start"]
            .groupby("event_id")["TimestampInt"]
            .max()
            .to_dict()
        )
        phase_two_start_times = (
            df.loc[df["event"] == "Phase_Two_Start"]
            .groupby("event_id")["TimestampInt"]
            .max()
            .to_dict()
        )
        phase_two_end_times = (
            df.loc[df["event"] == "Phase_Two_End"]
            .groupby("event_id")["TimestampInt"]
            .max()
            .to_dict()
        )

        # Apply phase boundary times
        df["phase_one_start_time"] = df["event_id"].map(lambda x: phase_one_start_times.get(x, np.nan))
        df["phase_two_start_time"] = df["event_id"].map(lambda x: phase_two_start_times.get(x, np.nan))
        df["phase_two_end_time"] = df["event_id"].map(lambda x: phase_two_end_times.get(x, np.nan))

        # Calculate phase correctly
        df["phase"] = np.where(
            (df["TimestampInt"] >= df["phase_one_start_time"]) & (df["TimestampInt"] < df["phase_two_start_time"]),
            1,  # Phase 1
            np.where(
                (df["TimestampInt"] >= df["phase_two_start_time"]) & (df["TimestampInt"] <= df["phase_two_end_time"]),
                2,  # Phase 2
                0  # Not in either phase
            )
        )

        print(df)

        # Count rows with phase=0 before dropping
        phase_zero_count = (df["phase"] == 0).sum()
        print(f"Found {phase_zero_count:,} rows with phase=0 (outside of both phases).")

        # Drop rows that aren't in either phase
        df = df[df["phase"] > 0]
        print(f"After dropping phase=0 rows: {len(df):,} rows remaining.")

        # ---------------------------------------------------------------------
        # 9. Final processing: add month/year and help indicators.
        # ---------------------------------------------------------------------
        print("Adding help indicators...")
        df["month"] = df["timestamp"].dt.month.astype(np.int16)
        df["year"] = df["timestamp"].dt.year.astype(np.int16)
        df["receivedHelpEver"] = (df["numHelpReceivedAT"] > 0).astype(np.int8)
        df["provided_answer"] = df["event_history"].map({"Window_Answer": 1}).fillna(0).astype(np.int8)
        df["numHelped"] = df["provided_answer"].astype(np.int8)

        # ---------------------------------------------------------------------
        # 10. Group and aggregate data by event_id and phase.
        # ---------------------------------------------------------------------
        print("Grouping and aggregating data...")
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
            "numHelpProvidedEver": "first",
            "receivedHelpEver": "first",
            "year": "first",
            "month": "first",
        }

        # Add all metric columns to aggregation
        for col in cols_to_propagate:
            if col in df.columns:
                agg_dict[col] = "first"

        # Group and aggregate
        agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()

        agg_df["hasHelped"] = (agg_df["numHelped"] > 0).astype(np.int8)

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
        print("Demeaning depedent variables...")
        if "numHelped" in agg_df.columns:
            agg_df["meaned_numHelped"] = agg_df.groupby("user_id")["numHelped"].transform(lambda x: x - x.mean())
        if "hasHelped" in agg_df.columns:
            agg_df["meaned_hasHelped"] = agg_df.groupby("user_id")["hasHelped"].transform(lambda x: x - x.mean())

        # ---------------------------------------------------------------------
        # 13. Export the aggregated data.
        # ---------------------------------------------------------------------
        out_file = os.path.splitext(file_name)[0] + "_processed.parquet"
        out_path = os.path.join(output_folder, out_file)
        agg_df.to_parquet(out_path, index=False)
        print(f"Finished processing {file_name}. Results saved to: {out_path}")

        del df, agg_df
        gc.collect()

    print("\nAll parquet files processed successfully.")


def main():
    input_folder = "./testing"  # Adjust input folder path as needed.
    output_folder = "./03_processed_datasets"  # Adjust output folder path as needed.
    cutoff_date = "2025-01-03"  # Phase two end cutoff.
    process_parquet_files(input_folder, output_folder, cutoff_date)
    sys.exit(0)


if __name__ == "__main__":
    main()