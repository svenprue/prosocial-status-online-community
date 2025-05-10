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
    help_provided_at = user_history_before["answer"].sum()
    accepted_answers_received_at = user_history_before["acceptedAnswerReceived"].sum()
    accepted_votes_received_at = user_history_before["acceptedVoteReceived"].sum()
    accepted_answers_posted_at = user_history_before["acceptedAnswerPosted"].sum()
    help_provided_ever = 1 if help_provided_at > 0 else 0

    # Calculate 30-day window
    cutoff_30d = target_time - pd.Timedelta(days=30)
    window_30d = user_history_before[user_history_before["timestamp"] >= cutoff_30d]
    questions_asked_30d = window_30d["questionAsked"].sum()
    help_provided_30d = window_30d["answer"].sum()
    accepted_answers_received_30d = window_30d["acceptedAnswerReceived"].sum()
    accepted_votes_received_30d = window_30d["acceptedVoteReceived"].sum()

    # Calculate 14-day window
    cutoff_14d = target_time - pd.Timedelta(days=14)
    window_14d = user_history_before[user_history_before["timestamp"] >= cutoff_14d]
    questions_asked_14d = window_14d["questionAsked"].sum()
    help_provided_14d = window_14d["answer"].sum()
    accepted_answers_received_14d = window_14d["acceptedAnswerReceived"].sum()
    accepted_votes_received_14d = window_14d["acceptedVoteReceived"].sum()

    # Calculate 7-day window
    cutoff_7d = target_time - pd.Timedelta(days=7)
    window_7d = user_history_before[user_history_before["timestamp"] >= cutoff_7d]
    questions_asked_7d = window_7d["questionAsked"].sum()
    help_provided_7d = window_7d["answer"].sum()
    accepted_answers_received_7d = window_7d["acceptedAnswerReceived"].sum()
    accepted_votes_received_7d = window_7d["acceptedVoteReceived"].sum()

    # Calculate 3-day window
    cutoff_3d = target_time - pd.Timedelta(days=3)
    window_3d = user_history_before[user_history_before["timestamp"] >= cutoff_3d]
    questions_asked_3d = window_3d["questionAsked"].sum()
    help_provided_3d = window_3d["answer"].sum()
    accepted_answers_received_3d = window_3d["acceptedAnswerReceived"].sum()
    accepted_votes_received_3d = window_3d["acceptedVoteReceived"].sum()

    # Calculate initial experience receiving
    initialExperienceReceiving = "unknown"
    if questions_asked_at == 0:
        initialExperienceReceiving = "no help seeked"
    elif questions_asked_at > 0 and accepted_answers_received_at == 0:
        initialExperienceReceiving = "help seeked"
    elif questions_asked_at > 0 and accepted_answers_received_at > 0:
        initialExperienceReceiving = "help received"

    # Calculate initial experience giving
    initialExperienceGiving = "unknown"
    if help_provided_at == 0:
        initialExperienceGiving = "no help attempted"
    elif help_provided_at > 0 and accepted_answers_posted_at == 0:
        initialExperienceGiving = "help attempted"
    elif accepted_answers_posted_at > 0:
        initialExperienceGiving = "helped"

    return (
        questions_asked_at, help_provided_at, help_provided_ever,
        accepted_answers_received_at, accepted_votes_received_at,
        questions_asked_30d, help_provided_30d,
        accepted_answers_received_30d, accepted_votes_received_30d,
        questions_asked_14d, help_provided_14d,
        accepted_answers_received_14d, accepted_votes_received_14d,
        questions_asked_7d, help_provided_7d,
        accepted_answers_received_7d, accepted_votes_received_7d,
        questions_asked_3d, help_provided_3d,
        accepted_answers_received_3d, accepted_votes_received_3d,
        accepted_answers_posted_at, initialExperienceReceiving, initialExperienceGiving
    )


def calculate_reciprocity_activation(user_histories):
    """
    Calculate reciprocity activation for each user, considering event timing

    Args:
        user_histories: Dictionary of user histories keyed by user_id

    Returns:
        reciprocity_status: Dictionary with user_id as key and (activated, activation_timestamp) as value
    """
    reciprocity_status = {}

    print("Calculating reciprocity activation...")

    for user_id, user_history in tqdm(user_histories.items(), desc="Processing reciprocity", position=1, leave=False):
        # Get the first provided answer (Answer or Window_Answer)
        first_answer = user_history[user_history['answer'] == 1].sort_values('timestamp').head(1)

        if first_answer.empty:
            # User never provided an answer
            reciprocity_status[user_id] = (0, pd.NaT)
            continue

        first_answer_time = first_answer['timestamp'].iloc[0]

        # Check for accepted answers within 7 days before the first provided answer
        accepted_answers = user_history[
            (user_history['acceptedAnswer'] == 1) &
            (user_history['timestamp'] < first_answer_time) &
            (user_history['timestamp'] > first_answer_time - pd.Timedelta(days=7))
            ]

        # Only mark as activated if there were accepted answers in the window
        if not accepted_answers.empty:
            reciprocity_status[user_id] = (1, first_answer_time)
        else:
            reciprocity_status[user_id] = (0, pd.NaT)

    return reciprocity_status


def process_question_dataset(input_file: str, output_file: str, chunk_size: int = 1000,
                             cutoff_date: str = "2025-04-01") -> None:
    """
    Process the question-centered dataset to calculate metrics for each user.
    Metrics are calculated only at Phase_One_Start events and applied to all rows with the same event_id.
    """
    cutoff = pd.to_datetime(cutoff_date)

    # Get unique user IDs
    user_ids_original = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()

    # Create a copy and shuffle it
    user_ids = np.copy(user_ids_original)
    np.random.shuffle(user_ids)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if output file exists and remove if it does
    if os.path.exists(output_file):
        os.remove(output_file)

    # Calculate number of chunks
    num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
    all_result_count = 0

    # Process users in chunks
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks", total=num_chunks, position=0, leave=False):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(user_ids))
        chunk_user_ids = user_ids[start_idx:end_idx]

        # Read only data for current chunk of users
        user_ids_str = ", ".join(str(id) for id in chunk_user_ids)

        # Use DuckDB to efficiently filter the parquet file
        df = duckdb.query(f"""
           SELECT * FROM read_parquet('{input_file}')
           WHERE user_id IN ({user_ids_str})
       """).to_df()

        # Ensure the timestamp is in datetime format
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["phase_one_start"] = pd.to_datetime(df["phase_one_start"], errors="coerce")
        df["phase_two_end"] = pd.to_datetime(df["phase_two_end"], errors="coerce")

        # Create numeric flags from event type with minimal memory usage
        df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["answer"] = df["event"].map({"Answer": 1, "Window_Answer": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswerReceived"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["acceptedVoteReceived"] = df["event"].map({"AcceptedAnswerVote": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswerPosted"] = df["event"].map({"AcceptedAnswerPosted": 1}).fillna(0).astype(np.int8)

        # Calculate time-to-first-answer in hours if both timestamps exist
        df["timeToFirstAnswerHours"] = None
        mask = ~df["first_answer_timestamp"].isna() & ~df["question_timestamp"].isna()
        if any(mask):
            df.loc[mask, "timeToFirstAnswerHours"] = (
                    (pd.to_datetime(df.loc[mask, "first_answer_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "question_timestamp"])).dt.total_seconds() / 3600
            )

        # Calculate time-to-accepted-answer in hours if both timestamps exist
        df["timeToAcceptedAnswerHours"] = None
        mask = ~df["accepted_answer_timestamp"].isna() & ~df["question_timestamp"].isna()
        if any(mask):
            df.loc[mask, "timeToAcceptedAnswerHours"] = (
                    (pd.to_datetime(df.loc[mask, "accepted_answer_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "question_timestamp"])).dt.total_seconds() / 3600
            )

        # Calculate time-to-accept-vote in hours if both timestamps exist
        df["timeToAcceptVoteHours"] = None
        mask = ~df["accepted_answer_vote_timestamp"].isna() & ~df["accepted_answer_timestamp"].isna()
        if any(mask):
            df.loc[mask, "timeToAcceptVoteHours"] = (
                    (pd.to_datetime(df.loc[mask, "accepted_answer_vote_timestamp"]) -
                     pd.to_datetime(df.loc[mask, "accepted_answer_timestamp"])).dt.total_seconds() / 3600
            )

        # Separate history and non-history data
        history_df = df[df["is_history"] == 1].copy()
        non_history_df = df[df["is_history"] == 0].copy()

        # Free memory
        del df
        gc.collect()

        if len(non_history_df) == 0:
            continue

        # Build user history data
        print("Building user history cache...")
        user_histories = {}

        for user_id, group in tqdm(history_df.groupby("user_id"), desc="Preprocessing users", position=1, leave=False):
            user_histories[user_id] = group

        # Calculate reciprocity activation
        reciprocity_status = calculate_reciprocity_activation(user_histories)

        # Free memory
        del history_df
        gc.collect()

        # Find all Phase_One_Start events
        phase_one_starts = non_history_df[non_history_df["event"] == "Phase_One_Start"]

        # Get all unique event IDs
        all_event_ids = non_history_df["event_id"].unique()

        # Calculate metrics for Phase_One_Start events
        event_metrics = {}
        for _, row in tqdm(phase_one_starts.iterrows(), total=len(phase_one_starts),
                           desc="Calculating metrics at Phase_One_Start", position=1, leave=False):
            user_id = row["user_id"]
            event_id = row["event_id"]
            target_time = row["timestamp"]

            # Calculate metrics
            if user_id not in user_histories:
                metrics = (0, 0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, "no help seeked", "no help attempted")
            else:
                metrics = calculate_metrics_pandas(user_histories[user_id], target_time)

            # Store metrics for this event_id
            event_metrics[event_id] = metrics

        # Check for missing event_ids
        missing_event_ids = set(all_event_ids) - set(event_metrics.keys())
        if missing_event_ids:
            default_metrics = (0, 0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, "no help seeked", "no help attempted")
            for eid in missing_event_ids:
                event_metrics[eid] = default_metrics

        # Create metric columns with default zeros
        metric_columns = [
            "numQuestionsAskedAT", "numHelpProvidedAT", "helpProvidedEver",
            "numAcceptedAnswersReceivedAT", "numAcceptedVotesReceivedAT",
            "numQuestionsAsked30D", "numHelpProvided30D",
            "numAcceptedAnswersReceived30D", "numAcceptedVotesReceived30D",
            "numQuestionsAsked14D", "numHelpProvided14D",
            "numAcceptedAnswersReceived14D", "numAcceptedVotesReceived14D",
            "numQuestionsAsked7D", "numHelpProvided7D",
            "numAcceptedAnswersReceived7D", "numAcceptedVotesReceived7D",
            "numQuestionsAsked3D", "numHelpProvided3D",
            "numAcceptedAnswersReceived3D", "numAcceptedVotesReceived3D",
            "numAcceptedAnswersPostedAT", "initialExperienceReceiving", "initialExperienceGiving"
        ]

        # Convert event_metrics dictionary to a DataFrame
        metrics_data = []
        for event_id, metrics in event_metrics.items():
            metrics_data.append([event_id] + list(metrics))

        metrics_df = pd.DataFrame(metrics_data, columns=['event_id'] + metric_columns)

        # Merge with non_history_df instead of iterating
        non_history_df = pd.merge(non_history_df, metrics_df, on='event_id', how='left')

        # Add reciprocity activation data
        reciprocity_data = []
        for user_id, (activated, activation_time) in reciprocity_status.items():
            # For each row with this user_id, determine if they're activated based on timestamp
            user_rows = non_history_df[non_history_df['user_id'] == user_id]
            for idx, row in user_rows.iterrows():
                if pd.isna(activation_time):
                    is_activated_at_time = 0
                else:
                    is_activated_at_time = 1 if row['timestamp'] >= activation_time else 0

                reciprocity_data.append({
                    'index': idx,
                    'reciprocityActivated': is_activated_at_time,
                    'reciprocityActivatedTimestamp': activation_time if is_activated_at_time else pd.NaT
                })

        # Apply reciprocity data to DataFrame
        if reciprocity_data:
            reciprocity_df = pd.DataFrame(reciprocity_data)
            reciprocity_df.set_index('index', inplace=True)
            non_history_df.loc[reciprocity_df.index, 'reciprocityActivated'] = reciprocity_df['reciprocityActivated']
            non_history_df.loc[reciprocity_df.index, 'reciprocityActivatedTimestamp'] = reciprocity_df[
                'reciprocityActivatedTimestamp']
        else:
            # If no reciprocity data, set default values
            non_history_df['reciprocityActivated'] = 0
            non_history_df['reciprocityActivatedTimestamp'] = pd.NaT

        # Set the numHelped column based on answer events
        non_history_df["numHelped"] = np.where(non_history_df["event"] == "Window_Answer", 1, 0)
        non_history_df["hasAnswer"] = (non_history_df["has_answer"] > 0).astype(int)

        # Determine phase (1 or 2) based on timestamps
        non_history_df["timestampInt"] = pd.to_numeric(non_history_df["timestamp"], errors="coerce", downcast="integer")
        # Get Phase_Two_Start timestamps for each event_id
        phase_two_start_times = (
            non_history_df.loc[non_history_df["event"] == "Phase_Two_Start"]
            .groupby("event_id")["timestampInt"]
            .max()
        )

        # Get Phase_Two_End timestamps for each event_id
        end_times = (
            non_history_df.loc[non_history_df["event"] == "Phase_Two_End"]
            .groupby("event_id")["timestampInt"]
            .max()
        )

        # Map these timestamps to all rows by event_id
        non_history_df["phaseTwoStartTime"] = non_history_df["event_id"].map(phase_two_start_times)
        non_history_df["endTime"] = non_history_df["event_id"].map(end_times)

        # Set isPhase based on the comparison with Phase_Two_Start timestamp
        # 0 for before Phase_Two_Start, 1 for Phase_Two_Start and after
        non_history_df["isPhase"] = np.where(
            non_history_df["timestampInt"] >= non_history_df["phaseTwoStartTime"],
            1,  # Phase 2 (at or after Phase_Two_Start)
            0  # Phase 1 (before Phase_Two_Start)
        )

        # Handle any rows that might be after the end_time
        non_history_df["isPhase"] = np.where(
            (non_history_df["timestampInt"] > non_history_df["endTime"]) &
            (non_history_df["endTime"].notna()),
            0,  # Set to 0 if after end_time
            non_history_df["isPhase"]
        )
        non_history_df["phase"] = non_history_df["isPhase"].replace({0: 1, 1: 2})

        # Add additional derived metrics
        non_history_df["receivedAcceptedAnswerEver"] = (non_history_df["numAcceptedAnswersReceivedAT"] > 0).astype(int)
        non_history_df["receivedAcceptedVoteEver"] = (non_history_df["numAcceptedVotesReceivedAT"] > 0).astype(int)
        non_history_df["month"] = non_history_df["timestamp"].dt.month
        non_history_df["year"] = non_history_df["timestamp"].dt.year

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
            "timeToFirstAnswerHours": "first",
            "timeToAcceptedAnswerHours": "first",
            "timeToAcceptVoteHours": "first",
            "numHelped": "sum",
            "hasAnswer": "max",
            "helpProvidedEver": "first",
            "receivedAcceptedAnswerEver": "first",
            "receivedAcceptedVoteEver": "first",
            "year": "first",
            "month": "first",
            "numQuestionsAskedAT": "first",
            "numHelpProvidedAT": "first",
            "numAcceptedAnswersReceivedAT": "first",
            "numAcceptedVotesReceivedAT": "first",
            "numQuestionsAsked30D": "first",
            "numHelpProvided30D": "first",
            "numAcceptedAnswersReceived30D": "first",
            "numAcceptedVotesReceived30D": "first",
            "numQuestionsAsked14D": "first",
            "numHelpProvided14D": "first",
            "numAcceptedAnswersReceived14D": "first",
            "numAcceptedVotesReceived14D": "first",
            "numQuestionsAsked7D": "first",
            "numHelpProvided7D": "first",
            "numAcceptedAnswersReceived7D": "first",
            "numAcceptedVotesReceived7D": "first",
            "numQuestionsAsked3D": "first",
            "numHelpProvided3D": "first",
            "numAcceptedAnswersReceived3D": "first",
            "numAcceptedVotesReceived3D": "first",
            "numAcceptedAnswersPostedAT": "first",
            "initialExperienceReceiving": "first",
            "initialExperienceGiving": "first",
            "reciprocityActivated": "first",
            "reciprocityActivatedTimestamp": "first",
        }

        # Apply aggregation
        non_history_df = non_history_df.sort_values(["event_id", "timestamp"])
        agg_df = non_history_df.groupby(group_cols).agg(agg_dict).reset_index()

        # Remove any rows with phase_two_end after the cutoff date
        initial_count = len(agg_df)
        agg_df = agg_df[agg_df["phase_two_end"] <= cutoff]
        removed = initial_count - len(agg_df)

        agg_df = agg_df.convert_dtypes()  # Convert columns to best possible dtypes
        for col in agg_df.select_dtypes(include=["Int32", "Int64"]).columns:  # Find nullable integer columns
            agg_df[col] = agg_df[col].astype("int64")  # Convert to standard int64
        for col in agg_df.select_dtypes(include=["Float32"]).columns:  # Handle nullable floats if needed
            agg_df[col] = agg_df[col].astype("float64")  # Convert to standard float64

        agg_df['hasHelped'] = (agg_df['numHelped'] > 0).astype(int)
        agg_df['lnNumHelped'] = np.log(agg_df['numHelped'] + 1)

        # Fixed effects for numHelped
        agg_df['userFeNumHelped'] = agg_df.groupby("user_id")["numHelped"].transform(lambda x: x - x.mean())
        agg_df['questionFeNumHelped'] = agg_df.groupby("event_id")["numHelped"].transform(lambda x: x - x.mean())

        # Fixed effects for hasHelped
        agg_df['userFeHasHelped'] = agg_df.groupby("user_id")["hasHelped"].transform(lambda x: x - x.mean())
        agg_df['questionFeHasHelped'] = agg_df.groupby("event_id")["hasHelped"].transform(lambda x: x - x.mean())

        # Fixed effects for lnNumHelped
        agg_df['userFeLnNumHelped'] = agg_df.groupby("user_id")["lnNumHelped"].transform(lambda x: x - x.mean())
        agg_df['questionFeLnNumHelped'] = agg_df.groupby("event_id")["lnNumHelped"].transform(lambda x: x - x.mean())

        # Count the number of rows in this chunk
        chunk_count = len(agg_df)
        all_result_count += chunk_count

        # Create a directory for chunks
        output_dir = f"{output_file}_chunks"
        os.makedirs(output_dir, exist_ok=True)

        # Save chunk to a separate file
        chunk_file = f"{output_dir}/chunk_{chunk_idx:02d}.parquet"
        agg_df.to_parquet(chunk_file, index=False)
        print(f"Saved chunk {chunk_idx + 1}/{num_chunks} with {chunk_count:,} rows to {chunk_file}")

        del non_history_df
        del agg_df
        del user_histories
        del event_metrics
        del reciprocity_status
        gc.collect()

    print(f"Processed a total of {all_result_count:,} rows")
    print(f"Saved output to {output_file}")


if __name__ == "__main__":
    input_folder = "../data/input"
    output_folder = "../data/study_datasets"
    cutoff_date = "2025-04-01"

    # Process each question-centered model dataset
    for days in [7]:
        input_file = f"{input_folder}/question_centered_model_{days}d_all_questions.parquet"
        output_file = f"{output_folder}/question_centered_model_{days}d_processed.parquet"

        process_question_dataset(
            input_file=input_file,
            output_file=output_file,
            chunk_size=1000,
            cutoff_date=cutoff_date
        )