import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from numba import njit
import duckdb

# Set display options for better debugging output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


@njit
def calculate_metrics_jit(
        timestamps, questionAsked, answer, acceptedAnswerReceived,
        acceptedVoteReceived, acceptedAnswerPosted, target_time
):
    """
    Calculate user metrics across multiple time windows using JIT compilation
    for significantly improved performance.
    """
    # Initialize variables for all time windows
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

    # Set cutoff times for different windows (in microseconds)
    cutoff_30d = target_time - 30 * 86400 * 1_000_000
    cutoff_14d = target_time - 14 * 86400 * 1_000_000
    cutoff_7d = target_time - 7 * 86400 * 1_000_000
    cutoff_3d = target_time - 3 * 86400 * 1_000_000

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
        accepted_answers_received_at += acceptedAnswerReceived[i]
        accepted_votes_received_at += acceptedVoteReceived[i]
        accepted_answers_posted_at += acceptedAnswerPosted[i]

        # Add to 30-day window metrics
        if timestamps[i] >= cutoff_30d:
            questions_asked_30d += questionAsked[i]
            help_provided_30d += answer[i]
            accepted_answers_received_30d += acceptedAnswerReceived[i]
            accepted_votes_received_30d += acceptedVoteReceived[i]
            accepted_answers_posted_30d += acceptedAnswerPosted[i]

        # Add to 14-day window metrics
        if timestamps[i] >= cutoff_14d:
            questions_asked_14d += questionAsked[i]
            help_provided_14d += answer[i]
            accepted_answers_received_14d += acceptedAnswerReceived[i]
            accepted_votes_received_14d += acceptedVoteReceived[i]
            accepted_answers_posted_14d += acceptedAnswerPosted[i]

        # Add to 7-day window metrics
        if timestamps[i] >= cutoff_7d:
            questions_asked_7d += questionAsked[i]
            help_provided_7d += answer[i]
            accepted_answers_received_7d += acceptedAnswerReceived[i]
            accepted_votes_received_7d += acceptedVoteReceived[i]
            accepted_answers_posted_7d += acceptedAnswerPosted[i]

        # Add to 3-day window metrics
        if timestamps[i] >= cutoff_3d:
            questions_asked_3d += questionAsked[i]
            help_provided_3d += answer[i]
            accepted_answers_received_3d += acceptedAnswerReceived[i]
            accepted_votes_received_3d += acceptedVoteReceived[i]
            accepted_answers_posted_3d += acceptedAnswerPosted[i]

    # Calculate derived metrics
    help_provided_ever = 1 if help_provided_at > 0 else 0

    # Determine user's initial experience as help receiver
    if questions_asked_at == 0:
        initialExperienceReceiving_code = 0
    elif questions_asked_at > 0 and accepted_answers_received_at == 0:
        initialExperienceReceiving_code = 1
    elif questions_asked_at > 0 and accepted_answers_received_at > 0:
        initialExperienceReceiving_code = 2
    else:
        initialExperienceReceiving_code = 3

    # Determine user's initial experience as help provider
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
    acceptedAnswerReceived = user_history_df["acceptedAnswerReceived"].values
    acceptedVoteReceived = user_history_df["acceptedVoteReceived"].values
    acceptedAnswerPosted = user_history_df["acceptedAnswerPosted"].values

    return (timestamps, questionAsked, answer, acceptedAnswerReceived,
            acceptedVoteReceived, acceptedAnswerPosted)


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


def process_question_centered_dataset(input_file: str, output_file: str, chunk_size: int = 1000,
                                       cutoff_date: str = "2025-04-01") -> None:
    """
    Process the question-centered dataset to calculate metrics for each user.
    Dataset is centered on when users post their question.
    Self-answers are excluded from all metrics.
    Uses JIT compilation for performance optimization.
    """
    print(f"\n=== Processing {input_file} ===")
    cutoff = pd.to_datetime(cutoff_date)

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

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    # Process users in chunks
    num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
    all_result_count = 0

    chunk_progress = tqdm(
        total=len(user_ids),
        desc="Overall progress",
        unit="users",
        position=0
    )

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(user_ids))
        chunk_user_ids = user_ids[start_idx:end_idx]

        print(f"\nProcessing users {start_idx:,} to {end_idx:,} ({len(chunk_user_ids):,} users)")

        # Use DuckDB for efficient filtering by user_id
        user_ids_str = ", ".join(str(id) for id in chunk_user_ids)
        df = duckdb.query(f"""
           SELECT * FROM read_parquet('{input_file}')
           WHERE user_id IN ({user_ids_str})
        """).to_df()

        print(f"Loaded {len(df):,} total rows for this chunk of users")

        # Convert timestamps to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["phase_one_start"] = pd.to_datetime(df["phase_one_start"], errors="coerce")
        df["phase_two_end"] = pd.to_datetime(df["phase_two_end"], errors="coerce")

        # Create binary event flags using minimal memory (int8)
        # Note: All Answer, AcceptedAnswer, and AcceptedAnswerPosted events in raw data exclude self-answers
        df["questionAsked"] = df["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswer"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["answer"] = df["event"].map({"Answer": 1, "Window_Answer": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswerReceived"] = df["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        df["acceptedVoteReceived"] = df["event"].map({"AcceptedAnswerVote": 1}).fillna(0).astype(np.int8)
        df["acceptedAnswerPosted"] = df["event"].map({"AcceptedAnswerPosted": 1}).fillna(0).astype(np.int8)

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
            chunk_progress.update(len(chunk_user_ids))
            print("No non-history data in this chunk, skipping...")
            continue

        # Build JIT-compatible user history cache
        print("Building JIT-compatible user history cache for this chunk...")
        user_histories_jit = {}
        user_histories_pandas = {}

        for user_id, user_data in tqdm(history_chunk.groupby("user_id"), desc="Preprocessing users", position=1,
                                       leave=False):
            user_histories_jit[user_id] = prepare_user_history_for_jit(user_data)
            user_histories_pandas[user_id] = user_data

        # Calculate reciprocity activation
        reciprocity_status = calculate_reciprocity_activation(user_histories_pandas)

        del history_chunk
        del user_histories_pandas

        # Find all Phase_One_Start events
        phase_one_starts = non_history_chunk[non_history_chunk["event"] == "Phase_One_Start"]
        all_event_ids = non_history_chunk["event_id"].unique()

        # Calculate metrics for each Phase_One_Start event using JIT
        print("Calculating metrics for each Phase_One_Start event...")
        event_metrics = {}

        phase_one_progress = tqdm(
            total=len(phase_one_starts),
            desc="Processing Phase_One_Start events with JIT",
            unit="events",
            position=1,
            leave=False
        )

        for _, row in phase_one_starts.iterrows():
            user_id = row["user_id"]
            event_id = row["event_id"]
            target_time = row["timestamp"].to_numpy().astype(np.int64)  # Convert to int64 for JIT

            # Calculate metrics using JIT
            if user_id not in user_histories_jit:
                # Default metrics if no history
                metrics_values = (0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0.0)  # Add default value for time_since_first_activity_days

                # Convert the last two values (experience codes) to strings
                initialExperienceReceiving, initialExperienceGiving = "no help seeked", "no help attempted"

                # Combine numeric metrics with string experience values and timeSinceFirstActivityDays
                metrics = list(metrics_values[:-3]) + [initialExperienceReceiving, initialExperienceGiving,
                                                       metrics_values[-1]]
            else:
                # Calculate metrics using JIT
                metrics_values = calculate_metrics_jit(*user_histories_jit[user_id], target_time)

                # Convert experience codes to strings
                initialExperienceReceiving, initialExperienceGiving = convert_experience_code_to_string(
                    metrics_values[-3], metrics_values[-2]
                )

                # Replace codes with strings in the metrics and keep timeSinceFirstActivityDays
                metrics = list(metrics_values[:-3]) + [initialExperienceReceiving, initialExperienceGiving,
                                                       metrics_values[-1]]

            event_metrics[event_id] = tuple(metrics)
            phase_one_progress.update(1)

        phase_one_progress.close()

        # Set default metrics for missing event_ids
        missing_event_ids = set(all_event_ids) - set(event_metrics.keys())
        if missing_event_ids:
            default_metrics = (0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0,
                               0, "no help seeked", "no help attempted",
                               0.0)
            for eid in missing_event_ids:
                event_metrics[eid] = default_metrics

        # Define metric column names
        metric_columns = [
            "numQuestionsAskedAT", "numHelpProvidedAT", "helpProvidedEver",
            "numAcceptedAnswersReceivedAT", "numAcceptedVotesReceivedAT",
            "numQuestionsAsked30D", "numHelpProvided30D",
            "numAcceptedAnswersReceived30D", "numAcceptedVotesReceived30D",
            "numAcceptedAnswersPosted30D",
            "numQuestionsAsked14D", "numHelpProvided14D",
            "numAcceptedAnswersReceived14D", "numAcceptedVotesReceived14D",
            "numAcceptedAnswersPosted14D",
            "numQuestionsAsked7D", "numHelpProvided7D",
            "numAcceptedAnswersReceived7D", "numAcceptedVotesReceived7D",
            "numAcceptedAnswersPosted7D",
            "numQuestionsAsked3D", "numHelpProvided3D",
            "numAcceptedAnswersReceived3D", "numAcceptedVotesReceived3D",
            "numAcceptedAnswersPosted3D",
            "numAcceptedAnswersPostedAT", "initialExperienceReceiving", "initialExperienceGiving",
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
        print("Adding reciprocity activation data...")
        non_history_chunk['reciprocityActivated'] = 0
        non_history_chunk['reciprocityActivatedTimestamp'] = pd.NaT

        for user_id, (activated, activation_time) in reciprocity_status.items():
            user_rows = non_history_chunk[non_history_chunk['user_id'] == user_id]
            if not user_rows.empty:
                if pd.isna(activation_time):
                    non_history_chunk.loc[user_rows.index, 'reciprocityActivated'] = 0
                else:
                    mask = non_history_chunk.index.isin(user_rows.index) & (
                            non_history_chunk['timestamp'] > activation_time)
                    non_history_chunk.loc[mask, 'reciprocityActivated'] = 1
                    non_history_chunk.loc[mask, 'reciprocityActivatedTimestamp'] = activation_time

        # Mark Window_Answer events for numHelped calculation
        non_history_chunk["numHelped"] = np.where(non_history_chunk["event"] == "Window_Answer", 1, 0)

        # Determine phase (1 or 2) based on timestamps
        print("Determining phase for each event...")
        non_history_chunk["timestampInt"] = pd.to_numeric(non_history_chunk["timestamp"], errors="coerce",
                                                          downcast="integer")

        # Get phase boundary timestamps for each event_id
        phase_two_start_times = (
            non_history_chunk.loc[non_history_chunk["event"] == "Phase_Two_Start"]
            .groupby("event_id")["timestampInt"]
            .max()
        )
        end_times = (
            non_history_chunk.loc[non_history_chunk["event"] == "Phase_Two_End"]
            .groupby("event_id")["timestampInt"]
            .max()
        )

        # Map phase boundary timestamps to all rows by event_id
        non_history_chunk["phaseTwoStartTime"] = non_history_chunk["event_id"].map(phase_two_start_times)
        non_history_chunk["endTime"] = non_history_chunk["event_id"].map(end_times)

        # Assign phases based on timestamp relationships
        non_history_chunk["isPhase"] = np.where(
            non_history_chunk["timestampInt"] >= non_history_chunk["phaseTwoStartTime"],
            1,  # Phase 2 (at or after Phase_Two_Start)
            0  # Phase 1 (before Phase_Two_Start)
        )

        # Handle post-end events
        non_history_chunk["isPhase"] = np.where(
            (non_history_chunk["timestampInt"] > non_history_chunk["endTime"]) &
            (non_history_chunk["endTime"].notna()),
            0,  # Set to 0 if after end_time
            non_history_chunk["isPhase"]
        )
        non_history_chunk["phase"] = non_history_chunk["isPhase"].replace({0: 1, 1: 2})

        # Add derived metrics
        non_history_chunk["receivedAcceptedAnswerEver"] = (
                non_history_chunk["numAcceptedAnswersReceivedAT"] > 0).astype(int)
        non_history_chunk["receivedAcceptedVoteEver"] = (non_history_chunk["numAcceptedVotesReceivedAT"] > 0).astype(
            int)

        # Add reciprocity activation flag for before Phase 1
        non_history_chunk["reciprocityActivatedBeforePhase1"] = (
            (non_history_chunk["reciprocityActivated"] == 1) &
            (non_history_chunk["reciprocityActivatedTimestamp"] < non_history_chunk["phase_one_start"])
        ).astype(int)

        non_history_chunk["month"] = non_history_chunk["timestamp"].dt.month
        non_history_chunk["year"] = non_history_chunk["timestamp"].dt.year

        # Define aggregation columns and functions
        print("Aggregating data by event_id and phase...")
        group_cols = ["event_id", "phase"]
        agg_dict = {
            "user_id": "first",
            "timestamp": "first",
            "event": "first",
            "question_id": "first",
            "phase_one_start": "first",
            "phase_two_end": "first",
            "has_answer": "first",
            "has_accepted_answer": "first",
            "has_self_answer": "first",
            "responseTimeHours": "first",
            "timeToAcceptedAnswerHours": "first",
            "timeToAcceptVoteHours": "first",
            "numHelped": "sum",
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
            "numAcceptedAnswersPosted30D": "first",
            "numQuestionsAsked14D": "first",
            "numHelpProvided14D": "first",
            "numAcceptedAnswersReceived14D": "first",
            "numAcceptedVotesReceived14D": "first",
            "numAcceptedAnswersPosted14D": "first",
            "numQuestionsAsked7D": "first",
            "numHelpProvided7D": "first",
            "numAcceptedAnswersReceived7D": "first",
            "numAcceptedVotesReceived7D": "first",
            "numAcceptedAnswersPosted7D": "first",
            "numQuestionsAsked3D": "first",
            "numHelpProvided3D": "first",
            "numAcceptedAnswersReceived3D": "first",
            "numAcceptedVotesReceived3D": "first",
            "numAcceptedAnswersPosted3D": "first",
            "numAcceptedAnswersPostedAT": "first",
            "initialExperienceReceiving": "first",
            "initialExperienceGiving": "first",
            "timeSinceFirstActivityDays": "first",
            "reciprocityActivated": "first",
            "reciprocityActivatedTimestamp": "first",
            "reciprocityActivatedBeforePhase1": "first",
            "autobiography_received": "first",
            "registration_date": "first",
            "autobiography_active_phase_one_start": "first",
            "autobiography_active_phase_two_start": "first",
            "days_since_registration_at_phase_one_start": "first",
            "helps_given_between_question_and_answer": "first"
        }

        # Apply aggregation
        non_history_chunk = non_history_chunk.sort_values(["event_id", "timestamp"])
        agg_df = non_history_chunk.groupby(group_cols).agg(agg_dict).reset_index()

        # Filter out rows beyond cutoff date
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
                agg_df[col] = agg_df[col].astype("int64")  # Convert nullable integers
        for col in agg_df.select_dtypes(include=["Float32"]).columns:
            agg_df[col] = agg_df[col].astype("float64")  # Convert nullable floats

        # Create additional metrics for analysis
        print("Calculating additional metrics and fixed effects...")
        agg_df['hasHelped'] = (agg_df['numHelped'] > 0).astype(int)
        agg_df['lnNumHelped'] = np.log(agg_df['numHelped'] + 1)

        # Calculate fixed effects (user, question, and month-level deviations)
        agg_df['userFeNumHelped'] = agg_df.groupby("user_id")["numHelped"].transform(lambda x: x - x.mean())
        agg_df['questionFeNumHelped'] = agg_df.groupby("event_id")["numHelped"].transform(lambda x: x - x.mean())
        agg_df['userFeHasHelped'] = agg_df.groupby("user_id")["hasHelped"].transform(lambda x: x - x.mean())
        agg_df['questionFeHasHelped'] = agg_df.groupby("event_id")["hasHelped"].transform(lambda x: x - x.mean())
        agg_df['userFeLnNumHelped'] = agg_df.groupby("user_id")["lnNumHelped"].transform(lambda x: x - x.mean())
        agg_df['questionFeLnNumHelped'] = agg_df.groupby("event_id")["lnNumHelped"].transform(lambda x: x - x.mean())

        # Calculate month fixed effects (seasonal deviations)
        #agg_df['monthFeNumHelped'] = agg_df.groupby("month")["numHelped"].transform(lambda x: x - x.mean())
        #agg_df['monthFeHasHelped'] = agg_df.groupby("month")["hasHelped"].transform(lambda x: x - x.mean())
        #agg_df['monthFeLnNumHelped'] = agg_df.groupby("month")["lnNumHelped"].transform(lambda x: x - x.mean())

        # Track processing counts
        chunk_count = len(agg_df)
        all_result_count += chunk_count

        # Standardize column naming
        print("Standardizing column names...")
        column_rename_map = {
            'user_id': 'userId',
            'event_id': 'eventId',
            'question_id': 'questionId',
            'phase_one_start': 'phaseOneStart',
            'phase_two_end': 'phaseTwoEnd',
            'has_answer': 'hasAnswer',
            'has_accepted_answer': 'hasAcceptedAnswer',
            'has_self_answer': 'hasSelfAnswer',
        }
        agg_df = agg_df.rename(columns=column_rename_map)

        # Save output using append for all chunks after the first
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if chunk_idx == 0:
            # For first chunk, create a new file
            agg_df.to_parquet(output_file, index=False)
            print(f"Created output file {output_file} with {chunk_count:,} rows")
        else:
            # For subsequent chunks, append to the existing file
            agg_df.to_parquet(
                output_file,
                index=False,
                append=True,
                engine="fastparquet"
            )
            print(f"Appended chunk {chunk_idx + 1}/{num_chunks} with {chunk_count:,} rows to {output_file}")

        # Free memory
        del non_history_chunk
        del user_histories_jit
        del agg_df
        del event_metrics
        del reciprocity_status
        gc.collect()

        chunk_progress.update(len(chunk_user_ids))

    chunk_progress.close()

    print(f"\nCompleted processing {all_result_count:,} total rows")
    print(f"Final output saved to {output_file}")


if __name__ == "__main__":
    input_folder = "../data/input"
    output_folder = "../data/study_datasets"
    cutoff_date = "2025-04-01"

    # Set this to True to process test mode files, False for full dataset
    test_mode = False
    test_user_limit = 100000  # Should match the limit used in creating script

    # Process each question-centered model dataset
    for days in [7]:
        # Build filename based on test mode
        if test_mode:
            input_file = f"{input_folder}/question_centered_model_{days}d_all_questions_TEST{test_user_limit}.parquet"
            output_file = f"{output_folder}/question_centered_model_{days}d_processed_TEST{test_user_limit}.parquet"
        else:
            input_file = f"{input_folder}/question_centered_model_{days}d_all_questions.parquet"
            output_file = f"{output_folder}/question_centered_model_{days}d_processed.parquet"

        process_question_centered_dataset(
            input_file=input_file,
            output_file=output_file,
            chunk_size=100000,
            cutoff_date=cutoff_date
        )