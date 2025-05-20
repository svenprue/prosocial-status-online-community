import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from numba import njit

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


@njit
def calculate_metrics_jit(
        timestamps, questionAsked, answerPosted, acceptedAnswerReceived,
        acceptedVoteReceived, acceptedAnswerPosted, target_time
):
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

    cutoff_30d = target_time - 30 * 86400 * 1_000_000
    cutoff_14d = target_time - 14 * 86400 * 1_000_000
    cutoff_7d = target_time - 7 * 86400 * 1_000_000
    cutoff_3d = target_time - 3 * 86400 * 1_000_000

    # Find the timestamp of first activity (question asked or answer provided)
    first_activity_timestamp = np.int64(0)  # Initialize to 0 (no activity)
    for i in range(len(timestamps)):
        if (questionAsked[i] == 1 or answerPosted[i] == 1) and timestamps[i] < target_time:
            if first_activity_timestamp == 0 or timestamps[i] < first_activity_timestamp:
                first_activity_timestamp = timestamps[i]

    # Calculate time since first activity in days
    time_since_first_activity_days = 0.0
    if first_activity_timestamp > 0:  # If there was activity
        # Convert microseconds to days (86400 seconds per day, 1_000_000 microseconds per second)
        time_since_first_activity_days = (target_time - first_activity_timestamp) / (86400 * 1_000_000)

    for i in range(len(timestamps)):
        if timestamps[i] >= target_time:
            continue

        questions_asked_at += questionAsked[i]
        help_provided_at += answerPosted[i]
        accepted_answers_received_at += acceptedAnswerReceived[i]
        accepted_votes_received_at += acceptedVoteReceived[i]
        accepted_answers_posted_at += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_30d:
            questions_asked_30d += questionAsked[i]
            help_provided_30d += answerPosted[i]
            accepted_answers_received_30d += acceptedAnswerReceived[i]
            accepted_votes_received_30d += acceptedVoteReceived[i]
            accepted_answers_posted_30d += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_14d:
            questions_asked_14d += questionAsked[i]
            help_provided_14d += answerPosted[i]
            accepted_answers_received_14d += acceptedAnswerReceived[i]
            accepted_votes_received_14d += acceptedVoteReceived[i]
            accepted_answers_posted_14d += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_7d:
            questions_asked_7d += questionAsked[i]
            help_provided_7d += answerPosted[i]
            accepted_answers_received_7d += acceptedAnswerReceived[i]
            accepted_votes_received_7d += acceptedVoteReceived[i]
            accepted_answers_posted_7d += acceptedAnswerPosted[i]

        if timestamps[i] >= cutoff_3d:
            questions_asked_3d += questionAsked[i]
            help_provided_3d += answerPosted[i]
            accepted_answers_received_3d += acceptedAnswerReceived[i]
            accepted_votes_received_3d += acceptedVoteReceived[i]
            accepted_answers_posted_3d += acceptedAnswerPosted[i]

    help_provided_ever = 1 if help_provided_at > 0 else 0

    if questions_asked_at == 0:
        initialExperienceReceiving_code = 0
    elif questions_asked_at > 0 and accepted_answers_received_at == 0:
        initialExperienceReceiving_code = 1
    elif questions_asked_at > 0 and accepted_answers_received_at > 0:
        initialExperienceReceiving_code = 2
    else:
        initialExperienceReceiving_code = 3

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
    user_history_df = user_history_df.sort_values("timestamp")
    timestamps = user_history_df["timestamp"].astype(np.int64).values
    questionAsked = user_history_df["questionAsked"].values
    answerPosted = user_history_df["answerPosted"].values
    acceptedAnswerReceived = user_history_df["acceptedAnswerReceived"].values
    acceptedVoteReceived = user_history_df["acceptedVoteReceived"].values
    acceptedAnswerPosted = user_history_df["acceptedAnswerPosted"].values

    return (timestamps, questionAsked, answerPosted, acceptedAnswerReceived,
            acceptedVoteReceived, acceptedAnswerPosted)


def calculate_reciprocity_activation(user_histories):
    reciprocity_status = {}
    print("Calculating reciprocity activation...")

    for user_id, user_history in user_histories.items():
        first_answer = user_history[user_history['answerPosted'] == 1].sort_values('timestamp').head(1)

        if first_answer.empty:
            reciprocity_status[user_id] = (0, pd.NaT)
            continue

        first_answer_time = first_answer['timestamp'].iloc[0]

        accepted_answers = user_history[
            (user_history['acceptedAnswerReceived'] == 1) &
            (user_history['timestamp'] < first_answer_time) &
            (user_history['timestamp'] > first_answer_time - pd.Timedelta(days=7))
            ]

        if not accepted_answers.empty:
            reciprocity_status[user_id] = (1, first_answer_time)
        else:
            reciprocity_status[user_id] = (0, pd.NaT)

    return reciprocity_status


def process_bounty_dataset(input_file: str, output_file: str, chunk_size: int = 500000) -> None:
    print(f"\n=== Processing {input_file} ===")

    print("Reading unique user IDs...")
    all_user_ids = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()
    print(f"Found {len(all_user_ids):,} unique users")

    total_rows = pd.read_parquet(input_file, columns=["is_history"])
    total_non_history = len(total_rows[total_rows["is_history"] == 0])
    print(f"Total non-history rows: {total_non_history:,}")
    del total_rows

    first_chunk = True
    processed_rows = 0

    columns_to_read = [
        "user_id", "timestamp", "event", "answer_id",
        "question_id", "is_bounty", "bounty_amount", "answer_sequence", "is_history"
    ]

    # Only one progress bar for chunks, with estimated time
    chunk_progress = tqdm(
        total=len(all_user_ids),
        desc="Processing user chunks",
        unit="users"
    )

    for chunk_start in range(0, len(all_user_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_user_ids))
        chunk_user_ids = all_user_ids[chunk_start:chunk_end]

        print(f"\nProcessing users {chunk_start:,} to {chunk_end:,} ({len(chunk_user_ids):,} users)")

        chunk_filter = [("user_id", "in", list(chunk_user_ids))]
        user_data_chunk = pd.read_parquet(input_file, filters=chunk_filter, columns=columns_to_read)

        print(f"Loaded {len(user_data_chunk):,} total rows for this chunk of users")

        user_data_chunk["timestamp"] = pd.to_datetime(user_data_chunk["timestamp"], errors="coerce")

        history_chunk = user_data_chunk[user_data_chunk["is_history"] == 1].copy()
        non_history_chunk = user_data_chunk[user_data_chunk["is_history"] == 0].copy()

        print(f"History rows: {len(history_chunk):,}")
        print(f"Non-history rows: {len(non_history_chunk):,}")

        if len(non_history_chunk) == 0:
            chunk_progress.update(len(chunk_user_ids))
            print("No non-history data in this chunk, skipping...")
            continue

        history_chunk["questionAsked"] = history_chunk["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedVoteReceived"] = history_chunk["event"].map({"AcceptedAnswerVote": 1}).fillna(0).astype(
            np.int8)
        history_chunk["acceptedAnswerReceived"] = history_chunk["event"].map({"AcceptedAnswerReceived": 1}).fillna(
            0).astype(np.int8)
        history_chunk["acceptedAnswerPosted"] = history_chunk["event"].map({"AcceptedAnswerPosted": 1}).fillna(
            0).astype(np.int8)
        history_chunk["answerPosted"] = history_chunk["event"].map({"AnswerProvided": 1}).fillna(0).astype(np.int8)

        print("Building JIT-compatible user history cache for this chunk...")
        user_histories_jit = {}
        for user_id, user_data in history_chunk.groupby("user_id"):
            user_histories_jit[user_id] = prepare_user_history_for_jit(user_data)

        user_histories_pandas = {}
        for user_id, user_data in history_chunk.groupby("user_id"):
            user_histories_pandas[user_id] = user_data

        reciprocity_status = calculate_reciprocity_activation(user_histories_pandas)

        del history_chunk
        del user_data_chunk
        del user_histories_pandas

        non_history_chunk = non_history_chunk.sort_values(["user_id", "timestamp"])

        results = []

        print(f"Processing {len(non_history_chunk):,} rows with JIT...")
        for _, row in non_history_chunk.iterrows():
            user_id = row["user_id"]
            target_time = row["timestamp"].to_numpy().astype(np.int64)

            if user_id in user_histories_jit:
                user_history_arrays = user_histories_jit[user_id]

                (
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
                ) = calculate_metrics_jit(*user_history_arrays, target_time)

                initialExperienceReceiving, initialExperienceGiving = convert_experience_code_to_string(
                    initialExperienceReceiving_code, initialExperienceGiving_code
                )
            else:
                questions_asked_at = help_provided_at = help_provided_ever = 0
                accepted_answers_received_at = accepted_votes_received_at = 0
                questions_asked_30d = help_provided_30d = 0
                accepted_answers_received_30d = accepted_votes_received_30d = accepted_answers_posted_30d = 0
                questions_asked_14d = help_provided_14d = 0
                accepted_answers_received_14d = accepted_votes_received_14d = accepted_answers_posted_14d = 0
                questions_asked_7d = help_provided_7d = 0
                accepted_answers_received_7d = accepted_votes_received_7d = accepted_answers_posted_7d = 0
                questions_asked_3d = help_provided_3d = 0
                accepted_answers_received_3d = accepted_votes_received_3d = accepted_answers_posted_3d = 0
                accepted_answers_posted_at = 0
                initialExperienceReceiving = "no help seeked"
                initialExperienceGiving = "no help attempted"
                time_since_first_activity_days = 0.0

            if user_id in reciprocity_status:
                activated, activation_time = reciprocity_status[user_id]
                if pd.isna(activation_time):
                    is_activated_at_time = 0
                    activation_timestamp = pd.NaT
                else:
                    is_activated_at_time = 1 if row["timestamp"] > activation_time else 0
                    activation_timestamp = activation_time if is_activated_at_time else pd.NaT
            else:
                is_activated_at_time = 0
                activation_timestamp = pd.NaT

            result = {
                "userId": user_id,
                "timestamp": row["timestamp"],
                "event": row["event"],
                "answerId": row.get("answer_id", None),
                "questionId": row.get("question_id", None),
                "isBounty": row.get("is_bounty", 0),
                "bountyAmount": row.get("bounty_amount", 0),
                "answerSequence": row.get("answer_sequence", None),

                "numQuestionsAskedAT": questions_asked_at,
                "numHelpProvidedAT": help_provided_at,
                "helpProvidedEver": help_provided_ever,
                "numAcceptedVotesReceivedAT": accepted_votes_received_at,
                "numAcceptedAnswersReceivedAT": accepted_answers_received_at,
                "numAcceptedAnswersPostedAT": accepted_answers_posted_at,

                "numQuestionsAsked30D": questions_asked_30d,
                "numHelpProvided30D": help_provided_30d,
                "numAcceptedVotesReceived30D": accepted_votes_received_30d,
                "numAcceptedAnswersReceived30D": accepted_answers_received_30d,
                "numAcceptedAnswersPosted30D": accepted_answers_posted_30d,

                "numQuestionsAsked14D": questions_asked_14d,
                "numHelpProvided14D": help_provided_14d,
                "numAcceptedVotesReceived14D": accepted_votes_received_14d,
                "numAcceptedAnswersReceived14D": accepted_answers_received_14d,
                "numAcceptedAnswersPosted14D": accepted_answers_posted_14d,

                "numQuestionsAsked7D": questions_asked_7d,
                "numHelpProvided7D": help_provided_7d,
                "numAcceptedVotesReceived7D": accepted_votes_received_7d,
                "numAcceptedAnswersReceived7D": accepted_answers_received_7d,
                "numAcceptedAnswersPosted7D": accepted_answers_posted_7d,

                "numQuestionsAsked3D": questions_asked_3d,
                "numHelpProvided3D": help_provided_3d,
                "numAcceptedVotesReceived3D": accepted_votes_received_3d,
                "numAcceptedAnswersReceived3D": accepted_answers_received_3d,
                "numAcceptedAnswersPosted3D": accepted_answers_posted_3d,

                "initialExperienceReceiving": initialExperienceReceiving,
                "initialExperienceGiving": initialExperienceGiving,
                "reciprocityActivated": is_activated_at_time,
                "reciprocityActivatedTimestamp": activation_timestamp,
                "timeSinceFirstActivityDays": time_since_first_activity_days
            }

            results.append(result)

        chunk_output_df = pd.DataFrame(results)

        chunk_output_df["receivedAcceptedAnswerEver"] = (chunk_output_df["numAcceptedAnswersReceivedAT"] > 0).astype(
            int)
        chunk_output_df["receivedAcceptedVoteEver"] = (chunk_output_df["numAcceptedVotesReceivedAT"] > 0).astype(int)
        chunk_output_df["month"] = chunk_output_df["timestamp"].dt.month
        chunk_output_df["year"] = chunk_output_df["timestamp"].dt.year

        if first_chunk:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            chunk_output_df.to_parquet(output_file, index=False)
            first_chunk = False
        else:
            chunk_output_df.to_parquet(
                output_file,
                index=False,
                append=True,
                engine="fastparquet"
            )

        processed_rows += len(chunk_output_df)
        print(f"Processed {len(chunk_output_df):,} rows in this chunk")
        print(
            f"Total processed so far: {processed_rows:,} / {total_non_history:,} ({processed_rows / total_non_history:.1%})")

        chunk_progress.update(len(chunk_user_ids))

        del user_histories_jit
        del non_history_chunk
        del chunk_output_df
        del reciprocity_status
        gc.collect()

    chunk_progress.close()

    print(f"\nCompleted processing {processed_rows:,} total rows")
    print(f"Final output saved to {output_file}")


if __name__ == "__main__":
    input_file = "../data/input/user_answers_bounty_dataset.parquet"
    output_file = "../data/study_datasets/user_answers_bounty_processed.parquet"

    process_bounty_dataset(
        input_file=input_file,
        output_file=output_file,
        chunk_size=100000
    )