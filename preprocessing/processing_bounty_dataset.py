import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba
from numba import jit


@jit(nopython=True)
def calculate_all_metrics(
        timestamps,
        question_asked,
        accepted_answer,
        answer,
        accepted_vote_received,
        accepted_answer_received,
        answers_received,
        accepted_answer_posted,
        target_time_ns
):
    """
    JIT-compiled function to calculate all metrics for all time windows in a single pass.
    """
    # Define time window cutoffs in nanoseconds
    day_ns = 24 * 60 * 60 * 1_000_000_000
    cutoff_30d = target_time_ns - (30 * day_ns)
    cutoff_14d = target_time_ns - (14 * day_ns)
    cutoff_7d = target_time_ns - (7 * day_ns)
    cutoff_3d = target_time_ns - (3 * day_ns)

    # Initialize counters for all windows
    qa_at = hr_at = hp_at = avr_at = aar_at = ar_at = aap_at = 0
    qa_30d = hr_30d = hp_30d = avr_30d = aar_30d = ar_30d = aap_30d = 0
    qa_14d = hr_14d = hp_14d = avr_14d = aar_14d = ar_14d = aap_14d = 0
    qa_7d = hr_7d = hp_7d = avr_7d = aar_7d = ar_7d = aap_7d = 0
    qa_3d = hr_3d = hp_3d = avr_3d = aar_3d = ar_3d = aap_3d = 0

    # Calculate metrics in a single pass through the data
    for i in range(len(timestamps)):
        ts = timestamps[i]

        if ts < target_time_ns:  # All-time
            qa_at += question_asked[i]
            hr_at += accepted_answer[i]
            hp_at += answer[i]
            avr_at += accepted_vote_received[i]
            aar_at += accepted_answer_received[i]
            ar_at += answers_received[i]
            aap_at += accepted_answer_posted[i]

            if ts >= cutoff_30d:  # 30-day window
                qa_30d += question_asked[i]
                hr_30d += accepted_answer[i]
                hp_30d += answer[i]
                avr_30d += accepted_vote_received[i]
                aar_30d += accepted_answer_received[i]
                ar_30d += answers_received[i]
                aap_30d += accepted_answer_posted[i]

                if ts >= cutoff_14d:  # 14-day window
                    qa_14d += question_asked[i]
                    hr_14d += accepted_answer[i]
                    hp_14d += answer[i]
                    avr_14d += accepted_vote_received[i]
                    aar_14d += accepted_answer_received[i]
                    ar_14d += answers_received[i]
                    aap_14d += accepted_answer_posted[i]

                    if ts >= cutoff_7d:  # 7-day window
                        qa_7d += question_asked[i]
                        hr_7d += accepted_answer[i]
                        hp_7d += answer[i]
                        avr_7d += accepted_vote_received[i]
                        aar_7d += accepted_answer_received[i]
                        ar_7d += answers_received[i]
                        aap_7d += accepted_answer_posted[i]

                        if ts >= cutoff_3d:  # 3-day window
                            qa_3d += question_asked[i]
                            hr_3d += accepted_answer[i]
                            hp_3d += answer[i]
                            avr_3d += accepted_vote_received[i]
                            aar_3d += accepted_answer_received[i]
                            ar_3d += answers_received[i]
                            aap_3d += accepted_answer_posted[i]

    # Return all metrics
    return (
        qa_at, hr_at, hp_at, avr_at, aar_at, ar_at, aap_at,
        qa_30d, hr_30d, hp_30d, avr_30d, aar_30d, ar_30d, aap_30d,
        qa_14d, hr_14d, hp_14d, avr_14d, aar_14d, ar_14d, aap_14d,
        qa_7d, hr_7d, hp_7d, avr_7d, aar_7d, ar_7d, aap_7d,
        qa_3d, hr_3d, hp_3d, avr_3d, aar_3d, ar_3d, aap_3d
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
        # Get the first provided answer (Answer)
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


def process_bounty_dataset(input_file: str, output_file: str, chunk_size: int = 100000) -> None:
    """
    Process the user_answers_bounty_dataset.parquet file to calculate metrics for each user.
    Uses a chunking approach to handle large datasets efficiently, building history only for each chunk.
    Fixed Numba implementation for correct calculations.
    """
    print(f"\n=== Processing {input_file} ===")

    # First, get all unique user IDs
    print("Reading unique user IDs...")
    all_user_ids = pd.read_parquet(input_file, columns=["user_id"])["user_id"].unique()
    print(f"Found {len(all_user_ids):,} unique users")

    # Get the total number of rows to estimate progress
    total_rows = pd.read_parquet(input_file, columns=["is_history"])
    total_non_history = len(total_rows[total_rows["is_history"] == 0])
    print(f"Total non-history rows: {total_non_history:,}")
    del total_rows

    # Set up for chunked processing
    first_chunk = True
    processed_rows = 0

    # Determine the list of columns to read for data
    columns_to_read = [
        "event_id", "user_id", "timestamp", "event", "answer_id",
        "question_id", "is_bounty", "bounty_amount", "answer_sequence", "is_history"
    ]

    # Create overall progress bar for chunks
    chunk_progress = tqdm(
        total=len(all_user_ids),
        desc="Overall progress",
        unit="users",
        position=0
    )

    # Process in chunks
    for chunk_start in range(0, len(all_user_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_user_ids))
        chunk_user_ids = all_user_ids[chunk_start:chunk_end]

        print(f"\nProcessing users {chunk_start:,} to {chunk_end:,} ({len(chunk_user_ids):,} users)")

        # Load ALL data for this chunk of users (both history and non-history)
        chunk_filter = [("user_id", "in", list(chunk_user_ids))]
        user_data_chunk = pd.read_parquet(input_file, filters=chunk_filter, columns=columns_to_read)

        print(f"Loaded {len(user_data_chunk):,} total rows for this chunk of users")

        # Ensure the timestamp is in datetime format
        user_data_chunk["timestamp"] = pd.to_datetime(user_data_chunk["timestamp"], errors="coerce")

        # Split into history and non-history
        history_chunk = user_data_chunk[user_data_chunk["is_history"] == 1].copy()
        non_history_chunk = user_data_chunk[user_data_chunk["is_history"] == 0].copy()

        print(f"History rows: {len(history_chunk):,}")
        print(f"Non-history rows: {len(non_history_chunk):,}")

        # Skip if no non-history data to process
        if len(non_history_chunk) == 0:
            chunk_progress.update(len(chunk_user_ids))
            print("No non-history data in this chunk, skipping...")
            continue

        # Create numeric flags from event type for history data - UPDATED to match new event names
        history_chunk["questionAsked"] = history_chunk["event"].map({"Question": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedAnswer"] = history_chunk["event"].map({"AcceptedAnswer": 1}).fillna(0).astype(np.int8)
        history_chunk["answer"] = history_chunk["event"].map({"Answer": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedVoteReceived"] = history_chunk["event"].map({"AcceptedAnswerVote": 1}).fillna(0).astype(
            np.int8)
        history_chunk["acceptedAnswerReceived"] = history_chunk["event"].map({"AcceptedAnswerReceived": 1}).fillna(
            0).astype(np.int8)
        history_chunk["answerReceived"] = history_chunk["event"].map({"AnswerReceived": 1}).fillna(0).astype(np.int8)
        history_chunk["acceptedAnswerPosted"] = history_chunk["event"].map({"AcceptedAnswerPosted": 1}).fillna(
            0).astype(np.int8)

        # Create a dictionary of user histories for this chunk
        print("Building user history cache for this chunk...")
        user_histories = {}
        for user_id, user_data in tqdm(history_chunk.groupby("user_id"), desc="Preprocessing users"):
            # Sort by timestamp
            user_data = user_data.sort_values("timestamp")
            user_histories[user_id] = user_data

        # Calculate reciprocity activation
        reciprocity_status = calculate_reciprocity_activation(user_histories)

        # Clear dataframes to free memory
        del history_chunk
        del user_data_chunk

        # Process rows by user_id for better cache efficiency
        non_history_chunk = non_history_chunk.sort_values(["user_id", "timestamp"])

        # Process the chunk
        results = []
        current_user_id = None
        user_history_df = None

        # Create a progress bar for this chunk
        chunk_row_progress = tqdm(
            total=len(non_history_chunk),
            desc="Processing rows",
            unit="rows",
            position=1,
            leave=False
        )

        # Pre-compile the JIT function by running it once with sample data
        dummy_timestamps = np.array([1000000000, 2000000000], dtype=np.int64)
        dummy_flags = np.array([1, 1], dtype=np.int8)
        dummy_target = np.int64(3000000000)
        calculate_all_metrics(
            dummy_timestamps, dummy_flags, dummy_flags, dummy_flags,
            dummy_flags, dummy_flags, dummy_flags, dummy_flags,
            dummy_target
        )
        print("JIT compilation completed")

        for _, row in non_history_chunk.iterrows():
            user_id = row["user_id"]
            target_time = row["timestamp"]

            # Fix timestamp conversion - use consistent method
            target_time_ns = int(target_time.timestamp() * 1_000_000_000)

            # Only fetch user history once for each user (caching optimization)
            if user_id != current_user_id:
                current_user_id = user_id
                if user_id in user_histories:
                    user_history_df = user_histories[user_id]

                    # Fix timestamp conversion
                    timestamps_array = np.array([
                        int(ts.timestamp() * 1_000_000_000)
                        for ts in user_history_df["timestamp"]
                    ], dtype=np.int64)

                    question_asked_array = user_history_df["questionAsked"].to_numpy()
                    accepted_answer_array = user_history_df["acceptedAnswer"].to_numpy()
                    answer_array = user_history_df["answer"].to_numpy()
                    accepted_vote_received_array = user_history_df["acceptedVoteReceived"].to_numpy()
                    accepted_answer_received_array = user_history_df["acceptedAnswerReceived"].to_numpy()
                    answers_received_array = user_history_df["answerReceived"].to_numpy()
                    accepted_answer_posted_array = user_history_df["acceptedAnswerPosted"].to_numpy()
                else:
                    # Create empty arrays if no history
                    timestamps_array = np.array([], dtype=np.int64)
                    question_asked_array = np.array([], dtype=np.int8)
                    accepted_answer_array = np.array([], dtype=np.int8)
                    answer_array = np.array([], dtype=np.int8)
                    accepted_vote_received_array = np.array([], dtype=np.int8)
                    accepted_answer_received_array = np.array([], dtype=np.int8)
                    answers_received_array = np.array([], dtype=np.int8)
                    accepted_answer_posted_array = np.array([], dtype=np.int8)

            # Calculate all metrics using Numba-accelerated function
            (
                questions_asked_at, help_received_at, help_provided_at, accepted_vote_received_at,
                accepted_answer_received_at, answers_received_at, accepted_answer_posted_at,
                questions_asked_30d, help_received_30d, help_provided_30d, accepted_vote_received_30d,
                accepted_answer_received_30d, answers_received_30d, accepted_answer_posted_30d,
                questions_asked_14d, help_received_14d, help_provided_14d, accepted_vote_received_14d,
                accepted_answer_received_14d, answers_received_14d, accepted_answer_posted_14d,
                questions_asked_7d, help_received_7d, help_provided_7d, accepted_vote_received_7d,
                accepted_answer_received_7d, answers_received_7d, accepted_answer_posted_7d,
                questions_asked_3d, help_received_3d, help_provided_3d, accepted_vote_received_3d,
                accepted_answer_received_3d, answers_received_3d, accepted_answer_posted_3d,
            ) = calculate_all_metrics(
                timestamps_array,
                question_asked_array,
                accepted_answer_array,
                answer_array,
                accepted_vote_received_array,
                accepted_answer_received_array,
                answers_received_array,
                accepted_answer_posted_array,
                target_time_ns
            )

            # Calculate help_provided_ever flag
            help_provided_ever = 1 if help_provided_at > 0 else 0

            # Calculate initial experience receiving
            initial_experience_receiving = "unknown"
            if questions_asked_at == 0:
                initial_experience_receiving = "no help seeked"
            elif questions_asked_at > 0 and accepted_answer_received_at == 0:
                initial_experience_receiving = "help seeked"
            elif questions_asked_at > 0 and accepted_answer_received_at > 0:
                initial_experience_receiving = "help received"

            # Calculate initial experience giving
            initial_experience_giving = "unknown"
            if help_provided_at == 0:
                initial_experience_giving = "no help attempted"
            elif help_provided_at > 0 and accepted_answer_posted_at == 0:
                initial_experience_giving = "help attempted"
            elif accepted_answer_posted_at > 0:
                initial_experience_giving = "helped"

            # Get reciprocity activation status
            if user_id in reciprocity_status:
                activated, activation_time = reciprocity_status[user_id]
                if pd.isna(activation_time):
                    is_activated_at_time = 0
                    activation_timestamp = pd.NaT
                else:
                    is_activated_at_time = 1 if target_time >= activation_time else 0
                    activation_timestamp = activation_time if is_activated_at_time else pd.NaT
            else:
                is_activated_at_time = 0
                activation_timestamp = pd.NaT

            # Create result
            result = {
                "eventId": row["event_id"],
                "userId": user_id,
                "timestamp": target_time,
                "event": row["event"],
                "answerId": row.get("answer_id", None),
                "questionId": row.get("question_id", None),
                "isBounty": row.get("is_bounty", 0),
                "bountyAmount": row.get("bounty_amount", 0),
                "answerSequence": row.get("answer_sequence", None),

                "numQuestionsAskedAT": questions_asked_at,
                "numHelpReceivedAT": help_received_at,
                "numHelpProvidedAT": help_provided_at,
                "helpProvidedEver": help_provided_ever,  # Changed from numHelpProvidedEver
                "numAcceptedVotesReceivedAT": accepted_vote_received_at,  # Changed to plural
                "numAcceptedAnswersReceivedAT": accepted_answer_received_at,  # Changed to plural
                "numAnswersReceivedAT": answers_received_at,
                "numAcceptedAnswersPostedAT": accepted_answer_posted_at,

                "numQuestionsAsked30D": questions_asked_30d,
                "numHelpReceived30D": help_received_30d,
                "numHelpProvided30D": help_provided_30d,
                "numAcceptedVotesReceived30D": accepted_vote_received_30d,  # Changed to plural
                "numAcceptedAnswersReceived30D": accepted_answer_received_30d,  # Changed to plural
                "numAnswersReceived30D": answers_received_30d,
                "numAcceptedAnswersPosted30D": accepted_answer_posted_30d,

                "numQuestionsAsked14D": questions_asked_14d,
                "numHelpReceived14D": help_received_14d,
                "numHelpProvided14D": help_provided_14d,
                "numAcceptedVotesReceived14D": accepted_vote_received_14d,  # Changed to plural
                "numAcceptedAnswersReceived14D": accepted_answer_received_14d,  # Changed to plural
                "numAnswersReceived14D": answers_received_14d,
                "numAcceptedAnswersPosted14D": accepted_answer_posted_14d,

                "numQuestionsAsked7D": questions_asked_7d,
                "numHelpReceived7D": help_received_7d,
                "numHelpProvided7D": help_provided_7d,
                "numAcceptedVotesReceived7D": accepted_vote_received_7d,  # Changed to plural
                "numAcceptedAnswersReceived7D": accepted_answer_received_7d,  # Changed to plural
                "numAnswersReceived7D": answers_received_7d,
                "numAcceptedAnswersPosted7D": accepted_answer_posted_7d,

                "numQuestionsAsked3D": questions_asked_3d,
                "numHelpReceived3D": help_received_3d,
                "numHelpProvided3D": help_provided_3d,
                "numAcceptedVotesReceived3D": accepted_vote_received_3d,  # Changed to plural
                "numAcceptedAnswersReceived3D": accepted_answer_received_3d,  # Changed to plural
                "numAnswersReceived3D": answers_received_3d,
                "numAcceptedAnswersPosted3D": accepted_answer_posted_3d,

                "initialExperienceReceiving": initial_experience_receiving,
                "initialExperienceGiving": initial_experience_giving,

                # Reciprocity activation
                "reciprocityActivated": is_activated_at_time,
                "reciprocityActivatedTimestamp": activation_timestamp,
            }

            results.append(result)
            chunk_row_progress.update(1)

        # Close the row progress bar
        chunk_row_progress.close()

        # Create DataFrame from results
        chunk_output_df = pd.DataFrame(results)

        chunk_output_df["receivedAcceptedAnswerEver"] = (chunk_output_df["numAcceptedAnswersReceivedAT"] > 0).astype(
            int)
        chunk_output_df["receivedAcceptedVoteEver"] = (chunk_output_df["numAcceptedVotesReceivedAT"] > 0).astype(int)
        chunk_output_df["month"] = chunk_output_df["timestamp"].dt.month
        chunk_output_df["year"] = chunk_output_df["timestamp"].dt.year

        # Save or append to output file
        if first_chunk:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            chunk_output_df.to_parquet(output_file, index=False)
            first_chunk = False
        else:
            # Append to existing file
            chunk_output_df.to_parquet(
                output_file,
                index=False,
                append=True,
                engine="fastparquet"  # fastparquet supports append mode
            )

        processed_rows += len(chunk_output_df)
        print(f"Processed {len(chunk_output_df):,} rows in this chunk")
        print(
            f"Total processed so far: {processed_rows:,} / {total_non_history:,} ({processed_rows / total_non_history:.1%})")

        # Update main progress bar
        chunk_progress.update(len(chunk_user_ids))

        # Clear memory
        del user_histories
        del non_history_chunk
        del chunk_output_df
        del reciprocity_status

    # Close the main progress bar
    chunk_progress.close()

    print(f"\nCompleted processing {processed_rows:,} total rows")
    print(f"Final output saved to {output_file}")


if __name__ == "__main__":
    input_file = "../data/input/user_answers_bounty_dataset.parquet"
    output_file = "../study_datasets/user_answers_bounty_processed.parquet"

    process_bounty_dataset(
        input_file=input_file,
        output_file=output_file,
        chunk_size=100000
    )