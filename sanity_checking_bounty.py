import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings
import duckdb

warnings.filterwarnings('ignore')


def sanity_check_user_question_pair(processed_df, raw_df, user_id, question_id, case_description=None):
    """
    Performs sanity checks on the processing of a specific user-question pair.

    Args:
        processed_df: The final processed dataframe
        raw_df: The original raw dataframe
        user_id: The user ID to check
        question_id: The question ID to check
        case_description: Optional description of the test case

    Returns:
        dict: Results of various checks
    """
    if case_description:
        print(f"\n=== {case_description} ===")
    print(f"Performing sanity check for user_id={user_id}, question_id={question_id}")

    # Get the processed rows for this user-question pair
    processed_rows = processed_df[(processed_df['user_id'] == user_id) &
                                  (processed_df['question_id'] == question_id)].copy()

    if len(processed_rows) == 0:
        print("❌ No processed rows found for this user-question pair")
        return

    print(f"Found {len(processed_rows)} processed rows")

    # Get relevant raw data
    raw_user_history = raw_df[(raw_df['user_id'] == user_id) & (raw_df['is_history'] == 1)].copy()
    raw_user_current = raw_df[(raw_df['user_id'] == user_id) &
                              (raw_df['question_id'] == question_id) &
                              (raw_df['is_history'] == 0)].copy()

    # Check phase coverage
    has_before = processed_rows['after_bounty'].eq(0).any()
    has_after = processed_rows['after_bounty'].eq(1).any()
    print(f"✓ Has before phase: {has_before}")
    print(f"✓ Has after phase: {has_after}")

    # Check metrics for each phase
    for phase, phase_name in [(0, "before bounty"), (1, "after bounty")]:
        phase_row = processed_rows[processed_rows['after_bounty'] == phase]
        if len(phase_row) == 0:
            print(f"❌ Missing {phase_name} phase")
            continue

        phase_row = phase_row.iloc[0]
        timestamp = phase_row['timestamp'] if 'timestamp' in phase_row else None

        print(f"\nChecking {phase_name} phase (timestamp: {timestamp}):")

        # Recalculate metrics at this timestamp
        history_before_timestamp = raw_user_history[
            raw_user_history['effective_timestamp'] < timestamp
            ] if timestamp is not None else pd.DataFrame()

        # Count metrics
        questions_asked = history_before_timestamp['questionAsked'].sum()
        answers_received = history_before_timestamp['acceptedAnswer'].sum()
        answers_provided = history_before_timestamp['answer'].sum()
        accepted_answers_provided = history_before_timestamp['acceptedAnswerProvided'].sum()
        has_provided = answers_provided > 0

        # Check counts
        checks = {
            'num_questions_asked': (phase_row['num_questions_asked'] == questions_asked,
                                    f"Expected: {questions_asked}, Got: {phase_row['num_questions_asked']}"),
            'num_accepted_answers_received': (phase_row['num_accepted_answers_received'] == answers_received,
                                              f"Expected: {answers_received}, Got: {phase_row['num_accepted_answers_received']}"),
            'num_answers_provided': (phase_row['num_answers_provided'] == answers_provided,
                                     f"Expected: {answers_provided}, Got: {phase_row['num_answers_provided']}"),
            'num_accepted_answers_provided': (phase_row['num_accepted_answers_provided'] == accepted_answers_provided,
                                              f"Expected: {accepted_answers_provided}, Got: {phase_row['num_accepted_answers_provided']}"),
            'has_provided_answers': (phase_row['has_provided_answers'] == int(has_provided),
                                     f"Expected: {int(has_provided)}, Got: {phase_row['has_provided_answers']}")
        }

        for metric, (is_correct, message) in checks.items():
            print(f"{'✓' if is_correct else '❌'} {metric}: {message}")

        # Check reciprocity activation
        if 'reciprocity_activated' in phase_row and 'reciprocity_activated_timestamp' in phase_row:
            # Find first answer
            first_answer = raw_user_history[raw_user_history['event_history'] == 'Answer'].sort_values(
                'effective_timestamp').iloc[0] if len(
                raw_user_history[raw_user_history['event_history'] == 'Answer']) > 0 else None

            if first_answer is not None:
                first_answer_time = first_answer['effective_timestamp']

                # Check for accepted answers 7 days before
                accepted_answers_7d_before = raw_user_history[
                    (raw_user_history['event_history'] == 'AcceptedAnswer') &
                    (raw_user_history['effective_timestamp'] < first_answer_time) &
                    (raw_user_history['effective_timestamp'] > first_answer_time - pd.Timedelta(days=7))
                    ]

                expected_activation = int(len(accepted_answers_7d_before) > 0 and timestamp >= first_answer_time)
                is_activation_correct = phase_row['reciprocity_activated'] == expected_activation

                print(f"{'✓' if is_activation_correct else '❌'} reciprocity_activated: " +
                      f"Expected: {expected_activation}, Got: {phase_row['reciprocity_activated']}")

                if not is_activation_correct:
                    print(f"  First answer time: {first_answer_time}")
                    print(f"  Accepted answers 7d before: {len(accepted_answers_7d_before)}")
                    print(f"  Current timestamp: {timestamp}")

                # Check timestamp
                if expected_activation == 1:
                    expected_timestamp = first_answer_time
                    is_timestamp_correct = pd.isna(phase_row['reciprocity_activated_timestamp']) == pd.isna(
                        expected_timestamp)
                    if not pd.isna(phase_row['reciprocity_activated_timestamp']) and not pd.isna(expected_timestamp):
                        is_timestamp_correct = phase_row['reciprocity_activated_timestamp'] == expected_timestamp

                    print(f"{'✓' if is_timestamp_correct else '❌'} reciprocity_activated_timestamp: " +
                          f"Expected: {expected_timestamp}, Got: {phase_row['reciprocity_activated_timestamp']}")
            else:
                print("ℹ️ No answers provided by user, reciprocity should be 0")
                print(f"{'✓' if phase_row['reciprocity_activated'] == 0 else '❌'} reciprocity_activated: " +
                      f"Expected: 0, Got: {phase_row['reciprocity_activated']}")

        # Check numHelped
        answered_in_phase = raw_user_current[raw_user_current['after_bounty'] == phase]
        expected_numHelped = len(answered_in_phase)
        is_numHelped_correct = phase_row['numHelped'] == expected_numHelped

        print(f"{'✓' if is_numHelped_correct else '❌'} numHelped: " +
              f"Expected: {expected_numHelped}, Got: {phase_row['numHelped']}")

        # Check derived metrics
        hasHelped = int(expected_numHelped > 0)
        is_hasHelped_correct = phase_row['hasHelped'] == hasHelped

        print(f"{'✓' if is_hasHelped_correct else '❌'} hasHelped: " +
              f"Expected: {hasHelped}, Got: {phase_row['hasHelped']}")

    return processed_rows


def prepare_raw_data(raw_df):
    """Prepare raw data for sanity checking by calculating effective_timestamp"""
    print("Preparing raw data...")

    # Create flag columns based on event_history if they don't exist
    if 'event_history' in raw_df.columns:
        raw_df['questionAsked'] = (raw_df['event_history'] == 'Question').astype(np.int32)
        raw_df['acceptedAnswer'] = (raw_df['event_history'] == 'AcceptedAnswer').astype(np.int32)
        raw_df['answer'] = (raw_df['event_history'] == 'Answer').astype(np.int32)
        raw_df['acceptedAnswerProvided'] = (raw_df['event_history'] == 'AcceptedAnswerProvided').astype(np.int32)
    else:
        print("Warning: 'event_history' column not found in raw data")
        for col in ['questionAsked', 'acceptedAnswer', 'answer', 'acceptedAnswerProvided']:
            raw_df[col] = 0

    # Ensure timestamps are datetime objects
    for col in ['timestamp', 'acceptance_date']:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in raw data")
            raw_df[col] = pd.NaT

    if 'is_history' not in raw_df.columns:
        print("Warning: 'is_history' column not found in raw data")
        raw_df['is_history'] = 0

    # Calculate effective_timestamp
    mask = (raw_df['acceptedAnswerProvided'] == 1) & (raw_df['acceptance_date'].notna())
    raw_df.loc[mask, 'effective_timestamp'] = raw_df.loc[mask, 'acceptance_date'] + pd.Timedelta(days=1)
    raw_df.loc[~mask, 'effective_timestamp'] = raw_df.loc[~mask, 'timestamp']

    return raw_df


def find_test_cases(processed_file_path):
    """Find suitable test cases for different scenarios using DuckDB"""
    print("Finding specific test cases using DuckDB...")

    # Create a connection
    con = duckdb.connect(database=':memory:')
    test_cases = {}

    # Register the parquet file as a view
    con.execute(f"CREATE VIEW processed AS SELECT * FROM '{processed_file_path}'")

    # Find users with answers in both periods
    helped_both_periods_query = """
    SELECT 
        t1.user_id, t1.question_id
    FROM 
        (SELECT user_id, question_id, after_bounty, "numHelped" 
         FROM processed 
         WHERE after_bounty = 0 AND "numHelped" > 0) t1
    JOIN 
        (SELECT user_id, question_id, after_bounty, "numHelped" 
         FROM processed 
         WHERE after_bounty = 1 AND "numHelped" > 0) t2
    ON 
        t1.user_id = t2.user_id AND t1.question_id = t2.question_id
    LIMIT 1
    """
    result = con.execute(helped_both_periods_query).fetchall()
    if result:
        test_cases['helped_both_periods'] = (result[0][0], result[0][1])
        print(
            f"Found case with help in both periods: user_id={test_cases['helped_both_periods'][0]}, question_id={test_cases['helped_both_periods'][1]}")

    # Find users who helped in only one period
    helped_one_period_query = """
    SELECT 
        t1.user_id, t1.question_id
    FROM 
        (SELECT user_id, question_id, "numHelped" 
         FROM processed 
         WHERE after_bounty = 0) t1
    JOIN 
        (SELECT user_id, question_id, "numHelped" 
         FROM processed 
         WHERE after_bounty = 1) t2
    ON 
        t1.user_id = t2.user_id AND t1.question_id = t2.question_id
    WHERE 
        (t1."numHelped" > 0 AND t2."numHelped" = 0) OR
        (t1."numHelped" = 0 AND t2."numHelped" > 0)
    LIMIT 1
    """
    result = con.execute(helped_one_period_query).fetchall()
    if result:
        test_cases['helped_one_period'] = (result[0][0], result[0][1])
        print(
            f"Found case with help in only one period: user_id={test_cases['helped_one_period'][0]}, question_id={test_cases['helped_one_period'][1]}")

    # Users with reciprocity activated
    reciprocity_activated_query = """
    SELECT 
        user_id, question_id
    FROM 
        processed 
    WHERE 
        reciprocity_activated = 1
    LIMIT 1
    """
    result = con.execute(reciprocity_activated_query).fetchall()
    if result:
        test_cases['reciprocity_activated'] = (result[0][0], result[0][1])
        print(
            f"Found case with reciprocity activated: user_id={test_cases['reciprocity_activated'][0]}, question_id={test_cases['reciprocity_activated'][1]}")

    # Users with no reciprocity activation
    reciprocity_not_activated_query = """
    SELECT 
        user_id, question_id
    FROM 
        processed 
    WHERE 
        reciprocity_activated = 0
    LIMIT 1
    """
    result = con.execute(reciprocity_not_activated_query).fetchall()
    if result:
        test_cases['reciprocity_not_activated'] = (result[0][0], result[0][1])
        print(
            f"Found case with no reciprocity activation: user_id={test_cases['reciprocity_not_activated'][0]}, question_id={test_cases['reciprocity_not_activated'][1]}")

    # Close connection
    con.close()

    return test_cases


def main():
    """Main function to run the sanity checks with predefined test cases"""
    # Default paths - change these if your files are stored elsewhere
    processed_path = r"./03_processed_datasets/processed_bounty_dataset.parquet"
    raw_path = r"./02_raw_datasets/bounty_raw_dataset.parquet"

    # Check if files exist, otherwise use command line arguments
    if not os.path.exists(processed_path) or not os.path.exists(raw_path):
        parser = argparse.ArgumentParser(description='Sanity check bounty dataset processing')
        parser.add_argument('--processed', type=str, required=True,
                            help='Path to processed dataset parquet file')
        parser.add_argument('--raw', type=str, required=True,
                            help='Path to raw dataset parquet file')
        args = parser.parse_args()
        processed_path = args.processed
        raw_path = args.raw

    # Find test cases using DuckDB (faster)
    test_cases = find_test_cases(processed_path)

    # Load dataframes after finding test cases
    print(f"Loading processed dataset from {processed_path}")
    processed_df = pd.read_parquet(processed_path)

    print(f"Loading raw dataset from {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    raw_df = prepare_raw_data(raw_df)

    if test_cases:
        # Case 1: User provided help in both periods
        if 'helped_both_periods' in test_cases:
            print("\n" + "=" * 80)
            sanity_check_user_question_pair(
                processed_df, raw_df,
                test_cases['helped_both_periods'][0],
                test_cases['helped_both_periods'][1],
                "CASE 1: User provided help in both periods"
            )

        # Case 2: User provided help in only one period
        if 'helped_one_period' in test_cases:
            print("\n" + "=" * 80)
            sanity_check_user_question_pair(
                processed_df, raw_df,
                test_cases['helped_one_period'][0],
                test_cases['helped_one_period'][1],
                "CASE 2: User provided help in only one period"
            )

        # Case 3: User with reciprocity activated
        if 'reciprocity_activated' in test_cases:
            print("\n" + "=" * 80)
            sanity_check_user_question_pair(
                processed_df, raw_df,
                test_cases['reciprocity_activated'][0],
                test_cases['reciprocity_activated'][1],
                "CASE 3: User with reciprocity activated"
            )

        # Case 4: User with no reciprocity activation
        if 'reciprocity_not_activated' in test_cases:
            print("\n" + "=" * 80)
            sanity_check_user_question_pair(
                processed_df, raw_df,
                test_cases['reciprocity_not_activated'][0],
                test_cases['reciprocity_not_activated'][1],
                "CASE 4: User with no reciprocity activation"
            )
    else:
        print("No suitable test cases found. Checking random samples instead.")
        # Check random samples as fallback
        unique_pairs = processed_df[['user_id', 'question_id']].drop_duplicates()
        if len(unique_pairs) > 0:
            sample_pairs = unique_pairs.sample(min(3, len(unique_pairs)))
            for i, (_, pair) in enumerate(sample_pairs.iterrows()):
                print("\n" + "=" * 80)
                sanity_check_user_question_pair(
                    processed_df, raw_df,
                    pair['user_id'], pair['question_id'],
                    f"RANDOM CASE {i + 1}"
                )
        else:
            print("No user-question pairs found in processed data")

    print("\n" + "=" * 80)
    print("Sanity check complete!")


if __name__ == "__main__":
    main()