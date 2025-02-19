import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def process_bounty_dataset(file_path):
    """
    Process bounty dataset to calculate key metrics for analysis.

    Key processing steps:
    1. Load dataset with original column names
    2. Convert all timestamp columns to datetime objects
    3. Calculate historical user activity (questions asked, answers provided, help received)
    4. Identify reciprocity activation based on 7-day window before first answer
    5. Count number of answers per user-question-phase combination
    6. Ensure both phases (before/after bounty) exist for each user-question pair
    7. Calculate derivative metrics (hasHelped, demeaned values)

    Args:
        file_path: Path to the bounty dataset parquet file

    Returns:
        processed_df: Processed dataframe with all metrics calculated
    """
    print("Loading dataset...")
    df = pd.read_parquet(file_path)

    # Map eventHistory to numerical flags for historical events
    df['questionAsked'] = (df['event_history'] == 'Question').astype(np.int32)
    df['acceptedAnswer'] = (df['event_history'] == 'AcceptedAnswer').astype(np.int32)
    df['answer'] = (df['event_history'] == 'Answer').astype(np.int32)
    df['acceptedAnswerProvided'] = (df['event_history'] == 'AcceptedAnswerProvided').astype(np.int32)

    # Ensure timestamps are datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['question_posted'] = pd.to_datetime(df['question_posted'], errors='coerce')
    df['bounty_start'] = pd.to_datetime(df['bounty_start'], errors='coerce')
    df['bounty_end'] = pd.to_datetime(df['bounty_end'], errors='coerce')
    df['acceptance_date'] = pd.to_datetime(df['acceptance_date'], errors='coerce')

    # Add one day to acceptance_date when it exists for accepted answer provided events
    # Reasoning: vote timestamp is beginning of day,so +1 to conserative approach
    # Add one day to acceptance_date when it exists for accepted answer provided events
    # Reasoning: vote timestamp is beginning of day, so +1 for conservative approach
    mask = (df['acceptedAnswerProvided'] == 1) & (df['acceptance_date'].notna())
    df.loc[mask, 'effective_timestamp'] = pd.to_datetime(df.loc[mask, 'acceptance_date']) + pd.Timedelta(days=1)
    df.loc[~mask, 'effective_timestamp'] = pd.to_datetime(df.loc[~mask, 'timestamp'])
    print("Calculating metrics...")

    # Calculate prior history counts for each user
    processed_df = calculate_prior_history(df)

    # Calculate reciprocity activation
    processed_df = calculate_reciprocity_activation(processed_df, df)

    # Count number of answers per user per question per phase
    processed_df['numHelped'] = (
        processed_df.groupby(['question_id', 'user_id', 'after_bounty'])['user_id']
        .transform('count')
    )

    # Deduplicate - keep first row per user-question-phase
    processed_df = processed_df.groupby(['question_id', 'user_id', 'after_bounty']).first().reset_index()

    # Add missing phase rows
    processed_df = add_missing_phases(processed_df)

    # Calculate derived metrics
    processed_df['hasHelped'] = (processed_df['numHelped'] > 0).astype(int)
    processed_df['demeaned_hasHelped'] = processed_df['hasHelped'] - processed_df.groupby(
        ['question_id', 'user_id'])['hasHelped'].transform('mean')
    processed_df['demeaned_numHelped'] = processed_df['numHelped'] - processed_df.groupby(
        ['question_id', 'user_id'])['numHelped'].transform('mean')
    processed_df['before_bounty'] = processed_df['after_bounty'].apply(lambda x: 1 if x == 0 else 0)

    # Clean up unnecessary columns
    columns_to_drop = ['questionAsked', 'acceptedAnswer', 'answer', 'acceptedAnswerProvided',
                       'event_history', 'effective_timestamp']
    processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')

    # Rename columns to match requested naming convention
    column_mapping = {
        'numQuestionsAskedAT': 'num_questions_asked',
        'numHelpReceivedAT': 'num_accepted_answers_received',
        'numHelpProvidedAT': 'num_answers_provided',
        'numAcceptedAnswersProvidedAT': 'num_accepted_answers_provided',
        'numHelpProvidedEver': 'has_provided_answers'
    }
    processed_df = processed_df.rename(columns=column_mapping)

    # Print summary stats
    print(f"Processed dataset contains {processed_df['question_id'].nunique()} unique questions")
    print(f"Processed dataset contains {processed_df['user_id'].nunique()} unique users")
    print(f"Total records in processed dataset: {len(processed_df)}")

    return processed_df


def calculate_prior_history(df):
    """
    Calculate prior history metrics for each user at each timestamp
    """
    # Filter for rows where is_history == 0 (the "current" answers we care about)
    df_filtered = df[df['is_history'] == 0].copy()

    # Prepare columns for results
    for col in ['numQuestionsAskedAT', 'numHelpReceivedAT', 'numHelpProvidedAT',
                'numAcceptedAnswersProvidedAT', 'numHelpProvidedEver']:
        df_filtered[col] = 0

    # Group by user
    user_groups = {}
    grouped = df.groupby('user_id')

    print("Building user history lookup tables...")
    # For each user, precompute history
    for user_id, group in tqdm(grouped, desc="Processing user groups"):
        history = group[group['is_history'] == 1].copy()

        # Use effective_timestamp for all historical events
        history = history.sort_values(by='effective_timestamp')

        if not history.empty:
            # Store timestamps and cumulative sums
            times = history['effective_timestamp'].astype(np.int64).values / 1e9
            q_cs = history['questionAsked'].cumsum().values
            acc_cs = history['acceptedAnswer'].cumsum().values
            ans_cs = history['answer'].cumsum().values
            acc_ans_cs = history['acceptedAnswerProvided'].cumsum().values

            user_groups[user_id] = (times, q_cs, acc_cs, ans_cs, acc_ans_cs)

    print("Calculating metrics for each answer...")
    # Fill in metrics for each filtered row
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Calculating Metrics"):
        user_id = row['user_id']

        if pd.notnull(row['timestamp']) and user_id in user_groups:
            target_time = row['timestamp'].timestamp()
            times, q_cs, acc_cs, ans_cs, acc_ans_cs = user_groups[user_id]

            # Binary search for insertion index
            idx_time = np.searchsorted(times, target_time, side='left')

            if idx_time > 0:
                df_filtered.at[idx, 'numQuestionsAskedAT'] = q_cs[idx_time - 1]
                df_filtered.at[idx, 'numHelpReceivedAT'] = acc_cs[idx_time - 1]
                df_filtered.at[idx, 'numHelpProvidedAT'] = ans_cs[idx_time - 1]
                df_filtered.at[idx, 'numAcceptedAnswersProvidedAT'] = acc_ans_cs[idx_time - 1]
                df_filtered.at[idx, 'numHelpProvidedEver'] = 1 if ans_cs[idx_time - 1] > 0 else 0

    return df_filtered


def calculate_reciprocity_activation(df_filtered, all_data):
    """
    Calculate reciprocity activation for each user, considering event timing
    """
    print("Calculating reciprocity activation...")
    reciprocity_results = []

    # Process each user
    for user_id, user_data in tqdm(all_data.groupby('user_id'), desc="Processing Users"):
        user_history = user_data[user_data["is_history"] == 1].sort_values(by='effective_timestamp')

        # Get the first provided answer
        first_provided_answer = user_history[user_history['event_history'] == 'Answer'].head(1)
        if first_provided_answer.empty:
            continue

        first_answer_time = first_provided_answer['effective_timestamp'].iloc[0]

        # Check for accepted answers within 7 days before the first provided answer
        accepted_answers = user_history[
            (user_history['event_history'] == 'AcceptedAnswer') &
            (user_history['effective_timestamp'] < first_answer_time) &
            (user_history['effective_timestamp'] > first_answer_time - pd.Timedelta(days=7))
            ]

        # Only record activation if there were accepted answers in the window
        if not accepted_answers.empty:
            reciprocity_results.append({
                'user_id': user_id,
                'activation_timestamp': first_answer_time
            })

    # Create activation DataFrame
    if reciprocity_results:
        activation_df = pd.DataFrame(reciprocity_results)

        # Apply activation based on timestamp comparison
        # First, set all to 0
        df_filtered['reciprocity_activated'] = 0
        df_filtered['reciprocity_activated_timestamp'] = pd.NaT

        # For each user with activation, set values correctly
        for _, activation_row in activation_df.iterrows():
            user_id = activation_row['user_id']
            activation_time = activation_row['activation_timestamp']

            # Set activation=1 only for events after activation timestamp
            mask = (df_filtered['user_id'] == user_id) & (df_filtered['timestamp'] >= activation_time)
            df_filtered.loc[mask, 'reciprocity_activated'] = 1
            df_filtered.loc[mask, 'reciprocity_activated_timestamp'] = activation_time
    else:
        # No activations found
        df_filtered['reciprocity_activated'] = 0
        df_filtered['reciprocity_activated_timestamp'] = pd.NaT

    return df_filtered


def add_missing_phases(df_filtered):
    """
    Add missing phase rows to ensure each question-user has both phases
    """
    print("Adding missing phase rows...")
    new_rows = []

    # Iterate over grouped data to check for missing phases
    for (user_id, question_id), group in tqdm(df_filtered.groupby(['user_id', 'question_id']),
                                              desc="Processing groups"):
        has_after_bounty_1 = (group['after_bounty'] == 1).any()
        has_after_bounty_0 = (group['after_bounty'] == 0).any()

        if not has_after_bounty_0:  # If after_bounty=0 (before phase) is missing
            new_row = group.iloc[0].copy()
            new_row['after_bounty'] = 0
            # new_row['timestamp'] = new_row['question_posted']
            new_row['numHelped'] = 0
            new_rows.append(new_row)

        if not has_after_bounty_1:  # If after_bounty=1 (after phase) is missing
            new_row = group.iloc[0].copy()
            new_row['after_bounty'] = 1
            # new_row['timestamp'] = new_row['bounty_start']
            new_row['numHelped'] = 0
            new_rows.append(new_row)

    # Append the new rows to the original dataframe
    if new_rows:
        df_filtered = pd.concat([df_filtered, pd.DataFrame(new_rows)], ignore_index=True)

    return df_filtered


def test_data_consistency(processed_df):
    """
    Test function to verify data consistency
    """
    # Check that each user-question pair has exactly two rows (before and after bounty)
    pair_counts = processed_df.groupby(['user_id', 'question_id']).size()
    incomplete_pairs = pair_counts[pair_counts != 2]

    if len(incomplete_pairs) > 0:
        print(f"WARNING: {len(incomplete_pairs)} user-question pairs don't have exactly 2 rows")
        print(f"Sample of incomplete pairs: {incomplete_pairs.head()}")
    else:
        print("✓ All user-question pairs have exactly 2 rows (before and after bounty)")

    return True


if __name__ == "__main__":
    # Example usage
    input_file = r".\02_raw_datasets\bounty_raw_dataset.parquet"
    processed_df = process_bounty_dataset(input_file)

    # Test data consistency
    test_data_consistency(processed_df)

    # Save processed data
    output_path = r".\03_processed_datasets\processed_bounty_dataset.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_parquet(output_path, compression='gzip')
    print(f"Processed dataset saved to {output_path}")