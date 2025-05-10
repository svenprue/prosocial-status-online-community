import pandas as pd
import datetime

# Load the parquet file
df = pd.read_parquet('../02_raw_datasets/question_centered_model_7d_all_questions.parquet')

# Convert timestamp columns to datetime if needed
timestamp_cols = ['timestamp', 'phase_one_start', 'phase_two_end', 'first_answer_timestamp',
                  'accepted_answer_timestamp', 'accepted_answer_vote_timestamp', 'question_timestamp']
for col in timestamp_cols:
    if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
        df[col] = pd.to_datetime(df[col])

# Find window answers 5 days prior to phase_two_end
window_answers = df[(df['event'] == 'Window_Answer') &
                    (df['timestamp'] >= df['phase_two_end'] - pd.Timedelta(days=5)) &
                    (df['timestamp'] <= df['phase_two_end'])]

# Count how many such window answers exist
print(f"Number of window answers within 5 days of phase_two_end: {len(window_answers)}")

# Get a sample event_id with such window answers
if not window_answers.empty:
    sample_event_id = window_answers['event_id'].iloc[0]

    # Print one complete example row
    print(f"\nExample row for event_id {sample_event_id}:")
    sample_row = window_answers[window_answers['event_id'] == sample_event_id]

    # Print each column value
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    # Print as a Series to show all values clearly
    print(sample_row)

    # Print the dataframe format for the same row to show all cells
    print("\nSame row in DataFrame format:")
    print(window_answers[window_answers['event_id'] == sample_event_id].reset_index(drop=True))
else:
    print("No window answers found within 5 days of phase_two_end")