import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
df = pd.read_parquet("03_processed_datasets/question_centered_model_allQ_7d.parquet")

# Filter by your conditions
filtered_df = df[(df["event_id"]==1056)]

# Get the first event_id that meets the conditions
if not filtered_df.empty:
    first_event_id = filtered_df["event_id"].iloc[0]
    print(f"First matching event_id: {first_event_id}")

    # Print all rows with this event_id
    print(df[df["event_id"] == first_event_id])
else:
    print("No events match the specified conditions")