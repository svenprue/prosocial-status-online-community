import pandas as pd

# Set display options to show all rows and columns with full width
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of each cell

# Load the dataset
df = pd.read_parquet("03_processed_datasets/user_answers_bounty_processed.parquet")

#df = df[df["is_bounty"]==1].head(10)
df = df[df["user_id"]==243]
print(df)