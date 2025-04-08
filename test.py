import pandas as pd
import duckdb
from tqdm import tqdm

# Load the processed dataset with the issue
df = pd.read_parquet("03_processed_datasets/user_answers_bounty_processed_sampled_100k.parquet")

# Connect to DuckDB for querying bounty data
con = duckdb.connect(database=':memory:')

# Load the necessary data sources
bounty_votes_path = "01_input_data/processed_data_dump/Votes.parquet"
answers_path = "01_input_data/processed_data_dump/posts_answers.parquet"

# Create a view of the bounty votes
con.execute(f"""
    CREATE TABLE bounty_votes AS
    SELECT
        PostId AS answer_id,
        CAST(BountyAmount AS INTEGER) AS bounty_amount
    FROM '{bounty_votes_path}'
    WHERE VoteTypeId = 9
""")

# Create a view of answers to link answer_id to question_id
con.execute(f"""
    CREATE TABLE answers AS
    SELECT
        Id AS answer_id,
        ParentId AS question_id
    FROM '{answers_path}'
""")


# Function to look up bounty amount for a given question_id
def get_bounty_amount(question_id, answer_id=None):
    # If we have both question_id and answer_id, we can be more specific
    query = f"""
            SELECT b.bounty_amount
            FROM bounty_votes b
            JOIN answers a ON b.answer_id = a.answer_id
            WHERE a.question_id = {question_id}
     """

    result = con.execute(query).fetchone()
    return result[0] if result else 0


# Create a copy of the DataFrame to avoid modifying the original
result_df = df.copy()

# Filter to only rows where is_bounty is 1 but bounty_amount is 0
bounty_rows = result_df[(result_df['is_bounty'] == 1) & (result_df['bounty_amount'] == 0)]
print(f"Found {len(bounty_rows)} rows marked as bounty with amount 0")

# Only update these rows
for idx in tqdm(bounty_rows.index, desc="Updating bounty amounts"):
    row = result_df.loc[idx]
    question_id = row['question_id']
    answer_id = row['answer_id']

    # Look up the correct bounty amount
    bounty_amount = get_bounty_amount(question_id, answer_id)
    # Update the bounty amount in the result DataFrame
    result_df.at[idx, 'bounty_amount'] = bounty_amount

# Print summary
updated_count = len(result_df[(result_df['is_bounty'] == 1) & (result_df['bounty_amount'] > 0)])
print(f"Updated {updated_count} rows with proper bounty amounts")
print(f"Total bounty amount: {result_df['bounty_amount'].sum()}")

# Save the corrected dataset
result_df.to_parquet("03_processed_datasets/user_answers_bounty_processed_fixed.parquet", index=False)
print(f"Fixed dataset saved to 03_processed_datasets/user_answers_bounty_processed_fixed.parquet")