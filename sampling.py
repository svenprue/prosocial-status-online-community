import pandas as pd
import numpy as np
import random
import os


def filter_dataset(input_file: str, output_file: str, random_seed: int = 42, sample_size: int = 1_000_000):
    """
    Filter the dataset to include:
    1. All rows with is_history=0
    2. All rows from a random selection of 1 million users with is_history=0

    Args:
        input_file: Path to the input parquet file
        output_file: Path where the filtered file will be saved
        random_seed: Random seed for reproducibility
        sample_size: Number of users to randomly select (default: 1 million)
    """
    print(f"\n=== Processing {input_file} ===")

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows.")

    # Get all user_ids with is_history=0
    print("Finding users with is_history=0...")
    non_history_users = df[df["is_history"] == 0]["user_id"].unique()
    print(f"Found {len(non_history_users):,} unique users with is_history=0.")

    # Check if we need to sample or keep all users
    if len(non_history_users) <= sample_size:
        print(
            f"Number of users ({len(non_history_users):,}) is less than or equal to sample size ({sample_size:,}). Keeping all users.")
        selected_users = non_history_users
    else:
        # Randomly select 1 million users
        print(f"Randomly selecting {sample_size:,} users...")
        selected_users = np.random.choice(non_history_users, size=sample_size, replace=False)
        print(f"Selected {len(selected_users):,} users.")

    # Filter the dataset to keep:
    # 1. All rows with is_history=1
    # 2. All rows from the randomly selected users
    print("Filtering dataset...")
    filtered_df = df[(df["user_id"].isin(selected_users))]

    print(f"Original dataset: {len(df):,} rows")
    print(f"Filtered dataset: {len(filtered_df):,} rows")
    print(f"Reduction: {(1 - len(filtered_df) / len(df)) * 100:.2f}%")

    # Save the filtered dataset
    print(f"Saving filtered dataset to {output_file}...")
    filtered_df.to_parquet(output_file, index=False)
    print("Done!")


if __name__ == "__main__":
    input_file = "02_raw_datasets/user_answers_bounty_dataset.parquet"
    output_file = "02_raw_datasets/user_answers_bounty_dataset_sampled_100K.parquet"

    filter_dataset(
        input_file=input_file,
        output_file=output_file,
        random_seed=42,
        sample_size=100000
    )