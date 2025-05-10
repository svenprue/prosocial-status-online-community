import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm  # For progress bar

# Directory paths
input_dir = r"E:\Data"
output_dir = r"03_processed_datasets"
os.makedirs(output_dir, exist_ok=True)

# List all parquet files
parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))

# Initialize an empty list to store processed dataframes
all_dfs = []

# Process each parquet file with progress bar
for file_path in tqdm(parquet_files, total=len(parquet_files), desc="Processing parquet files"):
    print(f"\nProcessing file: {os.path.basename(file_path)}")

    # Read the parquet file
    df = pd.read_parquet(file_path)
    print(f"Read parquet file: {df.shape[0]} rows, {df.shape[1]} columns")

    # Convert columns to best possible dtypes
    df = df.convert_dtypes()
    print("Converted columns to best possible dtypes")

    # Convert nullable integer columns to standard int64
    int_cols = df.select_dtypes(include=["Int32", "Int64"]).columns.tolist()
    for col in int_cols:
        df[col] = df[col].astype("int64")
    print(f"Converted {len(int_cols)} nullable integer columns to standard int64")

    # Convert nullable float columns to standard float64
    float_cols = df.select_dtypes(include=["Float32"]).columns.tolist()
    for col in float_cols:
        df[col] = df[col].astype("float64")
    print(f"Converted {len(float_cols)} nullable float columns to standard float64")

    # Create derived columns
    df['has_helped'] = (df['numHelped'] > 0).astype(int)
    df['ln_numHelped'] = np.log(df['numHelped'] + 1)
    print("Created derived columns: has_helped and ln_numHelped")

    # Fixed effects for numHelped
    df['user_fe_numHelped'] = df.groupby("user_id")["numHelped"].transform(lambda x: x - x.mean())
    df['question_fe_numHelped'] = df.groupby("event_id")["numHelped"].transform(lambda x: x - x.mean())
    print("Created fixed effects for numHelped")

    # Fixed effects for has_helped
    df['user_fe_has_helped'] = df.groupby("user_id")["has_helped"].transform(lambda x: x - x.mean())
    df['question_fe_has_helped'] = df.groupby("event_id")["has_helped"].transform(lambda x: x - x.mean())
    print("Created fixed effects for has_helped")

    # Fixed effects for ln_numHelped
    df['user_fe_ln_numHelped'] = df.groupby("user_id")["ln_numHelped"].transform(lambda x: x - x.mean())
    df['question_fe_ln_numHelped'] = df.groupby("event_id")["ln_numHelped"].transform(lambda x: x - x.mean())
    print("Created fixed effects for ln_numHelped")

    # Add to list of processed dataframes
    all_dfs.append(df)
    print(f"Added processed dataframe to list: now have {len(all_dfs)} dataframes")
    print(f"Current processed dataframe shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("Concatenating all dataframes...")
# Concatenate all processed dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined dataframe shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

# Count unique user_ids
unique_user_count = combined_df['user_id'].nunique()
print(f"Number of unique users: {unique_user_count}")

print(f"Saving final dataset...")
# Save to output file with count in filename
output_file = os.path.join(output_dir, f"reciprocity_model_7d_{unique_user_count}_users.parquet")
combined_df.to_parquet(output_file)

print(f"Processed data saved to {output_file}")