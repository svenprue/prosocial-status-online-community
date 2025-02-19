import pandas as pd

def clean_bounty_timeline(input_path: str, output_path: str) -> None:
    """
    Cleans the bounty timeline data by:
    - Parsing and converting timestamp columns ('bounty_start', 'bounty_end') from string format.
    - Removing rows where either 'bounty_start' or 'bounty_end' is missing.
    - Saving the cleaned data back to a Parquet file.
    """
    # Load the data
    df = pd.read_parquet(input_path)

    # Clean and parse timestamps
    def parse_timestamp(timestamp):
        if isinstance(timestamp, str):
            try:
                # Strip unwanted characters and convert to datetime
                return pd.to_datetime(timestamp.strip("[]'\"Z"), utc=True)
            except Exception:
                return pd.NaT
        return pd.NaT

    # Apply timestamp conversion
    df['bounty_start'] = df['bounty_start'].astype(str).apply(parse_timestamp)
    df['bounty_end'] = df['bounty_end'].astype(str).apply(parse_timestamp)

    # Print the number of rows before cleaning
    print(f"Rows before cleaning: {len(df)}")

    # Remove rows where either 'bounty_start' or 'bounty_end' is missing
    df_cleaned = df.dropna(subset=['bounty_start', 'bounty_end'], how='any')

    # Print the number of rows after cleaning
    print(f"Rows after cleaning: {len(df_cleaned)}")

    # Save the cleaned DataFrame back to a Parquet file
    df_cleaned.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Cleaned data saved to: {output_path}")

def main():
    input_path = './bounty_timeline_results.parquet'
    output_path = 'bounty_timeline_results_cleaned.parquet'
    clean_bounty_timeline(input_path, output_path)

if __name__ == "__main__":
    main()