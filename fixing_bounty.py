import pandas as pd
import os


def add_experience_categories(input_file, output_file=None):
    """
    Adds experience categorization columns to the complete processed dataset.

    Args:
        input_file: Path to the processed dataset
        output_file: Path to save the enhanced dataset (defaults to overwriting input)
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows")

    # Create experience receiving categorization
    print("Creating experience receiving categories...")
    df['experience_receiving'] = 'uncategorized'  # Default value

    # Apply rules in vectorized fashion for better performance
    df.loc[df['numQuestionsAskedAT'] == 0, 'experience_receiving'] = 'no help seeked'
    df.loc[(df['numQuestionsAskedAT'] > 0) & (df['numAcceptedAnswerReceivedAT'] == 0),
           'experience_receiving'] = 'help seeked'
    df.loc[(df['numQuestionsAskedAT'] > 0) & (df['numAcceptedAnswerReceivedAT'] > 0),
           'experience_receiving'] = 'help received'

    # Convert to ordered categorical
    receiving_categories_order = ['no help seeked', 'help seeked', 'help received']
    df['experience_receiving'] = pd.Categorical(
        df['experience_receiving'],
        categories=receiving_categories_order + ['uncategorized'],
        ordered=True
    )

    # Create experience giving categorization
    print("Creating experience giving categories...")
    df['experience_giving'] = 'uncategorized'  # Default value

    # Apply rules in vectorized fashion
    df.loc[df['numHelpProvidedAT'] == 0, 'experience_giving'] = 'no help attempted'
    df.loc[(df['numHelpProvidedAT'] > 0) & (df['numAcceptedVoteReceivedAT'] == 0),
           'experience_giving'] = 'help attempted'
    df.loc[df['numAcceptedVoteReceivedAT'] > 0, 'experience_giving'] = 'helped'

    # Convert to ordered categorical
    giving_categories_order = ['no help attempted', 'help attempted', 'helped']
    df['experience_giving'] = pd.Categorical(
        df['experience_giving'],
        categories=giving_categories_order + ['uncategorized'],
        ordered=True
    )

    # Add combined experience categories (for common filtering patterns)
    # FIX: Convert categorical to string before concatenation
    print("Creating combined experience categories...")
    df['experience_level'] = df['experience_receiving'].astype(str) + ' + ' + df['experience_giving'].astype(str)

    # Add beginner flags (useful for filtering)
    df['is_new_user'] = ((df['numQuestionsAskedAT'] <= 1) &
                         (df['numHelpProvidedAT'] == 0)).astype(int)

    # Show summary statistics
    print("\nExperience Receiving Breakdown:")
    print(df['experience_receiving'].value_counts(sort=False))

    print("\nExperience Giving Breakdown:")
    print(df['experience_giving'].value_counts(sort=False))

    print("\nNew Users:")
    print(f"Count: {df['is_new_user'].sum():,} ({df['is_new_user'].mean():.1%} of dataset)")

    # Save the enhanced dataset
    if output_file is None:
        output_file = input_file

    print(f"\nSaving enhanced dataset to {output_file}...")
    df.to_parquet(output_file)
    print("Done!")

    return df


if __name__ == "__main__":
    input_file = "03_processed_datasets/user_answers_bounty_processed.parquet"
    output_file = "03_processed_datasets/user_answers_bounty_categorized.parquet"
    add_experience_categories(input_file, output_file)