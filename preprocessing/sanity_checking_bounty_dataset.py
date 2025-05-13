def focused_sanity_check(processed_file_path):
    """
    Loads processed dataframe and prints targeted sanity checks for key metrics.
    """
    import pandas as pd
    import numpy as np

    # Set display options to show all columns without truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)

    # Load the processed data
    print(f"Loading processed data from {processed_file_path}...")
    df = pd.read_parquet(processed_file_path)

    # 1. Basic distribution statistics
    print("\n=== Basic Dataset Statistics ===")
    print(f"Total events: {len(df):,}")
    print(f"Unique users: {df['userId'].nunique():,}")
    print(f"Unique questions: {df['questionId'].nunique():,}")

    # Check bounty distribution
    bounty_counts = df['isBounty'].value_counts()
    print("\nBounty distribution:")
    for value, count in bounty_counts.items():
        print(f"  isBounty={value}: {count:,} events ({count / len(df):.1%})")

    # Check for missing values in key columns
    key_columns = ['userId', 'eventId', 'timestamp', 'event', 'questionId',
                   'numQuestionsAskedAT', 'numHelpProvidedAT', 'reciprocityActivated']

    print("\nMissing values in key columns:")
    for col in key_columns:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing:,} missing values ({missing / len(df):.1%})")

    # 2. Find specific example cases
    print("\n=== Specific Example Cases ===")

    # Case 1: All metrics non-zero, reciprocity activated, isBounty=1
    case1 = df[
        (df['numQuestionsAskedAT'] > 0) &
        (df['numHelpProvidedAT'] > 0) &
        (df['numAcceptedAnswersReceivedAT'] > 0) &
        (df['numAcceptedAnswersPostedAT'] > 0) &
        (df['reciprocityActivated'] == 1) &
        (df['isBounty'] == 1)
        ]

    print("\nCase 1: All key metrics non-zero, reciprocity activated, isBounty=1")
    print(f"Found {len(case1):,} matching events")

    if len(case1) > 0:
        print_full_event_details(case1.iloc[0])
    else:
        print("  No matching events found.")

        # Try with bounty=0 as fallback
        fallback = df[
            (df['numQuestionsAskedAT'] > 0) &
            (df['numHelpProvidedAT'] > 0) &
            (df['numAcceptedAnswersReceivedAT'] > 0) &
            (df['numAcceptedAnswersPostedAT'] > 0) &
            (df['reciprocityActivated'] == 1)
            ]
        if len(fallback) > 0:
            print("  Fallback: Found example with same conditions but isBounty=0")
            print_full_event_details(fallback.iloc[0])

    # Case 2: Reciprocity not activated, isBounty=0
    case2 = df[
        (df['reciprocityActivated'] == 0) &
        (df['isBounty'] == 0)
        ]

    print("\nCase 2: Reciprocity not activated, isBounty=0")
    print(f"Found {len(case2):,} matching events")

    if len(case2) > 0:
        print_full_event_details(case2.iloc[0])
    else:
        print("  No matching events found.")

    # 3. Find examples for each experience case
    print("\n=== Experience Cases ===")

    # initialExperienceReceiving cases
    receiving_values = ["no help seeked", "help seeked", "help received"]
    print("\ninitialExperienceReceiving examples:")

    for value in receiving_values:
        matching = df[df['initialExperienceReceiving'] == value]
        print(f"  {value}: {len(matching):,} events ({len(matching) / len(df):.1%})")

        if len(matching) > 0:
            event_id = matching.iloc[0]['eventId']
            user_id = matching.iloc[0]['userId']
            print(f"    Example eventId: {event_id}, userId: {user_id}")
            # Print full details for the first example of each value
            print("    Full details for this example:")
            print_full_event_details(matching.iloc[0])

    # initialExperienceGiving cases
    giving_values = ["no help attempted", "help attempted", "helped"]
    print("\ninitialExperienceGiving examples:")

    for value in giving_values:
        matching = df[df['initialExperienceGiving'] == value]
        print(f"  {value}: {len(matching):,} events ({len(matching) / len(df):.1%})")

        if len(matching) > 0:
            event_id = matching.iloc[0]['eventId']
            user_id = matching.iloc[0]['userId']
            print(f"    Example eventId: {event_id}, userId: {user_id}")
            # Print full details for the first example of each value
            print("    Full details for this example:")
            print_full_event_details(matching.iloc[0])


def print_full_event_details(row):
    """Helper function to print ALL details of an event in a readable format"""
    print("\n  === FULL EVENT DETAILS ===")
    print("  Event ID:", row['eventId'])

    # Print all columns for this event
    for column, value in row.items():
        print(f"  {column}: {value}")


# Keep the original function for reference or additional use
def print_event_details(row):
    """Helper function to print key details of an event"""
    print(f"  Event ID: {row['eventId']}")
    print(f"  User ID: {row['userId']}")
    print(f"  Event Type: {row['event']}")
    print(f"  Question ID: {row['questionId']}")
    print(f"  isBounty: {row['isBounty']}")

    # Print key metrics
    print("  Key metrics:")
    metrics = [
        ("Questions asked (AT)", "numQuestionsAskedAT"),
        ("Help provided (AT)", "numHelpProvidedAT"),
        ("Accepted answers received (AT)", "numAcceptedAnswersReceivedAT"),
        ("Accepted answers posted (AT)", "numAcceptedAnswersPostedAT"),
        ("Reciprocity activated", "reciprocityActivated"),
        ("Initial receiving experience", "initialExperienceReceiving"),
        ("Initial giving experience", "initialExperienceGiving")
    ]

    for label, field in metrics:
        print(f"    {label}: {row[field]}")


if __name__ == "__main__":
    processed_file_path = "../data/study_datasets/user_answers_bounty_processed.parquet"
    focused_sanity_check(processed_file_path)