import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the processed reciprocity dataset
# Update this path to match your actual file location
df = pd.read_parquet("../data/study_datasets/first_answer_centered_model_7d_processed_TEST100000.parquet")

print("=" * 80)
print("PROCESSED RECIPROCITY DATASET SANITY CHECK")
print("=" * 80)
print(f"\nTotal rows: {len(df):,}")
print(f"Total unique events: {df['eventId'].nunique():,}")
print(f"Total unique users: {df['userId'].nunique():,}")
print("\nPhase distribution:")
print(df['phase'].value_counts().sort_index())
print("\nHasAcceptedAnswer distribution:")
print(df['hasAcceptedAnswer'].value_counts())

# Find one event WITH accepted answer
with_accepted = df[df['hasAcceptedAnswer'] == True].head(1)

# Find one event WITHOUT accepted answer
without_accepted = df[df['hasAcceptedAnswer'] == False].head(1)

def print_event_details(event_df, title):
    """Print all columns and values for an event in a readable format"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if event_df.empty:
        print("No events found matching criteria")
        return

    # Get the first row as a Series
    event = event_df.iloc[0]

    # Print event metadata
    print("\n--- EVENT METADATA ---")
    metadata_cols = ['eventId', 'userId', 'questionId', 'phase', 'timestamp',
                     'phaseOneStart', 'phaseTwoEnd']
    for col in metadata_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print question/answer flags
    print("\n--- QUESTION/ANSWER FLAGS ---")
    flag_cols = ['hasAnswer', 'hasAcceptedAnswer']
    for col in flag_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print time metrics
    print("\n--- TIME METRICS ---")
    time_cols = ['timeToFirstAnswerHours', 'timeToAcceptedAnswerHours',
                 'timeToAcceptVoteHours', 'timeSinceFirstActivityDays']
    for col in time_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print all-time metrics
    print("\n--- ALL-TIME METRICS (AT) ---")
    at_cols = ['numQuestionsAskedAT', 'numHelpProvidedAT', 'helpProvidedEver',
               'numAcceptedAnswersReceivedAT', 'numAcceptedVotesReceivedAT',
               'numAcceptedAnswersPostedAT']
    for col in at_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print 30-day metrics
    print("\n--- 30-DAY METRICS (30D) ---")
    d30_cols = ['numQuestionsAsked30D', 'numHelpProvided30D',
                'numAcceptedAnswersReceived30D', 'numAcceptedVotesReceived30D',
                'numAcceptedAnswersPosted30D']
    for col in d30_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print 14-day metrics
    print("\n--- 14-DAY METRICS (14D) ---")
    d14_cols = ['numQuestionsAsked14D', 'numHelpProvided14D',
                'numAcceptedAnswersReceived14D', 'numAcceptedVotesReceived14D',
                'numAcceptedAnswersPosted14D']
    for col in d14_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print 7-day metrics
    print("\n--- 7-DAY METRICS (7D) ---")
    d7_cols = ['numQuestionsAsked7D', 'numHelpProvided7D',
               'numAcceptedAnswersReceived7D', 'numAcceptedVotesReceived7D',
               'numAcceptedAnswersPosted7D']
    for col in d7_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print 3-day metrics
    print("\n--- 3-DAY METRICS (3D) ---")
    d3_cols = ['numQuestionsAsked3D', 'numHelpProvided3D',
               'numAcceptedAnswersReceived3D', 'numAcceptedVotesReceived3D',
               'numAcceptedAnswersPosted3D']
    for col in d3_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print experience categories
    print("\n--- EXPERIENCE CATEGORIES ---")
    exp_cols = ['initialExperienceReceiving', 'initialExperienceGiving']
    for col in exp_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print reciprocity metrics
    print("\n--- RECIPROCITY METRICS ---")
    recip_cols = ['reciprocityActivated', 'reciprocityActivatedTimestamp']
    for col in recip_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print dependent variables
    print("\n--- DEPENDENT VARIABLES ---")
    dv_cols = ['numHelped', 'hasHelped', 'lnNumHelped']
    for col in dv_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print fixed effects
    print("\n--- FIXED EFFECTS ---")
    fe_cols = ['userFeNumHelped', 'questionFeNumHelped',
               'userFeHasHelped', 'questionFeHasHelped',
               'userFeLnNumHelped', 'questionFeLnNumHelped']
    for col in fe_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print temporal variables
    print("\n--- TEMPORAL VARIABLES ---")
    temp_cols = ['year', 'month', 'receivedAcceptedAnswerEver', 'receivedAcceptedVoteEver']
    for col in temp_cols:
        if col in event.index:
            print(f"{col:35s}: {event[col]}")

    # Print any remaining columns not captured above
    all_printed_cols = (metadata_cols + flag_cols + time_cols + at_cols + d30_cols +
                       d14_cols + d7_cols + d3_cols + exp_cols + recip_cols +
                       dv_cols + fe_cols + temp_cols)
    remaining_cols = [col for col in event.index if col not in all_printed_cols]
    if remaining_cols:
        print("\n--- OTHER COLUMNS ---")
        for col in remaining_cols:
            print(f"{col:35s}: {event[col]}")

# Print example with accepted answer
print_event_details(with_accepted, "EXAMPLE EVENT #1: WITH ACCEPTED ANSWER")

# Print example without accepted answer
print_event_details(without_accepted, "EXAMPLE EVENT #2: WITHOUT ACCEPTED ANSWER")

# Print summary comparison
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

if not with_accepted.empty and not without_accepted.empty:
    event_with = with_accepted.iloc[0]
    event_without = without_accepted.iloc[0]

    print("\nKey differences:")
    print(f"\n{'Metric':<40} {'With Accepted':<20} {'Without Accepted':<20}")
    print("-" * 80)

    key_metrics = ['hasAcceptedAnswer', 'timeToAcceptedAnswerHours',
                   'numAcceptedAnswersReceivedAT', 'receivedAcceptedAnswerEver',
                   'numHelped', 'phase']

    for metric in key_metrics:
        if metric in event_with.index and metric in event_without.index:
            print(f"{metric:<40} {str(event_with[metric]):<20} {str(event_without[metric]):<20}")

print("\n" + "=" * 80)
print("SANITY CHECK COMPLETE")
print("=" * 80)
