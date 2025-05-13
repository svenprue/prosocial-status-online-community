def question_centered_sanity_check(processed_file_path):
    """
    Loads processed question-centered dataframe and prints targeted sanity checks for key metrics.
    Focuses on phase distribution, reciprocity activation, and help metrics.
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
    print(f"Unique questions: {df['eventId'].nunique():,}")

    # Check phase distribution
    phase_counts = df['phase'].value_counts().sort_index()
    print("\nPhase distribution:")
    for phase, count in phase_counts.items():
        print(f"  Phase {phase}: {count:,} events ({count / len(df):.1%})")

    # Check for missing values in key columns
    key_columns = ['userId', 'eventId', 'timestamp', 'event', 'questionId',
                   'numQuestionsAskedAT', 'numHelpProvidedAT', 'numHelped',
                   'reciprocityActivated', 'phase']

    print("\nMissing values in key columns:")
    for col in key_columns:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing:,} missing values ({missing / len(df):.1%})")

    # 2. Find specific example cases
    print("\n=== Specific Example Cases ===")

    # Case 1: All metrics non-zero, reciprocity activated, phase=1
    case1 = df[
        (df['numQuestionsAskedAT'] > 0) &
        (df['numHelpProvidedAT'] > 0) &
        (df['numAcceptedAnswersReceivedAT'] > 0) &
        (df['numAcceptedAnswersPostedAT'] > 0) &
        (df['reciprocityActivated'] == 1) &
        (df['phase'] == 1)
        ]

    print("\nCase 1: All key metrics non-zero, reciprocity activated, phase=1")
    print(f"Found {len(case1):,} matching events")

    if len(case1) > 0:
        print_full_event_details(case1.iloc[0])
    else:
        print("  No matching events found.")

    # Case 2: All metrics non-zero, reciprocity activated, phase=2
    case2 = df[
        (df['numQuestionsAskedAT'] > 0) &
        (df['numHelpProvidedAT'] > 0) &
        (df['numAcceptedAnswersReceivedAT'] > 0) &
        (df['numAcceptedAnswersPostedAT'] > 0) &
        (df['reciprocityActivated'] == 1) &
        (df['phase'] == 2)
        ]

    print("\nCase 2: All key metrics non-zero, reciprocity activated, phase=2")
    print(f"Found {len(case2):,} matching events")

    if len(case2) > 0:
        print_full_event_details(case2.iloc[0])
    else:
        print("  No matching events found.")

    # Case 3: Reciprocity not activated, phase=1
    case3 = df[
        (df['reciprocityActivated'] == 0) &
        (df['phase'] == 1)
        ]

    print("\nCase 3: Reciprocity not activated, phase=1")
    print(f"Found {len(case3):,} matching events")

    if len(case3) > 0:
        print_full_event_details(case3.iloc[0])
    else:
        print("  No matching events found.")

    # Case 4: Reciprocity not activated, phase=2
    case4 = df[
        (df['reciprocityActivated'] == 0) &
        (df['phase'] == 2)
        ]

    print("\nCase 4: Reciprocity not activated, phase=2")
    print(f"Found {len(case4):,} matching events")

    if len(case4) > 0:
        print_full_event_details(case4.iloc[0])
    else:
        print("  No matching events found.")

    # Case 5: numHelped > 0 in both phase 1 and phase 2 for the same eventId
    # First, find eventIds that appear in both phases
    both_phases_events = df.groupby('eventId')['phase'].nunique()
    both_phases_events = both_phases_events[both_phases_events == 2].index.tolist()

    # Create a filtered dataframe of just these events
    both_phases_df = df[df['eventId'].isin(both_phases_events)]

    # Create a pivot to see numHelped for each phase by eventId
    helped_pivot = both_phases_df.pivot_table(
        index='eventId',
        columns='phase',
        values='numHelped',
        aggfunc='sum'
    ).fillna(0)

    # Find events where numHelped > 0 in both phases
    helped_both_phases = helped_pivot[(helped_pivot[1] > 0) & (helped_pivot[2] > 0)]

    print("\nCase 5: numHelped > 0 in both phase 1 and phase 2 for the same eventId")
    print(f"Found {len(helped_both_phases):,} matching events")

    if len(helped_both_phases) > 0:
        example_event_id = helped_both_phases.index[0]
        phase1_example = df[(df['eventId'] == example_event_id) & (df['phase'] == 1)].iloc[0]
        phase2_example = df[(df['eventId'] == example_event_id) & (df['phase'] == 2)].iloc[0]

        print(f"\n  === EVENT {example_event_id} IN PHASE 1 ===")
        print_full_event_details(phase1_example)

        print(f"\n  === SAME EVENT {example_event_id} IN PHASE 2 ===")
        print_full_event_details(phase2_example)
    else:
        print("  No matching events found.")

    # Case 6: numHelped = 0 in phase 1 but > 0 in phase 2
    helped_only_phase2 = helped_pivot[(helped_pivot[1] == 0) & (helped_pivot[2] > 0)]

    print("\nCase 6: numHelped = 0 in phase 1 but > 0 in phase 2")
    print(f"Found {len(helped_only_phase2):,} matching events")

    if len(helped_only_phase2) > 0:
        example_event_id = helped_only_phase2.index[0]
        phase1_example = df[(df['eventId'] == example_event_id) & (df['phase'] == 1)].iloc[0]
        phase2_example = df[(df['eventId'] == example_event_id) & (df['phase'] == 2)].iloc[0]

        print(f"\n  === EVENT {example_event_id} IN PHASE 1 (numHelped = 0) ===")
        print_full_event_details(phase1_example)

        print(f"\n  === SAME EVENT {example_event_id} IN PHASE 2 (numHelped > 0) ===")
        print_full_event_details(phase2_example)
    else:
        print("  No matching events found.")

    # 3. Distribution of reciprocity activation by phase
    print("\n=== Reciprocity Activation by Phase ===")
    recip_by_phase = df.groupby('phase')['reciprocityActivated'].value_counts().unstack().fillna(0)

    for phase in recip_by_phase.index:
        activated = recip_by_phase.loc[phase, 1]
        not_activated = recip_by_phase.loc[phase, 0]
        total = activated + not_activated

        print(f"Phase {phase}:")
        print(f"  Reciprocity activated: {activated:,} ({activated / total:.1%})")
        print(f"  Reciprocity not activated: {not_activated:,} ({not_activated / total:.1%})")

    # 4. Find examples for each experience case
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

    # 5. Examine distribution of numHelped by phase
    print("\n=== Distribution of numHelped by Phase ===")
    helped_stats = df.groupby('phase')['numHelped'].agg(
        ['min', 'max', 'mean', 'median', 'sum', lambda x: (x > 0).mean()])
    helped_stats = helped_stats.rename(columns={'<lambda_0>': 'prop_helped'})

    print(helped_stats)

    # Distribution of numHelped values
    print("\nDistribution of numHelped values by phase:")
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]['numHelped']
        value_counts = phase_data.value_counts().sort_index().head(10)  # Show top 10 most common values

        print(f"\nPhase {phase}:")
        for value, count in value_counts.items():
            print(f"  numHelped={value}: {count:,} events ({count / len(phase_data):.1%})")


def print_full_event_details(row):
    """Helper function to print ALL details of an event in a readable format"""
    print("\n  === FULL EVENT DETAILS ===")

    # Group related metrics for better readability
    field_groups = {
        "Event Information": [
            "eventId", "userId", "questionId", "event", "timestamp",
            "phase", "phaseOneStart", "phaseTwoEnd"
        ],
        "Basic Metrics": [
            "numHelped", "hasHelped", "lnNumHelped", "helpProvidedEver",
            "hasAnswer", "hasAcceptedAnswer"
        ],
        "Time Metrics": [
            "timeToFirstAnswerHours", "timeToAcceptedAnswerHours", "timeToAcceptVoteHours"
        ],
        "Reciprocity": [
            "reciprocityActivated", "reciprocityActivatedTimestamp"
        ],
        "User Experience": [
            "initialExperienceReceiving", "initialExperienceGiving",
            "receivedAcceptedAnswerEver", "receivedAcceptedVoteEver"
        ],
        "All-Time Metrics": [
            "numQuestionsAskedAT", "numHelpProvidedAT",
            "numAcceptedAnswersReceivedAT", "numAcceptedVotesReceivedAT",
            "numAcceptedAnswersPostedAT"
        ],
        "30-Day Metrics": [
            "numQuestionsAsked30D", "numHelpProvided30D",
            "numAcceptedAnswersReceived30D", "numAcceptedVotesReceived30D",
            "numAcceptedAnswersPosted30D"
        ],
        "14-Day Metrics": [
            "numQuestionsAsked14D", "numHelpProvided14D",
            "numAcceptedAnswersReceived14D", "numAcceptedVotesReceived14D",
            "numAcceptedAnswersPosted14D"
        ],
        "7-Day Metrics": [
            "numQuestionsAsked7D", "numHelpProvided7D",
            "numAcceptedAnswersReceived7D", "numAcceptedVotesReceived7D",
            "numAcceptedAnswersPosted7D"
        ],
        "3-Day Metrics": [
            "numQuestionsAsked3D", "numHelpProvided3D",
            "numAcceptedAnswersReceived3D", "numAcceptedVotesReceived3D",
            "numAcceptedAnswersPosted3D"
        ],
        "Fixed Effects": [
            "userFeNumHelped", "questionFeNumHelped",
            "userFeHasHelped", "questionFeHasHelped",
            "userFeLnNumHelped", "questionFeLnNumHelped"
        ]
    }

    # Print each group of fields
    for group_name, fields in field_groups.items():
        print(f"\n  --- {group_name} ---")
        for field in fields:
            if field in row.index:
                print(f"  {field}: {row[field]}")
            else:
                print(f"  {field}: [Not available]")

    # Print any remaining fields that weren't in the groups
    all_grouped_fields = [field for fields in field_groups.values() for field in fields]
    remaining_fields = [field for field in row.index if field not in all_grouped_fields]

    if remaining_fields:
        print("\n  --- Other Fields ---")
        for field in remaining_fields:
            print(f"  {field}: {row[field]}")


if __name__ == "__main__":
    processed_file_path = "../data/study_datasets/question_centered_model_7d_processed.parquet"
    question_centered_sanity_check(processed_file_path)