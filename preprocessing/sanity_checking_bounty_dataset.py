def focused_sanity_check(processed_file_path):
    """
    Uses DuckDB to perform targeted sanity checks on key metrics without loading the full dataset into memory.
    """
    import duckdb

    # Create DuckDB connection with memory settings
    print(f"Connecting to data at {processed_file_path}...")
    conn = duckdb.connect(database=':memory:')

    # Set memory limit and optimize for lower memory usage
    conn.execute("SET memory_limit='4GB'")
    conn.execute("PRAGMA temp_directory='/tmp'")

    # 1. Basic distribution statistics
    print("\n=== Basic Dataset Statistics ===")

    # Count distinct eventIds to verify that each is unique
    result = conn.execute(f"SELECT COUNT(DISTINCT eventId) AS unique_eventIds FROM '{processed_file_path}'").fetchone()
    print(f"Unique eventIds: {result[0]:,}")

    # Unique users count
    result = conn.execute(f"SELECT COUNT(DISTINCT userId) AS unique_users FROM '{processed_file_path}'").fetchone()
    print(f"Unique users: {result[0]:,}")

    # Unique questions count
    result = conn.execute(
        f"SELECT COUNT(DISTINCT answerId) AS unique_answers FROM '{processed_file_path}'").fetchone()
    print(f"Unique answerId: {result[0]:,}")

    # Check bounty distribution - use COUNT(eventId) since we want all events
    bounty_counts = conn.execute(f"""
        WITH total AS (SELECT COUNT(eventId) AS total FROM '{processed_file_path}')
        SELECT 
            isBounty, 
            COUNT(eventId) AS count, 
            CAST(COUNT(eventId) * 100.0 / (SELECT total FROM total) AS FLOAT) AS percentage
        FROM (
            SELECT eventId, isBounty 
            FROM '{processed_file_path}'
        )
        GROUP BY isBounty
        ORDER BY isBounty
    """).fetchall()

    print("\nBounty distribution:")
    for row in bounty_counts:
        print(f"  isBounty={row[0]}: {row[1]:,} eventIds ({row[2]:.1f}%)")

    # Check for missing values in key columns
    key_columns = ['userId', 'eventId', 'timestamp', 'event', 'questionId',
                   'numQuestionsAskedAT', 'numHelpProvidedAT', 'reciprocityActivated']

    print("\nMissing values in key columns:")

    # Create a single pass query to check all nulls in one go
    null_check_query = f"""
        WITH total AS (SELECT COUNT(eventId) AS total FROM '{processed_file_path}')
        SELECT 
            {', '.join([f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_null" for col in key_columns])},
            (SELECT total FROM total) AS total_count
        FROM (
            SELECT {', '.join(key_columns)} 
            FROM '{processed_file_path}'
        )
    """
    null_results = conn.execute(null_check_query).fetchone()

    # Get total count
    total_count = null_results[-1]

    # Print missing values for each column
    for i, col in enumerate(key_columns):
        missing_count = null_results[i]
        percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
        print(f"  {col}: {missing_count:,} missing values ({percentage:.1f}%)")

    # 2. Find specific example cases
    print("\n=== Specific Example Cases ===")

    # Case 1: Count matching eventIds
    needed_columns_case1 = ['eventId', 'numQuestionsAskedAT', 'numHelpProvidedAT',
                            'numAcceptedAnswersReceivedAT', 'numAcceptedAnswersPostedAT',
                            'reciprocityActivated', 'isBounty']

    case1_count = conn.execute(f"""
        SELECT COUNT(eventId)
        FROM (
            SELECT {', '.join(needed_columns_case1)} 
            FROM '{processed_file_path}'
            WHERE numQuestionsAskedAT > 0
              AND numHelpProvidedAT > 0
              AND numAcceptedAnswersReceivedAT > 0
              AND numAcceptedAnswersPostedAT > 0
              AND reciprocityActivated = 1
              AND isBounty = 1
        )
    """).fetchone()[0]

    print("\nCase 1: All key metrics non-zero, reciprocity activated, isBounty=1")
    print(f"Found {case1_count:,} matching eventIds")

    if case1_count > 0:
        # Only select essential columns for the example
        case1_columns = [
            'eventId', 'userId', 'event', 'questionId', 'isBounty',
            'numQuestionsAskedAT', 'numHelpProvidedAT', 'numAcceptedAnswersReceivedAT',
            'numAcceptedAnswersPostedAT', 'reciprocityActivated',
            'initialExperienceReceiving', 'initialExperienceGiving'
        ]

        case1 = conn.execute(f"""
            SELECT {', '.join(case1_columns)}
            FROM '{processed_file_path}'
            WHERE numQuestionsAskedAT > 0
              AND numHelpProvidedAT > 0
              AND numAcceptedAnswersReceivedAT > 0
              AND numAcceptedAnswersPostedAT > 0
              AND reciprocityActivated = 1
              AND isBounty = 1 
            LIMIT 1
        """).fetchone()

        if case1:
            print_event_details(dict(zip(case1_columns, case1)))
    else:
        print("  No matching eventIds found.")

        # Try with bounty=0 as fallback
        fallback_columns = [
            'eventId', 'userId', 'event', 'questionId', 'isBounty',
            'numQuestionsAskedAT', 'numHelpProvidedAT', 'numAcceptedAnswersReceivedAT',
            'numAcceptedAnswersPostedAT', 'reciprocityActivated',
            'initialExperienceReceiving', 'initialExperienceGiving'
        ]

        fallback = conn.execute(f"""
            SELECT {', '.join(fallback_columns)}
            FROM '{processed_file_path}'
            WHERE numQuestionsAskedAT > 0
              AND numHelpProvidedAT > 0
              AND numAcceptedAnswersReceivedAT > 0
              AND numAcceptedAnswersPostedAT > 0
              AND reciprocityActivated = 1
            LIMIT 1
        """).fetchone()

        if fallback:
            print("  Fallback: Found example with same conditions but isBounty=0")
            print_event_details(dict(zip(fallback_columns, fallback)))

    # Case 2: Reciprocity not activated, isBounty=0
    needed_columns_case2 = ['eventId', 'reciprocityActivated', 'isBounty']

    case2_count = conn.execute(f"""
        SELECT COUNT(eventId)
        FROM (
            SELECT {', '.join(needed_columns_case2)}
            FROM '{processed_file_path}'
            WHERE reciprocityActivated = 0
              AND isBounty = 0
        )
    """).fetchone()[0]

    print("\nCase 2: Reciprocity not activated, isBounty=0")
    print(f"Found {case2_count:,} matching eventIds")

    if case2_count > 0:
        # Only select essential columns
        case2_columns = [
            'eventId', 'userId', 'event', 'questionId', 'isBounty',
            'numQuestionsAskedAT', 'numHelpProvidedAT', 'reciprocityActivated',
            'initialExperienceReceiving', 'initialExperienceGiving'
        ]

        case2 = conn.execute(f"""
            SELECT {', '.join(case2_columns)}
            FROM '{processed_file_path}'
            WHERE reciprocityActivated = 0
              AND isBounty = 0 
            LIMIT 1
        """).fetchone()

        if case2:
            print_event_details(dict(zip(case2_columns, case2)))
    else:
        print("  No matching eventIds found.")

    # 3. Find examples for each experience case
    print("\n=== Experience Cases ===")

    # Get all initialExperienceReceiving distributions
    # First get counts per unique value
    receiving_values = ["no help seeked", "help seeked", "help received"]

    # Get the total count only once
    total_count = conn.execute(f"SELECT COUNT(eventId) FROM '{processed_file_path}'").fetchone()[0]

    # Get all value counts in one query rather than separate ones
    receiving_query = f"""
        SELECT 
            initialExperienceReceiving,
            COUNT(eventId) AS count
        FROM (
            SELECT eventId, initialExperienceReceiving 
            FROM '{processed_file_path}'
            WHERE initialExperienceReceiving IN ({', '.join([f"'{v}'" for v in receiving_values])})
        )
        GROUP BY initialExperienceReceiving
    """
    receiving_counts = conn.execute(receiving_query).fetchall()

    # Create a dictionary of counts
    receiving_count_dict = {row[0]: row[1] for row in receiving_counts}

    print("\ninitialExperienceReceiving examples:")

    for value in receiving_values:
        count = receiving_count_dict.get(value, 0)
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        print(f"  {value}: {count:,} eventIds ({percentage:.1f}%)")

        if count > 0:
            # Only select minimal columns for the example
            example_columns = [
                'eventId', 'userId', 'initialExperienceReceiving',
                'numQuestionsAskedAT', 'numHelpProvidedAT'
            ]

            example = conn.execute(f"""
                SELECT {', '.join(example_columns)}
                FROM '{processed_file_path}'
                WHERE initialExperienceReceiving = '{value}'
                LIMIT 1
            """).fetchone()

            if example:
                event_id = example[0]  # eventId is the first column
                user_id = example[1]  # userId is the second column
                print(f"    Example eventId: {event_id}, userId: {user_id}")

                # Print only key metrics
                example_dict = dict(zip(example_columns, example))
                print(f"    Questions asked: {example_dict.get('numQuestionsAskedAT', 'N/A')}")
                print(f"    Help provided: {example_dict.get('numHelpProvidedAT', 'N/A')}")

    # Same optimized approach for initialExperienceGiving
    giving_values = ["no help attempted", "help attempted", "helped"]

    # Reuse the total count from before

    # Get all value counts in one query
    giving_query = f"""
        SELECT 
            initialExperienceGiving,
            COUNT(eventId) AS count
        FROM (
            SELECT eventId, initialExperienceGiving 
            FROM '{processed_file_path}'
            WHERE initialExperienceGiving IN ({', '.join([f"'{v}'" for v in giving_values])})
        )
        GROUP BY initialExperienceGiving
    """
    giving_counts = conn.execute(giving_query).fetchall()

    # Create a dictionary of counts
    giving_count_dict = {row[0]: row[1] for row in giving_counts}

    print("\ninitialExperienceGiving examples:")

    for value in giving_values:
        count = giving_count_dict.get(value, 0)
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        print(f"  {value}: {count:,} eventIds ({percentage:.1f}%)")

        if count > 0:
            # Only select minimal columns for the example
            example_columns = [
                'eventId', 'userId', 'initialExperienceGiving',
                'numQuestionsAskedAT', 'numHelpProvidedAT'
            ]

            example = conn.execute(f"""
                SELECT {', '.join(example_columns)}
                FROM '{processed_file_path}'
                WHERE initialExperienceGiving = '{value}'
                LIMIT 1
            """).fetchone()

            if example:
                event_id = example[0]  # eventId is the first column
                user_id = example[1]  # userId is the second column
                print(f"    Example eventId: {event_id}, userId: {user_id}")

                # Print only key metrics
                example_dict = dict(zip(example_columns, example))
                print(f"    Questions asked: {example_dict.get('numQuestionsAskedAT', 'N/A')}")
                print(f"    Help provided: {example_dict.get('numHelpProvidedAT', 'N/A')}")

    # Close the connection
    conn.close()


def print_event_details(row_dict):
    """Helper function to print key details of an eventId"""
    print(f"  Event ID: {row_dict.get('eventId', 'N/A')}")
    print(f"  User ID: {row_dict.get('userId', 'N/A')}")
    print(f"  Event Type: {row_dict.get('event', 'N/A')}")
    print(f"  Question ID: {row_dict.get('questionId', 'N/A')}")
    print(f"  isBounty: {row_dict.get('isBounty', 'N/A')}")

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
        if field in row_dict:
            print(f"    {label}: {row_dict.get(field, 'N/A')}")

if __name__ == "__main__":
    processed_file_path = "../data/study_datasets/user_answers_bounty_processed.parquet"
    focused_sanity_check(processed_file_path)