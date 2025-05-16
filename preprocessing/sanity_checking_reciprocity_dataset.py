def question_centered_sanity_check(processed_file_path):
    """
    Uses DuckDB to perform targeted sanity checks on key metrics without loading the full dataset into memory.
    Focuses on phase distribution, reciprocity activation, and help metrics.
    """
    import duckdb

    # Create DuckDB connection
    print(f"Connecting to data at {processed_file_path}...")
    conn = duckdb.connect(database=':memory:')

    # Register parquet file as a view
    conn.execute(f"""
        CREATE VIEW df AS 
        SELECT * FROM '{processed_file_path}';
    """)

    # 1. Basic distribution statistics
    print("\n=== Basic Dataset Statistics ===")

    # Total events count
    result = conn.execute("SELECT COUNT(*) AS total_events FROM df").fetchone()
    print(f"Total events: {result[0]:,}")

    # Unique users count
    result = conn.execute("SELECT COUNT(DISTINCT userId) AS unique_users FROM df").fetchone()
    print(f"Unique users: {result[0]:,}")

    # Unique questions count
    result = conn.execute("SELECT COUNT(DISTINCT eventId) AS unique_questions FROM df").fetchone()
    print(f"Unique questions: {result[0]:,}")

    # Check phase distribution
    phase_counts = conn.execute("""
                                SELECT phase,
                                       COUNT(*) AS count, 
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) AS percentage
                                FROM df
                                GROUP BY phase
                                ORDER BY phase
                                """).fetchall()

    print("\nPhase distribution:")
    for row in phase_counts:
        print(f"  Phase {row[0]}: {row[1]:,} events ({row[2]:.1f}%)")

    # Check for missing values in key columns
    key_columns = ['userId', 'eventId', 'timestamp', 'event', 'questionId',
                   'numQuestionsAskedAT', 'numHelpProvidedAT', 'numHelped',
                   'reciprocityActivated', 'phase']

    print("\nMissing values in key columns:")
    for col in key_columns:
        result = conn.execute(f"""
            SELECT COUNT(*) AS missing, 
                   CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) AS percentage
            FROM df
            WHERE {col} IS NULL
        """).fetchone()
        print(f"  {col}: {result[0]:,} missing values ({result[1]:.1f}%)")

    # 2. Find specific example cases
    print("\n=== Specific Example Cases ===")

    # Case 1: All metrics non-zero, reciprocity activated, phase=1
    case1_count = conn.execute("""
                               SELECT COUNT(*)
                               FROM df
                               WHERE numQuestionsAskedAT > 0
                                 AND numHelpProvidedAT > 0
                                 AND numAcceptedAnswersReceivedAT > 0
                                 AND numAcceptedAnswersPostedAT > 0
                                 AND reciprocityActivated = 1
                                 AND phase = 1
                                 AND numHelped > 0
                               """).fetchone()[0]

    print("\nCase 1: All key metrics non-zero, reciprocity activated, phase=1")
    print(f"Found {case1_count:,} matching events")

    case1 = conn.execute("""
                         SELECT *
                         FROM df
                         WHERE numQuestionsAskedAT > 0
                           AND numHelpProvidedAT > 0
                           AND numAcceptedAnswersReceivedAT > 0
                           AND numAcceptedAnswersPostedAT > 0
                           AND reciprocityActivated = 1
                           AND numHelped > 0
                           AND phase = 1 LIMIT 1
                         """).fetchone()

    if case1:
        print_full_event_details_from_dict(dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, case1)))
    else:
        print("  No matching events found.")

    # Case 2: All metrics non-zero, reciprocity activated, phase=2
    case2_count = conn.execute("""
                               SELECT COUNT(*)
                               FROM df
                               WHERE numQuestionsAskedAT > 0
                                 AND numHelpProvidedAT > 0
                                 AND numAcceptedAnswersReceivedAT > 0
                                 AND numAcceptedAnswersPostedAT > 0
                                 AND reciprocityActivated = 1
                                 AND phase = 2
                               """).fetchone()[0]

    print("\nCase 2: All key metrics non-zero, reciprocity activated, phase=2")
    print(f"Found {case2_count:,} matching events")

    case2 = conn.execute("""
                         SELECT *
                         FROM df
                         WHERE numQuestionsAskedAT > 0
                           AND numHelpProvidedAT > 0
                           AND numAcceptedAnswersReceivedAT > 0
                           AND numAcceptedAnswersPostedAT > 0
                           AND reciprocityActivated = 1
                           AND phase = 2 LIMIT 1
                         """).fetchone()

    if case2:
        print_full_event_details_from_dict(dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, case2)))
    else:
        print("  No matching events found.")

    # Case 3: Reciprocity not activated, phase=1
    case3_count = conn.execute("""
                               SELECT COUNT(*)
                               FROM df
                               WHERE reciprocityActivated = 0
                                 AND phase = 1
                               """).fetchone()[0]

    print("\nCase 3: Reciprocity not activated, phase=1")
    print(f"Found {case3_count:,} matching events")

    case3 = conn.execute("""
                         SELECT *
                         FROM df
                         WHERE reciprocityActivated = 0
                           AND phase = 1 LIMIT 1
                         """).fetchone()

    if case3:
        print_full_event_details_from_dict(dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, case3)))
    else:
        print("  No matching events found.")

    # Case 4: Reciprocity not activated, phase=2
    case4_count = conn.execute("""
                               SELECT COUNT(*)
                               FROM df
                               WHERE reciprocityActivated = 0
                                 AND phase = 2
                               """).fetchone()[0]

    print("\nCase 4: Reciprocity not activated, phase=2")
    print(f"Found {case4_count:,} matching events")

    case4 = conn.execute("""
                         SELECT *
                         FROM df
                         WHERE reciprocityActivated = 0
                           AND phase = 2 LIMIT 1
                         """).fetchone()

    if case4:
        print_full_event_details_from_dict(dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, case4)))
    else:
        print("  No matching events found.")

    # Case 5: numHelped > 0 in both phase 1 and phase 2 for the same eventId
    # First, find eventIds that appear in both phases with numHelped > 0
    both_phases_query = """
                        WITH phase_summary AS (SELECT eventId, \
                                                      SUM(CASE WHEN phase = 1 THEN numHelped ELSE 0 END) AS phase1_helped, \
                                                      SUM(CASE WHEN phase = 2 THEN numHelped ELSE 0 END) AS phase2_helped, \
                                                      COUNT(DISTINCT phase)                              AS num_phases \
                                               FROM df \
                                               GROUP BY eventId \
                                               HAVING COUNT(DISTINCT phase) = 2)
                        SELECT eventId
                        FROM phase_summary
                        WHERE phase1_helped > 0 \
                          AND phase2_helped > 0 LIMIT 1 \
                        """

    example_event_id_result = conn.execute(both_phases_query).fetchone()

    print("\nCase 5: numHelped > 0 in both phase 1 and phase 2 for the same eventId")

    if example_event_id_result:
        example_event_id = example_event_id_result[0]

        # Count total matching events
        matching_count = conn.execute("""
                                      WITH phase_summary AS (SELECT eventId,
                                                                    SUM(CASE WHEN phase = 1 THEN numHelped ELSE 0 END) AS phase1_helped,
                                                                    SUM(CASE WHEN phase = 2 THEN numHelped ELSE 0 END) AS phase2_helped,
                                                                    COUNT(DISTINCT phase)                              AS num_phases
                                                             FROM df
                                                             GROUP BY eventId
                                                             HAVING COUNT(DISTINCT phase) = 2)
                                      SELECT COUNT(*)
                                      FROM phase_summary
                                      WHERE phase1_helped > 0
                                        AND phase2_helped > 0
                                      """).fetchone()[0]

        print(f"Found {matching_count:,} matching events")

        # Get phase 1 example
        phase1_example = conn.execute(f"""
            SELECT *
            FROM df
            WHERE eventId = '{example_event_id}' AND phase = 1
            LIMIT 1
        """).fetchone()

        # Get phase 2 example
        phase2_example = conn.execute(f"""
            SELECT *
            FROM df
            WHERE eventId = '{example_event_id}' AND phase = 2
            LIMIT 1
        """).fetchone()

        print(f"\n  === EVENT {example_event_id} IN PHASE 1 ===")
        print_full_event_details_from_dict(
            dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, phase1_example)))

        print(f"\n  === SAME EVENT {example_event_id} IN PHASE 2 ===")
        print_full_event_details_from_dict(
            dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, phase2_example)))
    else:
        print("  No matching events found.")

    # Case 6: numHelped = 0 in phase 1 but > 0 in phase 2
    phase2_only_query = """
                        WITH phase_summary AS (SELECT eventId, \
                                                      SUM(CASE WHEN phase = 1 THEN numHelped ELSE 0 END) AS phase1_helped, \
                                                      SUM(CASE WHEN phase = 2 THEN numHelped ELSE 0 END) AS phase2_helped, \
                                                      COUNT(DISTINCT phase)                              AS num_phases \
                                               FROM df \
                                               GROUP BY eventId \
                                               HAVING COUNT(DISTINCT phase) = 2)
                        SELECT eventId
                        FROM phase_summary
                        WHERE phase1_helped = 0 \
                          AND phase2_helped > 0 LIMIT 1 \
                        """

    example_event_id_result = conn.execute(phase2_only_query).fetchone()

    print("\nCase 6: numHelped = 0 in phase 1 but > 0 in phase 2")

    if example_event_id_result:
        example_event_id = example_event_id_result[0]

        # Count total matching events
        matching_count = conn.execute("""
                                      WITH phase_summary AS (SELECT eventId,
                                                                    SUM(CASE WHEN phase = 1 THEN numHelped ELSE 0 END) AS phase1_helped,
                                                                    SUM(CASE WHEN phase = 2 THEN numHelped ELSE 0 END) AS phase2_helped,
                                                                    COUNT(DISTINCT phase)                              AS num_phases
                                                             FROM df
                                                             GROUP BY eventId
                                                             HAVING COUNT(DISTINCT phase) = 2)
                                      SELECT COUNT(*)
                                      FROM phase_summary
                                      WHERE phase1_helped = 0
                                        AND phase2_helped > 0
                                      """).fetchone()[0]

        print(f"Found {matching_count:,} matching events")

        # Get phase 1 example
        phase1_example = conn.execute(f"""
            SELECT *
            FROM df
            WHERE eventId = '{example_event_id}' AND phase = 1
            LIMIT 1
        """).fetchone()

        # Get phase 2 example
        phase2_example = conn.execute(f"""
            SELECT *
            FROM df
            WHERE eventId = '{example_event_id}' AND phase = 2
            LIMIT 1
        """).fetchone()

        print(f"\n  === EVENT {example_event_id} IN PHASE 1 (numHelped = 0) ===")
        print_full_event_details_from_dict(
            dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, phase1_example)))

        print(f"\n  === SAME EVENT {example_event_id} IN PHASE 2 (numHelped > 0) ===")
        print_full_event_details_from_dict(
            dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, phase2_example)))
    else:
        print("  No matching events found.")

    # 3. Distribution of reciprocity activation by phase
    print("\n=== Reciprocity Activation by Phase ===")

    recip_by_phase = conn.execute("""
                                  SELECT phase,
                                         SUM(CASE WHEN reciprocityActivated = 1 THEN 1 ELSE 0 END) AS activated,
                                         SUM(CASE WHEN reciprocityActivated = 0 THEN 1 ELSE 0 END) AS not_activated,
                                         COUNT(*)                                                  AS total
                                  FROM df
                                  GROUP BY phase
                                  ORDER BY phase
                                  """).fetchall()

    for row in recip_by_phase:
        phase, activated, not_activated, total = row
        activated_pct = (activated / total) * 100 if total > 0 else 0
        not_activated_pct = (not_activated / total) * 100 if total > 0 else 0

        print(f"Phase {phase}:")
        print(f"  Reciprocity activated: {activated:,} ({activated_pct:.1f}%)")
        print(f"  Reciprocity not activated: {not_activated:,} ({not_activated_pct:.1f}%)")

    # 4. Find examples for each experience case
    print("\n=== Experience Cases ===")

    # initialExperienceReceiving cases
    receiving_values = ["no help seeked", "help seeked", "help received"]
    print("\ninitialExperienceReceiving examples:")

    for value in receiving_values:
        # Count matches
        matching_count = conn.execute(f"""
            SELECT COUNT(*) 
            FROM df 
            WHERE initialExperienceReceiving = '{value}'
        """).fetchone()[0]

        # Calculate percentage
        total_count = conn.execute("SELECT COUNT(*) FROM df").fetchone()[0]
        percentage = (matching_count / total_count) * 100 if total_count > 0 else 0

        print(f"  {value}: {matching_count:,} events ({percentage:.1f}%)")

        if matching_count > 0:
            # Get an example
            example = conn.execute(f"""
                SELECT *
                FROM df
                WHERE initialExperienceReceiving = '{value}'
                LIMIT 1
            """).fetchone()

            column_names = [desc[0] for desc in conn.execute("SELECT * FROM df LIMIT 0").description]
            example_dict = dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, example))

            # Extract eventId and userId using the simplified dictionary approach for display
            simplified_dict = {}
            for key, value in example_dict.items():
                if isinstance(key, tuple) and len(key) > 0:
                    simplified_key = key[0]
                    simplified_dict[simplified_key] = value
                else:
                    simplified_dict[key] = value

            event_id = simplified_dict.get('eventId', 'Unknown')
            user_id = simplified_dict.get('userId', 'Unknown')
            print(f"    Example eventId: {event_id}, userId: {user_id}")

            # Print full details for the example
            print("    Full details for this example:")
            print_full_event_details_from_dict(example_dict)

    # initialExperienceGiving cases
    giving_values = ["no help attempted", "help attempted", "helped"]
    print("\ninitialExperienceGiving examples:")

    for value in giving_values:
        # Count matches
        matching_count = conn.execute(f"""
            SELECT COUNT(*) 
            FROM df 
            WHERE initialExperienceGiving = '{value}'
        """).fetchone()[0]

        # Calculate percentage
        total_count = conn.execute("SELECT COUNT(*) FROM df").fetchone()[0]
        percentage = (matching_count / total_count) * 100 if total_count > 0 else 0

        print(f"  {value}: {matching_count:,} events ({percentage:.1f}%)")

        if matching_count > 0:
            # Get an example
            example = conn.execute(f"""
                SELECT *
                FROM df
                WHERE initialExperienceGiving = '{value}'
                LIMIT 1
            """).fetchone()

            column_names = [desc[0] for desc in conn.execute("SELECT * FROM df LIMIT 0").description]
            example_dict = dict(zip(conn.execute("SELECT * FROM df LIMIT 0").description, example))

            # Extract eventId and userId using the simplified dictionary approach for display
            simplified_dict = {}
            for key, value in example_dict.items():
                if isinstance(key, tuple) and len(key) > 0:
                    simplified_key = key[0]
                    simplified_dict[simplified_key] = value
                else:
                    simplified_dict[key] = value

            event_id = simplified_dict.get('eventId', 'Unknown')
            user_id = simplified_dict.get('userId', 'Unknown')
            print(f"    Example eventId: {event_id}, userId: {user_id}")

            # Print full details for the example
            print("    Full details for this example:")
            print_full_event_details_from_dict(example_dict)

    # 5. Examine distribution of numHelped by phase
    print("\n=== Distribution of numHelped by Phase ===")

    helped_stats = conn.execute("""
                                SELECT phase,
                                       MIN(numHelped) AS min,
            MAX(numHelped) AS max,
            AVG(numHelped) AS mean,
            MEDIAN(numHelped) AS median,
            SUM(numHelped) AS sum,
            AVG(CASE WHEN numHelped > 0 THEN 1.0 ELSE 0.0 END) AS prop_helped
                                FROM df
                                GROUP BY phase
                                ORDER BY phase
                                """).fetchall()

    # Create a formatted table for the statistics
    print("\nPhase\tMin\tMax\tMean\tMedian\tSum\tProp>0")
    print("-" * 70)
    for row in helped_stats:
        phase, min_val, max_val, mean, median, sum_val, prop = row
        print(f"{phase}\t{min_val}\t{max_val}\t{mean:.2f}\t{median:.1f}\t{sum_val}\t{prop:.2f}")

    # Distribution of numHelped values by phase
    print("\nDistribution of numHelped values by phase:")

    # Get unique phases
    phases = conn.execute("SELECT DISTINCT phase FROM df ORDER BY phase").fetchall()

    for phase_row in phases:
        phase = phase_row[0]

        # Get value counts for this phase (top 10)
        value_counts = conn.execute(f"""
            WITH counts AS (
                SELECT 
                    numHelped, 
                    COUNT(*) AS count,
                    CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df WHERE phase = {phase}) AS FLOAT) AS percentage
                FROM df
                WHERE phase = {phase}
                GROUP BY numHelped
                ORDER BY numHelped
                LIMIT 10
            )
            SELECT * FROM counts
        """).fetchall()

        print(f"\nPhase {phase}:")
        for row in value_counts:
            value, count, percentage = row
            print(f"  numHelped={value}: {count:,} events ({percentage:.1f}%)")

    # Close the connection
    conn.close()


def print_full_event_details_from_dict(row_dict):
    """Helper function to print ALL details of an event in a readable format from a dictionary"""
    print("\n  === FULL EVENT DETAILS ===")

    # Extract simple field names from the complex tuple keys
    simplified_dict = {}
    for key, value in row_dict.items():
        if isinstance(key, tuple) and len(key) > 0:
            # Use the first element of the tuple (column name) as the simplified key
            simplified_key = key[0]
            simplified_dict[simplified_key] = value
        else:
            simplified_dict[key] = value

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

    # Print each group of fields using the simplified dictionary
    for group_name, fields in field_groups.items():
        print(f"\n  --- {group_name} ---")
        for field in fields:
            if field in simplified_dict:
                print(f"  {field}: {simplified_dict[field]}")
            else:
                print(f"  {field}: [Not available]")

    # Print any remaining fields that weren't in the groups
    all_grouped_fields = [field for fields in field_groups.values() for field in fields]
    remaining_fields = [field for field in simplified_dict if field not in all_grouped_fields]

    if remaining_fields:
        print("\n  --- Other Fields ---")
        for field in remaining_fields:
            print(f"  {field}: {simplified_dict[field]}")


if __name__ == "__main__":
    processed_file_path = "../data/study_datasets/question_centered_model_7d_processed.parquet"
    question_centered_sanity_check(processed_file_path)