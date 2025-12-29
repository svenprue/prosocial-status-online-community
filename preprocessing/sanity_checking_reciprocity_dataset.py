"""
Comprehensive sanity check for processed reciprocity dataset.
Tests ALL variables, checks for nulls, validates logical consistency,
and provides detailed examples for comparison.
"""
import duckdb
import sys


def print_all_columns(row_tuple, column_names, title):
    """Print all columns for a single event in organized groups"""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    # Convert to dictionary for easier access
    row_dict = {}
    for i, col_name in enumerate(column_names):
        # Handle tuple column names from DuckDB
        if isinstance(col_name, tuple) and len(col_name) > 0:
            key = col_name[0]
        else:
            key = col_name
        row_dict[key] = row_tuple[i]

    # Define column groups for organized output
    groups = {
        "EVENT METADATA": [
            "eventId", "userId", "questionId", "phase", "event",
            "timestamp", "phaseOneStart", "phaseTwoEnd"
        ],

        "ANSWER STATUS FLAGS": [
            "hasAnswer", "hasAcceptedAnswer"
        ],

        "RESPONSE TIME METRICS (Hours)": [
            "responseTimeHours", "timeToAcceptedAnswerHours", "timeToAcceptVoteHours"
        ],

        "ALL-TIME METRICS (AT) - Before Phase 1 Start": [
            "numQuestionsAskedAT", "numHelpProvidedAT", "helpProvidedEver",
            "numAcceptedAnswersReceivedAT", "numAcceptedVotesReceivedAT",
            "numAcceptedAnswersPostedAT"
        ],

        "30-DAY WINDOW METRICS (30D)": [
            "numQuestionsAsked30D", "numHelpProvided30D",
            "numAcceptedAnswersReceived30D", "numAcceptedVotesReceived30D",
            "numAcceptedAnswersPosted30D"
        ],

        "14-DAY WINDOW METRICS (14D)": [
            "numQuestionsAsked14D", "numHelpProvided14D",
            "numAcceptedAnswersReceived14D", "numAcceptedVotesReceived14D",
            "numAcceptedAnswersPosted14D"
        ],

        "7-DAY WINDOW METRICS (7D)": [
            "numQuestionsAsked7D", "numHelpProvided7D",
            "numAcceptedAnswersReceived7D", "numAcceptedVotesReceived7D",
            "numAcceptedAnswersPosted7D"
        ],

        "3-DAY WINDOW METRICS (3D)": [
            "numQuestionsAsked3D", "numHelpProvided3D",
            "numAcceptedAnswersReceived3D", "numAcceptedVotesReceived3D",
            "numAcceptedAnswersPosted3D"
        ],

        "EXPERIENCE CATEGORIES": [
            "initialExperienceReceiving", "initialExperienceGiving",
            "timeSinceFirstActivityDays"
        ],

        "RECIPROCITY ACTIVATION": [
            "reciprocityActivated", "reciprocityActivatedTimestamp",
            "reciprocityActivatedBeforePhase1"
        ],

        "HELP GIVEN BETWEEN QUESTION AND FIRST ANSWER": [
            "helps_given_between_question_and_answer"
        ],

        "DEPENDENT VARIABLES (Helping in this Phase)": [
            "numHelped", "hasHelped", "lnNumHelped"
        ],

        "FIXED EFFECTS": [
            "userFeNumHelped", "questionFeNumHelped",
            "userFeHasHelped", "questionFeHasHelped",
            "userFeLnNumHelped", "questionFeLnNumHelped",
        ],

        "TEMPORAL VARIABLES": [
            "year", "month", "receivedAcceptedAnswerEver", "receivedAcceptedVoteEver"
        ],

        "USER METADATA": [
            "registration_date", "days_since_registration_at_phase_one_start",
            "autobiography_received", "autobiography_active_phase_one_start",
            "autobiography_active_phase_two_start"
        ]
    }

    # Print each group
    for group_name, columns in groups.items():
        print(f"\n{group_name}:")
        print("-" * 120)
        for col in columns:
            if col in row_dict:
                value = row_dict[col]
                null_marker = " [NULL]" if value is None else ""
                print(f"  {col:45s} = {value}{null_marker}")
            else:
                print(f"  {col:45s} = [COLUMN NOT FOUND]")

    # Print any columns not in groups
    all_grouped = [col for cols in groups.values() for col in cols]
    ungrouped = [col for col in row_dict.keys() if col not in all_grouped]

    if ungrouped:
        print(f"\nOTHER COLUMNS:")
        print("-" * 120)
        for col in ungrouped:
            value = row_dict[col]
            null_marker = " [NULL]" if value is None else ""
            print(f"  {col:45s} = {value}{null_marker}")


def main():
    """Main sanity check function with comprehensive validation"""

    # File path - UPDATE THIS if your file is in a different location
    file_path = "../data/study_datasets/question_centered_model_7d_processed_TEST10000.parquet"

    print("=" * 120)
    print("COMPREHENSIVE RECIPROCITY DATASET SANITY CHECK")
    print("=" * 120)
    print(f"\nFile: {file_path}")

    try:
        # Connect to DuckDB
        conn = duckdb.connect(database=':memory:')

        # Create view
        conn.execute(f"CREATE VIEW df AS SELECT * FROM '{file_path}';")

    except Exception as e:
        print(f"\n❌ ERROR: Could not load file: {e}")
        print(f"\nAttempting alternate filename pattern...")

        # Try alternate filename
        file_path = "../data/study_datasets/first_answer_centered_model_7d_processed_TEST100000.parquet"
        print(f"Trying: {file_path}")

        try:
            conn = duckdb.connect(database=':memory:')
            conn.execute(f"CREATE VIEW df AS SELECT * FROM '{file_path}';")
            print("✓ Successfully loaded alternate file")
        except Exception as e2:
            print(f"❌ ERROR: Could not load alternate file either: {e2}")
            sys.exit(1)

    # ========================================
    # PART 1: BASIC STATISTICS
    # ========================================
    print("\n" + "=" * 120)
    print("PART 1: BASIC DATASET STATISTICS")
    print("=" * 120)

    total_rows = conn.execute("SELECT COUNT(*) FROM df").fetchone()[0]
    print(f"\nTotal rows: {total_rows:,}")

    unique_events = conn.execute("SELECT COUNT(DISTINCT eventId) FROM df").fetchone()[0]
    print(f"Unique events (questions): {unique_events:,}")

    unique_users = conn.execute("SELECT COUNT(DISTINCT userId) FROM df").fetchone()[0]
    print(f"Unique users: {unique_users:,}")

    avg_events_per_user = total_rows / unique_users if unique_users > 0 else 0
    print(f"Average rows per user: {avg_events_per_user:.2f}")

    print("\nPhase distribution:")
    phase_dist = conn.execute("""
        SELECT phase, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY phase
        ORDER BY phase
    """).fetchall()
    for phase, count, pct in phase_dist:
        print(f"  Phase {phase}: {count:,} rows ({pct:.1f}%)")

    print("\nhasAcceptedAnswer distribution:")
    accepted_dist = conn.execute("""
        SELECT hasAcceptedAnswer, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY hasAcceptedAnswer
        ORDER BY hasAcceptedAnswer
    """).fetchall()
    for has_accepted, count, pct in accepted_dist:
        print(f"  hasAcceptedAnswer={has_accepted}: {count:,} rows ({pct:.1f}%)")

    # ========================================
    # PART 2: COMPREHENSIVE NULL CHECK FOR ALL COLUMNS
    # ========================================
    print("\n" + "=" * 120)
    print("PART 2: NULL VALUE ANALYSIS FOR ALL COLUMNS")
    print("=" * 120)

    column_names = [desc[0] for desc in conn.execute("SELECT * FROM df LIMIT 0").description]
    print(f"\nTotal columns: {len(column_names)}")

    # Check ALL columns for NULLs
    print("\nNULL value count by column:")
    print(f"{'Column Name':<50} {'NULL Count':>15} {'NULL %':>10} {'Status':>10}")
    print("-" * 120)

    null_summary = []
    for col in column_names:
        null_count = conn.execute(f"SELECT COUNT(*) FROM df WHERE \"{col}\" IS NULL").fetchone()[0]
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        status = "✓ OK" if null_count == 0 else "⚠ NULLS"

        null_summary.append((col, null_count, null_pct, status))
        print(f"{col:<50} {null_count:>15,} {null_pct:>9.1f}% {status:>10}")

    # Summary of columns with NULLs
    columns_with_nulls = [item for item in null_summary if item[1] > 0]
    print(f"\n{'='*120}")
    print(f"Summary: {len(columns_with_nulls)} out of {len(column_names)} columns have NULL values")
    if columns_with_nulls:
        print("\nColumns with NULLs (sorted by NULL count):")
        for col, count, pct, _ in sorted(columns_with_nulls, key=lambda x: x[1], reverse=True):
            print(f"  • {col}: {count:,} NULLs ({pct:.1f}%)")

    # ========================================
    # PART 3: VALUE RANGE AND DISTRIBUTION ANALYSIS
    # ========================================
    print("\n" + "=" * 120)
    print("PART 3: VALUE RANGE AND DISTRIBUTION ANALYSIS")
    print("=" * 120)

    # Numeric columns to analyze
    numeric_cols = [
        "numQuestionsAskedAT", "numHelpProvidedAT", "numAcceptedAnswersReceivedAT",
        "numAcceptedVotesReceivedAT", "numAcceptedAnswersPostedAT",
        "numQuestionsAsked30D", "numHelpProvided30D", "numAcceptedAnswersReceived30D",
        "numQuestionsAsked7D", "numHelpProvided7D", "numAcceptedAnswersReceived7D",
        "numHelped", "hasHelped", "helpProvidedEver",
        "responseTimeHours", "timeToAcceptedAnswerHours",
        "timeSinceFirstActivityDays", "days_since_registration_at_phase_one_start",
        "helps_given_between_question_and_answer"
    ]

    print("\nNumeric column statistics:")
    print(f"{'Column':<45} {'Min':>10} {'Max':>10} {'Mean':>12} {'Median':>12}")
    print("-" * 120)

    for col in numeric_cols:
        try:
            stats = conn.execute(f"""
                SELECT
                    MIN(\"{col}\") as min_val,
                    MAX(\"{col}\") as max_val,
                    AVG(CAST(\"{col}\" AS FLOAT)) as mean_val,
                    MEDIAN(\"{col}\") as median_val
                FROM df
                WHERE \"{col}\" IS NOT NULL
            """).fetchone()

            if stats and stats[0] is not None:
                min_val, max_val, mean_val, median_val = stats
                print(f"{col:<45} {min_val:>10.2f} {max_val:>10.2f} {mean_val:>12.2f} {median_val:>12.2f}")
            else:
                print(f"{col:<45} {'[ALL NULL]':>10}")
        except Exception as e:
            print(f"{col:<45} [ERROR: {str(e)[:40]}]")

    # ========================================
    # PART 4: CATEGORICAL VARIABLE DISTRIBUTIONS
    # ========================================
    print("\n" + "=" * 120)
    print("PART 4: CATEGORICAL VARIABLE DISTRIBUTIONS")
    print("=" * 120)

    print("\ninitialExperienceReceiving distribution:")
    exp_recv_dist = conn.execute("""
        SELECT initialExperienceReceiving, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY initialExperienceReceiving
        ORDER BY count DESC
    """).fetchall()
    for val, count, pct in exp_recv_dist:
        print(f"  {str(val):<30}: {count:>10,} ({pct:>5.1f}%)")

    print("\ninitialExperienceGiving distribution:")
    exp_give_dist = conn.execute("""
        SELECT initialExperienceGiving, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY initialExperienceGiving
        ORDER BY count DESC
    """).fetchall()
    for val, count, pct in exp_give_dist:
        print(f"  {str(val):<30}: {count:>10,} ({pct:>5.1f}%)")

    print("\nreciprocityActivated distribution:")
    recip_dist = conn.execute("""
        SELECT reciprocityActivated, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY reciprocityActivated
        ORDER BY reciprocityActivated
    """).fetchall()
    for val, count, pct in recip_dist:
        print(f"  reciprocityActivated={val}: {count:,} ({pct:.1f}%)")

    print("\nreciprocityActivatedBeforePhase1 distribution:")
    recip_before_dist = conn.execute("""
        SELECT reciprocityActivatedBeforePhase1, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY reciprocityActivatedBeforePhase1
        ORDER BY reciprocityActivatedBeforePhase1
    """).fetchall()
    for val, count, pct in recip_before_dist:
        print(f"  reciprocityActivatedBeforePhase1={val}: {count:,} ({pct:.1f}%)")

    print("\nnumHelped distribution (top 10 values):")
    helped_dist = conn.execute("""
        SELECT numHelped, COUNT(*) as count,
               CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df) AS FLOAT) as pct
        FROM df
        GROUP BY numHelped
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()
    for helped, count, pct in helped_dist:
        print(f"  numHelped={helped}: {count:,} rows ({pct:.1f}%)")

    # ========================================
    # PART 5: LOGICAL CONSISTENCY CHECKS
    # ========================================
    print("\n" + "=" * 120)
    print("PART 5: LOGICAL CONSISTENCY CHECKS")
    print("=" * 120)

    # Check 1: Time window nesting
    print("\n1. Time window nesting check (3D <= 7D <= 14D <= 30D <= AT):")
    invalid_nesting = conn.execute("""
        SELECT COUNT(*) FROM df
        WHERE numQuestionsAsked3D > numQuestionsAsked7D
           OR numQuestionsAsked7D > numQuestionsAsked14D
           OR numQuestionsAsked14D > numQuestionsAsked30D
           OR numQuestionsAsked30D > numQuestionsAskedAT
    """).fetchone()[0]
    if invalid_nesting == 0:
        print(f"  ✓ All time windows properly nested (0 violations)")
    else:
        print(f"  ⚠️  Found {invalid_nesting:,} rows with invalid time window nesting")

    # Check 2: hasAcceptedAnswer consistency
    print("\n2. hasAcceptedAnswer consistency check:")
    invalid_accepted = conn.execute("""
        SELECT COUNT(*) FROM df
        WHERE (hasAcceptedAnswer = 1 AND timeToAcceptedAnswerHours IS NULL)
           OR (hasAcceptedAnswer = 0 AND timeToAcceptedAnswerHours IS NOT NULL)
    """).fetchone()[0]
    if invalid_accepted == 0:
        print(f"  ✓ hasAcceptedAnswer consistent with timeToAcceptedAnswerHours (0 violations)")
    else:
        print(f"  ⚠️  Found {invalid_accepted:,} inconsistencies")

    # Check 3: Phase consistency per event
    print("\n3. Phase consistency check:")
    both_phases = conn.execute("""
        SELECT COUNT(DISTINCT eventId) as events_with_both
        FROM (
            SELECT eventId, COUNT(DISTINCT phase) as num_phases
            FROM df
            GROUP BY eventId
            HAVING COUNT(DISTINCT phase) = 2
        )
    """).fetchone()[0]
    single_phase = unique_events - both_phases
    print(f"  Events with both phases: {both_phases:,} ({both_phases/unique_events*100:.1f}%)")
    print(f"  Events with single phase: {single_phase:,} ({single_phase/unique_events*100:.1f}%)")

    # Check 4: helpProvidedEver consistency
    print("\n4. helpProvidedEver consistency check:")
    invalid_help_ever = conn.execute("""
        SELECT COUNT(*) FROM df
        WHERE (helpProvidedEver = 1 AND numHelpProvidedAT = 0)
           OR (helpProvidedEver = 0 AND numHelpProvidedAT > 0)
    """).fetchone()[0]
    if invalid_help_ever == 0:
        print(f"  ✓ helpProvidedEver consistent with numHelpProvidedAT (0 violations)")
    else:
        print(f"  ⚠️  Found {invalid_help_ever:,} inconsistencies")

    # Check 5: hasHelped consistency
    print("\n5. hasHelped consistency check:")
    invalid_has_helped = conn.execute("""
        SELECT COUNT(*) FROM df
        WHERE (hasHelped = 1 AND numHelped = 0)
           OR (hasHelped = 0 AND numHelped > 0)
    """).fetchone()[0]
    if invalid_has_helped == 0:
        print(f"  ✓ hasHelped consistent with numHelped (0 violations)")
    else:
        print(f"  ⚠️  Found {invalid_has_helped:,} inconsistencies")

    # Check 6: Reciprocity timestamp consistency
    print("\n6. reciprocityActivatedTimestamp consistency check:")
    invalid_recip_timestamp = conn.execute("""
        SELECT COUNT(*) FROM df
        WHERE (reciprocityActivated = 1 AND reciprocityActivatedTimestamp IS NULL)
           OR (reciprocityActivated = 0 AND reciprocityActivatedTimestamp IS NOT NULL)
    """).fetchone()[0]
    if invalid_recip_timestamp == 0:
        print(f"  ✓ reciprocityActivatedTimestamp consistent with reciprocityActivated (0 violations)")
    else:
        print(f"  ⚠️  Found {invalid_recip_timestamp:,} inconsistencies")

    # Check 7: Fixed effects should sum to zero within groups
    print("\n7. Fixed effects zero-sum property check:")
    user_fe_sum = conn.execute("SELECT ABS(SUM(userFeNumHelped)) FROM df").fetchone()[0]
    question_fe_sum = conn.execute("SELECT ABS(SUM(questionFeNumHelped)) FROM df").fetchone()[0]
    # month_fe_sum = conn.execute("SELECT ABS(SUM(monthFeNumHelped)) FROM df").fetchone()[0]
    print(f"  Sum of userFeNumHelped: {user_fe_sum:.6f} (should be ~0)")
    print(f"  Sum of questionFeNumHelped: {question_fe_sum:.6f} (should be ~0)")
    # print(f"  Sum of monthFeNumHelped: {month_fe_sum:.6f} (should be ~0)")
    if abs(user_fe_sum) < 0.01 and abs(question_fe_sum) < 0.01:
        print(f"  ✓ Fixed effects properly centered")
    else:
        print(f"  ⚠️  Fixed effects may not be properly centered")

    # Check 8: helps_given_between_question_and_answer should be non-negative
    print("\n8. helps_given_between_question_and_answer validation check:")
    invalid_helps_given = conn.execute("""
        SELECT COUNT(*) FROM df
        WHERE helps_given_between_question_and_answer < 0
           OR helps_given_between_question_and_answer IS NULL
    """).fetchone()[0]
    if invalid_helps_given == 0:
        print(f"  ✓ All helps_given_between_question_and_answer values are non-negative and non-null (0 violations)")
    else:
        print(f"  ⚠️  Found {invalid_helps_given:,} rows with invalid helps_given_between_question_and_answer")

    # ========================================
    # PART 6: DETAILED EXAMPLES
    # ========================================
    print("\n" + "=" * 120)
    print("PART 6: DETAILED EXAMPLES FOR COMPARISON")
    print("=" * 120)
    print("\nFinding diverse examples to illustrate different scenarios...")

    # Example 1: WITH accepted answer and high engagement
    example1 = conn.execute("""
        SELECT *
        FROM df
        WHERE hasAcceptedAnswer = 1
          AND numHelped > 0
          AND numQuestionsAskedAT > 0
          AND numHelpProvidedAT > 0
          AND numAcceptedAnswersReceivedAT > 0
          AND reciprocityActivated = 1
        LIMIT 1
    """).fetchone()

    if example1:
        print_all_columns(
            example1,
            column_names,
            "EXAMPLE 1: High engagement user WITH accepted answer AND reciprocity activated"
        )
    else:
        print("\n⚠️  Could not find Example 1")

    # Example 2: WITHOUT accepted answer but still helping
    example2 = conn.execute("""
        SELECT *
        FROM df
        WHERE hasAcceptedAnswer = 0
          AND numHelped > 0
          AND numQuestionsAskedAT > 0
          AND numHelpProvidedAT > 0
        LIMIT 1
    """).fetchone()

    if example2:
        print_all_columns(
            example2,
            column_names,
            "EXAMPLE 2: User WITHOUT accepted answer but still helping others"
        )
    else:
        print("\n⚠️  Could not find Example 2")

    # Example 3: New user (low activity)
    example3 = conn.execute("""
        SELECT *
        FROM df
        WHERE numQuestionsAskedAT <= 1
          AND numHelpProvidedAT = 0
          AND phase = 1
        LIMIT 1
    """).fetchone()

    if example3:
        print_all_columns(
            example3,
            column_names,
            "EXAMPLE 3: New/low-activity user (Phase 1)"
        )
    else:
        print("\n⚠️  Could not find Example 3")

    # Example 4: Reciprocity NOT activated
    example4 = conn.execute("""
        SELECT *
        FROM df
        WHERE reciprocityActivated = 0
          AND numHelped > 0
          AND numAcceptedAnswersReceivedAT > 0
        LIMIT 1
    """).fetchone()

    if example4:
        print_all_columns(
            example4,
            column_names,
            "EXAMPLE 4: User helping but reciprocity NOT activated"
        )
    else:
        print("\n⚠️  Could not find Example 4")

    # ========================================
    # PART 7: SIDE-BY-SIDE COMPARISONS
    # ========================================
    if example1 and example2:
        print("\n" + "=" * 120)
        print("PART 7: SIDE-BY-SIDE COMPARISON - WITH vs WITHOUT ACCEPTED ANSWER")
        print("=" * 120)

        # Convert to dicts
        ex1_dict = {column_names[i][0] if isinstance(column_names[i], tuple) else column_names[i]: example1[i]
                    for i in range(len(column_names))}
        ex2_dict = {column_names[i][0] if isinstance(column_names[i], tuple) else column_names[i]: example2[i]
                    for i in range(len(column_names))}

        comparison_metrics = [
            # Identifiers
            ("eventId", "Event ID"),
            ("userId", "User ID"),
            ("phase", "Phase"),

            # Key flags
            ("hasAnswer", "Has Answer"),
            ("hasAcceptedAnswer", "Has Accepted Answer"),

            # Response times
            ("responseTimeHours", "Response Time (hrs)"),
            ("timeToAcceptedAnswerHours", "Time to Accepted Answer (hrs)"),

            # Historical metrics
            ("numQuestionsAskedAT", "Questions Asked (all-time)"),
            ("numHelpProvidedAT", "Help Provided (all-time)"),
            ("numAcceptedAnswersReceivedAT", "Accepted Answers Received (AT)"),
            ("numAcceptedAnswersPostedAT", "Accepted Answers Posted (AT)"),

            # 7-day metrics
            ("numQuestionsAsked7D", "Questions Asked (7D)"),
            ("numHelpProvided7D", "Help Provided (7D)"),
            ("numAcceptedAnswersReceived7D", "Accepted Answers Received (7D)"),

            # Current phase behavior
            ("numHelped", "Number Helped This Phase"),
            ("hasHelped", "Has Helped This Phase"),

            # Reciprocity
            ("reciprocityActivated", "Reciprocity Activated"),
            ("reciprocityActivatedBeforePhase1", "Reciprocity Before Phase 1"),

            # Experience
            ("initialExperienceReceiving", "Initial Experience Receiving"),
            ("initialExperienceGiving", "Initial Experience Giving"),
            ("timeSinceFirstActivityDays", "Time Since First Activity (days)"),

            # User metadata
            ("days_since_registration_at_phase_one_start", "Days Since Registration"),
            ("autobiography_active_phase_one_start", "Autobiography Badge Active"),
        ]

        print(f"\n{'Metric':<50} {'WITH Accepted':<35} {'WITHOUT Accepted':<35}")
        print("=" * 120)

        for col_name, display_name in comparison_metrics:
            val1 = ex1_dict.get(col_name, "N/A")
            val2 = ex2_dict.get(col_name, "N/A")

            # Format values
            val1_str = "NULL" if val1 is None else str(val1)
            val2_str = "NULL" if val2 is None else str(val2)

            # Truncate if too long
            val1_str = val1_str[:33] if len(val1_str) > 33 else val1_str
            val2_str = val2_str[:33] if len(val2_str) > 33 else val2_str

            print(f"{display_name:<50} {val1_str:<35} {val2_str:<35}")

    # ========================================
    # PART 8: SUMMARY AND RECOMMENDATIONS
    # ========================================
    print("\n" + "=" * 120)
    print("PART 8: SUMMARY AND RECOMMENDATIONS")
    print("=" * 120)

    print("\n✓ CHECKS COMPLETED:")
    print(f"  • Loaded {total_rows:,} rows from {unique_events:,} events for {unique_users:,} users")
    print(f"  • Analyzed {len(column_names)} columns for NULL values")
    print(f"  • Validated {len(numeric_cols)} numeric columns for range")
    print(f"  • Performed 8 logical consistency checks")
    print(f"  • Displayed 4 diverse examples for manual inspection")

    # Flag potential issues
    issues_found = []
    if len(columns_with_nulls) > 0:
        issues_found.append(f"{len(columns_with_nulls)} columns have NULL values")
    if invalid_nesting > 0:
        issues_found.append(f"Time window nesting violations: {invalid_nesting:,}")
    if invalid_accepted > 0:
        issues_found.append(f"hasAcceptedAnswer inconsistencies: {invalid_accepted:,}")
    if invalid_help_ever > 0:
        issues_found.append(f"helpProvidedEver inconsistencies: {invalid_help_ever:,}")
    if invalid_has_helped > 0:
        issues_found.append(f"hasHelped inconsistencies: {invalid_has_helped:,}")
    if invalid_recip_timestamp > 0:
        issues_found.append(f"Reciprocity timestamp inconsistencies: {invalid_recip_timestamp:,}")
    if invalid_helps_given > 0:
        issues_found.append(f"Invalid helps_given_between_question_and_answer: {invalid_helps_given:,}")

    if issues_found:
        print("\n⚠️  POTENTIAL ISSUES FOUND:")
        for issue in issues_found:
            print(f"  • {issue}")
    else:
        print("\n✓ NO MAJOR ISSUES DETECTED")

    print("\n💡 RECOMMENDATIONS:")
    print("  1. Review the examples above to verify metrics make intuitive sense")
    print("  2. Check NULL values are expected (e.g., timeToAcceptedAnswerHours when hasAcceptedAnswer=0)")
    print("  3. Verify time window metrics properly nest (3D ≤ 7D ≤ 14D ≤ 30D ≤ AT)")
    print("  4. Confirm reciprocity activation logic matches research design")
    print("  5. Validate fixed effects are properly centered (sums ≈ 0)")

    # Close connection
    conn.close()

    print("\n" + "=" * 120)
    print("COMPREHENSIVE SANITY CHECK COMPLETE ✓")
    print("=" * 120)


if __name__ == "__main__":
    main()
