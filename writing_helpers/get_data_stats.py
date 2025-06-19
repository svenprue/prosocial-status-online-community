import duckdb
import os
import re  # For parsing filename (though not strictly needed for the new specific check, kept if other parts use it)


# --- calculate_descriptive_stats, print_descriptive_stats, ---
# --- calculate_reciprocity_descriptives, calculate_bounty_descriptives ---
# --- (These functions remain largely the same as in the previous version, ensure they use read_parquet) ---

def calculate_descriptive_stats(conn, file_path, column_name, condition=""):
    """
    Calculate descriptive statistics for a numerical column
    """
    where_clause = f"WHERE {condition}" if condition else ""
    query = f"""
        SELECT 
            COUNT({column_name}) as count,
            COUNT(CASE WHEN {column_name} IS NOT NULL THEN 1 END) as non_null_count,
            AVG({column_name}) as mean,
            MEDIAN({column_name}) as median,
            STDDEV({column_name}) as std_dev,
            MIN({column_name}) as min_val,
            MAX({column_name}) as max_val,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column_name}) as q25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column_name}) as q75
        FROM read_parquet('{file_path}') /* Using read_parquet for robustness */
        {where_clause}
    """
    result = conn.execute(query).fetchone()
    return {
        'count': result[0], 'non_null_count': result[1], 'mean': result[2],
        'median': result[3], 'std_dev': result[4], 'min': result[5],
        'max': result[6], 'q25': result[7], 'q75': result[8]
    }


def print_descriptive_stats(stats_dict, variable_name, indent="  "):
    """
    Print descriptive statistics in a formatted way
    """
    print(f"{indent}{variable_name}:")
    print(f"{indent}  Count: {stats_dict['count']:,}")
    print(f"{indent}  Non-null: {stats_dict['non_null_count']:,}")
    if stats_dict['non_null_count'] > 0 and stats_dict['mean'] is not None:
        print(f"{indent}  Mean: {stats_dict['mean']:.3f}")
        print(f"{indent}  Median: {stats_dict['median']:.3f}")
        print(f"{indent}  Std Dev: {stats_dict['std_dev']:.3f}")
        print(f"{indent}  Min: {stats_dict['min']:.3f}")
        print(f"{indent}  Max: {stats_dict['max']:.3f}")
        print(f"{indent}  Q25: {stats_dict['q25']:.3f}")
        print(f"{indent}  Q75: {stats_dict['q75']:.3f}")
    else:
        print(f"{indent}  No non-null values found for descriptive statistics.")


def calculate_reciprocity_descriptives(conn, file_path):
    """
    Calculate descriptive statistics for reciprocity dataset variables
    (Descriptives from the primary processed reciprocity file)
    """
    print("\n📈 RECIPROCITY DESCRIPTIVE STATISTICS (from primary processed file)")
    print("=" * 60)
    # numHelped (overall)
    try:
        stats = calculate_descriptive_stats(conn, file_path, "numHelped")
        print_descriptive_stats(stats, "numHelped (Overall)")
    except Exception as e:
        print(f"  Error calculating numHelped: {e}")
    # ... (other descriptive stats for numHelped phase 1, phase 2, hasAnswer, numHelpProvidedAT) ...
    # (Ensure these parts are present from your full script)
    # Example for hasAnswer:
    try:
        hasAnswer_stats = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN hasAnswer = TRUE THEN 1 ELSE 0 END) as has_answer_count,
                AVG(CAST(hasAnswer AS FLOAT)) as proportion_with_answer
            FROM read_parquet('{file_path}')
            WHERE hasAnswer IS NOT NULL
        """).fetchone()
        print("  hasAnswer:")
        print(f"    Total records with hasAnswer info: {hasAnswer_stats[0]:,}")
        print(f"    Count with answer: {hasAnswer_stats[1]:,}")
        if hasAnswer_stats[0] > 0:
            print(f"    Proportion with answer: {hasAnswer_stats[2]:.3f}")
    except Exception as e:
        print(f"  Error calculating hasAnswer: {e}")

    try:
        stats = calculate_descriptive_stats(conn, file_path, "numHelpProvidedAT")
        print_descriptive_stats(stats, "numHelpProvidedAT")
    except Exception as e:
        print(f"  Error calculating numHelpProvidedAT: {e}")


def calculate_bounty_descriptives(conn, file_path):
    """
    Calculate descriptive statistics for bounty dataset variables
    """
    print("\n💰 BOUNTY DESCRIPTIVE STATISTICS (from processed file)")
    print("=" * 60)
    # ... (content of this function remains as is from your full script) ...
    # Example for isBounty:
    try:
        bounty_stats = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN isBounty = TRUE THEN 1 ELSE 0 END) as is_bounty_count,
                AVG(CAST(isBounty as FLOAT)) as proportion_bounty
            FROM read_parquet('{file_path}')
            WHERE isBounty IS NOT NULL
        """).fetchone()
        print("  isBounty:")
        print(f"    Total records with isBounty info: {bounty_stats[0]:,}")
        print(f"    Is bounty count: {bounty_stats[1]:,}")
        if bounty_stats[0] > 0:
            print(f"    Proportion bounty: {bounty_stats[2]:.3f}")
    except Exception as e:
        print(f"  Error calculating isBounty: {e}")
    # ... (other descriptives for bounty file)


def _get_phase2_cutoff_count_from_specific_file(conn, specific_file_path):
    """
    Queries a specific parquet file to count rows where 'phase_two_start' is after a cutoff date.
    """
    cutoff_date = "2025-04-01"
    stats = {"file_total_rows": "N/A", "questions_phase2_after_cutoff": "N/A", "phase2_cutoff_date": cutoff_date}

    if not os.path.exists(specific_file_path):
        print(f"    File not found: {specific_file_path}")
        return stats

    try:
        # Single query to get both counts
        query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(CASE WHEN CAST(phase_two_start AS TIMESTAMP) > CAST('{cutoff_date}' AS TIMESTAMP) THEN 1 END) as phase2_after_cutoff
            FROM read_parquet('{specific_file_path}')
        """
        result = conn.execute(query).fetchone()
        stats["file_total_rows"] = result[0]
        stats["questions_phase2_after_cutoff"] = result[1]
        print(stats)

    except Exception as e:
        print(f"    Error querying {specific_file_path}: {e}")
        stats["file_total_rows"] = "Error"
        stats["questions_phase2_after_cutoff"] = "Error"

    return stats


def print_basic_dataset_statistics(conn, reciprocity_file, bounty_file):
    """
    Print basic dataset statistics (counts, unique values, breakdowns)
    including the new specific check for the reciprocity-related data.
    """
    print("=== BASIC DATASET STATISTICS ===\n")

    # Reciprocity Dataset Basic Statistics
    if os.path.exists(reciprocity_file):
        print("📊 RECIPROCITY DATASET (Question-Centered Model from primary processed file)")
        print("=" * 70)

        total_rows_processed = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{reciprocity_file}')").fetchone()[0]
        print(f"Total rows (in '{os.path.basename(reciprocity_file)}'): {total_rows_processed:,}")

        unique_questions_processed = conn.execute(
            f"SELECT COUNT(DISTINCT questionId) FROM read_parquet('{reciprocity_file}') WHERE questionId IS NOT NULL").fetchone()[
            0]
        print(f"Unique question IDs (in '{os.path.basename(reciprocity_file)}'): {unique_questions_processed:,}")

        unique_users_processed = conn.execute(
            f"SELECT COUNT(DISTINCT userId) FROM read_parquet('{reciprocity_file}') WHERE userId IS NOT NULL").fetchone()[
            0]
        print(f"Unique user IDs (in '{os.path.basename(reciprocity_file)}'): {unique_users_processed:,}")

        phase_breakdown = conn.execute(f"""
            SELECT phase, COUNT(*) as count
            FROM read_parquet('{reciprocity_file}')
            GROUP BY phase ORDER BY phase
        """).fetchall()
        print(f"\nPhase breakdown (in '{os.path.basename(reciprocity_file)}'):")
        for phase, count in phase_breakdown:
            print(f"  Phase {phase}: {count:,} rows")

        # --- New Specific Check as per your request ---
        print("\nSpecific Date Condition Check (for context):")
        # User specified this exact path for the check. Using raw string for Windows path.
        phase2_check_file_path = r"C:\Users\svenp\PycharmProjects\prosocial-status-online-community\data\input\question_centered_model_7d_all_questions.parquet"

        phase2_cutoff_stats = _get_phase2_cutoff_count_from_specific_file(conn, phase2_check_file_path)

        print(f"  Reference file for this check: '{os.path.basename(phase2_check_file_path)}'")
        if os.path.exists(phase2_check_file_path):  # Check again here for cleaner print structure
            print(f"    Total rows in this reference file: {phase2_cutoff_stats['file_total_rows']:,}" if isinstance(
                phase2_cutoff_stats['file_total_rows'],
                int) else f"    Total rows in this reference file: {phase2_cutoff_stats['file_total_rows']}")
            print(f"    Phase 2 cutoff date considered: {phase2_cutoff_stats['phase2_cutoff_date']}")
            print(
                f"    Count in reference file with 'phase_two_start' > {phase2_cutoff_stats['phase2_cutoff_date']}: {phase2_cutoff_stats['questions_phase2_after_cutoff']:,}" if isinstance(
                    phase2_cutoff_stats['questions_phase2_after_cutoff'],
                    int) else f"    Count in reference file with 'phase_two_start' > {phase2_cutoff_stats['phase2_cutoff_date']}: {phase2_cutoff_stats['questions_phase2_after_cutoff']}")
        # If the file wasn't found, the helper function already printed a message.
        # --- End of New Specific Check ---

    else:
        print(f"❌ Primary reciprocity file not found: {reciprocity_file}")

    print("\n" + "=" * 70 + "\n")

    # Bounty Dataset Basic Statistics (remains unchanged)
    if os.path.exists(bounty_file):
        print("💰 BOUNTY DATASET (User Answers)")
        print("=" * 50)
        # ... (content as before) ...
        total_rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{bounty_file}')").fetchone()[0]
        print(f"Total rows: {total_rows:,}")
        # ... (rest of bounty stats) ...
    else:
        print(f"❌ Bounty file not found: {bounty_file}")


def print_dataset_statistics():
    """
    Main function to print comprehensive dataset statistics.
    """
    reciprocity_file = "../data/study_datasets/question_centered_model_7d_processed.parquet"
    bounty_file = "../data/study_datasets/user_answers_bounty_processed.parquet"
    conn = None
    try:
        conn = duckdb.connect(database=':memory:')
        conn.execute("SET TimeZone='UTC';")

        print_basic_dataset_statistics(conn, reciprocity_file, bounty_file)

        if os.path.exists(reciprocity_file):
            calculate_reciprocity_descriptives(conn, reciprocity_file)

        print("\n" + "=" * 70 + "\n")

        if os.path.exists(bounty_file):
            calculate_bounty_descriptives(conn, bounty_file)

    except Exception as e:
        print(f"An error occurred in print_dataset_statistics: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    print_dataset_statistics()