import os
import duckdb
import pandas as pd


def process_accepted_answer_data(
        input_folder: str,
        output_folder: str,
        window_length: int,
        include_all_questions: bool = False,
        test_mode: bool = False,
        test_user_limit: int = 500000
) -> None:
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA max_temp_directory_size='200GiB'")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar;")

    questions_path = os.path.join(input_folder, 'posts_questions.parquet')
    answers_path = os.path.join(input_folder, 'posts_answers.parquet')
    votes_path = os.path.join(input_folder, 'Votes.parquet')
    badges_path = os.path.join(input_folder, 'Badges.parquet')
    users_path = os.path.join(input_folder, 'Users.parquet')

    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            AcceptedAnswerId AS accepted_answer_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{questions_path}'
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            OwnerUserId AS owner_user_id,
            ParentId AS parent_question_id,
            CAST(Score AS INTEGER) AS score,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{answers_path}'
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW votes AS
        SELECT
            PostId AS post_id,
            CAST(CreationDate AS TIMESTAMP) AS vote_date
        FROM '{votes_path}'
        WHERE VoteTypeId = 1;
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW autobiography_badges AS
        SELECT
            UserId AS user_id,
            CAST(Date AS TIMESTAMP) AS autobiography_received
        FROM '{badges_path}'
        WHERE Name = 'Autobiographer';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW users AS
        SELECT
            Id AS user_id,
            CAST(CreationDate AS TIMESTAMP) AS registration_date
        FROM '{users_path}'
    """)

    # Create a test users subset if in test mode
    if test_mode:
        con.execute(f"""
            CREATE TEMPORARY TABLE test_users AS
            SELECT DISTINCT owner_user_id
            FROM questions
            WHERE owner_user_id IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {test_user_limit};
        """)
        user_filter = "AND q.owner_user_id IN (SELECT owner_user_id FROM test_users)"
        print(f"TEST MODE: Randomly sampling {test_user_limit} users")
    else:
        user_filter = ""

    # Define eligible questions - INCLUDING ALL QUESTIONS from ALL users (registration data optional)
    if include_all_questions:
        con.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW eligible_questions AS
            WITH first_answers AS (
                SELECT
                    a.parent_question_id AS question_id,
                    MIN(a.creation_date) AS first_answer_timestamp
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                WHERE a.score >= 0  -- Only consider non-negative score answers
                  AND (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-answers
                GROUP BY a.parent_question_id
            ),
            accepted_answers AS (
                SELECT
                    q.question_id,
                    a.creation_date AS accepted_answer_timestamp,
                    v.vote_date AS accepted_answer_vote_timestamp
                FROM questions q
                INNER JOIN answers a ON q.accepted_answer_id = a.answer_id
                LEFT JOIN votes v ON a.answer_id = v.post_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-accepted answers
            ),
            answer_counts AS (
                SELECT
                    a.parent_question_id AS question_id,
                    COUNT(*) AS answer_count
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                WHERE a.score >= 0  -- Only count answers with non-negative scores
                  AND (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-answers
                GROUP BY a.parent_question_id
            ),
            self_answers AS (
                SELECT
                    a.parent_question_id AS question_id,
                    COUNT(*) AS self_answer_count
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                WHERE a.owner_user_id = q.owner_user_id  -- Only self-answers
                GROUP BY a.parent_question_id
            )
            SELECT
                q.question_id,
                q.owner_user_id,
                q.creation_date AS question_timestamp,
                COALESCE(ac.answer_count > 0, FALSE) AS has_answer,
                aa.accepted_answer_timestamp IS NOT NULL AS has_accepted_answer,
                COALESCE(sa.self_answer_count > 0, FALSE) AS has_self_answer,
                fa.first_answer_timestamp,
                aa.accepted_answer_timestamp,
                aa.accepted_answer_vote_timestamp
            FROM questions q
            LEFT JOIN first_answers fa ON q.question_id = fa.question_id  -- LEFT JOIN to include ALL questions
            LEFT JOIN users u ON q.owner_user_id = u.user_id  -- LEFT JOIN to include users even without registration data
            LEFT JOIN accepted_answers aa ON q.question_id = aa.question_id
            LEFT JOIN answer_counts ac ON q.question_id = ac.question_id
            LEFT JOIN self_answers sa ON q.question_id = sa.question_id
            WHERE q.owner_user_id IS NOT NULL
              {user_filter};
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW eligible_questions AS
            WITH first_answers AS (
                SELECT
                    a.parent_question_id AS question_id,
                    MIN(a.creation_date) AS first_answer_timestamp
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                WHERE a.score >= 0  -- Only consider non-negative score answers
                  AND (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-answers
                GROUP BY a.parent_question_id
            ),
            accepted_answers AS (
                SELECT
                    q.question_id,
                    a.creation_date AS accepted_answer_timestamp,
                    v.vote_date AS accepted_answer_vote_timestamp
                FROM questions q
                INNER JOIN answers a ON q.accepted_answer_id = a.answer_id
                LEFT JOIN votes v ON a.answer_id = v.post_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-accepted answers
            ),
            answer_counts AS (
                SELECT
                    a.parent_question_id AS question_id,
                    COUNT(*) AS answer_count
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                WHERE a.score >= 0  -- Only count answers with non-negative scores
                  AND (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-answers
                GROUP BY a.parent_question_id
            ),
            self_answers AS (
                SELECT
                    a.parent_question_id AS question_id,
                    COUNT(*) AS self_answer_count
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                WHERE a.owner_user_id = q.owner_user_id  -- Only self-answers
                GROUP BY a.parent_question_id
            ),
            raw AS (
                SELECT
                    q.question_id,
                    q.owner_user_id,
                    q.creation_date AS question_timestamp,
                    COALESCE(ac.answer_count > 0, FALSE) AS has_answer,
                    q.accepted_answer_id IS NOT NULL AS has_accepted_answer,
                    COALESCE(sa.self_answer_count > 0, FALSE) AS has_self_answer,
                    fa.first_answer_timestamp,
                    aa.accepted_answer_timestamp,
                    aa.accepted_answer_vote_timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY q.owner_user_id
                        ORDER BY RANDOM()
                    ) AS rn
                FROM questions q
                LEFT JOIN first_answers fa ON q.question_id = fa.question_id  -- LEFT JOIN to include ALL questions
                LEFT JOIN users u ON q.owner_user_id = u.user_id  -- LEFT JOIN to include users even without registration data
                LEFT JOIN accepted_answers aa ON q.question_id = aa.question_id
                LEFT JOIN answer_counts ac ON q.question_id = ac.question_id
                LEFT JOIN self_answers sa ON q.question_id = sa.question_id
                WHERE q.owner_user_id IS NOT NULL
                  {user_filter}
            )
            SELECT
                question_id,
                owner_user_id,
                question_timestamp,
                has_answer,
                has_accepted_answer,
                has_self_answer,
                first_answer_timestamp,
                accepted_answer_timestamp,
                accepted_answer_vote_timestamp
            FROM raw
            WHERE rn = 1;
        """)

    # Add the summary print statements here
    question_count = con.execute("SELECT COUNT(*) FROM eligible_questions").fetchone()[0]
    user_count = con.execute("SELECT COUNT(DISTINCT owner_user_id) FROM eligible_questions").fetchone()[0]

    # Check how many users have registration data (now optional since we use LEFT JOIN)
    users_with_registration = con.execute("""
        SELECT COUNT(DISTINCT eq.owner_user_id)
        FROM eligible_questions eq
        INNER JOIN users u ON eq.owner_user_id = u.user_id
        WHERE u.registration_date IS NOT NULL
    """).fetchone()[0]

    users_without_registration = user_count - users_with_registration
    pct_with_registration = (users_with_registration / user_count * 100) if user_count > 0 else 0

    print(f"Summary for window_length={window_length}d (Question-Centered):")
    print(f"  - Including {question_count:,} questions from {user_count:,} unique users")
    print(f"  - Users with registration data: {users_with_registration:,} ({pct_with_registration:.1f}%)")
    if users_without_registration > 0:
        print(f"  - Users without registration data: {users_without_registration:,} ({100-pct_with_registration:.1f}%)")
    print(f"  - {'All questions' if include_all_questions else 'One question'} per user mode")
    if test_mode:
        print(f"  - TEST MODE ACTIVE: Limited to {test_user_limit} users")

    # Calculate helps_given_between_question_and_first_answer and join autobiography badge
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        WITH helps_given AS (
            SELECT
                eq.question_id,
                COUNT(a.answer_id) AS helps_given_between_question_and_answer
            FROM eligible_questions eq
            LEFT JOIN answers a
              ON a.owner_user_id = eq.owner_user_id
             AND a.creation_date >= eq.question_timestamp
             AND a.creation_date < eq.first_answer_timestamp
            LEFT JOIN questions q
              ON a.parent_question_id = q.question_id
            WHERE (a.owner_user_id IS NULL
                   OR a.owner_user_id <> q.owner_user_id
                   OR q.owner_user_id IS NULL)  -- Exclude self-answers
            GROUP BY eq.question_id
        )
        SELECT
            eq.question_id,
            eq.owner_user_id,
            eq.question_timestamp,
            eq.has_answer,
            eq.has_accepted_answer,
            eq.has_self_answer,
            eq.first_answer_timestamp,
            eq.accepted_answer_timestamp,
            eq.accepted_answer_vote_timestamp,
            COALESCE(hg.helps_given_between_question_and_answer, 0) AS helps_given_between_question_and_answer,
            ab.autobiography_received,
            u.registration_date,
            DENSE_RANK() OVER (ORDER BY eq.owner_user_id, eq.question_id) AS event_id,
            eq.question_timestamp AS phase_two_start  -- Phase 2 starts when question is posted
        FROM eligible_questions eq
        LEFT JOIN helps_given hg ON eq.question_id = hg.question_id
        LEFT JOIN autobiography_badges ab ON eq.owner_user_id = ab.user_id
        LEFT JOIN users u ON eq.owner_user_id = u.user_id;
    """)

    # Add phase boundaries based on question posted timestamp
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            *,
            (phase_two_start - INTERVAL '{window_length} DAYS') AS phase_one_start,
            (phase_two_start + INTERVAL '{window_length} DAYS') AS phase_two_end,
            CASE
                WHEN autobiography_received IS NOT NULL
                     AND autobiography_received <= (phase_two_start - INTERVAL '{window_length} DAYS')
                THEN 1
                ELSE 0
            END AS autobiography_active_phase_one_start,
            CASE
                WHEN autobiography_received IS NOT NULL
                     AND autobiography_received <= phase_two_start
                THEN 1
                ELSE 0
            END AS autobiography_active_phase_two_start,
            CASE
                WHEN registration_date IS NOT NULL
                THEN CAST(EXTRACT(EPOCH FROM ((phase_two_start - INTERVAL '{window_length} DAYS') - registration_date)) / 86400 AS INTEGER)
                ELSE NULL
            END AS days_since_registration_at_phase_one_start
        FROM phase_definitions;
    """)

    # Historical events: Question Asked
    questions_asked_df = con.execute("""
                                     SELECT NULL            AS event_id,
                                            q.owner_user_id AS user_id,
                                            q.creation_date AS timestamp,
            'Question' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'Question' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS has_self_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp,
            NULL AS helps_given_between_question_and_answer,
            NULL AS autobiography_received,
            NULL AS registration_date,
            NULL AS autobiography_active_phase_one_start,
            NULL AS autobiography_active_phase_two_start,
            NULL AS days_since_registration_at_phase_one_start
                                     FROM questions q
                                     WHERE q.owner_user_id IN (
                                         SELECT DISTINCT owner_user_id FROM eligible_questions
                                         );
                                     """).fetchdf()

    # Historical events: Answers provided
    answers_provided_df = con.execute("""
                                      SELECT NULL            AS event_id,
                                             a.owner_user_id AS user_id,
                                             a.creation_date AS timestamp,
            'Answer' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'Answer' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS has_self_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp,
            NULL AS helps_given_between_question_and_answer,
            NULL AS autobiography_received,
            NULL AS registration_date,
            NULL AS autobiography_active_phase_one_start,
            NULL AS autobiography_active_phase_two_start,
            NULL AS days_since_registration_at_phase_one_start
                                      FROM answers a
                                          JOIN questions q
                                      ON a.parent_question_id = q.question_id
                                      WHERE a.owner_user_id IN (
                                          SELECT DISTINCT owner_user_id FROM eligible_questions
                                          )
                                        AND (a.owner_user_id <> q.owner_user_id
                                         OR q.owner_user_id IS NULL)
                                      """).fetchdf()

    # Historical events: AcceptedAnswers received
    accepted_answers_df = con.execute("""
                                      SELECT NULL            AS event_id,
                                             q.owner_user_id AS user_id,
                                             a.creation_date AS timestamp,
            'AcceptedAnswer' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'AcceptedAnswer' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS has_self_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp,
            NULL AS helps_given_between_question_and_answer,
            NULL AS autobiography_received,
            NULL AS registration_date,
            NULL AS autobiography_active_phase_one_start,
            NULL AS autobiography_active_phase_two_start,
            NULL AS days_since_registration_at_phase_one_start
                                      FROM questions q
                                          JOIN answers a
                                      ON q.accepted_answer_id = a.answer_id
                                      WHERE q.owner_user_id IN (
                                          SELECT DISTINCT owner_user_id FROM eligible_questions
                                          )
                                      """).fetchdf()

    # Historical events: Accepted Answer Vote received
    answer_votes_df = con.execute("""
                                  SELECT NULL            AS event_id,
                                         a.owner_user_id AS user_id,
                                         v.vote_date AS timestamp,
            'AcceptedAnswerVote' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'AcceptedAnswerVote' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS has_self_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp,
            NULL AS helps_given_between_question_and_answer,
            NULL AS autobiography_received,
            NULL AS registration_date,
            NULL AS autobiography_active_phase_one_start,
            NULL AS autobiography_active_phase_two_start,
            NULL AS days_since_registration_at_phase_one_start
                                  FROM answers a
                                      JOIN votes v
                                  ON a.answer_id = v.post_id
                                  WHERE a.owner_user_id IN (
                                      SELECT DISTINCT owner_user_id FROM eligible_questions
                                      )
                                  """).fetchdf()

    # Historical events: Accepted Answers posted by the user
    accepted_answers_posted_df = con.execute("""
                                             SELECT NULL            AS event_id,
                                                    a.owner_user_id AS user_id,
                                                    a.creation_date AS timestamp,
            'AcceptedAnswerPosted' AS event,
            NULL AS question_id,
            NULL AS phase_one_start,
            NULL AS phase_two_end,
            'AcceptedAnswerPosted' AS event_history,
            1 AS is_history,
            NULL AS has_answer,
            NULL AS has_accepted_answer,
            NULL AS has_self_answer,
            NULL AS first_answer_timestamp,
            NULL AS accepted_answer_timestamp,
            NULL AS accepted_answer_vote_timestamp,
            NULL AS question_timestamp,
            NULL AS helps_given_between_question_and_answer,
            NULL AS autobiography_received,
            NULL AS registration_date,
            NULL AS autobiography_active_phase_one_start,
            NULL AS autobiography_active_phase_two_start,
            NULL AS days_since_registration_at_phase_one_start
                                             FROM answers a
                                                 JOIN questions q
                                             ON q.accepted_answer_id = a.answer_id
                                             WHERE a.owner_user_id IN (
                                                 SELECT DISTINCT owner_user_id FROM eligible_questions
                                                 )
                                               AND (a.owner_user_id <> q.owner_user_id
                                                OR q.owner_user_id IS NULL)
                                             """).fetchdf()

    historical_events_df = pd.concat([
        questions_asked_df,
        answers_provided_df,
        accepted_answers_df,
        answer_votes_df,
        accepted_answers_posted_df
    ], ignore_index=True)

    con.register("historical_events_df", historical_events_df)
    con.execute("CREATE TEMPORARY TABLE historical_events AS SELECT * FROM historical_events_df;")

    # Current events
    con.execute("""
        CREATE TEMPORARY TABLE phase_one_events AS
        SELECT
            event_id,
            owner_user_id AS user_id,
            phase_one_start AS timestamp,
            'Phase_One_Start' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_One_Start' AS event_history,
            0 AS is_history,
            has_answer,
            has_accepted_answer,
            has_self_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp,
            helps_given_between_question_and_answer,
            autobiography_received,
            registration_date,
            autobiography_active_phase_one_start,
            autobiography_active_phase_two_start,
            days_since_registration_at_phase_one_start
        FROM phase_definitions;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE phase_two_start_events AS
        SELECT
            event_id,
            owner_user_id AS user_id,
            phase_two_start AS timestamp,
            'Phase_Two_Start' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_Start' AS event_history,
            0 AS is_history,
            has_answer,
            has_accepted_answer,
            has_self_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp,
            helps_given_between_question_and_answer,
            autobiography_received,
            registration_date,
            autobiography_active_phase_one_start,
            autobiography_active_phase_two_start,
            days_since_registration_at_phase_one_start
        FROM phase_definitions;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE phase_two_end_events AS
        SELECT
            event_id,
            owner_user_id AS user_id,
            phase_two_end AS timestamp,
            'Phase_Two_End' AS event,
            question_id,
            phase_one_start,
            phase_two_end,
            'Phase_Two_End' AS event_history,
            0 AS is_history,
            has_answer,
            has_accepted_answer,
            has_self_answer,
            first_answer_timestamp,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp,
            helps_given_between_question_and_answer,
            autobiography_received,
            registration_date,
            autobiography_active_phase_one_start,
            autobiography_active_phase_two_start,
            days_since_registration_at_phase_one_start
        FROM phase_definitions;
    """)

    con.execute("""
                CREATE
                TEMPORARY TABLE window_answers AS
                SELECT p.event_id,
                       a.owner_user_id AS user_id,
                       a.creation_date AS timestamp,
            'Window_Answer' AS event,
            p.question_id,
            p.phase_one_start,
            p.phase_two_end,
            'Window_Answer' AS event_history,
            0 AS is_history,
            p.has_answer,
            p.has_accepted_answer,
            p.has_self_answer,
            p.first_answer_timestamp,
            p.accepted_answer_timestamp,
            p.accepted_answer_vote_timestamp,
            p.question_timestamp,
            p.helps_given_between_question_and_answer,
            p.autobiography_received,
            p.registration_date,
            p.autobiography_active_phase_one_start,
            p.autobiography_active_phase_two_start,
            p.days_since_registration_at_phase_one_start
                FROM phase_definitions p
                    JOIN answers a
                ON a.owner_user_id = p.owner_user_id
                    AND a.creation_date BETWEEN p.phase_one_start AND p.phase_two_end
                    JOIN questions q
                    ON a.parent_question_id = q.question_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL);
                """)

    con.execute("""
        CREATE TEMPORARY TABLE current_events AS
        SELECT * FROM phase_one_events
        UNION ALL
        SELECT * FROM phase_two_start_events
        UNION ALL
        SELECT * FROM phase_two_end_events
        UNION ALL
        SELECT * FROM window_answers;
    """)

    con.execute("""
        CREATE TEMPORARY TABLE all_events AS
        SELECT * FROM historical_events
        UNION ALL
        SELECT * FROM current_events;
    """)

    # Add test mode indicator to filename
    test_suffix = f"_TEST{test_user_limit}" if test_mode else ""
    output_path = os.path.join(
        output_folder,
        f"question_centered_model_{window_length}d_{'all_questions' if include_all_questions else 'one_question'}{test_suffix}.parquet"
    )

    con.execute(f"""
        COPY (
            SELECT
                event_id,
                user_id,
                timestamp,
                event,
                question_id,
                phase_one_start,
                phase_two_end,
                event_history,
                is_history,
                has_answer,
                has_accepted_answer,
                has_self_answer,
                first_answer_timestamp,
                accepted_answer_timestamp,
                accepted_answer_vote_timestamp,
                question_timestamp,
                helps_given_between_question_and_answer,
                autobiography_received,
                registration_date,
                autobiography_active_phase_one_start,
                autobiography_active_phase_two_start,
                days_since_registration_at_phase_one_start
            FROM all_events
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.close()
    print(f"Done! Saved dataset to {output_path}")


if __name__ == "__main__":
    input_data_folder = r"..\data\input"
    output_data_folder = r"..\data\input"
    os.makedirs(output_data_folder, exist_ok=True)

    # Test mode - run with random sample of 100k users first
    # print("=" * 60)
    # print("RUNNING TEST MODE (500k random users)")
    # print("=" * 60)
    # for days in [7]:
    #    process_accepted_answer_data(
    #        input_folder=input_data_folder,
    #       output_folder=output_data_folder,
    #      window_length=days,
    #       include_all_questions=True,
    #       test_mode=True,
    #       test_user_limit=100000
    #   )

    # Full mode - uncomment to run on full dataset after testing
    print("\n" + "=" * 60)
    print("RUNNING FULL MODE")
    print("=" * 60)
    for days in [7]:
         process_accepted_answer_data(
             input_folder=input_data_folder,
             output_folder=output_data_folder,
             window_length=days,
             include_all_questions=True,
             test_mode=False
        )