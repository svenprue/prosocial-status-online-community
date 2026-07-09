import os
import duckdb
import pandas as pd
from pathlib import Path

from input_paths import resolve_input_file


def process_accepted_answer_data(
        input_folder: str,
        output_folder: str,
        window_length: int,
        include_all_questions: bool = False,
        test_mode: bool = False,
        test_user_limit: int = 500000
) -> None:
    # Use on-disk DB + temp directory so DuckDB can spill large intermediates.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    duckdb_tmp_dir = Path(output_folder) / "duckdb_tmp"
    duckdb_tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=str(duckdb_tmp_dir / "working.duckdb"))
    # Machine has plenty of RAM (e.g. 1TB); allow DuckDB to use a large chunk, leave headroom for OS/spilling.
    con.execute("PRAGMA memory_limit='200GB';")
    con.execute(f"PRAGMA temp_directory='{duckdb_tmp_dir}';")
    con.execute("PRAGMA max_temp_directory_size='200GiB';")
    con.execute("PRAGMA threads=28;")
    con.execute("PRAGMA enable_progress_bar;")

    questions_path = resolve_input_file("posts_questions")
    answers_path = resolve_input_file("posts_answers")
    votes_path = resolve_input_file("votes")
    badges_path = resolve_input_file("badges")
    users_path = resolve_input_file("users")

    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            OwnerUserId AS owner_user_id,
            AcceptedAnswerId AS accepted_answer_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date,
            Tags AS tags,
            TRY_CAST(ViewCount AS DOUBLE) AS view_count,
            TRY_CAST(BodyLenChars AS INTEGER) AS body_len_chars,
            TRY_CAST(BodyLenWords AS INTEGER) AS body_len_words,
            TRY_CAST(TitleLenChars AS INTEGER) AS title_len_chars,
            TRY_CAST(TitleLenWords AS INTEGER) AS title_len_words,
            TRY_CAST(NCodeBlocks AS INTEGER) AS n_code_blocks
        FROM '{questions_path}'
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            OwnerUserId AS owner_user_id,
            ParentId AS parent_question_id,
            CAST(Score AS INTEGER) AS score,
            CAST(CreationDate AS TIMESTAMP) AS creation_date,
            TRY_CAST(BodyLenChars AS INTEGER) AS body_len_chars
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

    # Autobiographer badge disabled for now
    # con.execute(f"""
    #     CREATE TEMPORARY VIEW autobiography_badges AS
    #     SELECT
    #         UserId AS user_id,
    #         CAST(Date AS TIMESTAMP) AS autobiography_received
    #     FROM '{badges_path}'
    #     WHERE Name = 'Autobiographer';
    # """)

    con.execute(f"""
        CREATE TEMPORARY VIEW users AS
        SELECT
            Id AS user_id,
            CAST(CreationDate AS TIMESTAMP) AS registration_date,
            TRY_CAST(Reputation AS INTEGER) AS reputation
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
            WITH answers_with_votes AS (
                SELECT
                    a.parent_question_id AS question_id,
                    a.answer_id,
                    a.creation_date,
                    a.score,
                    q.creation_date AS question_timestamp,
                    COUNT(v.post_id) AS vote_count
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                LEFT JOIN votes v ON a.answer_id = v.post_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-answers
                GROUP BY a.parent_question_id, a.answer_id, a.creation_date, a.score, q.creation_date
            ),
            non_downvoted_within_7d AS (
                SELECT
                    question_id,
                    answer_id,
                    creation_date,
                    score,
                    vote_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY question_id
                        ORDER BY creation_date
                    ) AS rn
                FROM answers_with_votes
                WHERE score >= 0  -- Non-downvoted
                  AND creation_date <= question_timestamp + INTERVAL '7 DAYS'  -- Within 7 days
            ),
            first_answer_fallback AS (
                SELECT
                    question_id,
                    answer_id,
                    creation_date,
                    score,
                    vote_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY question_id
                        ORDER BY creation_date
                    ) AS rn
                FROM answers_with_votes
                WHERE question_id NOT IN (SELECT question_id FROM non_downvoted_within_7d WHERE rn = 1)
            ),
            first_answer_candidates AS (
                SELECT * FROM non_downvoted_within_7d WHERE rn = 1
                UNION ALL
                SELECT * FROM first_answer_fallback WHERE rn = 1
            ),
            first_answers AS (
                SELECT
                    question_id,
                    answer_id,
                    creation_date AS first_answer_timestamp,
                    score AS first_answer_score,
                    vote_count AS first_answer_vote_count
                FROM first_answer_candidates
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
            downvoted_within_7d AS (
                SELECT DISTINCT q.question_id
                FROM questions q
                JOIN answers a ON a.parent_question_id = q.question_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)
                  AND a.score < 0
                  AND a.creation_date <= q.creation_date + INTERVAL '7 DAYS'
            )
            SELECT
                q.question_id,
                q.owner_user_id,
                q.creation_date AS question_timestamp,
                COALESCE(ac.answer_count > 0, FALSE) AS has_answer,
                (NOT COALESCE(ac.answer_count > 0, FALSE) AND d.question_id IS NOT NULL) AS has_unhelpful_answer,
                aa.accepted_answer_timestamp IS NOT NULL AS has_accepted_answer,
                COALESCE(sa.self_answer_count > 0, FALSE) AS has_self_answer,
                fa.first_answer_timestamp,
                fa.first_answer_score,
                fa.first_answer_vote_count,
                fa.answer_id AS first_answer_id,
                abl.body_len_chars AS first_answer_body_len_chars,
                aa.accepted_answer_timestamp,
                aa.accepted_answer_vote_timestamp,
                q.view_count,
                q.body_len_chars,
                q.title_len_chars,
                q.n_code_blocks,
                u.reputation AS owner_reputation
            FROM questions q
            LEFT JOIN first_answers fa ON q.question_id = fa.question_id
            LEFT JOIN answers abl ON fa.answer_id = abl.answer_id
            LEFT JOIN users u ON q.owner_user_id = u.user_id
            LEFT JOIN accepted_answers aa ON q.question_id = aa.question_id
            LEFT JOIN answer_counts ac ON q.question_id = ac.question_id
            LEFT JOIN self_answers sa ON q.question_id = sa.question_id
            LEFT JOIN downvoted_within_7d d ON q.question_id = d.question_id
            WHERE q.owner_user_id IS NOT NULL
              {user_filter};
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW eligible_questions AS
            WITH answers_with_votes AS (
                SELECT
                    a.parent_question_id AS question_id,
                    a.answer_id,
                    a.creation_date,
                    a.score,
                    q.creation_date AS question_timestamp,
                    COUNT(v.post_id) AS vote_count
                FROM answers a
                JOIN questions q ON a.parent_question_id = q.question_id
                LEFT JOIN votes v ON a.answer_id = v.post_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)  -- Exclude self-answers
                GROUP BY a.parent_question_id, a.answer_id, a.creation_date, a.score, q.creation_date
            ),
            non_downvoted_within_7d AS (
                SELECT
                    question_id,
                    answer_id,
                    creation_date,
                    score,
                    vote_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY question_id
                        ORDER BY creation_date
                    ) AS rn
                FROM answers_with_votes
                WHERE score >= 0  -- Non-downvoted
                  AND creation_date <= question_timestamp + INTERVAL '7 DAYS'  -- Within 7 days
            ),
            first_answer_fallback AS (
                SELECT
                    question_id,
                    answer_id,
                    creation_date,
                    score,
                    vote_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY question_id
                        ORDER BY creation_date
                    ) AS rn
                FROM answers_with_votes
                WHERE question_id NOT IN (SELECT question_id FROM non_downvoted_within_7d WHERE rn = 1)
            ),
            first_answer_candidates AS (
                SELECT * FROM non_downvoted_within_7d WHERE rn = 1
                UNION ALL
                SELECT * FROM first_answer_fallback WHERE rn = 1
            ),
            first_answers AS (
                SELECT
                    question_id,
                    answer_id,
                    creation_date AS first_answer_timestamp,
                    score AS first_answer_score,
                    vote_count AS first_answer_vote_count
                FROM first_answer_candidates
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
            downvoted_within_7d AS (
                SELECT DISTINCT q.question_id
                FROM questions q
                JOIN answers a ON a.parent_question_id = q.question_id
                WHERE (a.owner_user_id <> q.owner_user_id OR q.owner_user_id IS NULL)
                  AND a.score < 0
                  AND a.creation_date <= q.creation_date + INTERVAL '7 DAYS'
            ),
            raw AS (
                SELECT
                    q.question_id,
                    q.owner_user_id,
                    q.creation_date AS question_timestamp,
                    COALESCE(ac.answer_count > 0, FALSE) AS has_answer,
                    (NOT COALESCE(ac.answer_count > 0, FALSE) AND d.question_id IS NOT NULL) AS has_unhelpful_answer,
                    q.accepted_answer_id IS NOT NULL AS has_accepted_answer,
                    COALESCE(sa.self_answer_count > 0, FALSE) AS has_self_answer,
                    fa.first_answer_timestamp,
                    fa.first_answer_score,
                    fa.first_answer_vote_count,
                    fa.answer_id AS first_answer_id,
                    abl.body_len_chars AS first_answer_body_len_chars,
                    aa.accepted_answer_timestamp,
                    aa.accepted_answer_vote_timestamp,
                    q.view_count,
                    q.body_len_chars,
                    q.title_len_chars,
                    q.n_code_blocks,
                    u.reputation AS owner_reputation,
                    ROW_NUMBER() OVER (
                        PARTITION BY q.owner_user_id
                        ORDER BY RANDOM()
                    ) AS rn
                FROM questions q
                LEFT JOIN first_answers fa ON q.question_id = fa.question_id
                LEFT JOIN answers abl ON fa.answer_id = abl.answer_id
                LEFT JOIN users u ON q.owner_user_id = u.user_id
                LEFT JOIN accepted_answers aa ON q.question_id = aa.question_id
                LEFT JOIN answer_counts ac ON q.question_id = ac.question_id
                LEFT JOIN self_answers sa ON q.question_id = sa.question_id
                LEFT JOIN downvoted_within_7d d ON q.question_id = d.question_id
                WHERE q.owner_user_id IS NOT NULL
                  {user_filter}
            )
            SELECT
                question_id,
                owner_user_id,
                question_timestamp,
                has_answer,
                has_unhelpful_answer,
                has_accepted_answer,
                has_self_answer,
                first_answer_timestamp,
                first_answer_score,
                first_answer_vote_count,
                first_answer_id,
                first_answer_body_len_chars,
                accepted_answer_timestamp,
                accepted_answer_vote_timestamp,
                view_count,
                body_len_chars,
                title_len_chars,
                n_code_blocks,
                owner_reputation
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

    # Build tag dictionary and per-question tag ID lists based on eligible questions.
    # Tags in posts_questions are angle-bracket delimited, e.g. <android><listview>.
    # Normalize to comma-separated (replace '><' with ',', strip '<' and '>'), then split.
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE question_tags_clean AS
        SELECT
            q.question_id,
            TRIM(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REPLACE(REPLACE(REPLACE(q.tags, '><', ','), '<', ''), '>', ''),
                        '\\\\[|\\\\]',
                        ''
                    ),
                    '''',
                    ''
                )
            ) AS tags_clean
        FROM questions q
        JOIN eligible_questions eq ON q.question_id = eq.question_id
        WHERE q.tags IS NOT NULL;
    """)

    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE tag_exploded AS
        SELECT
            question_id,
            TRIM(t.tag_val) AS tag_name
        FROM question_tags_clean,
             UNNEST(STRING_SPLIT(tags_clean, ',')) AS t(tag_val)
        WHERE TRIM(t.tag_val) != '';
    """)

    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE tag_dictionary AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY tag_name) AS tag_id,
            tag_name,
            COUNT(*) AS tag_frequency
        FROM tag_exploded
        GROUP BY tag_name;
    """)

    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE question_tag_ids AS
        SELECT
            t.question_id,
            LIST(td.tag_id ORDER BY td.tag_id) AS tag_ids
        FROM tag_exploded t
        JOIN tag_dictionary td ON t.tag_name = td.tag_name
        GROUP BY t.question_id;
    """)

    # Compute helps_given in user batches (with ample RAM we can use larger batches for speed).
    HELPS_GIVEN_BATCH_SIZE = 200_000  # users per batch
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE user_batches AS
        SELECT
            owner_user_id,
            CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY owner_user_id) - 1) / %d) AS INTEGER) AS batch_id
        FROM (SELECT DISTINCT owner_user_id FROM eligible_questions) t;
    """ % HELPS_GIVEN_BATCH_SIZE)
    max_batch_id = con.execute("SELECT COALESCE(MAX(batch_id), 0) FROM user_batches").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE helps_given (
            question_id BIGINT,
            helps_given_between_question_and_answer INTEGER
        );
    """)

    for batch_id in range(0, int(max_batch_id) + 1):
        print(f"  helps_given batch %d / %d ..." % (batch_id, int(max_batch_id)))
        con.execute("""
            INSERT INTO helps_given (question_id, helps_given_between_question_and_answer)
            WITH batch_eq AS (
                SELECT eq.*
                FROM eligible_questions eq
                JOIN user_batches ub ON eq.owner_user_id = ub.owner_user_id
                WHERE ub.batch_id = %d
                  AND eq.first_answer_timestamp IS NOT NULL
            ),
            batch_answers AS (
                SELECT a.*
                FROM answers a
                JOIN user_batches ub ON a.owner_user_id = ub.owner_user_id
                WHERE ub.batch_id = %d
            ),
            h AS (
                SELECT
                    batch_eq.question_id,
                    COUNT(a.answer_id) AS helps_given_between_question_and_answer
                FROM batch_eq
                LEFT JOIN batch_answers a
                  ON a.owner_user_id = batch_eq.owner_user_id
                 AND a.creation_date >= batch_eq.question_timestamp
                 AND a.creation_date < batch_eq.first_answer_timestamp
                LEFT JOIN questions q ON a.parent_question_id = q.question_id
                WHERE (a.owner_user_id IS NULL
                       OR a.owner_user_id <> q.owner_user_id
                       OR q.owner_user_id IS NULL)
                GROUP BY batch_eq.question_id
            )
            SELECT question_id, helps_given_between_question_and_answer FROM h;
        """ % (batch_id, batch_id))

    # Build phase_definitions from pre-computed helps_given (no heavy join).
    con.execute("""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            eq.question_id,
            eq.owner_user_id,
            eq.question_timestamp,
            eq.has_answer,
            eq.has_unhelpful_answer,
            eq.has_accepted_answer,
            eq.has_self_answer,
            eq.first_answer_timestamp,
            eq.first_answer_score,
            eq.first_answer_vote_count,
            eq.first_answer_id,
            eq.first_answer_body_len_chars,
            eq.accepted_answer_timestamp,
            eq.accepted_answer_vote_timestamp,
            eq.view_count,
            eq.body_len_chars,
            eq.title_len_chars,
            eq.n_code_blocks,
            eq.owner_reputation,
            COALESCE(hg.helps_given_between_question_and_answer, 0) AS helps_given_between_question_and_answer,
            u.registration_date,
            qt.tag_ids,
            DENSE_RANK() OVER (ORDER BY eq.owner_user_id, eq.question_id) AS event_id,
            eq.question_timestamp AS phase_two_start
        FROM eligible_questions eq
        LEFT JOIN helps_given hg ON eq.question_id = hg.question_id
        LEFT JOIN users u ON eq.owner_user_id = u.user_id
        LEFT JOIN question_tag_ids qt ON eq.question_id = qt.question_id;
    """)

    # Add phase boundaries based on question posted timestamp
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE phase_definitions AS
        SELECT
            *,
            (phase_two_start - INTERVAL '{window_length} DAYS') AS phase_one_start,
            (phase_two_start + INTERVAL '{window_length} DAYS') AS phase_two_end,
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
                                            NULL AS has_unhelpful_answer,
                                            NULL AS has_accepted_answer,
                                            NULL AS has_self_answer,
                                            NULL AS first_answer_timestamp,
                                            NULL AS first_answer_score,
                                            NULL AS first_answer_vote_count,
                                            NULL AS first_answer_id,
                                            NULL AS first_answer_body_len_chars,
                                            NULL AS view_count,
                                            NULL AS body_len_chars,
                                            NULL AS title_len_chars,
                                            NULL AS n_code_blocks,
                                            NULL AS owner_reputation,
                                            NULL AS accepted_answer_timestamp,
                                            NULL AS accepted_answer_vote_timestamp,
                                            NULL AS question_timestamp,
                                            NULL AS helps_given_between_question_and_answer,
                                            NULL AS registration_date,
                                            NULL AS days_since_registration_at_phase_one_start,
                                            NULL AS tag_ids
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
                                             NULL AS has_unhelpful_answer,
                                             NULL AS has_accepted_answer,
                                             NULL AS has_self_answer,
                                             NULL AS first_answer_timestamp,
                                             NULL AS first_answer_score,
                                             NULL AS first_answer_vote_count,
                                             NULL AS first_answer_id,
                                             NULL AS first_answer_body_len_chars,
                                             NULL AS view_count,
                                             NULL AS body_len_chars,
                                             NULL AS title_len_chars,
                                             NULL AS n_code_blocks,
                                             NULL AS owner_reputation,
                                             NULL AS accepted_answer_timestamp,
                                             NULL AS accepted_answer_vote_timestamp,
                                             NULL AS question_timestamp,
                                             NULL AS helps_given_between_question_and_answer,
                                             NULL AS registration_date,
                                             NULL AS days_since_registration_at_phase_one_start,
                                             NULL AS tag_ids
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
                                      NULL AS has_unhelpful_answer,
                                      NULL AS has_accepted_answer,
                                      NULL AS has_self_answer,
                                      NULL AS first_answer_timestamp,
                                      NULL AS first_answer_score,
                                      NULL AS first_answer_vote_count,
                                      NULL AS first_answer_id,
                                      NULL AS first_answer_body_len_chars,
                                      NULL AS view_count,
                                      NULL AS body_len_chars,
                                      NULL AS title_len_chars,
                                      NULL AS n_code_blocks,
                                      NULL AS owner_reputation,
                                      NULL AS accepted_answer_timestamp,
                                      NULL AS accepted_answer_vote_timestamp,
                                      NULL AS question_timestamp,
                                      NULL AS helps_given_between_question_and_answer,
                                      NULL AS registration_date,
                                      NULL AS days_since_registration_at_phase_one_start,
                                      NULL AS tag_ids
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
                                  NULL AS has_unhelpful_answer,
                                  NULL AS has_accepted_answer,
                                  NULL AS has_self_answer,
                                  NULL AS first_answer_timestamp,
                                  NULL AS first_answer_score,
                                  NULL AS first_answer_vote_count,
                                  NULL AS first_answer_id,
                                  NULL AS first_answer_body_len_chars,
                                  NULL AS view_count,
                                  NULL AS body_len_chars,
                                  NULL AS title_len_chars,
                                  NULL AS n_code_blocks,
                                  NULL AS owner_reputation,
                                  NULL AS accepted_answer_timestamp,
                                  NULL AS accepted_answer_vote_timestamp,
                                  NULL AS question_timestamp,
                                  NULL AS helps_given_between_question_and_answer,
                                  NULL AS registration_date,
                                  NULL AS days_since_registration_at_phase_one_start,
                                  NULL AS tag_ids
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
                                             NULL AS has_unhelpful_answer,
                                             NULL AS has_accepted_answer,
                                             NULL AS has_self_answer,
                                             NULL AS first_answer_timestamp,
                                             NULL AS first_answer_score,
                                             NULL AS first_answer_vote_count,
                                             NULL AS first_answer_id,
                                             NULL AS first_answer_body_len_chars,
                                             NULL AS view_count,
                                             NULL AS body_len_chars,
                                             NULL AS title_len_chars,
                                             NULL AS n_code_blocks,
                                             NULL AS owner_reputation,
                                             NULL AS accepted_answer_timestamp,
                                             NULL AS accepted_answer_vote_timestamp,
                                             NULL AS question_timestamp,
                                             NULL AS helps_given_between_question_and_answer,
                                             NULL AS registration_date,
                                             NULL AS days_since_registration_at_phase_one_start,
                                             NULL AS tag_ids
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
            has_unhelpful_answer,
            has_accepted_answer,
            has_self_answer,
            first_answer_timestamp,
            first_answer_score,
            first_answer_vote_count,
            first_answer_id,
            first_answer_body_len_chars,
            view_count,
            body_len_chars,
            title_len_chars,
            n_code_blocks,
            owner_reputation,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp,
            helps_given_between_question_and_answer,
            registration_date,
            days_since_registration_at_phase_one_start,
            tag_ids
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
            has_unhelpful_answer,
            has_accepted_answer,
            has_self_answer,
            first_answer_timestamp,
            first_answer_score,
            first_answer_vote_count,
            first_answer_id,
            first_answer_body_len_chars,
            view_count,
            body_len_chars,
            title_len_chars,
            n_code_blocks,
            owner_reputation,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp,
            helps_given_between_question_and_answer,
            registration_date,
            days_since_registration_at_phase_one_start,
            tag_ids
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
            has_unhelpful_answer,
            has_accepted_answer,
            has_self_answer,
            first_answer_timestamp,
            first_answer_score,
            first_answer_vote_count,
            first_answer_id,
            first_answer_body_len_chars,
            view_count,
            body_len_chars,
            title_len_chars,
            n_code_blocks,
            owner_reputation,
            accepted_answer_timestamp,
            accepted_answer_vote_timestamp,
            question_timestamp,
            helps_given_between_question_and_answer,
            registration_date,
            days_since_registration_at_phase_one_start,
            tag_ids
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
            p.has_unhelpful_answer,
            p.has_accepted_answer,
            p.has_self_answer,
            p.first_answer_timestamp,
            p.first_answer_score,
            p.first_answer_vote_count,
            p.first_answer_id,
            p.first_answer_body_len_chars,
            p.view_count,
            p.body_len_chars,
            p.title_len_chars,
            p.n_code_blocks,
            p.owner_reputation,
            p.accepted_answer_timestamp,
            p.accepted_answer_vote_timestamp,
            p.question_timestamp,
            p.helps_given_between_question_and_answer,
            p.registration_date,
            p.days_since_registration_at_phase_one_start,
            p.tag_ids
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

    tag_output_path = os.path.join(
        output_folder,
        f"question_centered_model_{window_length}d_{'all_questions' if include_all_questions else 'one_question'}{test_suffix}_tags.parquet"
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
                has_unhelpful_answer,
                has_accepted_answer,
                has_self_answer,
                first_answer_timestamp,
                first_answer_score,
                first_answer_vote_count,
                first_answer_id,
                first_answer_body_len_chars,
                view_count,
                body_len_chars,
                title_len_chars,
                n_code_blocks,
                owner_reputation,
                accepted_answer_timestamp,
                accepted_answer_vote_timestamp,
                question_timestamp,
                helps_given_between_question_and_answer,
                registration_date,
                days_since_registration_at_phase_one_start,
                tag_ids
            FROM all_events
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.execute(f"""
        COPY (
            SELECT
                tag_id,
                tag_name,
                tag_frequency
            FROM tag_dictionary
        )
        TO '{tag_output_path}'
        (FORMAT PARQUET, COMPRESSION 'GZIP');
    """)

    con.close()
    print(f"Done! Saved dataset to {output_path}")
    print(f"Saved tag dictionary to {tag_output_path}")


# Alias for main.py which calls process_question_data
process_question_data = process_accepted_answer_data

if __name__ == "__main__":
    # Resolve base directory relative to this file so paths work reliably on Linux
    base_dir = Path(__file__).resolve().parent.parent
    input_data_folder = base_dir / "data" / "input"
    output_data_folder = base_dir / "data" / "input"
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