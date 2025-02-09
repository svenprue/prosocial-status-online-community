import duckdb
import pandas as pd
import inflect

# Initialize the inflect engine
p = inflect.engine()

def create_views(con, input_folder):
    con.execute(f"""
        CREATE VIEW questions AS
        SELECT 
            Id AS question_id,
            OwnerUserId AS user_id,
            AcceptedAnswerId AS accepted_answer_id,
            CreationDate AS timestamp_of_question
        FROM parquet_scan('{input_folder}/posts_questions.parquet')
    """)

    con.execute(f"""
        CREATE VIEW answers AS
        SELECT 
            Id AS answer_id,
            ParentId AS question_id,
            OwnerUserId AS responder_id,
            CreationDate AS timestamp_of_answer
        FROM parquet_scan('{input_folder}/posts_answers.parquet')
    """)

def generate_timeline_events(con, window_length_days):
    return con.execute(f"""
        SELECT
            q.user_id,
            q.timestamp_of_question,
            a.timestamp_of_answer,
            DATE_DIFF('minute', q.timestamp_of_question, a.timestamp_of_answer) AS response_time,
            (a.timestamp_of_answer - INTERVAL {window_length_days} DAY) AS window_start,
            a.timestamp_of_answer AS answer_moment,
            (a.timestamp_of_answer + INTERVAL {window_length_days} DAY) AS window_end,
            0 AS is_history
        FROM questions q
        INNER JOIN answers a ON q.question_id = a.question_id
    """).df()

def generate_historical_questions(con, user_ids):
    user_ids_str = ', '.join(map(str, user_ids))
    return con.execute(f"""
        SELECT
            user_id,
            timestamp_of_question,
            NULL AS timestamp_of_answer,
            NULL AS response_time,
            1 AS is_history
        FROM questions
        WHERE user_id IN ({user_ids_str})
    """).df()

def generate_historical_accepted_answers(con, user_ids):
    user_ids_str = ', '.join(map(str, user_ids))
    return con.execute(f"""
        SELECT
            q.user_id,
            q.timestamp_of_question,
            a.timestamp_of_answer,
            DATE_DIFF('minute', q.timestamp_of_question, a.timestamp_of_answer) AS response_time,
            1 AS is_history
        FROM questions q
        INNER JOIN answers a ON q.accepted_answer_id = a.answer_id
        WHERE q.user_id IN ({user_ids_str})
    """).df()

def generate_historical_provided_answers(con, user_ids):
    user_ids_str = ', '.join(map(str, user_ids))
    return con.execute(f"""
        SELECT
            a.responder_id AS user_id,
            q.timestamp_of_question,
            a.timestamp_of_answer,
            DATE_DIFF('minute', q.timestamp_of_question, a.timestamp_of_answer) AS response_time,
            1 AS is_history
        FROM answers a
        INNER JOIN questions q ON a.question_id = q.question_id
        WHERE a.responder_id IN ({user_ids_str})
    """).df()

def process_data(input_folder, output_folder, window_length_days):
    con = duckdb.connect(database=':memory:')
    create_views(con, input_folder)

    timeline_events = generate_timeline_events(con, window_length_days)
    user_ids = timeline_events['user_id'].unique()

    historical_questions = generate_historical_questions(con, user_ids)
    historical_accepted_answers = generate_historical_accepted_answers(con, user_ids)
    historical_provided_answers = generate_historical_provided_answers(con, user_ids)

    final_dataset = (pd.concat([timeline_events, historical_questions, historical_accepted_answers, historical_provided_answers])
                     .sort_values(['user_id', 'timestamp_of_question'])
                     .reset_index(drop=True))

    final_dataset['timestamp_of_question'] = pd.to_datetime(final_dataset['timestamp_of_question'])
    final_dataset['timestamp_of_answer'] = pd.to_datetime(final_dataset['timestamp_of_answer'])

    # Convert the number of days to words
    window_length_days_word = p.number_to_words(window_length_days).replace('-', '_')

    final_dataset.to_parquet(f'{output_folder}/model_one_{window_length_days_word}_days.parquet')

def main():
    input_folder = './01_input_data'  # Specify your input folder here
    output_folder = './01_input_data'  # Specify your output folder here

    process_data(input_folder, output_folder, 3)
    process_data(input_folder, output_folder, 7)
    process_data(input_folder, output_folder, 14)
    process_data(input_folder, output_folder, 30)

if __name__ == "__main__":
    main()