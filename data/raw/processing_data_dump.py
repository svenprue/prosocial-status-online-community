import pandas as pd
import lxml.etree as ET
from fastparquet import write
from tqdm import tqdm
import duckdb
import os


def parse_generic_row_posts(elem):
    attrib = elem.attrib
    try:
        post_type_id = attrib.get('PostTypeId', None)
        if post_type_id not in ["1", "2"]:
            return None

        return {
            'Id': attrib.get('Id', None),
            'PostTypeId': post_type_id,
            'CreationDate': attrib.get('CreationDate', None),
            'OwnerUserId': attrib.get('OwnerUserId', None),
            'Score': attrib.get('Score', None),
            'AcceptedAnswerId': attrib.get('AcceptedAnswerId', None),
            'ParentId': attrib.get('ParentId', None)
        }

    except Exception as e:
        print(f"Error parsing post row (ID: {attrib.get('Id', 'N/A')}): {e}")
        return None


def parse_generic_row_votes(elem):
    attrib = elem.attrib
    try:
        vote_type_id = attrib.get('VoteTypeId', None)
        if vote_type_id not in ["1", "2", "3", "8", "9"]:
            return None

        return {
            'Id': attrib.get('Id', None),
            'PostId': attrib.get('PostId', None),
            'VoteTypeId': vote_type_id,
            'CreationDate': attrib.get('CreationDate', None),
            'UserId': attrib.get('UserId', None),
            'BountyAmount': attrib.get('BountyAmount', None),
        }

    except Exception as e:
        print(f"Error parsing vote row (ID: {attrib.get('Id', 'N/A')}): {e}")
        return None


def parse_generic_row_users(elem):
    attrib = elem.attrib
    try:
        return {
            'Id': attrib.get('Id', None),
            'CreationDate': attrib.get('CreationDate', None)
        }

    except Exception as e:
        print(f"Error parsing user row (ID: {attrib.get('Id', 'N/A')}): {e}")
        return None


def parse_generic_row_badges(elem):
    attrib = elem.attrib
    try:
        return {
            'Id': attrib.get('Id', None),
            'UserId': attrib.get('UserId', None),
            'Name': attrib.get('Name', None),
            'Date': attrib.get('Date', None),
            'Class': attrib.get('Class', None)
        }

    except Exception as e:
        print(f"Error parsing badge row (ID: {attrib.get('Id', 'N/A')}): {e}")
        return None


def convert_xml_to_parquet(xml_path, parquet_path, parse_function, chunk_size=1000000):
    chunk_data = []
    first = True
    with open(xml_path, 'rb') as f:
        context = ET.iterparse(f, events=('end',), tag='row', recover=True)

        pbar = tqdm(desc=f"Processing {os.path.basename(xml_path)}", unit="rows")

        for event, elem in context:
            record = parse_function(elem)
            pbar.update(1)
            if record:
                chunk_data.append(record)

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

            if len(chunk_data) >= chunk_size:
                df = pd.DataFrame(chunk_data)
                write(parquet_path, df, compression='snappy', write_index=False, append=not first)
                first = False
                chunk_data = []

        if chunk_data:
            df = pd.DataFrame(chunk_data)
            write(parquet_path, df, compression='snappy', write_index=False, append=not first)
        pbar.close()


def split_parquet_by_column(parquet_file_path, output_folder, column, values, custom_names=None):
    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute(f"CREATE VIEW data AS SELECT * FROM '{parquet_file_path}'")

    for i, value in enumerate(values):
        df = con.execute(f"SELECT * FROM data WHERE {column} = {value}").df()

        if custom_names and i < len(custom_names):
            output_filename = custom_names[i]
        else:
            output_filename = f'{column}_{value}.parquet'

        output_path = os.path.join(output_folder, output_filename)
        df.to_parquet(output_path, index=False)

    con.close()


def process_posts(posts_file_path, output_folder):
    posts_parquet_path = os.path.join(output_folder, 'Posts.parquet')
    convert_xml_to_parquet(posts_file_path, posts_parquet_path, parse_generic_row_posts)
    split_parquet_by_column(
        posts_parquet_path,
        output_folder,
        'PostTypeId',
        [1, 2],
        custom_names=['posts_questions.parquet', 'posts_answers.parquet']
    )


def process_votes(votes_file_path, output_folder):
    votes_parquet_path = os.path.join(output_folder, 'Votes.parquet')
    convert_xml_to_parquet(votes_file_path, votes_parquet_path, parse_generic_row_votes)


def process_users(users_file_path, output_folder):
    users_parquet_path = os.path.join(output_folder, 'Users.parquet')
    convert_xml_to_parquet(users_file_path, users_parquet_path, parse_generic_row_users)


def process_badges(badges_file_path, output_folder):
    badges_parquet_path = os.path.join(output_folder, 'Badges.parquet')
    convert_xml_to_parquet(badges_file_path, badges_parquet_path, parse_generic_row_badges)


def fix_column_types(parquet_file_paths):
    con = duckdb.connect(database=':memory:', read_only=False)

    for parquet_file_path in parquet_file_paths:
        table_name = os.path.splitext(os.path.basename(parquet_file_path))[0]

        con.execute(f"CREATE TABLE temp AS SELECT * FROM '{parquet_file_path}'")

        # Cast ID columns to INTEGER
        id_columns = con.execute(f"PRAGMA table_info('temp')").fetchdf()
        for col in id_columns['name']:
            if any(id_keyword in col.lower() for id_keyword in
                   ['id', 'userid', 'owneruserid', 'AcceptedAnswerId', 'ParentId', 'Score']):
                con.execute(f"ALTER TABLE temp ALTER COLUMN {col} SET DATA TYPE INTEGER")

        # Cast date columns to TIMESTAMP
        for col in id_columns['name']:
            if any(date_keyword in col.lower() for date_keyword in ['date', 'creationdate']):
                con.execute(f"ALTER TABLE temp ALTER COLUMN {col} SET DATA TYPE TIMESTAMP")

        # Overwrite the original Parquet file
        con.execute(f"COPY temp TO '{parquet_file_path}' (FORMAT PARQUET, OVERWRITE TRUE)")
        con.execute("DROP TABLE temp")

    con.close()


def main():
    input_folder = './'  # Specify your input folder here
    output_folder = '../input/'  # Specify your output folder here

    posts_file_path = os.path.join(input_folder, 'Posts.xml')
    votes_file_path = os.path.join(input_folder, 'Votes.xml')
    users_file_path = os.path.join(input_folder, 'Users.xml')
    badges_file_path = os.path.join(input_folder, 'Badges.xml')

    process_posts(posts_file_path, output_folder)
    process_votes(votes_file_path, output_folder)
    process_users(users_file_path, output_folder)
    process_badges(badges_file_path, output_folder)

    # Fix column types for all processed parquet files
    parquet_files = [
        os.path.join(output_folder, 'posts_answers.parquet'),
        os.path.join(output_folder, 'posts_questions.parquet'),
        os.path.join(output_folder, 'Votes.parquet'),
        os.path.join(output_folder, 'Users.parquet'),
        os.path.join(output_folder, 'Badges.parquet')
    ]
    fix_column_types(parquet_files)


if __name__ == "__main__":
    main()