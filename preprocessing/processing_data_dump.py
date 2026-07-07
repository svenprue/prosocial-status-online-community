import html as _html
import re

import pandas as pd
import lxml.etree as ET
from fastparquet import write
from tqdm import tqdm
import duckdb
import os
from pathlib import Path

# Wh-words used to flag question-form titles
_WH_WORDS = {'how', 'what', 'why', 'when', 'where', 'who', 'which'}
_SENTENCE_RE = re.compile(r'[.!?]+')
# The dump's Body is machine-rendered, sanitized HTML (code is entity-escaped
# inside <pre><code>), so regex tag handling is safe and ~10x faster than a
# full HTML parse at this volume.
_PRE_RE = re.compile(r'<pre[^>]*>.*?</pre>', re.S)
_TAG_RE = re.compile(r'<[^>]+>')
_CODE_RE = re.compile(r'<code[\s>]')
_P_RE = re.compile(r'<p[\s>]')
_A_RE = re.compile(r'<a[\s>]')
_IMG_RE = re.compile(r'<img[\s>]')
_LIST_RE = re.compile(r'<[uo]l[\s>]')
_BQ_RE = re.compile(r'<blockquote[\s>]')

# Text metrics derived from Body/Title at parse time; the raw text itself is
# never persisted (Body is rendered HTML in the dump and too heavy to keep).
TEXT_METRIC_COLUMNS = [
    'BodyLenChars', 'BodyLenWords',
    'NCodeBlocks', 'NInlineCode', 'CodeLenChars',
    'NParagraphs', 'NLinks', 'NImages', 'NLists', 'NBlockquotes',
    'NSentences', 'NLongWords', 'AvgWordLenChars',
    'TitleLenChars', 'TitleLenWords', 'TitleIsQuestion',
]


def compute_text_metrics(body_html, title):
    m = dict.fromkeys(TEXT_METRIC_COLUMNS, None)

    if title:
        t = title.strip()
        words = t.split()
        m['TitleLenChars'] = len(t)
        m['TitleLenWords'] = len(words)
        first = words[0].lower() if words else ''
        m['TitleIsQuestion'] = int(t.endswith('?') or first in _WH_WORDS)

    if body_html:
        try:
            code_parts = _PRE_RE.findall(body_html)
            code_len = 0
            for c in code_parts:
                c = _TAG_RE.sub('', c)
                if '&' in c:
                    c = _html.unescape(c)
                code_len += len(c)

            no_pre = _PRE_RE.sub(' ', body_html)
            prose = _TAG_RE.sub(' ', no_pre)
            if '&' in prose:
                prose = _html.unescape(prose)

            words = prose.split()
            n_words = len(words)
            char_sum = sum(len(w) for w in words)
            # single-space-normalized prose length (tag removal inserts spaces)
            prose_len = char_sum + max(0, n_words - 1)

            m['BodyLenChars'] = prose_len + code_len
            m['BodyLenWords'] = n_words
            m['NCodeBlocks'] = len(code_parts)
            m['NInlineCode'] = len(_CODE_RE.findall(no_pre))
            m['CodeLenChars'] = code_len
            m['NParagraphs'] = len(_P_RE.findall(no_pre))
            m['NLinks'] = len(_A_RE.findall(no_pre))
            m['NImages'] = len(_IMG_RE.findall(no_pre))
            m['NLists'] = len(_LIST_RE.findall(no_pre))
            m['NBlockquotes'] = len(_BQ_RE.findall(no_pre))
            m['NSentences'] = len(_SENTENCE_RE.findall(prose))
            m['NLongWords'] = sum(1 for w in words if len(w) > 6)
            m['AvgWordLenChars'] = round(char_sum / n_words, 2) if n_words else 0.0
        except Exception:
            # Unexpected content: fall back to crude whole-string length
            m['BodyLenChars'] = len(body_html)

    return m


def parse_generic_row_posts(elem):
    attrib = elem.attrib
    try:
        post_type_id = attrib.get('PostTypeId', None)
        if post_type_id not in ["1", "2"]:
            return None

        view_count = attrib.get('ViewCount', None)
        record = {
            'Id': attrib.get('Id', None),
            'PostTypeId': post_type_id,
            'CreationDate': attrib.get('CreationDate', None),
            'OwnerUserId': attrib.get('OwnerUserId', None),
            'Score': attrib.get('Score', None),
            'AcceptedAnswerId': attrib.get('AcceptedAnswerId', None),
            'ParentId': attrib.get('ParentId', None),
            'Tags': attrib.get('Tags', None),
            'ViewCount': int(view_count) if view_count is not None else None,
        }
        record.update(compute_text_metrics(attrib.get('Body', None), attrib.get('Title', None)))
        return record

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
        # Reputation/UpVotes/DownVotes are dump-time snapshots, not historical values
        reputation = attrib.get('Reputation', None)
        up_votes = attrib.get('UpVotes', None)
        down_votes = attrib.get('DownVotes', None)
        return {
            'Id': attrib.get('Id', None),
            'CreationDate': attrib.get('CreationDate', None),
            'Reputation': int(reputation) if reputation is not None else None,
            'UpVotes': int(up_votes) if up_votes is not None else None,
            'DownVotes': int(down_votes) if down_votes is not None else None,
        }

    except Exception as e:
        print(f"Error parsing user row (ID: {attrib.get('Id', 'N/A')}): {e}")
        return None


def parse_generic_row_comments(elem):
    attrib = elem.attrib
    try:
        # UserId is absent when the commenting user was deleted; Text is skipped (heavy)
        score = attrib.get('Score', None)
        return {
            'Id': attrib.get('Id', None),
            'PostId': attrib.get('PostId', None),
            'UserId': attrib.get('UserId', None),
            'CreationDate': attrib.get('CreationDate', None),
            'Score': int(score) if score is not None else None,
        }

    except Exception as e:
        print(f"Error parsing comment row (ID: {attrib.get('Id', 'N/A')}): {e}")
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


def convert_xml_to_parquet(xml_path, parquet_path, parse_function, chunk_size=1000000, dtypes=None):
    def _write_chunk(records, is_first):
        df = pd.DataFrame(records)
        if dtypes:
            df = df.astype({col: dt for col, dt in dtypes.items() if col in df.columns})
        write(parquet_path, df, compression='snappy', write_index=False, append=not is_first)

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
                _write_chunk(chunk_data, first)
                first = False
                chunk_data = []

        if chunk_data:
            _write_chunk(chunk_data, first)
        pbar.close()


def split_parquet_by_column(parquet_file_path, output_folder, column, values, custom_names=None, columns="*"):
    con = duckdb.connect(database=':memory:', read_only=False)

    con.execute("SET memory_limit='6GB'")
    # Use a cross-platform temp directory instead of a hardcoded Windows path
    temp_dir = Path(os.getenv("DUCKDB_TEMP_DIR", Path.cwd() / "duckdb_temp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
    con.execute("SET threads=2")
    # Streaming COPY does not need input order preserved; saves a lot of memory
    con.execute("SET preserve_insertion_order=false")

    # Build column list for SELECT
    if isinstance(columns, (list, tuple)):
        select_cols = ", ".join(columns)
    else:
        select_cols = columns

    for i, value in enumerate(values):
        if custom_names and i < len(custom_names):
            output_filename = custom_names[i]
        else:
            output_filename = f'{column}_{value}.parquet'

        output_path = os.path.join(output_folder, output_filename)

        # Stream directly from source to output
        con.execute(f"""
            COPY (SELECT {select_cols} FROM '{parquet_file_path}' WHERE {column} = {value})
            TO '{output_path}' (FORMAT PARQUET)
        """)

    con.close()


def process_posts(posts_file_path, output_folder, convert=True):
    posts_parquet_path = os.path.join(output_folder, 'Posts.parquet')
    if convert:
        # Metric columns can be all-NULL within a chunk (e.g. title metrics on
        # answers), so pin them to float64 to keep appended chunks compatible
        metric_dtypes = {col: 'float64' for col in TEXT_METRIC_COLUMNS}
        metric_dtypes['ViewCount'] = 'float64'
        convert_xml_to_parquet(posts_file_path, posts_parquet_path, parse_generic_row_posts,
                               dtypes=metric_dtypes)
    split_parquet_by_column(
        posts_parquet_path,
        output_folder,
        'PostTypeId',
        [1, 2],
        custom_names=['posts_questions.parquet', 'posts_answers.parquet'],
        # Body/Title are already reduced to scalar metrics at parse time
        columns=[
            'Id',
            'PostTypeId',
            'CreationDate',
            'OwnerUserId',
            'Score',
            'AcceptedAnswerId',
            'ParentId',
            'Tags',
            'ViewCount',
        ] + TEXT_METRIC_COLUMNS
    )


def process_votes(votes_file_path, output_folder):
    votes_parquet_path = os.path.join(output_folder, 'Votes.parquet')
    convert_xml_to_parquet(votes_file_path, votes_parquet_path, parse_generic_row_votes)


def process_users(users_file_path, output_folder):
    users_parquet_path = os.path.join(output_folder, 'Users.parquet')
    convert_xml_to_parquet(users_file_path, users_parquet_path, parse_generic_row_users,
                           dtypes={'Reputation': 'float64', 'UpVotes': 'float64', 'DownVotes': 'float64'})


def process_badges(badges_file_path, output_folder):
    badges_parquet_path = os.path.join(output_folder, 'Badges.parquet')
    convert_xml_to_parquet(badges_file_path, badges_parquet_path, parse_generic_row_badges)


def process_comments(comments_file_path, output_folder):
    comments_parquet_path = os.path.join(output_folder, 'Comments.parquet')
    convert_xml_to_parquet(comments_file_path, comments_parquet_path, parse_generic_row_comments,
                           dtypes={'Score': 'float64'})


def fix_column_types(parquet_file_paths):
    con = duckdb.connect(database=':memory:', read_only=False)

    con.execute("SET memory_limit='8GB'")
    temp_dir = Path(os.getenv("DUCKDB_TEMP_DIR", Path.cwd() / "duckdb_temp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")

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
    # Resolve paths relative to this script's location so running from any CWD works
    base_dir = Path(__file__).resolve().parent
    input_folder = base_dir.parent / 'data' / 'input'
    output_folder = input_folder

    posts_file_path = input_folder / 'Posts.xml'
    votes_file_path = input_folder / 'Votes.xml'
    users_file_path = input_folder / 'Users.xml'
    comments_file_path = input_folder / 'Comments.xml'
    # Badges.xml intentionally not processed (only the archived bounty pipeline used it)

    # Resume-friendly: skip a conversion whose output parquet already exists
    # (delete the parquet to force a re-parse)
    posts_parquet = input_folder / 'Posts.parquet'
    if posts_parquet.exists():
        print(f"{posts_parquet} exists; skipping Posts.xml parse, re-running split only")
    process_posts(posts_file_path, output_folder, convert=not posts_parquet.exists())
    if (input_folder / 'Votes.parquet').exists():
        print("Votes.parquet exists; skipping")
    else:
        process_votes(votes_file_path, output_folder)
    if (input_folder / 'Users.parquet').exists():
        print("Users.parquet exists; skipping")
    else:
        process_users(users_file_path, output_folder)
    if (input_folder / 'Comments.parquet').exists():
        print("Comments.parquet exists; skipping")
    else:
        process_comments(comments_file_path, output_folder)

    # Fix column types for all processed parquet files
    parquet_files = [
        os.path.join(output_folder, 'posts_answers.parquet'),
        os.path.join(output_folder, 'posts_questions.parquet'),
        os.path.join(output_folder, 'Votes.parquet'),
        os.path.join(output_folder, 'Users.parquet'),
        os.path.join(output_folder, 'Comments.parquet'),
    ]
    fix_column_types(parquet_files)


if __name__ == "__main__":
    main()