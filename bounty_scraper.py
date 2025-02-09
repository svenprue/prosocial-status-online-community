import os
import time
import pandas as pd
import duckdb
import undetected_chromedriver as uc
from selenium.webdriver import ChromeOptions
from bs4 import BeautifulSoup
from tqdm import tqdm
from fastparquet import write

# Constants
WRITE_BATCH_SIZE = 10
SELENIUM_TIMEOUT = 15

# Utility Functions
def create_data_directory(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def load_processed_ids(processed_ids_path):
    if os.path.exists(processed_ids_path):
        try:
            processed_ids_df = pd.read_parquet(processed_ids_path, engine="fastparquet")
            return processed_ids_df['question_id'].tolist()
        except Exception as e:
            print(f"Error reading processed IDs file: {e}")
    return []

def append_to_parquet(dataframe, path):
    try:
        if os.path.exists(path):
            write(path, dataframe, append=True)
        else:
            write(path, dataframe)
    except Exception as e:
        print(f"Error writing to Parquet file '{path}': {e}")

def get_unprocessed_post_ids(bounty_votes_path, results_path):
    con = duckdb.connect()
    query = f"""
        SELECT DISTINCT b.PostId
        FROM parquet_scan('{bounty_votes_path}') b
        WHERE b.VoteTypeId = 8
          AND b.PostId NOT IN (
              SELECT DISTINCT r.question_id
              FROM parquet_scan('{results_path}') r
          )
    """
    post_ids_data = con.execute(query).fetchall()
    con.close()
    return [row[0] for row in post_ids_data]

def configure_driver():
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")

    driver = uc.Chrome(options=chrome_options)
    driver.set_page_load_timeout(SELENIUM_TIMEOUT)
    driver.set_script_timeout(SELENIUM_TIMEOUT)
    return driver

def scrape_timeline_events(driver, question_id):
    time.sleep(1)
    url = f'https://stackoverflow.com/posts/{question_id}/timeline'

    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        bounty_start, bounty_end = [], []
        bounty_rows = soup.find_all(
            'tr', attrs={'data-eventtype': 'history', 'class': lambda x: x and 'datehash' in x}
        )

        for row in bounty_rows:
            event_cell = row.find('td', class_='wmn1')
            date_cell = row.find('span', class_='relativetime')

            if event_cell and date_cell:
                event_text = event_cell.get_text(strip=True)
                date = date_cell.get('title')

                if 'bounty started' in event_text:
                    bounty_start.append(date)
                elif 'bounty ended' in event_text:
                    bounty_end.append(date)

        return {'question_id': question_id, 'bounty_start': bounty_start, 'bounty_end': bounty_end}

    except Exception as e:
        print(f"Error scraping timeline for question_id={question_id}: {e}")
        return None

def process_results(results, results_path):
    if results:
        df = pd.DataFrame(results)
        append_to_parquet(df, results_path)

def main(data_dir, bounty_votes_path, results_path, processed_ids_path):
    create_data_directory(data_dir)
    post_ids = get_unprocessed_post_ids(bounty_votes_path, results_path)

    if not post_ids:
        print("No unprocessed Post IDs found.")
        return

    driver = configure_driver()
    results = []

    with tqdm(total=len(post_ids), desc="Scraping All IDs") as pbar:
        for post_id in post_ids:
            result = scrape_timeline_events(driver, post_id)
            if result:
                results.append(result)

            if len(results) >= WRITE_BATCH_SIZE:
                process_results(results, results_path)
                results = []
            pbar.update(1)

    process_results(results, results_path)
    driver.quit()

if __name__ == "__main__":
    DATA_DIR = "./01_input_data/processed_SO_data_dump"
    RESULTS_PATH = os.path.join(DATA_DIR, "bounty.parquet")
    PROCESSED_IDS_PATH = os.path.join(DATA_DIR, "processed_question_ids.parquet")
    BOUNTY_VOTES_PATH = os.path.join(DATA_DIR, "Votes.parquet")

    main(DATA_DIR, BOUNTY_VOTES_PATH, RESULTS_PATH, PROCESSED_IDS_PATH)