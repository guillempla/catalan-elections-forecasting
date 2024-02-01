import os
import logging
from dotenv import load_dotenv
from download_data import DownloadData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.info("Loading environment variables.")

    load_dotenv()
    SOCRATA_DOMAIN = os.environ.get("SOCRATA_DOMAIN")
    SOCRATA_DATASET_ID = os.environ.get("SOCRATA_DATASET_ID")
    SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")
    SOCRATA_EMAIL = os.environ.get("SOCRATA_EMAIL")
    SOCRATA_PASSWORD = os.environ.get("SOCRATA_PASSWORD")

    POSTGRES_USERNAME = os.environ.get("POSTGRES_USER")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    POSTGRES_CATALAN_ELECTIONS_DATA_DB = os.environ.get(
        "POSTGRES_CATALAN_ELECTIONS_DATA_DB"
    )
    POSTGRES_CATALAN_ELECTIONS_DATA_TABLE = os.environ.get(
        "POSTGRES_CATALAN_ELECTIONS_DATA_TABLE"
    )

    CATALAN_ELECTIONS_DATA_CSV_PATH = os.environ.get("CATALAN_ELECTIONS_DATA_CSV_PATH")

    DownloadData().download_catalan_elections_data(
        SOCRATA_DOMAIN,
        SOCRATA_APP_TOKEN,
        SOCRATA_EMAIL,
        SOCRATA_PASSWORD,
        SOCRATA_DATASET_ID,
        CATALAN_ELECTIONS_DATA_CSV_PATH,
        POSTGRES_USERNAME,
        POSTGRES_PASSWORD,
        POSTGRES_HOST,
        POSTGRES_CATALAN_ELECTIONS_DATA_DB,
        POSTGRES_CATALAN_ELECTIONS_DATA_TABLE,
    )


if __name__ == "__main__":
    main()
