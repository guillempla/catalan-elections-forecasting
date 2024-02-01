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
    socrata_domain = os.environ.get("SOCRATA_DOMAIN")
    socrata_dataset_id = os.environ.get("SOCRATA_DATASET_ID")
    socrata_app_token = os.environ.get("SOCRATA_APP_TOKEN")
    socrata_email = os.environ.get("SOCRATA_EMAIL")
    socrata_password = os.environ.get("SOCRATA_PASSWORD")

    postgres_username = os.environ.get("POSTGRES_USER")
    postgres_password = os.environ.get("POSTGRES_PASSWORD")
    postgres_host = os.environ.get("POSTGRES_HOST")
    postgres_catalan_elections_data_db = os.environ.get(
        "POSTGRES_CATALAN_ELECTIONS_DATA_DB"
    )
    postgres_catalan_elections_data_table = os.environ.get(
        "POSTGRES_CATALAN_ELECTIONS_DATA_TABLE"
    )

    catalan_elections_data_csv_path = os.environ.get("CATALAN_ELECTIONS_DATA_CSV_PATH")

    DownloadData(
        socrata_domain,
        socrata_app_token,
        socrata_email,
        socrata_password,
        socrata_dataset_id,
        catalan_elections_data_csv_path
        # postgres_username,
        # postgres_password,
        # postgres_host,
        # postgres_catalan_elections_data_db,
        # postgres_catalan_elections_data_table,
    ).download_catalan_elections_data()


if __name__ == "__main__":
    main()
