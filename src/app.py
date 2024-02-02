import os
import argparse
import logging
from dotenv import load_dotenv
from download_data import DownloadData
from clean_data import CleanData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Download data from the Catalan Elections API."
    )
    # Add boolean argument for deciding if the data must be downloaded or not.
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download the data from the Catalan Elections API.",
    )
    # Add boolean argument for deciding if the data must be cleaned or not.
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clean the data from the Catalan Elections API.",
    )

    args = parser.parse_args()
    download_data = args.download
    clean_data = args.clean

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

    if download_data:
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

    if clean_data:
        CleanData().clean_elections_data()


if __name__ == "__main__":
    main()
