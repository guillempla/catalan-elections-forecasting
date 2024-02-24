import os
import argparse
import logging
from dotenv import load_dotenv
from download_data import DownloadData
from clean_data import CleanData
from group_parties import GroupParties
from typing import List

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
    # Add boolean argument for deciding if the parties must be grouped or not.
    parser.add_argument(
        "--group",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Group the parties from the Catalan Elections API.",
    )

    args = parser.parse_args()
    download_data = args.download
    clean_data = args.clean
    group_parties = args.group

    logging.info("Loading environment variables.")

    load_dotenv()
    socrata_domain = os.environ.get("SOCRATA_DOMAIN")
    socrata_elections_results_id = os.environ.get("SOCRATA_ELECTIONS_RESULTS_ID")
    socrata_elections_participation_id = os.environ.get(
        "SOCRATA_ELECTIONS_PARTICIPATION_ID"
    )
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

    catalan_elections_results_csv_path = (
        os.environ.get("CATALAN_ELECTIONS_RESULTS_FILENAME") + ".csv"
    )
    catalan_elections_results_pkl_path = (
        os.environ.get("CATALAN_ELECTIONS_RESULTS_FILENAME") + ".pkl"
    )
    catalan_elections_participation_csv_path = (
        os.environ.get("CATALAN_ELECTIONS_PARTICIPATION_FILENAME") + ".csv"
    )
    catalan_elections_participation_pkl_path = (
        os.environ.get("CATALAN_ELECTIONS_PARTICIPATION_FILENAME") + ".pkl"
    )

    if download_data:
        dataset_configs: List[dict] = []

        dataset_configs.append(
            {
                "socrata_domain": socrata_domain,
                "socrata_app_token": socrata_app_token,
                "socrata_username": socrata_email,
                "socrata_password": socrata_password,
                "socrata_dataset_id": socrata_elections_results_id,
                "csv_path": catalan_elections_results_csv_path,
                "pkl_path": catalan_elections_results_pkl_path,
            }
        )
        dataset_configs.append(
            {
                "socrata_domain": socrata_domain,
                "socrata_app_token": socrata_app_token,
                "socrata_username": socrata_email,
                "socrata_password": socrata_password,
                "socrata_dataset_id": socrata_elections_participation_id,
                "csv_path": catalan_elections_participation_csv_path,
                "pkl_path": catalan_elections_participation_pkl_path,
            }
        )

        DownloadData(
            dataset_configs=dataset_configs,
        ).download_catalan_elections_data()

    if clean_data:
        CleanData().clean_elections_data()

    if group_parties:
        GroupParties().group_parties()


if __name__ == "__main__":
    main()
