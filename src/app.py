import os
import argparse
import logging
from typing import List
from dotenv import load_dotenv
from download_data import DownloadData
from clean_data import CleanData
from group_parties import GroupParties

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Main - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Download and trasnform data from the Catalan Elections API."
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

    # postgres_username = os.environ.get("POSTGRES_USER")
    # postgres_password = os.environ.get("POSTGRES_PASSWORD")
    # postgres_host = os.environ.get("POSTGRES_HOST")
    # postgres_catalan_elections_data_db = os.environ.get(
    #     "POSTGRES_CATALAN_ELECTIONS_DATA_DB"
    # )
    # postgres_catalan_elections_data_table = os.environ.get(
    #     "POSTGRES_CATALAN_ELECTIONS_DATA_TABLE"
    # )

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
            {"dataset_type": "gis", "year": "2019", "output_path": "../data/raw/"}
        )
        dataset_configs.append(
            {
                "dataset_type": "socrata",
                "socrata_domain": socrata_domain,
                "socrata_app_token": socrata_app_token,
                "socrata_username": socrata_email,
                "socrata_password": socrata_password,
                "socrata_dataset_id": socrata_elections_participation_id,
                "csv_path": catalan_elections_participation_csv_path,
                "pkl_path": catalan_elections_participation_pkl_path,
            }
        )
        dataset_configs.append(
            {
                "dataset_type": "socrata",
                "socrata_domain": socrata_domain,
                "socrata_app_token": socrata_app_token,
                "socrata_username": socrata_email,
                "socrata_password": socrata_password,
                "socrata_dataset_id": socrata_elections_results_id,
                "csv_path": catalan_elections_results_csv_path,
                "pkl_path": catalan_elections_results_pkl_path,
            }
        )

        DownloadData(
            dataset_configs=dataset_configs,
        ).download_catalan_elections_data()

    if clean_data:
        clean_configs: List[dict] = []

        clean_configs.append(
            {
                "elections_data_filename": catalan_elections_participation_pkl_path,
                "elections_days_filename": "../data/processed/elections_days.csv",
                "output_filename": "../data/processed/catalan-elections-clean-participation",
                "create_party_column": False,
                "divide_id_eleccio": True,
                "elections_type": [
                    "M",
                    "E",
                    "A",
                    "G",
                ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals
                "create_date_column": True,
                "columns_to_drop": [
                    "vots_primer_avan",
                    "vots_segon_avan",
                    "hora_primer_avan",
                    "hora_segon_avan",
                    "vots_candidatures",
                    "abstencio",
                    "nombre_meses",
                ],
                "columns_types": {
                    "year": "int",
                    "month": "int",
                    "day": "int",
                    "seccio": "int",
                    "votants": "int",
                    "escons": "int",
                    "districte": "int",
                },
            }
        )
        clean_configs.append(
            {
                "elections_data_filename": catalan_elections_results_pkl_path,
                "elections_days_filename": "../data/processed/elections_days.csv",
                "elections_participation_filename": "../data/processed/catalan-elections-clean-participation.pkl",
                "output_filename": "../data/processed/catalan-elections-clean-data",
                "columns_to_drop": [
                    "candidat_posicio",
                    "candidatura_logotip",
                    "candidatura_codi",
                    "candidatura_denominacio",
                    "candidatura_sigles",
                    "candidatura_color",
                    "agrupacio_codi",
                    "agrupacio_denominacio",
                    "agrupacio_sigles",
                ],
                "columns_to_rename": {"secci_": "seccio"},
                "elections_type": [
                    "M",
                    "E",
                    "A",
                    "G",
                ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals
                "color_column": "candidatura_color",
                "color_default": "grey",
                "columns_types": {
                    "year": "int",
                    "month": "int",
                    "day": "int",
                    "seccio": "int",
                    "vots": "int",
                    "escons": "int",
                    "districte": "int",
                    "party_code": "int",
                },
                "columns_null_values": ["candidatura_sigles"],
                "create_party_column": True,
                "divide_id_eleccio": True,
                "create_date_column": True,
            }
        )

        CleanData(clean_configs=clean_configs)

    if group_parties:
        GroupParties(exclude_competed_together=False).group_parties()


if __name__ == "__main__":
    main()
