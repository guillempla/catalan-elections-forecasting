import os
import argparse
import logging
from typing import List
from dotenv import load_dotenv
from download_data import DownloadData
from clean_data import CleanData
from experiment.experiment_manager import ExperimentManager
from group_parties import GroupParties
from transform_data.transform_data import TransformData

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
    # Add boolean argument for deciding if the data must be transformed into ML formad or not.
    parser.add_argument(
        "--transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Transform the data from the Catalan Elections API into ML format.",
    )
    # Add boolean argument for deciding if the ML model must be trained or not.
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the ML model.",
    )

    args = parser.parse_args()
    download_data = args.download
    clean_data = args.clean
    group_parties = args.group
    transform_data = args.transform
    train_model = args.train

    logging.info("Loading environment variables.")

    load_dotenv()
    socrata_domain = os.environ.get("SOCRATA_DOMAIN")
    socrata_elections_results_id = os.environ.get("SOCRATA_ELECTIONS_RESULTS_ID")
    socrata_elections_participation_id = os.environ.get(
        "SOCRATA_ELECTIONS_PARTICIPATION_ID"
    )
    socrata_elections_dates_id = os.environ.get("SOCRATA_ELECTIONS_DATES_ID")
    socrata_app_token = os.environ.get("SOCRATA_APP_TOKEN")
    socrata_email = os.environ.get("SOCRATA_EMAIL")
    socrata_password = os.environ.get("SOCRATA_PASSWORD")

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
    catalan_elections_dates_csv_path = (
        os.environ.get("CATALAN_ELECTIONS_DATES_FILENAME") + ".csv"
    )
    catalan_elections_dates_pkl_path = (
        os.environ.get("CATALAN_ELECTIONS_DATES_FILENAME") + ".pkl"
    )

    if download_data:
        dataset_configs: List[dict] = []

        # dataset_configs.append(
        #     {"dataset_type": "gis", "year": "2019", "output_path": "../data/raw/"}
        # )
        dataset_configs.append(
            {
                "dataset_type": "idescat",
                "data_type": "socioeconomic_index",
                "output_path": "../data/raw/",
                "urls_info": [
                    {
                        "year": "2020",
                        "url": "https://www.idescat.cat/pub/?id=ist&n=14034&by=sec&t=2020&f=zip&fi=csv",
                    },
                    {
                        "year": "2019",
                        "url": "https://www.idescat.cat/pub/?id=ist&n=14034&by=sec&t=2019&f=zip&fi=csv",
                    },
                    {
                        "year": "2018",
                        "url": "https://www.idescat.cat/pub/?id=ist&n=14034&by=sec&t=2018&f=zip&fi=csv",
                    },
                    {
                        "year": "2017",
                        "url": "https://www.idescat.cat/pub/?id=ist&n=14034&by=sec&t=2017&f=zip&fi=csv",
                    },
                    {
                        "year": "2016",
                        "url": "https://www.idescat.cat/pub/?id=ist&n=14034&by=sec&t=2016&f=zip&fi=csv",
                    },
                    {
                        "year": "2015",
                        "url": "https://www.idescat.cat/pub/?id=ist&n=14034&by=sec&t=2015&f=zip&fi=csv",
                    },
                ],
            }
        )
        dataset_configs.append(
            {
                "dataset_type": "ine",
                "data_type": "mean_income",
                "output_path": "../data/raw/",
                "urls_info": [
                    {
                        "province": "Barcelona",
                        "url": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/30896.csv?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "url": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31016.csv?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "url": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31079.csv?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "url": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31223.csv?nocab=1",
                    },
                ],
            }
        )
        dataset_configs.append(
            {
                "dataset_type": "ine",
                "data_type": "place_of_birth",
                "output_path": "../data/raw/",
                "urls_info": [
                    {
                        "province": "Barcelona",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/4306.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/0806.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/1706.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/2506.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/4306.csv_bdsc?nocab=1",
                    },
                ],
            }
        )
        dataset_configs.append(
            {
                "dataset_type": "ine",
                "data_type": "age_groups",
                "output_path": "../data/raw/",
                "urls_info": [
                    {
                        "province": "Barcelona",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2022",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2022/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2021",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2021/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2020",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2020/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2019",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2019/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2018",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2018/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2017",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2017/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2016",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2016/l0/4301.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Barcelona",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/0801.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Girona",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/1701.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Lleida",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/2501.csv_bdsc?nocab=1",
                    },
                    {
                        "province": "Tarragona",
                        "year": "2015",
                        "url": "https://www.ine.es/jaxi/files/_px/es/csv_bdsc/t20/e245/p07/a2015/l0/4301.csv_bdsc?nocab=1",
                    },
                ],
            }
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
        dataset_configs.append(
            {
                "dataset_type": "socrata",
                "socrata_domain": socrata_domain,
                "socrata_app_token": socrata_app_token,
                "socrata_username": socrata_email,
                "socrata_password": socrata_password,
                "socrata_dataset_id": socrata_elections_dates_id,
                "csv_path": catalan_elections_dates_csv_path,
                "pkl_path": catalan_elections_dates_pkl_path,
            }
        )

        DownloadData(
            dataset_configs=dataset_configs,
        ).download_catalan_elections_data()

    if clean_data:
        clean_configs: List[dict] = []

        clean_configs.append(
            {
                "data_type": "elections_data",
                "elections_data_filename": catalan_elections_dates_pkl_path,
                "output_filename": "../data/processed/catalan-elections-clean-dates",
                "create_date_columns": True,
                "columns_to_drop": [
                    "id_eleccio",
                    "codi_tipus_eleccio",
                    "nom_tipus_eleccio",
                    "data_eleccio",
                    "nivell_circumscripcio",
                ],
            }
        )
        clean_configs.append(
            {
                "data_type": "elections_data",
                "elections_data_filename": catalan_elections_participation_pkl_path,
                "elections_days_filename": "../data/processed/catalan-elections-clean-dates.pkl",
                "output_filename": "../data/processed/catalan-elections-clean-participation",
                "create_party_column": False,
                "divide_id_eleccio": True,
                "elections_type": [
                    "E",
                    "A",
                    "G",
                ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals,
                "territorial_levels": ["SE"],  # SE: Secció censal
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
                "create_mundissec_column": True,
            }
        )
        clean_configs.append(
            {
                "data_type": "elections_data",
                "elections_data_filename": catalan_elections_results_pkl_path,
                "elections_days_filename": "../data/processed/catalan-elections-clean-dates.pkl",
                "elections_participation_filename": "../data/processed/catalan-elections-clean-participation.pkl",
                "output_filename": "../data/processed/catalan-elections-clean-data",
                "fix_party_codes": {
                    "1083": {
                        "code_column": "candidatura_codi",
                        "name_column": "candidatura_denominacio",
                    },
                    "1084": {
                        "code_column": "candidatura_codi",
                        "name_column": "candidatura_denominacio",
                    },
                },
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
                    "E",
                    "A",
                    "G",
                ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals
                "territorial_levels": ["SE"],  # SE: Secció censal
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
                "create_mundissec_column": True,
                "divide_id_eleccio": True,
                "create_date_column": True,
                "aggregate_duplicated_parties": True,
            }
        )
        clean_configs.append(
            {
                "data_type": "mean_income_data",
                "output_filename": "../data/processed/mean_income_clean_data",
                "mean_income_directory": "../data/raw/mean_income/",
                "fix_total_column": True,
                "remove_empty_rows": ["Secciones"],
                "divide_columns": [
                    {
                        "column": "Municipios",
                        "new_columns": ["municipal_code", "municipal_name"],
                        "regex_separator": r"^(\d+)\s+(.+)",  # Divide by the first space
                    },
                    {
                        "column": "Distritos",
                        "new_columns": ["district_code", "district_name"],
                        "regex_separator": r"^(\d+)\s+(.+)",  # Divide by the first space
                    },
                    {
                        "column": "Secciones",
                        "new_columns": ["mundissec", "section_name"],
                        "regex_separator": r"^(\d+)\s+(.+)",  # Divide by the first space
                    },
                ],
                "fix_mundissec": True,
                "filter_by_income": {
                    "column": "Indicadores de renta media y mediana",
                    "values": ["Renta neta media por hogar"],
                },
                "columns_to_rename": {
                    "Total": "mean_income",
                },
                "pivot_table": {
                    "index": ["mundissec"],
                    "columns": "Periodo",
                    "values": "mean_income",
                    "aggfunc": "sum",
                },
            }
        )
        clean_configs.append(
            {
                "data_type": "place_of_birth",
                "output_filename": "../data/processed/place_of_birth_clean_data",
                "place_of_birth_directory": "../data/raw/place_of_birth/",
                "create_column_on_load": {
                    "new_column": "year",
                    "regex": r"_(\d{4})\.\w+$",
                },
                "fix_mundissec": True,
                "remove_rows_by_values": [
                    {
                        "column": "Sección",
                        "value": "TOTAL",
                    },
                    {
                        "column": "Sexo",
                        "value": "Hombres",
                    },
                    {
                        "column": "Sexo",
                        "value": "Mujeres",
                    },
                ],
                "columns_to_rename": {
                    "Sección": "mundissec",
                },
                "calculate_p_born_abroad": True,
                "pivot_table": {
                    "index": ["mundissec"],
                    "columns": "year",
                    "values": "p_born_abroad",
                    "aggfunc": "sum",
                },
            }
        )
        clean_configs.append(
            {
                "data_type": "age_groups",
                "output_filename": "../data/processed/age_groups_clean_data",
                "age_groups_directory": "../data/raw/age_groups/",
                "create_column_on_load": {
                    "new_column": "year",
                    "regex": r"_(\d{4})\.\w+$",
                },
                "fix_mundissec": True,
                "remove_rows_by_values": [
                    {
                        "column": "Sección",
                        "value": "TOTAL",
                    },
                    {
                        "column": "Sexo",
                        "value": "Hombres",
                    },
                    {
                        "column": "Sexo",
                        "value": "Mujeres",
                    },
                ],
                "columns_to_rename": {
                    "Sección": "mundissec",
                },
                "group_age_groups": {
                    "indexes": ["year", "mundissec"],
                    "group_column": "Edad (grupos quinquenales)",
                    "new_column": "age_groups",
                    "groups": {
                        "child": ["0-4", "5-9", "10-14"],
                        "young": ["15-19", "20-24", "25-29", "30-34"],
                        "adult": ["35-39", "40-44", "45-49", "50-54", "55-59", "60-64"],
                        "senior": [
                            "65-69",
                            "70-74",
                            "75-79",
                            "80-84",
                            "85-89",
                            "90-94",
                            "95-99",
                            "100 y más",
                        ],
                    },
                    "agg_columns": ["Total"],
                },
                "calcutate_p_age_groups": True,
                "pivot_table": {
                    "index": ["mundissec"],
                    "columns": ["age_groups", "year"],
                    "values": "p_age_groups",
                    "aggfunc": "sum",
                },
            }
        )
        clean_configs.append(
            {
                "data_type": "socioeconomic_index",
                "output_filename": "../data/processed/socioeconomic_index_clean_data",
                "socioeconomic_index_directory": "../data/raw/socioeconomic_index/",
                "remove_rows_by_values": [
                    {
                        "column": "secció censal",
                        "value": "Catalunya",
                    },
                ],
                "columns_to_rename": {
                    "secció censal": "mundissec",
                    "any": "year",
                },
                "pivot_table": {
                    "index": ["mundissec"],
                    "columns": ["concepte", "year"],
                    "values": "valor",
                    "aggfunc": "sum",
                },
            }
        )

        CleanData(clean_configs=clean_configs)

    if group_parties:
        GroupParties(threshold=0.10).group_parties()

    if transform_data:
        start_years = [2003, 2010]
        add_election_types = [1, 2]
        add_combinations = [
            {
                "add_born_abroad": False,
                "add_age_groups": False,
                "add_mean_income": False,
                "add_socioeconomic_index": False,
            },
            {
                "add_born_abroad": False,
                "add_age_groups": True,
                "add_mean_income": False,
                "add_socioeconomic_index": True,
            },
            {
                "add_born_abroad": True,
                "add_age_groups": True,
                "add_mean_income": True,
                "add_socioeconomic_index": False,
            },
            {
                "add_born_abroad": True,
                "add_age_groups": True,
                "add_mean_income": True,
                "add_socioeconomic_index": True,
            },
        ]

        for start_year in start_years:
            for add_election_type in add_election_types:
                for add_combination in add_combinations:
                    TransformData(
                        censal_sections_path="../data/raw/bseccenv10sh1f1_20210101_2/bseccenv10sh1f1_20210101_2.shp",
                        results_path="../data/processed/catalan-elections-grouped-data.pkl",
                        start_year=start_year,
                        n_important_parties=6,
                        transform_to_timeseries=True,
                        add_election_type=add_election_type,
                        add_born_abroad=add_combination["add_born_abroad"],
                        add_age_groups=add_combination["add_age_groups"],
                        add_mean_income=add_combination["add_mean_income"],
                        add_socioeconomic_index=add_combination[
                            "add_socioeconomic_index"
                        ],
                    ).transform_data()

    if train_model:
        experiments_configs: List[dict] = []

        experiments_configs.append(
            {
                "experiment_name": "XGBoost Multi Output Regressor",
                "model_type": "xgboost",
                "dataset_params": [
                    {
                        "name": "only_votes",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "only_votes_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                ],
                "model_params": {
                    "n_estimators": 10000,  # Number of boosting rounds
                    "max_depth": 12,  # Typically 3-10. Higher values can lead to overfitting
                    "eta": 0.01,  # Learning rate, typically between 0.01 and 0.2
                    "objective": "reg:squarederror",  # Regression with squared loss
                    "eval_metric": "rmse",  # Root Mean Square Error for evaluation
                    "tree_method": "hist",  # Fast histogram optimized approximate greedy algorithm
                    "multi_strategy": "multi_output_tree",
                    "early_stopping_rounds": 5,
                    "reg_alpha": 100,  # L1 regularization term on weights
                    "reg_lambda": 50,  # L2 regularization term on weights
                    "n_jobs": -1,  # Use all available cores
                },
                "fit_params": {
                    "verbose": True,
                },
            }
        )
        experiments_configs.append(
            {
                "experiment_name": "XGBoost Single Output Regressor",
                "model_type": "xgboost",
                "dataset_params": [
                    {
                        "name": "only_votes",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "only_votes_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                ],
                "model_params": {
                    "n_estimators": 10000,  # Number of boosting rounds
                    "max_depth": 6,  # Typically 3-10. Higher values can lead to overfitting
                    "eta": 0.01,  # Learning rate, typically between 0.01 and 0.2
                    "objective": "reg:squarederror",  # Regression with squared loss
                    "eval_metric": "rmse",  # Root Mean Square Error for evaluation
                    "early_stopping_rounds": 10,
                    "reg_alpha": 10,  # L1 regularization term on weights
                    "reg_lambda": 100,  # L2 regularization term on weights
                    "n_jobs": -1,  # Use all available cores
                },
                "fit_params": {
                    "verbose": True,
                },
            },
        )
        experiments_configs.append(
            {
                "experiment_name": "KNN Regressor",
                "model_type": "knn",
                "dataset_params": [
                    {
                        "name": "only_votes",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "only_votes_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                ],
                "model_params": {
                    "algorithm": "kd_tree",
                    "n_neighbors": 29,
                    "weights": "distance",
                    "n_jobs": -1,  # Use all available cores
                },
            }
        )
        experiments_configs.append(
            {
                "experiment_name": "Decision Tree Regressor",
                "model_type": "decision_tree",
                "dataset_params": [
                    {
                        "name": "only_votes",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "only_votes_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                ],
                "model_params": {
                    "criterion": "poisson",
                    "max_depth": 10,
                    "max_features": "sqrt",
                    "min_samples_leaf": 4,
                    "min_samples_split": 10,
                    "splitter": "best",
                },
            }
        )
        experiments_configs.append(
            {
                "experiment_name": "Linear Regression",
                "model_type": "linear_regression",
                "dataset_params": [
                    {
                        "name": "only_votes",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist",
                        "path": "../data/output/timeseries_2010_2024_6_1_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete",
                        "path": "../data/output/timeseries_2010_2024_6_1_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3",
                        "path": "../data/output/timeseries_2003_2024_6_1_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "only_votes_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_False_False_False.pkl",
                    },
                    {
                        "name": "no_ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_False.pkl",
                    },
                    {
                        "name": "ist_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_False_True_False_True.pkl",
                    },
                    {
                        "name": "complete_dummy",
                        "path": "../data/output/timeseries_2010_2024_6_2_True_True_True_True.pkl",
                    },
                    {
                        "name": "only_votes_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_False_False_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "no_ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_False.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "ist_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_False_True_False_True.pkl",
                        "num_shifts": 3,
                    },
                    {
                        "name": "complete_shifts_3_dummy",
                        "path": "../data/output/timeseries_2003_2024_6_2_True_True_True_True.pkl",
                        "num_shifts": 3,
                    },
                ],
            }
        )

        ExperimentManager(experiments=experiments_configs).run_all_experiments()


if __name__ == "__main__":
    main()
