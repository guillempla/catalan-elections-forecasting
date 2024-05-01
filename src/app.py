import os
import argparse
import logging
from typing import List
from dotenv import load_dotenv
from download_data import DownloadData
from clean_data import CleanData
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

    args = parser.parse_args()
    download_data = args.download
    clean_data = args.clean
    group_parties = args.group
    transform_data = args.transform

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
        # dataset_configs.append(
        #     {
        #         "dataset_type": "socrata",
        #         "socrata_domain": socrata_domain,
        #         "socrata_app_token": socrata_app_token,
        #         "socrata_username": socrata_email,
        #         "socrata_password": socrata_password,
        #         "socrata_dataset_id": socrata_elections_participation_id,
        #         "csv_path": catalan_elections_participation_csv_path,
        #         "pkl_path": catalan_elections_participation_pkl_path,
        #     }
        # )
        # dataset_configs.append(
        #     {
        #         "dataset_type": "socrata",
        #         "socrata_domain": socrata_domain,
        #         "socrata_app_token": socrata_app_token,
        #         "socrata_username": socrata_email,
        #         "socrata_password": socrata_password,
        #         "socrata_dataset_id": socrata_elections_results_id,
        #         "csv_path": catalan_elections_results_csv_path,
        #         "pkl_path": catalan_elections_results_pkl_path,
        #     }
        # )
        # dataset_configs.append(
        #     {
        #         "dataset_type": "socrata",
        #         "socrata_domain": socrata_domain,
        #         "socrata_app_token": socrata_app_token,
        #         "socrata_username": socrata_email,
        #         "socrata_password": socrata_password,
        #         "socrata_dataset_id": socrata_elections_dates_id,
        #         "csv_path": catalan_elections_dates_csv_path,
        #         "pkl_path": catalan_elections_dates_pkl_path,
        #     }
        # )

        DownloadData(
            dataset_configs=dataset_configs,
        ).download_catalan_elections_data()

    if clean_data:
        clean_configs: List[dict] = []

        # clean_configs.append(
        #     {
        #         "elections_data_filename": catalan_elections_dates_pkl_path,
        #         "output_filename": "../data/processed/catalan-elections-clean-dates",
        #         "create_date_columns": True,
        #         "columns_to_drop": [
        #             "id_eleccio",
        #             "codi_tipus_eleccio",
        #             "nom_tipus_eleccio",
        #             "data_eleccio",
        #             "nivell_circumscripcio",
        #         ],
        #     }
        # )
        # clean_configs.append(
        #     {
        #         "elections_data_filename": catalan_elections_participation_pkl_path,
        #         "elections_days_filename": "../data/processed/catalan-elections-clean-dates.pkl",
        #         "output_filename": "../data/processed/catalan-elections-clean-participation",
        #         "create_party_column": False,
        #         "divide_id_eleccio": True,
        #         "elections_type": [
        #             "M",
        #             "E",
        #             "A",
        #             "G",
        #         ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals,
        #         "territorial_levels": ["SE"],  # SE: Secció censal
        #         "create_date_column": True,
        #         "columns_to_drop": [
        #             "vots_primer_avan",
        #             "vots_segon_avan",
        #             "hora_primer_avan",
        #             "hora_segon_avan",
        #             "vots_candidatures",
        #             "abstencio",
        #             "nombre_meses",
        #         ],
        #         "columns_types": {
        #             "year": "int",
        #             "month": "int",
        #             "day": "int",
        #             "seccio": "int",
        #             "votants": "int",
        #             "escons": "int",
        #             "districte": "int",
        #         },
        #         "create_mundissec_column": True,
        #     }
        # )
        # clean_configs.append(
        #     {
        #         "elections_data_filename": catalan_elections_results_pkl_path,
        #         "elections_days_filename": "../data/processed/catalan-elections-clean-dates.pkl",
        #         "elections_participation_filename": "../data/processed/catalan-elections-clean-participation.pkl",
        #         "output_filename": "../data/processed/catalan-elections-clean-data",
        #         "fix_party_codes": {
        #             "1083": {
        #                 "code_column": "candidatura_codi",
        #                 "name_column": "candidatura_denominacio",
        #             },
        #             "1084": {
        #                 "code_column": "candidatura_codi",
        #                 "name_column": "candidatura_denominacio",
        #             },
        #         },
        #         "columns_to_drop": [
        #             "candidat_posicio",
        #             "candidatura_logotip",
        #             "candidatura_codi",
        #             "candidatura_denominacio",
        #             "candidatura_sigles",
        #             "candidatura_color",
        #             "agrupacio_codi",
        #             "agrupacio_denominacio",
        #             "agrupacio_sigles",
        #         ],
        #         "columns_to_rename": {"secci_": "seccio"},
        #         "elections_type": [
        #             "M",
        #             "E",
        #             "A",
        #             "G",
        #         ],  # M: Municipals, E: Europees, A: Autonòmiques, G: Generals
        #         "territorial_levels": ["SE"],  # SE: Secció censal
        #         "color_column": "candidatura_color",
        #         "color_default": "grey",
        #         "columns_types": {
        #             "year": "int",
        #             "month": "int",
        #             "day": "int",
        #             "seccio": "int",
        #             "vots": "int",
        #             "escons": "int",
        #             "districte": "int",
        #             "party_code": "int",
        #         },
        #         "columns_null_values": ["candidatura_sigles"],
        #         "create_party_column": True,
        #         "create_mundissec_column": True,
        #         "divide_id_eleccio": True,
        #         "create_date_column": True,
        #         "aggregate_duplicated_parties": True,
        #     }
        # )
        clean_configs.append(
            {
                "output_filename": "../data/processed/mean_income_clean_data",
                "mean_income_filenames": [
                    "../data/raw/mean_income/mean_income_barcelona.csv",
                    "../data/raw/mean_income/mean_income_girona.csv",
                    "../data/raw/mean_income/mean_income_lleida.csv",
                    "../data/raw/mean_income/mean_income_tarragona.csv",
                ],
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
                "filter_by_income": {
                    "column": "Indicadores de renta media y mediana",
                    "values": ["Renta neta media por hogar"],
                },
                "pivot_table": {
                    "index": ["mundissec"],
                    "columns": "Periodo",
                    "values": "Total",
                    "aggfunc": "sum",
                },
            }
        )

        CleanData(clean_configs=clean_configs)

    if group_parties:
        GroupParties(threshold=0.2, only_abbr=True).group_parties()

    if transform_data:
        TransformData(
            censal_sections_path="../data/raw/bseccenv10sh1f1_20210101_2/bseccenv10sh1f1_20210101_2.shp",
            results_path="../data/processed/catalan-elections-grouped-data.pkl",
        ).transform_data()


if __name__ == "__main__":
    main()
