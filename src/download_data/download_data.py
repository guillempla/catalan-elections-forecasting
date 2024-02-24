"""
Module for downloading data.
"""

import logging
from typing import List
from .catalan_elections_data import DownloadCatalanElectionsData


class DownloadData:
    """
    Class for downloading data.

    Args:
        socrata_domain (str): The domain of the Socrata API.
        socrata_app_token (str): The app token for accessing the Socrata API.
        socrata_username (str): The username for authentication.
        socrata_password (str): The password for authentication.
        socrata_dataset_id (str): The ID of the Socrata dataset.
        csv_path (str, optional): The path to save the downloaded data as a CSV file.
            Defaults to None.
        pkl_path (str, optional): The path to save the downloaded data as a pickle file.
            Defaults to None.
    """

    def __init__(
        self,
        dataset_configs: List[dict],
    ) -> None:
        """
        Initializes a new instance of the DownloadData class.

        Args:
            dataset_configs (List[dict]): A list of dictionaries containing the configuration.
        """
        logging.info("Starting data download.")

        self.catalan_elections_data_downloaders: List[DownloadCatalanElectionsData] = []
        for dataset_config in dataset_configs:
            self.catalan_elections_data_downloaders.append(
                DownloadCatalanElectionsData(
                    dataset_config["socrata_domain"],
                    dataset_config["socrata_app_token"],
                    dataset_config["socrata_username"],
                    dataset_config["socrata_password"],
                    dataset_config["socrata_dataset_id"],
                    dataset_config["csv_path"],
                    dataset_config["pkl_path"],
                )
            )

    def download_catalan_elections_data(self) -> None:
        """
        Downloads the Catalan elections data using the configured parameters.
        """

        for downloader in self.catalan_elections_data_downloaders:
            downloader.download_catalan_elections_data()
