"""
Module for downloading data from a Socrata dataset.
"""

import logging
from .catalan_elections_data import DownloadCatalanElectionsData
import logging
from typing import Optional
from .catalan_elections_data import DownloadCatalanElectionsData


class DownloadData:
    """
    Class for downloading data from a Socrata dataset.

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
        socrata_domain: str,
        socrata_app_token: str,
        socrata_username: str,
        socrata_password: str,
        socrata_dataset_id: str,
        csv_path: Optional[str] = None,
        pkl_path: Optional[str] = None,
    ) -> None:
        """
        Initializes a new instance of the DownloadData class.

        Args:
            socrata_domain (str): The domain of the Socrata API.
            socrata_app_token (str): The app token for accessing the Socrata API.
            socrata_username (str): The username for authentication.
            socrata_password (str): The password for authentication.
            socrata_dataset_id (str): The ID of the Socrata dataset.
            csv_path (str, optional): The path to save the downloaded data as a CSV file. Defaults to None.
            pkl_path (str, optional): The path to save the downloaded data as a pickle file. Defaults to None.
        """
        logging.info("Starting data download.")

        self.catalan_elections_data_downloader = DownloadCatalanElectionsData(
            socrata_domain,
            socrata_app_token,
            socrata_username,
            socrata_password,
            socrata_dataset_id,
            csv_path,
            pkl_path,
        )

    def download_catalan_elections_data(self) -> None:
        """
        Downloads the Catalan elections data using the configured parameters.
        """
        self.catalan_elections_data_downloader.download_catalan_elections_data()
