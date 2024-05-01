"""
Download INE Data

This script downloads data from the INE website and saves it in the data/raw folder.
"""

import logging
from pathlib import Path
from typing import Dict
from utils.rw_files import download_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Download INE Data - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DownloadIneData:
    """
    A class for downloading mean income data from the INE website.

    Attributes:
        output_path (str): The path to save the data.

    Methods:
        __init__(self, provinces=None, output_path="../data/raw/") -> None:
            Initialize the DownloadMeanIncomeData object.
        download_data(self) -> None:
            Download data from the INE website and save it in the data/raw folder.
        run(self) -> None:
            Run the data download process.
    """

    def __init__(
        self,
        data_type: str,
        urls_info: Dict[str, str],
        output_path: str = "../data/raw/",
    ) -> None:
        """
        Initialize the DownloadMeanIncomeData object.

        Args:
            output_path (str, optional): The path to save the data.
                Defaults to "../data/raw/".
        """
        self.data_type = data_type
        self.urls_info = urls_info
        self.output_path = output_path

    def download(self) -> None:
        """
        Download data from the INE website and save it in the data/raw folder.
        """
        for url_info in self.urls_info:
            province = url_info.get("province")
            year = url_info.get("year")
            url = url_info.get("url")
            logging.info(
                "Downloading %s data for %s %s...", self.data_type, province, year
            )
            file_name = self.create_filename(self.data_type, province, year)
            download_file(
                url,
                Path(self.output_path, file_name),
            )

    @staticmethod
    def create_filename(data_type: str, province: str = None, year: str = None) -> str:
        """
        Create a filename for the downloaded data.

        Args:
            data_type (str): The type of data.
            province (str): The province of the data.
            year (str): The year of the data.

        Returns:
            str: The filename.
        """
        file_name = data_type
        if province:
            file_name = file_name + "_" + province
        if year:
            file_name = file_name + "_" + year
        return file_name
