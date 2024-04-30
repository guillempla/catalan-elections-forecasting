"""
Download INE Mean Income Data

This script downloads the mean income data from the INE website and saves it in the data/raw folder.
"""

import logging
from pathlib import Path
from typing import List
from utils.rw_files import download_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Download INE Mean Income Data - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# create a constant dictionary that stores a URL for each year and each province
URLS = {
    "Barcelona": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/30896.csv?nocab=1",
    "Girona": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31016.csv?nocab=1",
    "Lleida": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31079.csv?nocab=1",
    "Tarragona": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/31223.csv?nocab=1",
}


class DownloadMeanIncomeData:
    """
    A class for downloading mean income data from the INE website.

    Attributes:
        provinces (List[str]): The list of provinces for which to download the data.
        output_path (str): The path to save the data.

    Methods:
        __init__(self, provinces=None, output_path="../data/raw/") -> None:
            Initialize the DownloadMeanIncomeData object.
        download_data(self) -> None:
            Download the mean income data from the INE website and save it in the data/raw folder.
        run(self) -> None:
            Run the data download process.
    """

    def __init__(
        self, provinces: List[str] = None, output_path: str = "../data/raw/"
    ) -> None:
        """
        Initialize the DownloadMeanIncomeData object.

        Args:
            provinces (List[str], optional): The list of provinces for which to download the data.
                Defaults to ["Barcelona", "Girona", "Lleida", "Tarragona"].
            output_path (str, optional): The path to save the data.
                Defaults to "../data/raw/".
        """
        if provinces is None:
            provinces = ["Barcelona", "Girona", "Lleida", "Tarragona"]
        self.provinces = provinces
        self.output_path = output_path

    def download(self) -> None:
        """
        Download the mean income data from the INE website and save it in the data/raw folder.
        """
        for province in self.provinces:
            logging.info("Downloading the mean income data for %s...", province)
            download_file(
                URLS[province],
                Path(self.output_path, f"mean_income_{province.lower()}.csv"),
            )
