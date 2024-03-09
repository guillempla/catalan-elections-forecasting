"""
Module for downloading data.
"""

import logging
from typing import List
from .download_socrata_data import DownloadCatalanElectionsData
from .download_censal_sections_gis_data import DownloadCensalSectionsGisData


class DownloadData:
    """
    Class for downloading data.

    Args:
        datset_configs (List[dict]): A list of dictionaries containing the configuration.
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

        self.downloaders: List[
            DownloadCensalSectionsGisData | DownloadCatalanElectionsData
        ] = []
        for dataset_config in dataset_configs:
            dataset_type = dataset_config["dataset_type"]
            if dataset_type == "gis":
                self.downloaders.append(
                    DownloadCensalSectionsGisData(
                        dataset_config["year"],
                        dataset_config["output_path"],
                    )
                )
            if dataset_type == "socrata":
                self.downloaders.append(
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

        for downloader in self.downloaders:
            downloader.download()
