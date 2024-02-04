import logging
from .catalan_elections_data import DownloadCatalanElectionsData


class DownloadData:
    def __init__(
        self,
        socrata_domain,
        socrata_app_token,
        socrata_username,
        socrata_password,
        socrata_dataset_id,
        csv_path=None,
        pkl_path=None,
    ):
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

    def download_catalan_elections_data(self):
        self.catalan_elections_data_downloader.download_catalan_elections_data()
