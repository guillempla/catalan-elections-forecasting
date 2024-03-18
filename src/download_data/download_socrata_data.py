"""
Download data from Socrata Open Data API (SODA) and save it as a CSV file.

This module provides a class, DownloadCatalanElectionsData,
that allows you to download data from the Socrata Open Data API (SODA) and save it as a CSV file.
It also provides functionality to convert data types and load the data into a PostgreSQL database.

Example usage:
    downloader = DownloadCatalanElectionsData(
        socrata_domain='example.domain.com',
        socrata_app_token='your_app_token',
        socrata_username='your_username',
        socrata_password='your_password',
        dataset_id='your_dataset_id',
        csv_path='path/to/save/data.csv'
    )
    downloader.download_catalan_elections_data()
    downloader.load_into_postgres(
        username='your_postgres_username',
        password='your_postgres_password',
        host='your_postgres_host',
        db_name='your_database_name',
        table_name='your_table_name'
    )
"""

from typing import Optional
import logging
import pandas as pd
from sodapy import Socrata
from sqlalchemy import create_engine
from utils.rw_files import save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - Download Socrata Data - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DownloadCatalanElectionsData:
    def __init__(
        self,
        socrata_domain: str,
        socrata_app_token: str,
        socrata_username: str,
        socrata_password: str,
        dataset_id: str,
        csv_path: Optional[str] = None,
        pkl_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the DownloadCatalanElectionsData object.

        Args:
            socrata_domain (str): The domain of the Socrata API.
            socrata_app_token (str): The app token for accessing the Socrata API.
            socrata_username (str): The username for authentication.
            socrata_password (str): The password for authentication.
            dataset_id (str): The ID of the dataset to download.
            csv_path (Optional[str], optional): The path to save the downloaded data as a CSV file.
                Defaults to None.
            pkl_path (Optional[str], optional): The path to save the downloaded data as a CSV file.
                Defaults to None.
        """
        self.client: Socrata = self.initialize_client(
            socrata_domain, socrata_app_token, socrata_username, socrata_password
        )
        self.dataset_id: str = dataset_id
        self.csv_path: Optional[str] = csv_path
        self.pkl_path: Optional[str] = pkl_path
        self.path: Optional[str] = (
            self.csv_path.replace(".csv", "") if self.csv_path else None
        )
        self.results_df: Optional[pd.DataFrame] = None
        self.nrows: Optional[int] = None

    def download(self) -> None:
        """
        Download the Catalan elections data from the Socrata API.

        This method retrieves the data from the Socrata API, converts the data types,
        and saves the data as a CSV file if a path is provided.
        """
        logging.info("Starting download from Socrata Open Data API (SODA).")
        self.nrows = self.get_row_count()
        self.results_df = self.get_data()
        self.convert_data_types()
        save_csv = self.csv_path is not None
        save_pickle = self.pkl_path is not None
        save_data(
            self.results_df, self.path, save_csv=save_csv, save_pickle=save_pickle
        )

    def initialize_client(
        self, domain: str, app_token: str, username: str, password: str
    ) -> Socrata:
        """
        Initialize the Socrata client.

        Args:
            domain (str): The domain of the Socrata API.
            app_token (str): The app token for accessing the Socrata API.
            username (str): The username for authentication.
            password (str): The password for authentication.

        Returns:
            Socrata: The initialized Socrata client.
        """
        client = Socrata(
            domain=domain,
            app_token=app_token,
            username=username,
            password=password,
        )
        return client

    def get_row_count(self) -> int:
        """
        Get the row count of the dataset.

        Returns:
            int: The number of rows in the dataset.
        """
        # Query for row count
        results = self.client.get(self.dataset_id, select="count(*)")
        nrows = int(results[0]["count"])
        return nrows

    def get_data(self) -> pd.DataFrame:
        """
        Get the data from the dataset.

        Returns:
            pd.DataFrame: The downloaded data as a pandas DataFrame.
        """
        results = self.client.get(self.dataset_id, limit=self.nrows)
        return pd.DataFrame.from_records(results)

    def convert_data_types(self) -> None:
        """
        Convert the data types of the DataFrame.

        This method converts numeric columns to numeric types, datetime columns to datetime types, and timedelta columns to timedelta types.
        """
        for column in self.results_df.columns:
            self.results_df[column] = pd.to_numeric(
                self.results_df[column], errors="ignore"
            )
            if pd.api.types.is_datetime64_any_dtype(self.results_df[column]):
                self.results_df[column] = pd.to_datetime(
                    self.results_df[column], errors="ignore"
                )
            if pd.api.types.is_timedelta64_dtype(self.results_df[column]):
                self.results_df[column] = pd.to_timedelta(
                    self.results_df[column], errors="ignore"
                )
        self.results_df = self.results_df.convert_dtypes()

    def load_into_postgres(
        self, username: str, password: str, host: str, db_name: str, table_name: str
    ) -> None:
        """
        Load the data into a PostgreSQL database.

        Args:
            username (str): The username for PostgreSQL authentication.
            password (str): The password for PostgreSQL authentication.
            host (str): The host of the PostgreSQL database.
            db_name (str): The name of the PostgreSQL database.
            table_name (str): The name of the table to load the data into.
        """
        logging.info("Saving data into PostgreSQL.")
        db_url = f"postgresql://{username}:{password}@{host}:5432/{db_name}"
        engine = create_engine(db_url)
        self.results_df.to_sql(table_name, engine, if_exists="replace", index=False)
