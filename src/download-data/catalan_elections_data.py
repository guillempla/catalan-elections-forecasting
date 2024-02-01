"""
Download data from Socrata Open Data API (SODA) and save it as a CSV file.
"""
import logging
import pandas as pd
from sodapy import Socrata
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DownloadCatalanElectionsData:
    def __init__(
        self,
        socrata_domain,
        socrata_app_token,
        socrata_username,
        socrata_password,
        dataset_id,
        csv_path=None,
    ):
        logging.info("Starting download from Socrata Open Data API (SODA).")

        self.client = self.initialize_client(
            socrata_domain, socrata_app_token, socrata_username, socrata_password
        )
        self.dataset_id = dataset_id
        self.path = csv_path
        self.results_df = None
        self.nrows = None

    def download_catalan_elections_data(self):
        self.nrows = self.get_row_count()
        self.results_df = self.get_data()
        self.convert_data_types()
        if self.path is not None:
            self.save_as_csv(self.path)

    def initialize_client(self, domain, app_token, username, password):
        client = Socrata(
            domain=domain,
            app_token=app_token,
            username=username,
            password=password,
        )
        return client

    def get_row_count(self):
        # Query for row count
        results = self.client.get(self.dataset_id, select="count(*)")
        nrows = int(results[0]["count"])
        return nrows

    def get_data(self):
        results = self.client.get(self.dataset_id, limit=self.nrows)
        return pd.DataFrame.from_records(results)

    def convert_data_types(self):
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

    def save_as_csv(self, path):
        logging.info("Saving data as CSV.")
        self.results_df.to_csv(path, index=False)

    def load_into_postgres(self, username, password, host, db_name, table_name):
        logging.info("Saving data into PostgreSQL.")
        db_url = "postgresql://%s:%s@%s:5432/%s" % (username, password, host, db_name)
        engine = create_engine(db_url)
        self.results_df.to_sql(table_name, engine, if_exists="replace", index=False)
