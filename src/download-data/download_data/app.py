"""
Download data from Socrata Open Data API (SODA) and save it as a CSV file.
"""
import os
import pprint
import logging
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Starting download from Socrata Open Data API (SODA).")

logging.info("Loading environment variables.")
load_dotenv()
SOCRATA_DOMAIN = os.environ.get("SOCRATA_DOMAIN", "analisi.transparenciacatalunya.cat")
SOCRATA_DATASET_ID = os.environ.get("SOCRATA_DATASET_ID", "ntc4-rnwr")
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")
SOCRATA_EMAIL = os.environ.get("SOCRATA_EMAIL")
SOCRATA_PASSWORD = os.environ.get("SOCRATA_PASSWORD")


logging.info("Downloading data from SODA.")
client = Socrata(
    SOCRATA_DOMAIN,
    app_token=SOCRATA_APP_TOKEN,
    username=SOCRATA_EMAIL,
    password=SOCRATA_PASSWORD,
)

# Query for row count
results = client.get(SOCRATA_DATASET_ID, select="count(*)")
nrows = int(results[0]["count"])
logging.info("Number of rows: %s", nrows)

results = client.get(SOCRATA_DATASET_ID, limit=nrows)
results_df = pd.DataFrame.from_records(results)

# Automatically infer and convert data types for each column
for column in results_df.columns:
    results_df[column] = pd.to_numeric(results_df[column], errors="ignore")
    if pd.api.types.is_datetime64_any_dtype(results_df[column]):
        results_df[column] = pd.to_datetime(results_df[column], errors="ignore")
    if pd.api.types.is_timedelta64_dtype(results_df[column]):
        results_df[column] = pd.to_timedelta(results_df[column], errors="ignore")

results_df = results_df.convert_dtypes()
print(results_df.dtypes)


logging.info("Saving data as CSV.")
results_df.to_csv("data/raw/catalan-elections-data.csv", index=False)


logging.info("Loading data into PostgreSQL.")
db_url = "postgresql://postgres:+Pst1998gpb@localhost:5432/CatalanElectionsDB"
engine = create_engine(db_url)

table_name = "elections"
results_df.to_sql(table_name, engine, if_exists="replace", index=False)
