"""
Download data from Socrata Open Data API (SODA) and save it as a CSV file.
"""
import os
import pandas as pd
from sodapy import Socrata

SOCRATA_DOMAIN = os.environ.get("SOCRATA_DOMAIN", "analisi.transparenciacatalunya.cat")
SOCRATA_DATASET_ID = os.environ.get("SOCRATA_DATASET_ID", "ntc4-rnwr")
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")
SOCRATA_EMAIL = os.environ.get("SOCRATA_EMAIL")
SOCRATA_PASSWORD = os.environ.get("SOCRATA_PASSWORD")

client = Socrata(
    SOCRATA_DOMAIN,
    app_token=SOCRATA_APP_TOKEN,
    username=SOCRATA_EMAIL,
    password=SOCRATA_PASSWORD,
)

results = client.get(SOCRATA_DATASET_ID, limit=20000000)

results_df = pd.DataFrame.from_records(results)

results_df.to_csv("../../data/data.csv", index=False)
