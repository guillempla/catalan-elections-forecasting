"""
Transform censal sections data and results data into a single output dataframe
"""

from typing import List, Tuple

import logging

import pandas as pd

# import numpy as np
import geopandas as gpd
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def filter_dataframe_by_column(
    data: pd.DataFrame, column: str, values: List[str]
) -> pd.DataFrame:
    """
    Filter data by column values
    """
    return data[data[column].isin(values)]


def create_unique_id(row):
    for attribute in ["vots", "escons", "percentage"]:
        row[f"{row['clean_party_abbr'].lower()}_{row['party_code']}_{attribute}"] = row[
            attribute
        ]
    return row


class TransformData:
    """
    Transform censal sections data and results data into a single output dataframe
    """

    def __init__(self, censal_sections_path: str, results_path: str) -> None:
        """
        Initialize the class with the paths to the censal sections and results data
        """
        self.censal_sections_gdf = load_data(censal_sections_path)
        self.results_df = load_data(results_path)

    def transform_data(self) -> None:
        """
        Transform censal sections data and results data into a single output dataframe
        """
        # Filter results data by the column "id_nivell_territorial". Keep only "SE" values
        self.results_df = filter_dataframe_by_column(
            self.results_df, "id_nivell_territorial", ["SE"]
        )

        self.results_df = self.results_df.apply(create_unique_id, axis=1)

        # Select only the necessary columns (electoral section ID + newly created columns)
        columns_to_keep = ["mundissec"] + [
            col
            for col in self.results_df
            if col.startswith(
                tuple(
                    self.results_df["clean_party_abbr"].str.lower()
                    + "_"
                    + self.results_df["party_code"].astype(str)
                )
            )
        ]
        results_wide = self.results_df[columns_to_keep]

        # Pivot the DataFrame
        results_wide_df = results_wide.pivot_table(index="section_id", aggfunc="first")
        print(results_wide_df)

        # # Merge censal sections and results data
        # output = pd.merge(censal_sections, results, on="section_id", how="inner")

        # save_data(output, self.results_path)
        pass
