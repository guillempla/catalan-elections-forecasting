"""
Transform censal sections data and results data into a single output dataframe
"""

from typing import List, Tuple

import logging

import numpy as np
import pandas as pd

# import numpy as np
import geopandas as gpd
from utils.rw_files import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def select_important_parties(df):
    # Determine the top 10 parties per election
    top_parties = (
        set()
    )  # This will hold the joined codes of top parties across all elections
    grouped = df.groupby("nom_eleccio")

    for _, group in grouped:
        # Sort parties in each election by 'votes' and select the top 10
        top_in_election = (
            group.sort_values(by="vots", ascending=False)
            .head(10)["joined_code"]
            .unique()
        )
        top_parties.update(top_in_election)

    # Identify rows that are not in the top parties
    mask = ~df["joined_code"].isin(top_parties)

    # Replace details for non-top parties to aggregate them as 'Other Parties'
    df.loc[mask, "joined_code"] = 999999999
    df.loc[mask, "joined_name"] = "Other Parties"
    df.loc[mask, "joined_color"] = "#535353"
    df.loc[mask, "joined_abbr"] = "Other"

    return df


class TransformData:
    """
    Transform censal sections data and results data into a single output dataframe
    """

    def __init__(
        self,
        censal_sections_path: str,
        results_path: str,
        output_path: str = "../data/output/censal_sections_results",
    ) -> None:
        """
        Initialize the class with the paths to the censal sections and results data
        """
        self.censal_sections_gdf = load_data(censal_sections_path)
        self.results_df = load_data(results_path)
        self.output_path = output_path

    def transform_data(self) -> None:
        """
        Transform censal sections data and results data into a single output dataframe
        """
        important_parties = select_important_parties(self.results_df)

        # Pivot the DataFrame
        logging.info("Pivoting the DataFrame")
        results_wide_df = important_parties.pivot_table(
            index="mundissec",
            columns=["joined_code", "id_eleccio"],
            values=[
                "vots",
                "votants_percentage",
                "vots_valids_percentage",
                "cens_electoral_percentage",
            ],
        )
        logging.info("Pivoting done")

        # Flattening the MultiIndex Columns
        results_wide_df.columns = [
            "_".join(map(str, col)).strip() for col in results_wide_df.columns.values
        ]

        # Replace 'Infinity' values with a suitable finite number (e.g., 0)
        results_wide_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        logging.info("Replacing 'Infinity' values done")

        # save_data(results_wide_df, "../data/output/results_wide_df")

        # Merge censal sections and results data
        merged_gdf = self.censal_sections_gdf.merge(
            results_wide_df, left_on="MUNDISSEC", right_on="mundissec", how="inner"
        )

        logging.info("Merging done")

        # Convert all columns of type 'Float64Dtype' to 'float64' to avoid problems with GeoPandas
        for col in merged_gdf.select_dtypes(include=["Float64"]).columns:
            merged_gdf[col] = merged_gdf[col].fillna(0.0).astype("float64")

        logging.info("Converting columns to float64 done")

        merged_gdf.to_file("../data/output/merged_data.geojson", driver="GeoJSON")
        logging.info("Data transformation done")
