"""
Transform censal sections data and results data into a single output dataframe
"""

from typing import List, Tuple

import re
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


def select_important_parties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the top 10 parties per election and aggregate the rest as 'Other Parties'

    Args:
        df (pandas.DataFrame): DataFrame containing the election results data

    Returns:
        pandas.DataFrame: DataFrame with the top 10 parties per election and 'Other Parties' aggregated
    """
    logging.info("Selecting important parties")

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


def add_missing_party_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Adds missing party columns to the GeoDataFrame.
    Columns are identified by the pattern `VARIABLENAME_PARTYCODE_ELECTIONSID`.
    Missing columns are added with their values filled as `0.0`.

    :param gdf: The original GeoDataFrame with existing columns.
    :return: The GeoDataFrame with all possible party columns present.
    """
    logging.info("Add missing party columns")

    # Function to parse columns into three components
    def parse_column_name(col_name):
        parts = col_name.split("_")
        election_id = parts[-1]
        party_code = parts[-2]
        variable_name = "_".join(parts[:-2])  # Handle underscores within VARIABLENAME
        return variable_name, party_code, election_id

    # Extract unique patterns
    existing_patterns = {parse_column_name(col) for col in gdf.columns if "_" in col}

    # Create sets for unique components
    unique_vars = {x[0] for x in existing_patterns}
    unique_parties = {x[1] for x in existing_patterns}
    unique_elections = {x[2] for x in existing_patterns}

    # Generate all combinations
    all_combinations = {
        f"{var}_{party}_{election}"
        for var in unique_vars
        for party in unique_parties
        for election in unique_elections
    }

    # Determine which combinations are missing
    existing_columns = set(gdf.columns)
    missing_columns = all_combinations - existing_columns

    # Create a DataFrame to hold the missing columns
    missing_df = pd.DataFrame(0.0, index=gdf.index, columns=sorted(missing_columns))

    # Concatenate the new columns to the original GeoDataFrame
    gdf = pd.concat([gdf, missing_df], axis=1)

    return gdf


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

        Args:
            censal_sections_path (str): Path to the censal sections data file
            results_path (str): Path to the election results data file
            output_path (str, optional): Path to save the output dataframe. Defaults to "../data/output/censal_sections_results".
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

        merged_gdf = add_missing_party_columns(merged_gdf)

        logging.info("Saving output data with only past votes data")

        only_votes_df = merged_gdf.drop(columns=["geometry"])
        save_data(only_votes_df, "../data/output/only_votes.csv")

        logging.info("Saving output data with only past votes data done")

        # merged_gdf.to_file("../data/output/merged_data.geojson", driver="GeoJSON")
        logging.info("Data transformation done")
